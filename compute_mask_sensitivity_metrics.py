"""
Inpainting Metrics Mask Sensitivity Analysis

Computes the following metrics for each inpainting method:
  1. PSNR (Peak Signal-to-Noise Ratio)
  2. SSIM (Structural Similarity Index)
  3. T-LPIPS (Temporal Learned Perceptual Image Patch Similarity)
  4. FID (Fréchet Inception Distance) - per-scenario approximation
  5. Temporal Consistency (flow-warped frame difference)

Mask Sensitivity Analysis:
  - Shrink mask by 3px and 7px (erode)
  - Grow mask by 3px and 7px (dilate)
  - Shift mask randomly by ±5px and ±10px
  - Reports per-class and aggregate sensitivity

Outputs:
  - CSV summary tables
  - JSON raw results
  - Matplotlib graphs (bar charts, heatmaps, line plots)
"""

import cv2
import torch
import lpips
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
import warnings

warnings.filterwarnings('ignore')

DATA_ROOT = Path(__file__).parent / 'DAVIS'
COMPARISON_DIR = Path(__file__).parent / 'MVP-Boeing-Scholars'
METHODS = {'LaMa': COMPARISON_DIR / 'Output_Lama' / 'inpainted'}
SCENARIOS = ['bus', 'car-roundabout', 'car-shadow', 'car-turn', 'classic-car', 'crossing']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MASK_PERTURBATIONS = {
    'original':   {'type': 'none'},
    'shrink_3px': {'type': 'erode',  'pixels': 3},
    'shrink_7px': {'type': 'erode',  'pixels': 7},
    'grow_3px':   {'type': 'dilate', 'pixels': 3},
    'grow_7px':   {'type': 'dilate', 'pixels': 7},
    'shift_5px':  {'type': 'shift',  'pixels': 5},
    'shift_10px': {'type': 'shift',  'pixels': 10},
}

def perturb_mask(mask: np.ndarray, config: dict, rng: np.random.RandomState) -> np.ndarray:
    """Applies erode, dilate, or shift perturbations to a binary mask."""
    if config['type'] == 'none': return mask.copy()
    
    px = config.get('pixels', 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    
    if config['type'] == 'erode': return cv2.erode(mask, kernel, iterations=1)
    if config['type'] == 'dilate': return cv2.dilate(mask, kernel, iterations=1)
    if config['type'] == 'shift':
        M = np.float32([[1, 0, rng.randint(-px, px + 1)], [0, 1, rng.randint(-px, px + 1)]])
        return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    return mask.copy()

def load_frames(path: Path) -> list:
    """Loads images from a folder into a list of arrays."""
    return [cv2.imread(str(f)) for f in sorted(path.glob('*.[jp][pn]*g')) if cv2.imread(str(f)) is not None]

class MetricsComputer:
    def __init__(self, device=DEVICE):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    def _get_bbox(self, mask, pad=0):
        """Extracts bounding box coordinates from a mask."""
        ys, xs = np.where(mask > 127)
        if len(ys) < 50: return None
        return max(0, ys.min()-pad), min(mask.shape[0], ys.max()+1+pad), \
               max(0, xs.min()-pad), min(mask.shape[1], xs.max()+1+pad)

    def _to_tensor(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (torch.from_numpy(rgb).float() / 255.0 * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _get_optical_flow_warped(self, prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        map_x, map_y = (gx - flow[..., 0]).astype(np.float32), (gy - flow[..., 1]).astype(np.float32)
        return cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def compute_psnr_ssim(self, orig, inp, mask):
        is_masked = mask > 127
        if np.sum(is_masked) < 50: return np.nan, np.nan
        
        mse = np.mean((orig[is_masked].astype(float) - inp[is_masked].astype(float)) ** 2)
        psnr = 100.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
        
        bbox = self._get_bbox(mask)
        if not bbox: return psnr, np.nan
        y1, y2, x1, x2 = bbox
        
        try:
            ssim = ssim_metric(orig[y1:y2, x1:x2], inp[y1:y2, x1:x2], channel_axis=2, data_range=255)
        except: ssim = np.nan
        return psnr, ssim

    def compute_lpips(self, orig, inp, mask):
        bbox = self._get_bbox(mask)
        if not bbox or (bbox[1]-bbox[0]) < 64 or (bbox[3]-bbox[2]) < 64: return np.nan
        y1, y2, x1, x2 = bbox
        
        with torch.no_grad():
            t_orig, t_inp = self._to_tensor(orig[y1:y2, x1:x2]), self._to_tensor(inp[y1:y2, x1:x2])
            return self.lpips_model(t_orig, t_inp, normalize=True).item()

    def compute_temporal_metrics(self, prev_frame, curr_frame, mask):
        bbox = self._get_bbox(cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2))
        if not bbox or (bbox[1]-bbox[0]) < 64 or (bbox[3]-bbox[2]) < 64: return np.nan, np.nan
        y1, y2, x1, x2 = bbox
        
        warped_prev = self._get_optical_flow_warped(prev_frame, curr_frame)
        
        with torch.no_grad():
            t_curr, t_prev = self._to_tensor(curr_frame[y1:y2, x1:x2]), self._to_tensor(warped_prev[y1:y2, x1:x2])
            tlpips = self.lpips_model(t_curr, t_prev, normalize=True).item()
            
        is_masked = mask > 127
        tc = np.mean(np.abs(warped_prev[is_masked].astype(float) - curr_frame[is_masked].astype(float)))
        
        return tlpips, float(tc)

def run_analysis(data_root, methods, scenarios, perturbations, output_dir, max_frames=None, sample_stride=1):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_computer = MetricsComputer(DEVICE)
    rng = np.random.RandomState(42)
    all_results = []

    for method_name, method_dir in methods.items():
        print(f"\nMethod: {method_name}")
        
        for scenario in scenarios:
            orig_frames = load_frames(data_root / 'JPEGImages' / '480p' / scenario)
            inp_frames = load_frames(method_dir / scenario)
            masks = [cv2.threshold(cv2.imread(str(f), 0), 0, 255, cv2.THRESH_BINARY)[1] 
                     for f in sorted((data_root / 'Annotations' / '480p' / scenario).glob('*.png'))]
            
            if not orig_frames or not inp_frames or not masks: continue
            
            n_frames = min(len(orig_frames), len(inp_frames), len(masks))
            indices = list(range(0, n_frames, sample_stride))[:max_frames]

            for p_name, p_cfg in perturbations.items():
                res = {'Method': method_name, 'Scenario': scenario, 'Perturbation': p_name, 'Frames': len(indices),
                       'PSNR': [], 'SSIM': [], 'LPIPS': [], 'TLPIPS': [], 'TC': []}

                for pos, idx in enumerate(indices):
                    mask_p = cv2.threshold(perturb_mask(masks[idx], p_cfg, rng), 127, 255, cv2.THRESH_BINARY)[1]
                    
                    p, s = metrics_computer.compute_psnr_ssim(orig_frames[idx], inp_frames[idx], mask_p)
                    lp = metrics_computer.compute_lpips(orig_frames[idx], inp_frames[idx], mask_p)
                    res['PSNR'].append(p); res['SSIM'].append(s); res['LPIPS'].append(lp)

                    if pos > 0:
                        tl, tc = metrics_computer.compute_temporal_metrics(inp_frames[indices[pos-1]], inp_frames[idx], mask_p)
                        res['TLPIPS'].append(tl); res['TC'].append(tc)

                row = {k: np.nanmean(v) if v else np.nan for k, v in res.items() if isinstance(v, list)}
                row.update({k: v for k, v in res.items() if not isinstance(v, list)})
                all_results.append(row)

    return pd.DataFrame(all_results)

def generate_all_graphs(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ['PSNR', 'SSIM', 'LPIPS', 'TLPIPS', 'TC']
    
    df_orig = df[df['Perturbation'] == 'original']
    df_orig.groupby('Method')[metrics].mean().plot(kind='bar', subplots=True, layout=(1, 5), figsize=(20, 4), title="Baseline Metrics")
    plt.savefig(output_dir / '01_baseline_metrics.png', bbox_inches='tight')
    plt.close()

    for metric in metrics:
        df.pivot_table(index='Perturbation', columns='Method', values=metric, aggfunc='mean').plot(marker='o', figsize=(8, 5))
        plt.title(f"Sensitivity Analysis - {metric}")
        plt.savefig(output_dir / f'02_sensitivity_{metric}.png', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=str(DATA_ROOT))
    parser.add_argument('--comparison-dir', default=str(COMPARISON_DIR))
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--sample-stride', type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.comparison_dir) / 'metrics_sensitivity'
    
    t0 = time.perf_counter()
    df = run_analysis(Path(args.data_root), METHODS, SCENARIOS, MASK_PERTURBATIONS, out_dir, args.max_frames, args.sample_stride)
    
    if not df.empty:
        df.to_csv(out_dir / 'sensitivity_results.csv', index=False)
        df.groupby(['Method', 'Perturbation']).mean(numeric_only=True).to_csv(out_dir / 'agg_method_perturbation.csv')
        generate_all_graphs(df, out_dir / 'graphs')
        
    print(f"Done in {time.perf_counter() - t0:.1f}s. Results saved to {out_dir}")