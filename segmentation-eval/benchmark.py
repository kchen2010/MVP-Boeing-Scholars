"""
Generic segmentation benchmarking library.

Consumers provide two callbacks:

  pred_loader(img_path) -> PredResult
    Runs model inference on the image and returns the predicted instance masks
    plus the resolution at which they were produced.  All model-specific logic
    (class filtering, tensor→numpy, resizing, etc.) lives here.

  gt_loader(img_path, mask_h, mask_w) -> list[np.ndarray]
    Returns one boolean (mask_h, mask_w) array per GT instance in the image,
    already at the requested evaluation resolution.  The resolution is taken
    directly from the pred_loader result, so GT and predictions are always
    compared at the same scale regardless of which model is used.
    Return an empty list to skip the image.

Example — yolo26n-seg on Mapillary Vistas v2.0, car class::

    from benchmark import (
        run_benchmark, make_yolo_loader,
        extract_instances_16bit, print_report, generate_charts,
    )
    from ultralytics import YOLO
    from pathlib import Path
    import numpy as np, cv2
    from PIL import Image

    INSTANCES_DIR = Path("dataset/validation/v2.0/instances")
    CAR_LABEL_ID = 108

    def mapillary_car_loader(img_path, h, w):
        inst_path = INSTANCES_DIR / f"{img_path.stem}.png"
        if not inst_path.exists():
            return []
        raw = np.array(Image.open(inst_path), dtype=np.uint16)
        resized = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        return extract_instances_16bit(resized, CAR_LABEL_ID)

    pred_loader = make_yolo_loader(YOLO("yolo26n-seg.pt"), classes=[2])
    image_files = sorted(Path("dataset/validation/images").glob("*.jpg"))
    results = run_benchmark(image_files, pred_loader, mapillary_car_loader)
    print_report(results, label="yolo26n-seg · Mapillary · car")
    generate_charts(results, output_dir=Path("out"), label="yolo26n-seg · Mapillary · car")

Metrics computed
----------------
Semantic metrics (per-image combined mask, averaged across images):
  IoU, Dice, Boundary F1

Instance metric — Panoptic Quality (global across all images):
  SQ  = mean IoU of TP pairs
  RQ  = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
  PQ  = SQ * RQ
  A predicted mask is a TP only when IoU with its matched GT >= 0.5.

Pixel-level confusion matrix (global):
  Precision, Recall, F1 treating the class of interest as the positive.

All metrics are computed at the resolution returned by pred_loader; gt_loader
is called with that same resolution so the two are always spatially aligned.
"""

import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from metrics import iou, dice, boundary_f_score  # noqa: E402


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class PredResult(NamedTuple):
    """Return type for pred_loader callbacks."""
    masks:  list[np.ndarray]  # boolean (height, width) arrays, one per instance
    height: int               # evaluation resolution rows
    width:  int               # evaluation resolution columns


PredLoader = Callable[[Path], PredResult]
GtLoader   = Callable[[Path, int, int], list[np.ndarray]]


# ---------------------------------------------------------------------------
# Size-bin constants (used by run_benchmark and plot_histogram)
# ---------------------------------------------------------------------------

# Log-spaced bins covering the wide range of mask sizes.
# Defined in pixels at the ~480×640 evaluation resolution.
BIN_EDGES  = [0, 25, 100, 400, 1600, 6400, np.inf]
BIN_LABELS = ["1–25", "25–100", "100–400", "400–1600", "1600–6400", "6400+"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_instances_16bit(inst_small: np.ndarray, label_id: int) -> list[np.ndarray]:
    """
    Extract per-instance boolean masks from a 16-bit instance image that uses
    the encoding ``class_id = pixel_value >> 8`` (e.g. Mapillary Vistas).

    Returns a list of boolean (H, W) arrays, one per instance of ``label_id``.
    """
    label_arr  = inst_small >> 8
    class_mask = label_arr == label_id
    if not class_mask.any():
        return []
    return [inst_small == v for v in np.unique(inst_small[class_mask])]


def iou_matrix(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    """Compute an (n_gt, n_pred) IoU matrix via vectorised matrix multiplication."""
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float32)

    gt_flat   = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float32)
    pred_flat = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float32)

    inter    = gt_flat @ pred_flat.T
    gt_sum   = gt_flat.sum(axis=1, keepdims=True)
    pred_sum = pred_flat.sum(axis=1, keepdims=True)
    union    = gt_sum + pred_sum.T - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def greedy_match(iou_mat: np.ndarray) -> dict[int, int]:
    """
    Greedily match GT rows to prediction columns by descending IoU.
    Returns a dict {gt_index: pred_index}.
    """
    remaining = iou_mat.copy()
    matches: dict[int, int] = {}
    for _ in range(min(iou_mat.shape)):
        if remaining.max() <= 0:
            break
        i, j = np.unravel_index(remaining.argmax(), remaining.shape)
        matches[int(i)] = int(j)
        remaining[i, :] = -1.0
        remaining[:, j] = -1.0
    return matches


def mask_to_pixels(binary: np.ndarray) -> np.ndarray:
    """Convert a boolean (H, W) mask to an (N, 2) [row, col] coordinate array."""
    return np.column_stack(np.where(binary))


def pq_match(
    iou_mat: np.ndarray,
    threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Greedy PQ matching: accept pairs only when IoU >= threshold.

    Returns (tp_pairs, fp_indices, fn_indices).
    """
    n_gt, n_pred = iou_mat.shape
    remaining    = np.where(iou_mat >= threshold, iou_mat, 0.0).copy()
    matched_gt:   set[int] = set()
    matched_pred: set[int] = set()
    tp_pairs: list[tuple[int, int]] = []

    for _ in range(min(n_gt, n_pred)):
        if remaining.max() <= 0:
            break
        i, j = np.unravel_index(remaining.argmax(), remaining.shape)
        tp_pairs.append((int(i), int(j)))
        matched_gt.add(int(i))
        matched_pred.add(int(j))
        remaining[i, :] = 0.0
        remaining[:, j] = 0.0

    fn_indices = [i for i in range(n_gt)   if i not in matched_gt]
    fp_indices = [j for j in range(n_pred) if j not in matched_pred]
    return tp_pairs, fp_indices, fn_indices


# ---------------------------------------------------------------------------
# pred_loader factories
# ---------------------------------------------------------------------------

def make_yolo_loader(model, *, classes: list[int] | None = None) -> PredLoader:
    """
    Build a pred_loader for an ultralytics YOLO segmentation model.

    Parameters
    ----------
    model : ultralytics YOLO
        A loaded YOLO segmentation model instance.
    classes : list[int] or None
        YOLO class IDs to restrict inference to.  None = all classes.

    Notes
    -----
    When the model produces no masks for an image (no detections), the
    resolution is estimated from the image's original shape using YOLO's
    standard longest-side-to-640 scaling, matching the resolution that
    would have been used if detections had been found.
    """
    infer_kwargs = {} if classes is None else {"classes": classes}

    def loader(img_path: Path) -> PredResult:
        r = model(str(img_path), verbose=False, **infer_kwargs)[0]

        if r.masks is not None:
            mask_h, mask_w = r.masks.data.shape[1], r.masks.data.shape[2]
            masks = [
                mt.cpu().numpy().astype(bool)
                for mt in r.masks.data
                if mt.cpu().numpy().any()
            ]
        else:
            orig_h, orig_w = r.orig_shape
            scale  = 640.0 / max(orig_h, orig_w)
            mask_h = round(orig_h * scale)
            mask_w = round(orig_w * scale)
            masks  = []

        return PredResult(masks, mask_h, mask_w)

    return loader


def make_rfdetr_loader(
    model,
    *,
    classes: list[int] | None = None,
    threshold: float = 0.5,
    max_size: int = 640,
) -> PredLoader:
    """
    Build a pred_loader for an rfdetr segmentation model (e.g. RFDETRSegNano).

    Parameters
    ----------
    model : rfdetr RFDETRSegNano (or any rfdetr seg variant)
        A loaded rfdetr segmentation model instance.
    classes : list[int] or None
        COCO class IDs to keep.  None = all classes.
        RF-DETR uses 1-indexed COCO category IDs (car = 3), unlike
        YOLO which uses 0-indexed IDs (car = 2).
    threshold : float
        Detection confidence threshold passed to model.predict().
    max_size : int
        Longest side in pixels to resize input images to before inference.
        RF-DETR returns masks at the input resolution, so feeding full
        Mapillary images (2448×3264) would be very slow; resizing here
        keeps costs comparable to YOLO.  The returned height/width reflect
        the resized dimensions, so gt_loader is called at the same scale.
    """
    def loader(img_path: Path) -> PredResult:
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size  # PIL size is (width, height)
        scale = max_size / max(orig_w, orig_h)
        if scale < 1.0:
            img = img.resize((round(orig_w * scale), round(orig_h * scale)),
                             Image.LANCZOS)

        mask_w, mask_h = img.size  # after potential resize

        detections = model.predict(img, threshold=threshold)

        if detections.mask is None or len(detections.mask) == 0:
            return PredResult([], mask_h, mask_w)

        mask_arr = detections.mask  # (n, H, W) bool

        if classes is not None and detections.class_id is not None:
            keep = np.isin(detections.class_id, classes)
            mask_arr = mask_arr[keep]

        masks = [mask_arr[i] for i in range(len(mask_arr)) if mask_arr[i].any()]
        return PredResult(masks, mask_h, mask_w)

    return loader


# ---------------------------------------------------------------------------
# Core benchmark engine
# ---------------------------------------------------------------------------

def run_benchmark(
    image_files: Iterable[Path],
    pred_loader: PredLoader,
    gt_loader:   GtLoader,
) -> dict:
    """
    Run the full segmentation benchmark loop.

    Parameters
    ----------
    image_files : iterable of Path
        Paths to the images to evaluate.
    pred_loader : PredLoader
        ``(img_path,) -> PredResult``
        Runs model inference and returns predicted instance masks plus the
        resolution at which they were produced.
    gt_loader : GtLoader
        ``(img_path, mask_h, mask_w) -> list[np.ndarray]``
        Returns boolean (mask_h, mask_w) GT instance masks at the resolution
        used by pred_loader.  Return [] to skip an image.

    Returns
    -------
    dict with keys:
        all_iou, all_dice, all_bf1, all_seg_ratio  – per-image / per-pair lists
        pq_tp_count, pq_tp_iou_sum, pq_fp_count, pq_fn_count
        sq, rq, pq
        px_tp, px_fp, px_fn, px_tn
        px_precision, px_recall, px_f1
        images_with_gt, total_gt_instances, total_images
        instance_records  – list of (gt_size_px, iou, dice, boundary_f1) per GT
            instance, using unconstrained greedy matching; unmatched GTs scored 0
    """
    image_files = list(image_files)

    all_iou:       list[float] = []
    all_dice:      list[float] = []
    all_bf1:       list[float] = []
    all_seg_ratio: list[float] = []
    instance_records: list[tuple[int, float, float, float]] = []

    pq_tp_count:   int   = 0
    pq_tp_iou_sum: float = 0.0
    pq_fp_count:   int   = 0
    pq_fn_count:   int   = 0

    px_tp = px_fp = px_fn = px_tn = 0
    images_with_gt     = 0
    total_gt_instances = 0

    for img_path in tqdm(image_files, desc="Benchmarking"):
        pred = pred_loader(img_path)
        gt_masks = gt_loader(img_path, pred.height, pred.width)
        if not gt_masks:
            continue

        pred_masks = pred.masks
        images_with_gt     += 1
        total_gt_instances += len(gt_masks)

        # Semantic metrics: combine all instances into one mask per side
        gt_combined   = np.logical_or.reduce(gt_masks)
        pred_combined = (np.logical_or.reduce(pred_masks)
                         if pred_masks else np.zeros_like(gt_combined))
        gt_pix   = mask_to_pixels(gt_combined)
        pred_pix = (mask_to_pixels(pred_combined)
                    if pred_combined.any() else np.empty((0, 2), dtype=int))

        all_iou.append(iou(pred_pix, gt_pix))
        all_dice.append(dice(pred_pix, gt_pix))
        all_bf1.append(boundary_f_score(pred_pix, gt_pix) if pred_combined.any() else 0.0)

        # Pixel-level confusion matrix
        px_tp += int(( gt_combined &  pred_combined).sum())
        px_fp += int((~gt_combined &  pred_combined).sum())
        px_fn += int(( gt_combined & ~pred_combined).sum())
        px_tn += int((~gt_combined & ~pred_combined).sum())

        # PQ: instance-level matching
        iou_mat = iou_matrix(gt_masks, pred_masks)
        tp_pairs, fp_idx, fn_idx = pq_match(iou_mat, threshold=0.5)

        pq_tp_count   += len(tp_pairs)
        pq_tp_iou_sum += sum(float(iou_mat[i, j]) for i, j in tp_pairs)
        pq_fp_count   += len(fp_idx)
        pq_fn_count   += len(fn_idx)

        for gi, pi in tp_pairs:
            gt_px   = int(gt_masks[gi].sum())
            pred_px = int(pred_masks[pi].sum())
            all_seg_ratio.append(pred_px / gt_px if gt_px > 0 else 0.0)

        # Per-instance histogram records — unconstrained greedy match so every
        # GT gets its best available prediction score (not just IoU >= 0.5).
        hist_matches = greedy_match(iou_mat)
        for i, gt_mask in enumerate(gt_masks):
            gt_size = int(gt_mask.sum())
            if i in hist_matches:
                gt_pix   = mask_to_pixels(gt_mask)
                pred_pix = mask_to_pixels(pred_masks[hist_matches[i]])
                i_val = iou(pred_pix, gt_pix)
                d_val = dice(pred_pix, gt_pix)
                b_val = boundary_f_score(pred_pix, gt_pix)
            else:
                i_val = d_val = b_val = 0.0
            instance_records.append((gt_size, i_val, d_val, b_val))

    all_px = px_tp + px_fp + px_fn + px_tn
    px_tp_n = px_tp / all_px if all_px > 0 else 0.0
    px_fp_n = px_fp / all_px if all_px > 0 else 0.0
    px_fn_n = px_fn / all_px if all_px > 0 else 0.0
    px_tn_n = px_tn / all_px if all_px > 0 else 0.0

    # Derived scores
    sq = pq_tp_iou_sum / pq_tp_count if pq_tp_count > 0 else 0.0
    denom_rq = pq_tp_count + 0.5 * pq_fp_count + 0.5 * pq_fn_count
    rq = pq_tp_count / denom_rq if denom_rq > 0 else 0.0

    px_precision = px_tp / (px_tp + px_fp) if (px_tp + px_fp) > 0 else 0.0
    px_recall    = px_tp / (px_tp + px_fn) if (px_tp + px_fn) > 0 else 0.0
    px_f1 = (2 * px_precision * px_recall / (px_precision + px_recall)
             if (px_precision + px_recall) > 0 else 0.0)

    return {
        "all_iou":            all_iou,
        "all_dice":           all_dice,
        "all_bf1":            all_bf1,
        "all_seg_ratio":      all_seg_ratio,
        "pq_tp_count":        pq_tp_count,
        "pq_tp_iou_sum":      pq_tp_iou_sum,
        "pq_fp_count":        pq_fp_count,
        "pq_fn_count":        pq_fn_count,
        "sq":                 sq,
        "rq":                 rq,
        "pq":                 sq * rq,
        "px_tp":              px_tp,
        "px_fp":              px_fp,
        "px_fn":              px_fn,
        "px_tn":              px_tn,
        "px_precision":       px_precision,
        "px_recall":          px_recall,
        "px_f1":              px_f1,
        "images_with_gt":     images_with_gt,
        "total_gt_instances": total_gt_instances,
        "total_images":       len(image_files),
        "instance_records":   instance_records,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_results(results: dict, path: Path) -> None:
    """
    Save the results dict from run_benchmark() to a JSON file.

    Numpy scalar types are coerced to native Python types so the file is
    readable by any standard JSON parser.
    """
    import json  # noqa: PLC0415

    def _default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_default)
    print(f"Results saved to {path}")


def print_report(results: dict, *, label: str = "") -> None:
    """Print the benchmark summary to stdout."""
    r   = results
    sep = "=" * 50
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
    print(f"Images evaluated:   {r['images_with_gt']} / {r['total_images']}")
    print(f"GT instances:       {r['total_gt_instances']}")
    print(sep)
    print("--- Semantic (per-image combined mask, averaged) ---")
    print(f"Mean IoU:           {np.mean(r['all_iou']):.4f}")
    print(f"Mean Dice:          {np.mean(r['all_dice']):.4f}")
    print(f"Mean Boundary F1:   {np.mean(r['all_bf1']):.4f}")
    print(sep)
    sr           = r["all_seg_ratio"]
    mean_ratio   = float(np.mean(sr))   if sr else float("nan")
    median_ratio = float(np.median(sr)) if sr else float("nan")
    print("--- Seg ratio over matched pairs (pred px / GT px) ---")
    print(f"  n pairs:  {len(sr):,}")
    print(f"  mean:     {mean_ratio:.4f}  ({'over' if mean_ratio > 1 else 'under'}-segmented on average)")
    print(f"  median:   {median_ratio:.4f}  ({'over' if median_ratio > 1 else 'under'}-segmented on average)")
    over  = sum(1 for x in sr if x > 1)
    under = sum(1 for x in sr if x < 1)
    print(f"  over-segmented pairs:   {over} / {len(sr)}")
    print(f"  under-segmented pairs:  {under} / {len(sr)}")
    print(sep)
    print("--- Panoptic Quality (IoU >= 0.5 matching, global) ---")
    print(f"  TP={r['pq_tp_count']:,}  FP={r['pq_fp_count']:,}  FN={r['pq_fn_count']:,}")
    print(f"  SQ  (mean IoU of TP):  {r['sq']:.4f}")
    print(f"  RQ  (recognition):     {r['rq']:.4f}")
    print(f"  PQ  = SQ * RQ:         {r['pq']:.4f}")
    print(sep)
    px_total = r["px_tp"] + r["px_fp"] + r["px_fn"] + r["px_tn"]
    print(f"--- Pixel-level confusion matrix (global, {px_total:,} total px) ---")
    print(f"                 Pred pos    Pred neg")
    print(f"  GT pos    TP {r['px_tp']:>12,}  FN {r['px_fn']:>12,}")
    print(f"  GT neg    FP {r['px_fp']:>12,}  TN {r['px_tn']:>12,}")
    print(f"  Precision:  {r['px_precision']:.4f}")
    print(f"  Recall:     {r['px_recall']:.4f}")
    print(f"  F1:         {r['px_f1']:.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_charts(results: dict, output_dir: Path, *, label: str = "") -> None:
    """Save the three benchmark charts (semantic, PQ, confusion) to output_dir."""
    r        = results
    out      = Path(output_dir)
    subtitle = f" — {label}" if label else ""
    n_images = r["images_with_gt"]

    BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

    # 1. Semantic metrics
    fig, ax = plt.subplots(figsize=(6, 4.5))
    metrics = ["IoU", "Dice", "Boundary F1"]
    values  = [np.mean(r["all_iou"]), np.mean(r["all_dice"]), np.mean(r["all_bf1"])]
    bars = ax.bar(metrics, values, color=[BLUE, ORANGE, GREEN], width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean value (per-image combined mask)", fontsize=10)
    ax.set_title(f"Semantic segmentation metrics{subtitle}\n{n_images:,} images", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "chart_semantic.png", dpi=150)
    plt.close(fig)

    # 2. Panoptic Quality
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    pq_labels = ["SQ\n(segment quality)", "RQ\n(recognition quality)", "PQ\n(= SQ × RQ)"]
    pq_values = [r["sq"], r["rq"], r["pq"]]
    bars = ax.bar(pq_labels, pq_values, color=[BLUE, ORANGE, GREEN], width=0.5)
    for bar, v in zip(bars, pq_values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Panoptic Quality scores", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    counts = [r["pq_tp_count"], r["pq_fp_count"], r["pq_fn_count"]]
    bars = ax.bar(
        ["TP\n(matched)", "FP\n(false pred)", "FN\n(missed GT)"],
        counts, color=[GREEN, RED, ORANGE], width=0.5,
    )
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 50,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Instance count (global)", fontsize=10)
    ax.set_title("Instance matching breakdown (IoU ≥ 0.5)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Panoptic Quality{subtitle}", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "chart_pq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    cm = np.array([[r["px_tp"], r["px_fn"]], [r["px_fp"], r["px_tn"]]], dtype=np.float64)
    im = ax.imshow(np.log10(cm + 1), cmap="Blues", aspect="auto")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{int(cm[row, col]):,}",
                    ha="center", va="center", fontsize=11, color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred pos", "Pred neg"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["GT pos",   "GT neg"])
    ax.set_title("Pixel confusion matrix (counts, log colour scale)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("log₁₀(count)", fontsize=8)

    ax = axes[1]
    pr_values = [r["px_precision"], r["px_recall"], r["px_f1"]]
    bars = ax.bar(["Precision", "Recall", "F1"], pr_values,
                  color=[BLUE, ORANGE, GREEN], width=0.45)
    for bar, v in zip(bars, pr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Pixel-level precision / recall / F1", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Pixel-level confusion matrix{subtitle}", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "chart_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Charts saved to {out}/")


def plot_histogram(results: dict, output_dir: Path, *, label: str = "") -> None:
    """
    Plot average IoU / Dice / Boundary F1 by GT mask size bin and save to
    output_dir/mask_histogram.png.  Also prints the per-bin table to stdout.

    Uses ``results["instance_records"]`` produced by run_benchmark().
    """
    records = results["instance_records"]
    if not records:
        print("plot_histogram: no instance records, skipping.")
        return

    sizes = np.array([r[0] for r in records])
    ious  = np.array([r[1] for r in records])
    dices = np.array([r[2] for r in records])
    bf1s  = np.array([r[3] for r in records])

    n_bins   = len(BIN_LABELS)
    bin_iou  = np.zeros(n_bins)
    bin_dice = np.zeros(n_bins)
    bin_bf1  = np.zeros(n_bins)
    bin_cnt  = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        mask = (sizes > lo) & (sizes <= hi)
        bin_cnt[b] = mask.sum()
        if mask.sum() > 0:
            bin_iou[b]  = ious[mask].mean()
            bin_dice[b] = dices[mask].mean()
            bin_bf1[b]  = bf1s[mask].mean()

    x      = np.arange(n_bins)
    width  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    subtitle = f" — {label}" if label else ""

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_iou  = ax.bar(x - width, bin_iou,  width, label="IoU",          color=colors[0])
    bars_dice = ax.bar(x,         bin_dice, width, label="Dice",         color=colors[1])
    bars_bf1  = ax.bar(x + width, bin_bf1,  width, label="Boundary F1", color=colors[2])

    for bars in (bars_iou, bars_dice, bars_bf1):
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\n(n={cnt:,})" for lbl, cnt in zip(BIN_LABELS, bin_cnt)],
        fontsize=9,
    )
    ax.set_xlabel("GT mask size (pixels at ~480×640 resolution)", fontsize=11)
    ax.set_ylabel("Mean metric value", fontsize=11)
    ax.set_title(
        f"Segmentation quality by GT mask size{subtitle}\n"
        "Unmatched GT instances scored as 0",
        fontsize=11,
    )
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(output_dir)
    out_path = out / "mask_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Histogram saved to {out_path}")

    print(f"\n{'Bin':<12} {'Count':>7}  {'IoU':>6}  {'Dice':>6}  {'BF1':>6}")
    print("-" * 46)
    for b in range(n_bins):
        print(f"{BIN_LABELS[b]:<12} {bin_cnt[b]:>7,}  "
              f"{bin_iou[b]:>6.4f}  {bin_dice[b]:>6.4f}  {bin_bf1[b]:>6.4f}")


# ---------------------------------------------------------------------------
# Mapillary Vistas v2.0 entry point
# ---------------------------------------------------------------------------

def _make_mapillary_loader(instances_dir: Path, label_id: int) -> GtLoader:
    """
    Build a gt_loader for Mapillary Vistas v2.0 instance masks.

    Masks are 16-bit PNGs where ``class_id = pixel_value >> 8``.
    The loader resizes the raw mask to the requested resolution with
    nearest-neighbour interpolation before extracting instances.
    """
    def loader(img_path: Path, h: int, w: int) -> list[np.ndarray]:
        inst_path = instances_dir / f"{img_path.stem}.png"
        if not inst_path.exists():
            return []
        raw     = np.array(Image.open(inst_path), dtype=np.uint16)
        resized = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        return extract_instances_16bit(resized, label_id)
    return loader


def main() -> None:
    from ultralytics import YOLO  # noqa: PLC0415

    working_dir   = Path(__file__).parent
    dataset_root  = working_dir.parent
    instances_dir = dataset_root / "validation" / "v2.0" / "instances"
    images_dir    = dataset_root / "validation" / "images"
    car_label_id  = 108   # Mapillary 'object--vehicle--car'
    yolo_car_cls  = 2     # COCO 'car'

    pred_loader = make_yolo_loader(YOLO("yolo26n-seg.pt"), classes=[yolo_car_cls])
    gt_loader   = _make_mapillary_loader(instances_dir, car_label_id)
    image_files = sorted(images_dir.glob("*.jpg"))

    print(f"Found {len(image_files)} validation images.")
    results = run_benchmark(image_files, pred_loader, gt_loader)

    label = "yolo26n-seg · Mapillary Vistas v2.0 · car"
    print_report(results, label=label)
    save_results(results, working_dir / "benchmark_results.json")
    generate_charts(results, output_dir=working_dir, label=label)
    plot_histogram(results, output_dir=working_dir, label=label)


if __name__ == "__main__":
    main()
