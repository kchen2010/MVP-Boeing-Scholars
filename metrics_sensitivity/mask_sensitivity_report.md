# LaMa Inpainting — Mask Sensitivity Analysis Report

## Overview

This report evaluates the quality of LaMa video inpainting on 11 automotive scenarios from the DAVIS dataset, with a focus on how **mask size and position changes** affect inpainting quality. The central question: *how sensitive is the inpainting output to small errors or variations in the segmentation mask?*

---

## Methodology

### Dataset
- **Source:** DAVIS 2017 benchmark dataset
- **Resolution:** 480p
- **Scenarios (11):** bus, car-roundabout, car-shadow, car-turn, classic-car, crossing, drift-chicane, drift-straight, drift-turn, night-race, rallye
- **Masks:** DAVIS ground-truth segmentation annotations (binarized: object pixels → 255, background → 0)

### Inpainting Method
- **LaMa** (Large Mask inpainting) via `simple-lama-inpainting`
- Each frame inpainted independently (spatial only, no temporal model)
- Mask pre-processing: 9×9 elliptical dilation applied before inpainting to ensure clean edge coverage

### Mask Perturbations Tested

| Perturbation | Type | Description |
|---|---|---|
| `original` | None | Ground-truth mask as-is |
| `shrink_3px` | Erosion | Mask eroded by 3px elliptical kernel |
| `shrink_7px` | Erosion | Mask eroded by 7px elliptical kernel |
| `grow_3px` | Dilation | Mask dilated by 3px elliptical kernel |
| `grow_7px` | Dilation | Mask dilated by 7px elliptical kernel |
| `shift_5px` | Translation | Mask randomly shifted ±5px in x and y |
| `shift_10px` | Translation | Mask randomly shifted ±10px in x and y |

### Metrics

| Metric | Description | Better |
|---|---|---|
| **PSNR** (dB) | Peak Signal-to-Noise Ratio in the masked region | Higher |
| **SSIM** | Structural Similarity Index in the masked bounding box | Higher |
| **LPIPS** | Learned Perceptual Image Patch Similarity (AlexNet) | Lower |
| **T-LPIPS** | Temporal LPIPS — flow-warped previous vs. current inpainted frame | Lower |
| **Temporal Consistency** | Mean absolute L2 difference in masked region across warped frames | Lower |

---

## Aggregate Results (Averaged Across All 11 Scenarios)

| Perturbation | PSNR (dB) | SSIM | LPIPS | T-LPIPS | Temp. Consistency |
|---|---|---|---|---|---|
| **original** | 10.63 | 0.343 | 0.569 | 0.193 | 13.37 |
| shrink_3px | 10.55 | 0.320 | 0.582 | 0.197 | 13.28 |
| shrink_7px | 10.58 | 0.297 | 0.599 | 0.200 | 13.18 |
| grow_3px | 11.04 | 0.372 | 0.561 | 0.188 | 13.34 |
| grow_7px | **11.58** | **0.416** | **0.540** | **0.184** | **13.01** |
| shift_5px | 10.79 | 0.349 | 0.567 | 0.192 | 13.32 |
| shift_10px | 10.99 | 0.361 | 0.564 | 0.192 | 13.13 |

### Change Relative to Original Mask (Δ)

| Perturbation | ΔPSNR | ΔSSIM | ΔLPIPS | ΔT-LPIPS |
|---|---|---|---|---|
| shrink_3px | −0.08 | −0.023 | +0.013 | +0.004 |
| shrink_7px | −0.05 | −0.046 | +0.030 | +0.007 |
| grow_3px | +0.41 | +0.028 | −0.009 | −0.005 |
| grow_7px | +0.95 | +0.073 | −0.030 | −0.009 |
| shift_5px | +0.16 | +0.006 | −0.002 | −0.001 |
| shift_10px | +0.37 | +0.018 | −0.005 | −0.001 |

---

## Per-Scenario Baseline Results (Original Mask)

| Scenario | PSNR (dB) | SSIM | LPIPS | T-LPIPS |
|---|---|---|---|---|
| bus | 8.33 | 0.432 | 0.514 | 0.158 |
| car-roundabout | 9.58 | 0.264 | 0.558 | 0.198 |
| car-shadow | 9.47 | 0.320 | 0.618 | 0.166 |
| car-turn | 9.94 | 0.285 | 0.676 | 0.134 |
| classic-car | 9.21 | 0.471 | 0.524 | 0.393 |
| crossing | 10.05 | 0.455 | 0.482 | 0.090 |
| drift-chicane | **16.23** | 0.283 | 0.646 | 0.149 |
| drift-straight | 10.95 | 0.317 | 0.530 | 0.263 |
| drift-turn | 11.52 | 0.289 | 0.674 | 0.270 |
| night-race | 14.89 | **0.534** | **0.276** | **0.063** |
| rallye | 6.78 | 0.125 | 0.767 | 0.234 |

**Best performing scenarios:** `night-race` and `drift-chicane` — both achieve high PSNR and, in night-race's case, the lowest LPIPS and T-LPIPS in the set.

**Worst performing scenario:** `rallye` — lowest PSNR (6.78 dB) and highest LPIPS (0.767), likely due to complex road textures and fast motion making background reconstruction difficult.

---

## Key Findings

### 1. Mask Shrinkage Consistently Hurts Quality

Eroding the mask means the inpainting boundary tightens around the object. This leaves **visible seam artifacts at the object's edges** — pixels that belong to the object are not inpainted, so sharp transitions appear where the object boundary meets the filled region. All metrics degrade:

- SSIM drops by **−0.023** at shrink_3px and **−0.046** at shrink_7px
- LPIPS worsens by **+0.013** and **+0.030** respectively
- T-LPIPS worsens slightly, suggesting mild increase in temporal flickering at seams

The PSNR degradation is small (≈ −0.08 dB) because PSNR is computed only inside the masked region — the seam artifacts, which are outside the tighter mask, are not penalised by this metric. This highlights a limitation of using only PSNR for evaluating inpainting with imprecise masks.

### 2. Mask Growth Consistently Improves Measured Quality

Dilating the mask causes the inpainting to fill a slightly larger region including a border of background pixels around the object. This gives LaMa more context to blend with the surroundings and avoids sharp boundary transitions:

- At grow_7px: PSNR improves by **+0.95 dB**, SSIM by **+0.073**, LPIPS by **−0.030**
- Temporal consistency also improves (lower T-LPIPS), suggesting smoother frame-to-frame transitions

> **Note:** The metric improvements from mask growth do not necessarily mean the visual result is better from a perceptual standpoint. A larger mask means more of the original background is replaced, which can create over-inpainting. However, the metrics favour it because the filled region blends more smoothly with its expanded surroundings.

### 3. Mask Shifting Has Minimal Impact

Random spatial shifts of ±5px and ±10px produce surprisingly stable metrics — PSNR changes by only +0.16 to +0.37 dB and LPIPS barely moves. This indicates that **LaMa is robust to small positional errors in the mask**, likely because the model conditions broadly on surrounding context rather than precise boundary placement.

### 4. Scenario-Level Sensitivity Varies Significantly

Scenarios differ substantially in how sensitive they are to mask perturbation:

- **classic-car** shows very little sensitivity to any perturbation (PSNR range < 0.3 dB across all perturbations), suggesting a relatively uniform or simple background
- **rallye** shows high sensitivity to mask growth (PSNR jumps from 6.78 → 8.45 at grow_7px, SSIM from 0.125 → 0.292), indicating that the tight original mask is creating significant boundary artefacts
- **drift-chicane** benefits most from mask growth (+1.53 dB PSNR at grow_7px) due to the narrow, fast-moving car subject

---

## Observations on Metric Agreement

Across all perturbations, PSNR, SSIM, LPIPS, and T-LPIPS broadly agree in direction:

- Shrinking the mask → all metrics worsen
- Growing the mask → all metrics improve
- Shifting the mask → near-neutral, slight improvement

However, the **magnitude of sensitivity differs by metric**. SSIM is the most sensitive to shrinkage (proportionally), while PSNR is most sensitive to growth. T-LPIPS remains relatively stable across all mask size perturbations, suggesting that temporal coherence of LaMa's output is largely independent of small mask boundary changes.

---

## Outputs

All results saved to `MVP-Boeing-Scholars/metrics_sensitivity/`:

| File | Contents |
|---|---|
| `sensitivity_results.csv` | Full per-method × scenario × perturbation results |
| `aggregated_method_perturbation.csv` | Averages grouped by method and perturbation |
| `aggregated_scenario_perturbation.csv` | Averages grouped by scenario and perturbation |
| `sensitivity_results.json` | Raw results in JSON format |
| `graphs/01_baseline_metrics_comparison.png` | Bar chart: LaMa baseline across all 5 metrics |
| `graphs/02_per_scenario_PSNR_mean.png` (×5) | Per-scenario bars for each metric |
| `graphs/03_sensitivity_PSNR_mean.png` (×5) | Line plots: metric vs. mask perturbation |
| `graphs/04_heatmap_PSNR_mean.png` (×5) | Heatmaps: perturbation × method |
| `graphs/05_delta_from_baseline.png` | Δ metric change from original mask |
| `graphs/06_per_scenario_sensitivity.png` | Per-scenario sensitivity heatmaps |
| `graphs/07_degradation_summary.png` | Average % change from baseline per perturbation type |
