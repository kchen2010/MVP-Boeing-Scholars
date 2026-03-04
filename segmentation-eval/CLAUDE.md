# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python library for benchmarking instance segmentation models on arbitrary datasets. The Mapillary Vistas v2.0 dataset (car class, label ID 108) with `yolo26n-seg.pt` is the reference use case, but both the model and dataset are fully pluggable via callbacks.

## Commands

```bash
# Run main benchmark on the full validation set (generates charts + histogram)
python benchmark.py
```

**Dependencies:** `ultralytics`, `opencv-python`, `numpy`, `matplotlib`, `Pillow`, `tqdm`

## Dataset Layout

The Mapillary `main()` entry points expect this layout relative to `working/`:

```
parent_directory/
├── validation/
│   ├── images/          # 2,000 .jpg files
│   └── v2.0/
│       └── instances/   # 16-bit PNG instance masks
└── working/             # this directory
```

## Architecture

**`metrics.py`** — All metric computations (IoU, Dice, Boundary F1, Panoptic Quality components). Uses set-based operations on pixel coordinate arrays for efficiency. Boundary extraction uses 4-connectivity; boundary F1 uses taxicab distance tolerance.

**`utils.py`** — Converts multi-mask images to pixel coordinate arrays (`mask_img_to_pixel_lists`), flattens multiple masks (`flatten_pixel_lists`), and extracts boundary pixels (`mask_boundary`).

**`benchmark.py`** — Generic benchmarking library. Two callbacks decouple the engine from any specific model or dataset:

- `PredResult(masks, height, width)` — NamedTuple returned by a pred_loader. `masks` is a list of boolean `(height, width)` arrays, one per predicted instance.
- `PredLoader = Callable[[Path], PredResult]` — model adapter. Owns all inference logic: loading, class filtering, tensor→numpy, resolution. The resolution it reports (`height`, `width`) is passed directly to the gt_loader so both sides are always spatially aligned.
- `GtLoader = Callable[[Path, int, int], list[np.ndarray]]` — dataset adapter. Receives `(img_path, mask_h, mask_w)` and returns boolean instance masks at that resolution, or `[]` to skip the image.

The full public API:
- `run_benchmark(image_files, pred_loader, gt_loader) -> dict` — core loop; returns a results dict with all accumulators and derived scores, including `instance_records`: a list of `(gt_size_px, iou, dice, bf1)` per GT instance using unconstrained greedy matching.
- `make_yolo_loader(model, *, classes=None) -> PredLoader` — factory for ultralytics YOLO models; the only place YOLO-specific code lives.
- `make_rfdetr_loader(model, *, classes=None, threshold=0.5, max_size=640) -> PredLoader` — factory for rfdetr segmentation models. Note: RF-DETR uses 1-indexed COCO class IDs (car=3) unlike YOLO (car=2).
- `print_report(results, *, label="")` — prints the summary table.
- `save_results(results, path)` — writes the results dict to a JSON file; handles numpy scalar coercion automatically.
- `generate_charts(results, output_dir, *, label="")` — saves `chart_semantic.png`, `chart_pq.png`, `chart_confusion.png`.
- `plot_histogram(results, output_dir, *, label="")` — bins `instance_records` into 6 log-spaced size buckets and saves `mask_histogram.png`. Called automatically by `main()` after `generate_charts()`.
- `extract_instances_16bit(inst_small, label_id)` — helper for datasets using 16-bit PNG mask encoding (`class_id = pixel_value >> 8`).

The Mapillary adapter lives in `_make_mapillary_loader()` and `main()`. `ultralytics` is imported only inside `main()`, so the library can be imported without it installed.

## Key Technical Details

- **Resolution contract:** `pred_loader` returns the resolution it used; `gt_loader` is called with that exact resolution. GT and predictions are always compared at the model's native evaluation scale.
- **16-bit instance encoding:** `class_id = pixel_value >> 8`, instance ID in lower 8 bits (Mapillary format).
- **Instance matching:** Two passes over the same IoU matrix per image. `pq_match` (IoU ≥ 0.5 threshold) for Panoptic Quality. `greedy_match` (unconstrained) for histogram records so every GT gets its best available score rather than being zeroed if it falls below the PQ threshold.
- **Vectorized IoU:** numpy matrix multiplication (`A @ B.T`) for efficient pairwise computation across all GT/pred pairs per image.
