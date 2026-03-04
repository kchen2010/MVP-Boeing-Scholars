"""
Multi-model segmentation comparison on Mapillary Vistas v2.0 (car class).

Models evaluated:
  YOLO11n-seg, YOLO11s-seg   (ultralytics)
  YOLO26n-seg, YOLO26s-seg   (ultralytics)
  RFDETRSegNano, RFDETRSegSmall (rfdetr)

Per-model outputs under comparison_results/<slug>/:
  results.json, chart_semantic.png, chart_pq.png, chart_confusion.png,
  mask_histogram.png

Summary outputs at comparison_results/:
  comparison_metrics.png   – scalar metrics side-by-side
  comparison_histogram.png – IoU/Dice/BF1 by mask size, all models overlaid

Usage:
  python compare_models.py            # full 2000-image eval
  python compare_models.py --n 50     # quick test on first N images
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_WORKING_DIR = Path(__file__).parent
sys.path.insert(0, str(_WORKING_DIR))

from benchmark import (  # noqa: E402
    BIN_EDGES,
    BIN_LABELS,
    _make_mapillary_loader,
    generate_charts,
    make_rfdetr_loader,
    make_yolo_loader,
    plot_histogram,
    print_report,
    run_benchmark,
    save_results,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_ROOT  = _WORKING_DIR.parent
INSTANCES_DIR = DATASET_ROOT / "validation" / "v2.0" / "instances"
IMAGES_DIR    = DATASET_ROOT / "validation" / "images"
OUTPUT_DIR    = _WORKING_DIR / "comparison_results"
CAR_LABEL_ID  = 108

# Palette — one colour per model, consistent across all charts
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_pred_loaders() -> list[tuple[str, str, object]]:
    """
    Returns [(slug, label, pred_loader), ...].

    All six models are instantiated here so any weight-download errors surface
    before the long eval loop starts.
    """
    from ultralytics import YOLO           # noqa: PLC0415
    from rfdetr import RFDETRSegNano       # noqa: PLC0415
    from rfdetr import RFDETRSegSmall      # noqa: PLC0415

    return [
        ("yolo11n-seg",      "YOLO11n-seg",
         make_yolo_loader(YOLO("yolo11n-seg.pt"),  classes=[2])),
        ("yolo11s-seg",      "YOLO11s-seg",
         make_yolo_loader(YOLO("yolo11s-seg.pt"),  classes=[2])),
        ("yolo26n-seg",      "YOLO26n-seg",
         make_yolo_loader(YOLO("yolo26n-seg.pt"),  classes=[2])),
        ("yolo26s-seg",      "YOLO26s-seg",
         make_yolo_loader(YOLO("yolo26s-seg.pt"),  classes=[2])),
        ("rfdetr-seg-nano",  "RFDETRSeg-Nano",
         make_rfdetr_loader(RFDETRSegNano(pretrain_weights="rf-detr-seg-nano.pt"), classes=[3])),
        ("rfdetr-seg-small", "RFDETRSeg-Small",
         make_rfdetr_loader(RFDETRSegSmall(), classes=[3])),
    ]


# ---------------------------------------------------------------------------
# Comparison charts
# ---------------------------------------------------------------------------

def generate_comparison_charts(all_results: dict[str, dict], output_dir: Path) -> None:
    """
    Two charts written to output_dir:
      comparison_metrics.png   – three panels: Semantic / Instance / Pixel scalars
      comparison_histogram.png – three panels: IoU / Dice / BF1 by mask-size bin
    """
    slugs  = list(all_results.keys())
    labels = [all_results[s]["_label"] for s in slugs]
    n      = len(slugs)

    # ---- 1. Scalar metrics ------------------------------------------------
    metric_groups = [
        ("Semantic",
         ["Mean IoU", "Mean Dice", "Mean BF1"],
         [lambda r: float(np.mean(r["all_iou"])),
          lambda r: float(np.mean(r["all_dice"])),
          lambda r: float(np.mean(r["all_bf1"]))]),
        ("Instance (PQ)",
         ["PQ", "SQ", "RQ"],
         [lambda r: r["pq"], lambda r: r["sq"], lambda r: r["rq"]]),
        ("Pixel-level",
         ["Precision", "Recall", "F1"],
         [lambda r: r["px_precision"], lambda r: r["px_recall"], lambda r: r["px_f1"]]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    width   = 0.75 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for ax, (group_title, metric_names, extractors) in zip(axes, metric_groups):
        x = np.arange(len(metric_names))
        for i, (slug, color) in enumerate(zip(slugs, COLORS)):
            r    = all_results[slug]
            vals = [fn(r) for fn in extractors]
            bars = ax.bar(x + offsets[i], vals, width,
                          label=labels[i], color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{group_title} metrics", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Model comparison — Mapillary Vistas v2.0 · car class",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = output_dir / "comparison_metrics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # ---- 2. Size-bin histogram --------------------------------------------
    n_bins      = len(BIN_LABELS)
    metric_info = [("IoU", 1), ("Dice", 2), ("Boundary F1", 3)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric_name, rec_idx) in zip(axes, metric_info):
        x = np.arange(n_bins)
        for i, (slug, color) in enumerate(zip(slugs, COLORS)):
            records  = all_results[slug]["instance_records"]
            sizes    = np.array([rec[0] for rec in records])
            vals     = np.array([rec[rec_idx] for rec in records])
            bin_vals = np.zeros(n_bins)
            for b in range(n_bins):
                lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
                mask = (sizes > lo) & (sizes <= hi)
                if mask.sum() > 0:
                    bin_vals[b] = vals[mask].mean()
            ax.bar(x + offsets[i], bin_vals, width,
                   label=labels[i], color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("GT mask size (px at ~480×640)", fontsize=9)
        ax.set_ylabel(f"Mean {metric_name}", fontsize=10)
        ax.set_title(f"{metric_name} by GT mask size", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Size-binned quality — Mapillary Vistas v2.0 · car class\n"
        "Unmatched GT instances scored as 0",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = output_dir / "comparison_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N images (for quick testing)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    if args.n:
        image_files = image_files[: args.n]
    print(f"Images to evaluate: {len(image_files)}\n")

    gt_loader = _make_mapillary_loader(INSTANCES_DIR, CAR_LABEL_ID)

    print("Loading models (weights will auto-download if needed)...")
    model_list = build_pred_loaders()
    print(f"Loaded {len(model_list)} models.\n")

    all_results: dict[str, dict] = {}

    for slug, label, pred_loader in model_list:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

        out_dir = OUTPUT_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        results          = run_benchmark(image_files, pred_loader, gt_loader)
        results["_label"] = label

        print_report(results, label=label)
        save_results(results, out_dir / "results.json")
        generate_charts(results, output_dir=out_dir, label=label)
        plot_histogram(results, output_dir=out_dir, label=label)

        all_results[slug] = results

    print(f"\n{'=' * 60}")
    print("Generating comparison charts...")
    generate_comparison_charts(all_results, OUTPUT_DIR)
    print("\nAll done. Results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
