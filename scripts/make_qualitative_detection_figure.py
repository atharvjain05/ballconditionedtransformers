"""Assemble a 2×2 qualitative figure from detection visualization PNGs (paper-ready).

Example:

    python scripts/make_qualitative_detection_figure.py \\
      --inputs a.png b.png c.png d.png \\
      --out figures/heatmap_qualitative \\
      --fig-inches 11
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import image as mpimg


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs=4,
        required=True,
        help="Four PNGs in row-major order: top-left, top-right, bottom-left, bottom-right",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path without suffix; writes .png and .pdf",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--fig-inches",
        type=float,
        default=11.0,
        help="Width and height of the square figure in inches (panels fill canvas; legend below).",
    )
    args = p.parse_args(argv)

    raw = [mpimg.imread(str(path)) for path in args.inputs]
    normed = []
    for im in raw:
        if im.ndim != 3 or im.shape[2] not in (3, 4):
            raise ValueError(f"Expected RGB(A) image, got shape {im.shape}")
        if im.dtype == np.uint8:
            normed.append(im.astype(np.float32) / 255.0)
        else:
            normed.append(np.clip(im.astype(np.float32), 0.0, 1.0))
    imgs = normed

    letters = ("a", "b", "c", "d")
    sz = float(args.fig_inches)
    fig, axes = plt.subplots(2, 2, figsize=(sz, sz), constrained_layout=False)
    # Reserve more bottom margin for a 2-row legend with large text.
    fig.subplots_adjust(
        left=0.002, right=0.998, top=0.998, bottom=0.215, wspace=0.006, hspace=0.006
    )

    for ax, im, letter in zip(axes.flat, imgs, letters):
        ax.imshow(np.clip(im[..., :3], 0.0, 1.0))
        ax.margins(0)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
            color="white",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    # Match visualize_heatmap_predictions.py: lime / yellow GT, red / orange predictions.
    gt_pl = mpatches.Rectangle((0, 0), 1, 1, facecolor="lime", edgecolor="#333333", linewidth=0.5)
    gt_bl = mpatches.Rectangle((0, 0), 1, 1, facecolor="yellow", edgecolor="#333333", linewidth=0.5)
    pr_pl = mpatches.Rectangle(
        (0, 0), 1, 1, facecolor="#E41A1C", edgecolor="#333333", linewidth=0.5
    )
    pr_bl = mpatches.Rectangle(
        (0, 0), 1, 1, facecolor="#FF8C00", edgecolor="#333333", linewidth=0.5
    )
    fig.legend(
        handles=[gt_pl, gt_bl, pr_pl, pr_bl],
        labels=[
            "Ground truth (player)",
            "Ground truth (ball)",
            "Prediction (player)",
            "Prediction (ball)",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=True,
        fancybox=False,
        fontsize=22,
        handlelength=2.4,
        handleheight=1.45,
        borderaxespad=0.4,
        columnspacing=2.4,
        labelspacing=0.9,
    )

    base = args.out
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
