"""
analyze_nv_attractor.py -- E2.5: is the NV (Nevus) attractor found by
analyze_predicted_class.py actually distinctive to OOD, or just NV's normal
base rate?

analyze_predicted_class.py found 62-64% of OOD (PAD-UFES) samples get
nearest-centroid-assigned to NV, consistently across the primary ladder.
On its own this number is uninterpretable: NV is ISIC's majority class, so
if ID (ISIC-test) samples also get pulled toward the NV centroid at a
similar rate, there is no OOD-specific effect to explain -- it would just be
NV's normal pull on every point, source domain notwithstanding. The
distinguishing test is P(pred=NV | OOD) vs. P(pred=NV | ID): if OOD is
pulled toward NV much more than ID's own baseline, that supports the
"Mahalanobis reads PAD-UFES lesions as unusually typical-looking Nevus"
mechanism directly. If the two rates are close, the earlier 62-64% number
does not, by itself, indicate anything domain-specific.

Reads results/e2_distances/*.npz (predicted_class_id/ood, labels_id --
already captured, no rerun).

Produces:
  - ID confusion matrix (true label -> predicted centroid, row-normalized %)
    per rung, pooled across seeds -- shows whether NV pulls broadly from
    many true classes even within ID, or is confined to samples that are
    already true NV.
  - OOD predicted-centroid histogram per rung (same numbers already in
    analyze_predicted_class.py's output, reprinted here for side-by-side
    comparison against the ID confusion matrix's column marginal).
  - The headline comparison: P(pred=NV | ID) vs. P(pred=NV | OOD) per rung,
    with the absolute (percentage-point) and relative (ratio) gap -- the
    actual number that decides whether the attractor hypothesis survives
    this check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyze_e1 import PRIMARY_RUNGS
from analyze_norm import discover_checkpoints
from analyze_predicted_class import LABELS, class_distribution_pct, load_class_data

NV_INDEX = LABELS.index("NV")


def pooled_id_arrays(npz_dir: Path, manifest: pd.DataFrame, rung: str) -> tuple[np.ndarray, np.ndarray]:
    seeds = manifest[manifest["rung"] == rung]["seed"].tolist()
    predicted, labels = [], []
    for seed in seeds:
        d = load_class_data(npz_dir, rung, seed)
        predicted.append(d["predicted_class_id"])
        labels.append(d["labels_id"])
    return np.concatenate(predicted), np.concatenate(labels)


def pooled_ood_predicted(npz_dir: Path, manifest: pd.DataFrame, rung: str) -> np.ndarray:
    seeds = manifest[manifest["rung"] == rung]["seed"].tolist()
    return np.concatenate([load_class_data(npz_dir, rung, s)["predicted_class_ood"] for s in seeds])


def id_confusion_matrix_pct(predicted: np.ndarray, labels: np.ndarray, n_classes: int = 8) -> pd.DataFrame:
    """Row-normalized: for each TRUE class, % of its samples predicted into each class."""
    counts = np.zeros((n_classes, n_classes), dtype=float)
    for true_c in range(n_classes):
        mask = labels == true_c
        if mask.sum() == 0:
            continue
        counts[true_c] = np.bincount(predicted[mask], minlength=n_classes)[:n_classes]
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pct = 100.0 * counts / row_sums
    return pd.DataFrame(pct, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])


def make_confusion_heatmaps(matrices: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5.5))
    if len(matrices) == 1:
        axes = [axes]
    for ax, (rung, cm) in zip(axes, matrices.items()):
        im = ax.imshow(cm.values, cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("predicted (nearest centroid)")
        ax.set_ylabel("true label")
        ax.set_title(f"{rung} -- ID confusion matrix (%)", fontsize=10)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                v = cm.values[i, j]
                if v >= 1.0:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="white" if v < 50 else "black", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="% of true-class samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_id_confusion_matrix.png", dpi=200)
    fig.savefig(out_dir / "figure_id_confusion_matrix.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--npz_dir", default=str(results_dir / "e2_distances"))
    parser.add_argument("--output_dir", default=str(results_dir))
    parser.add_argument("--figures_dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    manifest = discover_checkpoints(npz_dir)
    if len(manifest) != 13:
        print(f"[analyze_nv_attractor] WARNING: found {len(manifest)} checkpoints in {npz_dir}, expected 13. "
              "Proceeding with what's available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrices = {}
    headline_rows = []
    for rung in PRIMARY_RUNGS:
        pred_id, labels_id = pooled_id_arrays(npz_dir, manifest, rung)
        pred_ood = pooled_ood_predicted(npz_dir, manifest, rung)

        cm = id_confusion_matrix_pct(pred_id, labels_id)
        matrices[rung] = cm
        cm.to_csv(out_dir / f"id_confusion_matrix_{rung}.csv")

        id_marginal = class_distribution_pct(pred_id)
        ood_marginal = class_distribution_pct(pred_ood)

        p_nv_id = id_marginal[NV_INDEX]
        p_nv_ood = ood_marginal[NV_INDEX]
        headline_rows.append({
            "rung": rung,
            "n_id": len(pred_id), "n_ood": len(pred_ood),
            "P_pred_NV_given_ID_pct": p_nv_id,
            "P_pred_NV_given_OOD_pct": p_nv_ood,
            "diff_pp": p_nv_ood - p_nv_id,
            "ratio_ood_over_id": p_nv_ood / p_nv_id if p_nv_id > 0 else np.inf,
        })

        print(f"\n=== {rung}: ID confusion matrix (true label -> predicted centroid, row %) ===")
        print(cm.round(1).to_string())

        print(f"\n=== {rung}: predicted-centroid marginals (%) ===")
        marg_df = pd.DataFrame({"ID": id_marginal, "OOD": ood_marginal}, index=LABELS)
        print(marg_df.round(1).to_string())

    headline = pd.DataFrame(headline_rows)
    headline.to_csv(out_dir / "nv_attractor_headline.csv", index=False)

    print("\n=== Headline: P(pred=NV | ID) vs. P(pred=NV | OOD), per rung ===")
    print(headline.round(2).to_string(index=False))

    print(
        "\nReading this: a large diff_pp (OOD well above ID's own NV baseline) supports the attractor "
        "hypothesis as domain-specific, not just NV's usual pull. A small diff_pp means the earlier "
        "62-64% OOD number does not, by itself, distinguish OOD from how the model already treats ID."
    )

    make_confusion_heatmaps(matrices, Path(args.figures_dir))
    print(f"\n[analyze_nv_attractor] Figure written to {args.figures_dir}/figure_id_confusion_matrix.{{png,pdf}}")
    print(f"[analyze_nv_attractor] Tables written to {out_dir}/id_confusion_matrix_<rung>.csv and nv_attractor_headline.csv")


if __name__ == "__main__":
    main()
