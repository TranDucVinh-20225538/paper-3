"""
analyze_e2_distances.py -- E2.5: ID vs OOD Mahalanobis distance distributions.

AUROC is a ranking statistic: AUROC<0.5 shows P(distance(ID) > distance(OOD))
> 0.5, and nothing else. It cannot, on its own, distinguish "PAD-UFES
genuinely sits closer to ISIC class centroids in this representation" (a
finding about the representation) from a Mahalanobis-implementation bug
(wrong min/argmax direction, a train/test transform mismatch, wrong
centroid classes, wrong feature-extraction hook) that would produce the
identical AUROC number. Several of those were ruled out by reading source
directly (see extract_auroc_e2.py's module docstring), but the actual
distance distributions were never inspected before this script existed.

Reads:
  - results/distance_summary.csv (per-checkpoint id/ood mean, median, p95 --
    written by extract_auroc_e2.py alongside e2_auroc.csv).
  - results/e2_distances/{rung}_s{seed}.npz (full per-sample s_id/s_ood
    arrays, same source).

Produces, for the primary ladder only (runA_grl/runB_orth1/runB -- same
scope as E1a/E2a, baseline_soft excluded per SPEC.md Sec 4):
  - The distance_summary.csv table, printed directly.
  - Per-rung verdict: does median(OOD) < median(ID) actually hold, checked
    explicitly, not inferred from AUROC.
  - Figure: histogram + KDE + ECDF of ID vs OOD distances, one column per
    rung, pooling all seeds within a rung (more samples per panel, and a
    genuine representation effect should show up consistently across seeds
    within a rung rather than needing to be teased out of one seed alone).
  - A boxplot across all three rungs for a compact side-by-side comparison.

If every rung shows median(OOD) < median(ID) clearly, with separated (not
just differently-tailed) distributions, that's the direct evidence needed
to treat AUROC<0.5 as a real property of the representation rather than an
unverified inference from a single summary number. If not, this is exactly
where a Mahalanobis-implementation bug would show up instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from analyze_e1 import PRIMARY_RUNGS, RUNG_LAMBDA


def load_distance_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["rung"].isin(PRIMARY_RUNGS)].copy()
    return df


def load_raw_distances(npz_dir: Path, rung: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_dir / f"{rung}_s{seed}.npz")
    return data["s_id"], data["s_ood"]


def pooled_distances_by_rung(npz_dir: Path, summary: pd.DataFrame) -> dict:
    """Pools per-sample ID/OOD distances across all seeds within each rung --
    a genuine representation effect should be visible consistently within a
    rung, not just in one seed."""
    pooled = {}
    for rung in PRIMARY_RUNGS:
        seeds = summary[summary["rung"] == rung]["seed"].tolist()
        all_id, all_ood = [], []
        for seed in seeds:
            s_id, s_ood = load_raw_distances(npz_dir, rung, seed)
            all_id.append(s_id)
            all_ood.append(s_ood)
        pooled[rung] = (np.concatenate(all_id), np.concatenate(all_ood))
    return pooled


def make_distribution_figure(pooled: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, len(PRIMARY_RUNGS), figsize=(5 * len(PRIMARY_RUNGS), 12))

    for col, rung in enumerate(PRIMARY_RUNGS):
        s_id, s_ood = pooled[rung]
        lo, hi = min(s_id.min(), s_ood.min()), max(s_id.max(), s_ood.max())
        bins = np.linspace(lo, hi, 40)

        ax = axes[0, col]
        ax.hist(s_id, bins=bins, alpha=0.5, label=f"ID (n={len(s_id)})", color="tab:blue", density=True)
        ax.hist(s_ood, bins=bins, alpha=0.5, label=f"OOD (n={len(s_ood)})", color="tab:red", density=True)
        ax.set_title(f"{rung} ($\\lambda$={RUNG_LAMBDA[rung]:g}) -- histogram", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel("Mahalanobis squared distance", fontsize=8)

        ax = axes[1, col]
        xs = np.linspace(lo, hi, 400)
        kde_id, kde_ood = gaussian_kde(s_id)(xs), gaussian_kde(s_ood)(xs)
        ax.plot(xs, kde_id, color="tab:blue", label="ID")
        ax.plot(xs, kde_ood, color="tab:red", label="OOD")
        ax.fill_between(xs, kde_id, alpha=0.2, color="tab:blue")
        ax.fill_between(xs, kde_ood, alpha=0.2, color="tab:red")
        ax.set_title(f"{rung} -- KDE", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel("Mahalanobis squared distance", fontsize=8)

        ax = axes[2, col]
        for data, color, label in [(s_id, "tab:blue", "ID"), (s_ood, "tab:red", "OOD")]:
            xs_sorted = np.sort(data)
            ys = np.arange(1, len(xs_sorted) + 1) / len(xs_sorted)
            ax.plot(xs_sorted, ys, color=color, label=label)
        ax.set_title(f"{rung} -- ECDF", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel("Mahalanobis squared distance", fontsize=8)
        ax.set_ylabel("cumulative probability", fontsize=8)

    fig.suptitle(
        "E2.5 -- ID vs OOD Mahalanobis distance distributions, pooled across seeds per rung\n"
        "(baseline_soft categorical reference not shown, same scope as E1a/E2a)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_e2_5_distance_distributions.png", dpi=200)
    fig.savefig(out_dir / "figure_e2_5_distance_distributions.pdf")
    plt.close(fig)


def make_boxplot(pooled: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors = [], [], []
    palette = {"ID": "tab:blue", "OOD": "tab:red"}
    for rung in PRIMARY_RUNGS:
        s_id, s_ood = pooled[rung]
        data += [s_id, s_ood]
        labels += [f"{rung}\nID", f"{rung}\nOOD"]
        colors += [palette["ID"], palette["OOD"]]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Mahalanobis squared distance")
    ax.set_title("ID vs OOD Mahalanobis distance, per rung (pooled across seeds)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_e2_5_boxplot.png", dpi=200)
    fig.savefig(out_dir / "figure_e2_5_boxplot.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--distance_summary_csv", default=str(results_dir / "distance_summary.csv"))
    parser.add_argument("--npz_dir", default=str(results_dir / "e2_distances"))
    parser.add_argument("--figures_dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = parser.parse_args()

    summary = load_distance_summary(Path(args.distance_summary_csv))

    print("=== distance_summary.csv (per checkpoint, primary ladder) ===")
    cols = ["rung", "seed", "id_mean", "id_median", "id_p95", "ood_mean", "ood_median", "ood_p95"]
    print(summary[cols].sort_values(["rung", "seed"]).to_string(index=False))

    npz_dir = Path(args.npz_dir)
    pooled = pooled_distances_by_rung(npz_dir, summary)

    print("\n=== Per-rung verdict: does median(OOD) < median(ID)? (checked directly, not inferred from AUROC) ===")
    all_ood_below_id = True
    for rung in PRIMARY_RUNGS:
        s_id, s_ood = pooled[rung]
        id_med, ood_med = float(np.median(s_id)), float(np.median(s_ood))
        holds = ood_med < id_med
        all_ood_below_id &= holds
        verdict = "OOD < ID (consistent with 'OOD sits closer to ID centroids')" if holds \
            else "OOD >= ID -- INCONSISTENT with that reading; re-examine before concluding anything"
        print(f"  {rung:12s} (pooled n_id={len(s_id)}, n_ood={len(s_ood)}): "
              f"median(ID)={id_med:.2f}  median(OOD)={ood_med:.2f}  ->  {verdict}")

    print(
        f"\nOverall: {'every rung shows median(OOD) < median(ID) -- consistent across the ladder, not a fluke of one rung' if all_ood_below_id else 'NOT every rung shows median(OOD) < median(ID) -- do not generalize the mechanism claim past what the data actually shows'}"
    )
    print(
        "\nThis verdict is about the DIRECTION of the median gap only. Whether the distributions are "
        "cleanly separated or heavily overlapping (which changes how strong a claim this supports) "
        "is a visual/shape question -- see the histogram/KDE/ECDF figure, not this printed line alone."
    )

    make_distribution_figure(pooled, Path(args.figures_dir))
    make_boxplot(pooled, Path(args.figures_dir))
    print(f"\n[analyze_e2_distances] Figures written to {args.figures_dir}/figure_e2_5_distance_distributions.{{png,pdf}} "
          f"and figure_e2_5_boxplot.{{png,pdf}}")


if __name__ == "__main__":
    main()
