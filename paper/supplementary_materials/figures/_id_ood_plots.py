"""
_id_ood_plots.py -- shared ID-vs-OOD distribution plotting.

Factored out because the same "histogram + KDE + ECDF, one column per rung,
pooled across seeds" comparison is now needed by both
analyze_e2_distances.py (Mahalanobis distance) and analyze_norm.py (feature
norm) -- implementing it twice would be exactly the kind of near-duplicate
drift this project has repeatedly flagged elsewhere (Fisher ratio, ECE,
feature-extraction collectors in the source repos). One implementation,
parametrized by which quantity and axis label, used by both.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def make_distribution_figure(
    pooled: dict, out_dir: Path, quantity_label: str, file_stem: str, rung_lambda: dict,
) -> None:
    """pooled: {rung: (id_values, ood_values)}."""
    rungs = list(pooled.keys())
    fig, axes = plt.subplots(3, len(rungs), figsize=(5 * len(rungs), 12))
    if len(rungs) == 1:
        axes = axes.reshape(3, 1)

    for col, rung in enumerate(rungs):
        v_id, v_ood = pooled[rung]
        lo, hi = min(v_id.min(), v_ood.min()), max(v_id.max(), v_ood.max())
        bins = np.linspace(lo, hi, 40)

        ax = axes[0, col]
        ax.hist(v_id, bins=bins, alpha=0.5, label=f"ID (n={len(v_id)})", color="tab:blue", density=True)
        ax.hist(v_ood, bins=bins, alpha=0.5, label=f"OOD (n={len(v_ood)})", color="tab:red", density=True)
        ax.set_title(f"{rung} ($\\lambda$={rung_lambda[rung]:g}) -- histogram", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel(quantity_label, fontsize=8)

        ax = axes[1, col]
        xs = np.linspace(lo, hi, 400)
        kde_id, kde_ood = gaussian_kde(v_id)(xs), gaussian_kde(v_ood)(xs)
        ax.plot(xs, kde_id, color="tab:blue", label="ID")
        ax.plot(xs, kde_ood, color="tab:red", label="OOD")
        ax.fill_between(xs, kde_id, alpha=0.2, color="tab:blue")
        ax.fill_between(xs, kde_ood, alpha=0.2, color="tab:red")
        ax.set_title(f"{rung} -- KDE", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel(quantity_label, fontsize=8)

        ax = axes[2, col]
        for data, color, label in [(v_id, "tab:blue", "ID"), (v_ood, "tab:red", "OOD")]:
            xs_sorted = np.sort(data)
            ys = np.arange(1, len(xs_sorted) + 1) / len(xs_sorted)
            ax.plot(xs_sorted, ys, color=color, label=label)
        ax.set_title(f"{rung} -- ECDF", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel(quantity_label, fontsize=8)
        ax.set_ylabel("cumulative probability", fontsize=8)

    fig.suptitle(
        f"ID vs OOD {quantity_label} distributions, pooled across seeds per rung\n"
        "(baseline_soft categorical reference not shown, same scope as E1a/E2a)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{file_stem}.png", dpi=200)
    fig.savefig(out_dir / f"{file_stem}.pdf")
    plt.close(fig)


def make_boxplot(pooled: dict, out_dir: Path, quantity_label: str, file_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors = [], [], []
    palette = {"ID": "tab:blue", "OOD": "tab:red"}
    for rung, (v_id, v_ood) in pooled.items():
        data += [v_id, v_ood]
        labels += [f"{rung}\nID", f"{rung}\nOOD"]
        colors += [palette["ID"], palette["OOD"]]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(quantity_label)
    ax.set_title(f"ID vs OOD {quantity_label}, per rung (pooled across seeds)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{file_stem}.png", dpi=200)
    fig.savefig(out_dir / f"{file_stem}.pdf")
    plt.close(fig)
