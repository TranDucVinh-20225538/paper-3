"""
make_figure_dumbbell.py -- the paper's headline figure: a dumbbell plot
connecting mean non-probing-scorer AUROC to mean domain-probe AUROC, per
rung. Reads results/table_e2_7_domain_vs_distance.csv (analyze_e2_7_domain_probe.py's
own output) rather than hardcoding numbers, so this figure can never drift
out of sync with the actual pooled statistics the way a hand-copied script
could. baseline_soft is deliberately NOT plotted -- descriptive reference
only, kept off this axis per the locked design (manuscript_blueprint.md
Figure 5 spec; unchanged by E2.8's scorer additions).

Writes directly into paper/figures/ (the manuscript's own copy), not a
separate top-level figures/ that would then need a manual copy step -- this
script is the first generator this figure has ever had checked into the
repo (previously produced ad hoc, outside version control).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PAPER_FIGURES_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"

RUNGS = ["runA_grl", "runB_orth1", "runB"]
LAMBDA = {"runA_grl": 0, "runB_orth1": 1, "runB": 5}
PROBE_COLS = ["logistic_regression", "linear_svm", "random_forest"]


def main():
    df = pd.read_csv(RESULTS_DIR / "table_e2_7_domain_vs_distance.csv")
    ladder = df[df["rung"].isin(RUNGS)]

    scorer_mean = {r: ladder[ladder["rung"] == r]["nonprobing_scorer_auroc_mean"].iloc[0] for r in RUNGS}
    probe_mean = {
        r: ladder[ladder["rung"] == r]["domain_probe_auroc_mean"].mean()
        for r in RUNGS
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(RUNGS))

    for i, rung in enumerate(RUNGS):
        ax.plot([i, i], [scorer_mean[rung], probe_mean[rung]], color="gray", zorder=1, linewidth=2)

    ax.scatter(x, [scorer_mean[r] for r in RUNGS], s=140, color="tab:red", zorder=3,
               label="Non-probing scorers (mean of 8)", edgecolor="black")
    ax.scatter(x, [probe_mean[r] for r in RUNGS], s=140, color="tab:blue", zorder=3,
               label="Domain probes (mean of 3)", edgecolor="black")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, zorder=0)
    ax.text(2.55, 0.505, "chance", fontsize=14, color="gray", va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n($\\lambda_{{orth}}$={LAMBDA[r]})" for r in RUNGS], fontsize=15)
    ax.set_ylabel("AUROC (ISIC-test vs. PAD-UFES)", fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0.30, 0.85)
    ax.set_xlim(-0.4, 2.6)
    ax.legend(loc="upper left", fontsize=13, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES_DIR / "figure_dumbbell.png", dpi=300)
    fig.savefig(PAPER_FIGURES_DIR / "figure_dumbbell.pdf")
    print("Saved to", PAPER_FIGURES_DIR)
    print("scorer_mean =", scorer_mean)
    print("probe_mean =", probe_mean)


if __name__ == "__main__":
    main()
