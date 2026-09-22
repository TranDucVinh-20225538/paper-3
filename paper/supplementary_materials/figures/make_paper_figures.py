#!/usr/bin/env python3
"""
Generate the manuscript's Figures 2-5 from results/.

These four had no generator in this repository: they were built once, by hand,
and went stale the moment the ladder grew from 13 checkpoints to 15 while the
text was updated and the images were not. Figures 6 and 7 already had
generators (paper/tmlr/generate_fig6.py, analysis/make_figure_power_tmlr.py)
and are not produced here.

Content and pixel dimensions follow the captions and the existing files, so the
LaTeX needs no change -- only the images are replaced.

    python3 analysis/make_paper_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNGS = ["runA_grl", "runB_orth1", "runB"]
LAMBDAS = {"runA_grl": 0, "runB_orth1": 1, "runB": 5}
RUNG_COLOUR = {"runA_grl": "tab:blue", "runB_orth1": "tab:orange", "runB": "tab:green"}
SCORERS = [("mahalanobis", "Mahalanobis"), ("cosine", "Cosine-to-centroid"),
           ("knn_k1", "$k$-NN ($k{=}1$)"), ("knn_k10", "$k$-NN ($k{=}10$)"),
           ("knn_k50", "$k$-NN ($k{=}50$)")]
XTICKS = [f"\\texttt{{{r}}}\n($\\lambda={LAMBDAS[r]}$)".replace("\\texttt{", "").replace("}", "", 1)
          for r in RUNGS]
XLABELS = [f"{r}\n($\\lambda$={LAMBDAS[r]})" for r in RUNGS]


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches=None)
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def strip_plot(ax, df, col, ylabel, title):
    """Per-checkpoint points in grey, rung mean +/- SD in blue -- the style the
    existing figure used, kept so the two are visually comparable."""
    for i, r in enumerate(RUNGS):
        v = df[df.rung == r][col].to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-0.07, 0.07, len(v)), v,
                   color="0.6", s=26, zorder=2)
    m = [df[df.rung == r][col].mean() for r in RUNGS]
    s = [df[df.rung == r][col].std() for r in RUNGS]
    ax.errorbar(range(3), m, yerr=s, color="tab:blue", marker="o", ms=7,
                capsize=4, lw=2, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(XLABELS, fontsize=9)
    ax.set_xlim(-0.45, 2.45); ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=0.25)


def fig2(out_dir: Path) -> None:
    g = pd.read_csv(RESULTS / "e1_geometry_metrics.csv")
    g = g[g.rung.isin(RUNGS)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.2))
    strip_plot(a, g, "condition_number", "condition number", "Covariance conditioning")
    strip_plot(b, g, "fisher_ratio_scalar", r"$\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$",
               "Fisher-ratio scalar")
    fig.tight_layout()
    save(fig, out_dir, "fig2_geometry_ladder")


def fig3(out_dir: Path) -> None:
    m = pd.read_csv(RESULTS / "e2_merged.csv")
    m = m[m.rung.isin(RUNGS)]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for r in RUNGS:
        sub = m[m.rung == r]
        ax.scatter(sub.condition_number, sub.auroc, s=52, alpha=0.9,
                   color=RUNG_COLOUR[r], label=f"{r} ($\\lambda$={LAMBDAS[r]})")
    ax.axhline(0.5, color="0.4", ls="--", lw=1)
    ax.text(ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]), 0.502,
            "chance", ha="left", va="bottom", fontsize=8, color="0.4")
    ax.set_xlabel("condition number", fontsize=10)
    ax.set_ylabel("Mahalanobis AUROC (ISIC-test vs. PAD-UFES)", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "fig3_geometry_vs_auroc")


def pooled_norm_medians() -> dict:
    """Median over every pooled sample, not the mean of per-seed medians."""
    out = {}
    for r in RUNGS:
        idv, oodv = [], []
        for f in sorted((RESULTS / "e2_distances").glob(f"{r}_s*.npz")):
            if f.stem.endswith("_z"):
                continue
            d = np.load(f, allow_pickle=True)
            idv.append(d["feature_norm_id"]); oodv.append(d["feature_norm_ood"])
        out[r] = (float(np.median(np.concatenate(idv))), float(np.median(np.concatenate(oodv))))
    return out


def fig4(out_dir: Path) -> None:
    norms = pooled_norm_medians()
    nv = pd.read_csv(RESULTS / "nv_attractor_headline.csv").set_index("rung")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.2))
    x = np.arange(3); w = 0.36
    a.bar(x - w/2, [norms[r][0] for r in RUNGS], w, label="ISIC-test", color="tab:blue")
    a.bar(x + w/2, [norms[r][1] for r in RUNGS], w, label="PAD-UFES", color="tab:red")
    a.set_ylabel(r"pooled median $\|z\|$", fontsize=10)
    a.set_title("(A) Feature norm", fontsize=11)
    lo = min(min(v) for v in norms.values()); hi = max(max(v) for v in norms.values())
    a.set_ylim(lo - 0.25 * (hi - lo), hi + 0.12 * (hi - lo))

    b.bar(x - w/2, [nv.loc[r, "P_pred_NV_given_ID_pct"] for r in RUNGS], w,
          label="ISIC-test", color="tab:blue")
    b.bar(x + w/2, [nv.loc[r, "P_pred_NV_given_OOD_pct"] for r in RUNGS], w,
          label="PAD-UFES", color="tab:red")
    b.set_ylabel("assigned to majority class (Nevus), %", fontsize=10)
    b.set_title("(B) Majority-class attraction", fontsize=11)
    b.set_ylim(0, 75)
    for ax in (a, b):
        ax.set_xticks(x); ax.set_xticklabels(XLABELS, fontsize=9)
        ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "fig4_norm_majority")


def fig5(out_dir: Path) -> None:
    c = pd.read_csv(RESULTS / "e2_6_scorer_comparison.csv")
    c = c[c.rung.isin(RUNGS)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(SCORERS)); w = 0.26
    for j, r in enumerate(RUNGS):
        m = [c[(c.scorer == s) & (c.rung == r)].auroc.mean() for s, _ in SCORERS]
        e = [c[(c.scorer == s) & (c.rung == r)].auroc.std() for s, _ in SCORERS]
        ax.errorbar(x + (j - 1) * w, m, yerr=e, fmt="o", ms=8, capsize=5, lw=2,
                    color=RUNG_COLOUR[r], label=f"{r} ($\\lambda$={LAMBDAS[r]})")
    ax.axhline(0.5, color="0.35", ls="--", lw=1.4)
    ax.text(len(SCORERS) - 0.6, 0.503, "chance (AUROC = 0.5)", ha="right", va="bottom",
            fontsize=9, color="0.35")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in SCORERS], fontsize=10)
    ax.set_ylabel("directed OOD AUROC (ISIC-test vs. PAD-UFES)", fontsize=10)
    ax.set_ylim(0.33, 0.55); ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "fig5_three_scorers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "tmlr" / "figs")
    args = ap.parse_args()
    n = pd.read_csv(RESULTS / "e1_geometry_metrics.csv")
    n = len(n[n.rung.isin(RUNGS)])
    print(f"sinh hình từ {n} checkpoint ->")
    fig2(args.out_dir); fig3(args.out_dir); fig4(args.out_dir); fig5(args.out_dir)


if __name__ == "__main__":
    main()
