"""
Power/detectability figure for the TMLR version of the manuscript.

analysis/analyze_power_and_ci.py computes every quantity plotted here and
writes them to results/{power_curve,power_design_summary,kendall_tau_ci}.csv.
This script only re-draws panel (B) over the subset of association tests the
TMLR manuscript actually reports: the eight scorers of the earlier draft were
cut to the five variants of three families (Mahalanobis, cosine-to-centroid,
pooled k-NN at k = 1, 10, 50), so Energy, ViM and the KDE scorer must not
appear in a figure attached to that version.

Nothing is recomputed: no bootstrap, no simulation, no checkpoint.

    python3 analysis/make_figure_power_tmlr.py --out-dir paper/tmlr/figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ALPHA = 0.05
PLAN_TAU = 0.3  # |tau| the analysis plan fixed as the effect size of interest

ROOT = Path(__file__).resolve().parents[1]

# (family, test) -> row label, in the order the manuscript introduces them.
# Anything not listed is not reported in the TMLR version and is dropped.
REPORTED = {
    ("E1", "condition_number"): "Condition number",
    ("E1", "fisher_ratio_HL"): "Fisher ratio (HL)",
    ("E1", "fisher_ratio_scalar"): "Fisher ratio (scalar)",
    ("E1", "mardia_kurtosis_b"): r"Mardia's kurtosis ($b$)",
    ("E1", "mardia_kurtosis_z"): r"Mardia's kurtosis ($z$)",
    ("E2", "condition_number"): "Condition number",
    ("E2", "fisher_ratio_HL"): "Fisher ratio (HL)",
    ("E2", "fisher_ratio_scalar"): "Fisher ratio (scalar)",
    ("E2", "mardia_kurtosis_b"): r"Mardia's kurtosis ($b$)",
    ("E2", "mardia_kurtosis_z"): r"Mardia's kurtosis ($z$)",
    ("E2.6", "mahalanobis"): "Mahalanobis",
    ("E2.6", "cosine"): "Cosine-to-centroid",
    ("E2.6", "knn_k1"): r"$k$-NN ($k=1$)",
    ("E2.6", "knn_k10"): r"$k$-NN ($k=10$)",
    ("E2.6", "knn_k50"): r"$k$-NN ($k=50$)",
    ("E2.7", "logistic_regression"): "Logistic regression",
    ("E2.7", "linear_svm"): "Linear SVM",
    ("E2.7", "random_forest"): "Random forest",
}

GROUP_LABEL = {
    "E1": r"geometry vs. $\lambda_{orth}$",
    "E2": "geometry vs. Mahalanobis AUROC",
    "E2.6": r"scorer AUROC vs. $\lambda_{orth}$",
    "E2.7": r"probe AUROC vs. $\lambda_{orth}$",
}
GROUP_COLOR = {"E1": "tab:purple", "E2": "tab:blue", "E2.6": "tab:orange", "E2.7": "tab:green"}

DESIGN_STYLE = {
    "A (n=13, continuous-continuous)": ("tab:blue", "-", "A: continuous--continuous, $n=13$"),
    "B (5/5/3 ladder, n=13)": ("tab:orange", "-", "B: 5/5/3 ordered ladder, $n=13$"),
    "B (3/3/3 common-seed, n=9)": ("tab:green", "--", "B: 3/3/3 common-seed subset, $n=9$"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "tmlr" / "figs")
    ap.add_argument("--stem", default="fig7_power")
    args = ap.parse_args()

    curve = pd.read_csv(args.results_dir / "power_curve.csv")
    summary = pd.read_csv(args.results_dir / "power_design_summary.csv")
    ci = pd.read_csv(args.results_dir / "kendall_tau_ci.csv")

    key = list(zip(ci["family"], ci["test"]))
    ci = ci[[k in REPORTED for k in key]].copy()
    ci["label"] = [REPORTED[k] for k in zip(ci["family"], ci["test"])]
    order = list(REPORTED)
    ci["_ord"] = [order.index(k) for k in zip(ci["family"], ci["test"])]
    ci = ci.sort_values("_ord", ascending=False).reset_index(drop=True)
    missing = [k for k in order if k not in set(zip(ci["family"], ci["test"]))]
    if missing:
        raise SystemExit(f"no interval on record for reported tests: {missing}")

    # stacked, not side by side: the manuscript is single-column, so a wide
    # two-panel figure scaled to \linewidth renders its labels unreadably small.
    fig, (ax_pow, ax_ci) = plt.subplots(
        2, 1, figsize=(7.4, 8.5), gridspec_kw={"height_ratios": [1.0, 1.32]}
    )

    # (A) power curves ----------------------------------------------------
    for design, (color, ls, label) in DESIGN_STYLE.items():
        sub = curve[curve["design"] == design].sort_values("expected_tau")
        ax_pow.plot(sub["expected_tau"], sub["power"], color=color, ls=ls, lw=2, label=label)
        crit = float(summary[summary["design"] == design].iloc[0]["tau_crit"])
        ax_pow.axvline(crit, color=color, lw=0.9, alpha=0.45, ls=":")

    ax_pow.axhline(0.80, color="gray", lw=0.9, ls="--")
    ax_pow.axhline(ALPHA, color="gray", lw=0.9, ls=":")
    ax_pow.axvline(PLAN_TAU, color="crimson", lw=1.3, ls="--")
    ax_pow.text(PLAN_TAU + 0.012, 0.60, r"$|\tau| = 0.3$", color="crimson",
                fontsize=8.5, rotation=90, va="top")
    ax_pow.text(0.855, 0.818, "80% power", color="gray", fontsize=8.5, ha="right")

    ann = []
    for design, (_, _, label) in DESIGN_STYLE.items():
        r = summary[summary["design"] == design].iloc[0]
        ann.append(f"{label}\n"
                   f"   power at $\\tau=0.3$: {r['power_at_tau_0.3']:.2f}"
                   f"   |   MDE$_{{80\\%}}$: {r['mde_power80']:.2f}"
                   f"   |   $\\tau_{{\\mathrm{{crit}}}}$: {r['tau_crit']:.2f}")
    ax_pow.text(0.02, 0.98, "\n".join(ann), transform=ax_pow.transAxes, fontsize=7.8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))

    ax_pow.set_xlabel(r"expected Kendall's $\tau$ under the alternative", fontsize=9.5)
    ax_pow.set_ylabel(r"power ($\alpha=0.05$, two-sided)", fontsize=9.5)
    ax_pow.set_xlim(0, 0.86)
    ax_pow.set_ylim(0, 1.0)
    ax_pow.set_title("(A) What this design could have detected", fontsize=10.5)
    ax_pow.legend(fontsize=8, loc="lower right")
    ax_pow.grid(alpha=0.25)

    # (B) observed tau with bootstrap intervals -----------------------------
    for k, row in ci.iterrows():
        c = GROUP_COLOR[row["family"]]
        if row["at_design_ceiling"]:
            ax_ci.plot(row["tau"], k, "*", color=c, ms=11, mec="k", mew=0.5)
            ax_ci.annotate("at design ceiling: no valid\n"
                           f"bootstrap CI (exact $p$={row['p_exact']:.1e})",
                           (row["tau"], k), textcoords="offset points", xytext=(-14, 7),
                           fontsize=7.2, ha="right", va="bottom", color=c)
            continue
        ax_ci.plot([row["ci_lo"], row["ci_hi"]], [k, k], color=c, lw=1.8,
                   alpha=0.85, solid_capstyle="butt")
        ax_ci.plot(row["tau"], k, "o", color=c, ms=5)

    crit_a = float(summary[summary["design"] == "A (n=13, continuous-continuous)"].iloc[0]["tau_crit"])
    crit_b = float(summary[summary["design"] == "B (5/5/3 ladder, n=13)"].iloc[0]["tau_crit"])
    ax_ci.axvspan(-crit_b, crit_b, color="crimson", alpha=0.055, zorder=0)
    for c in (-crit_b, crit_b):
        ax_ci.axvline(c, color="crimson", lw=1.0, ls="--", alpha=0.7, zorder=1)
    for c in (-crit_a, crit_a):
        ax_ci.axvline(c, color="tab:blue", lw=1.0, ls=":", alpha=0.7, zorder=1)
    ax_ci.axvline(0, color="k", lw=0.8, zorder=1)

    ax_ci.set_yticks(np.arange(len(ci)))
    ax_ci.set_yticklabels(ci["label"], fontsize=8.2)
    ax_ci.set_xlabel(r"Kendall's $\tau$ (point estimate and 95% bootstrap CI)", fontsize=9.5)
    ax_ci.set_xlim(-1.02, 1.02)
    ax_ci.set_ylim(-0.7, len(ci) + 1.35)
    ax_ci.set_title("(B) What the data exclude", fontsize=10.5)

    for tick, fam in zip(ax_ci.get_yticklabels(), ci["family"]):
        tick.set_color(GROUP_COLOR[fam])

    handles = [Line2D([], [], color=GROUP_COLOR[f], lw=2, label=lab)
               for f, lab in GROUP_LABEL.items()]
    ax_ci.legend(handles=handles, fontsize=7.8, loc="upper center",
                 bbox_to_anchor=(0.5, -0.105), ncol=2, frameon=False)
    ax_ci.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"{args.stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_dir / args.stem}.{{pdf,png}} over {len(ci)} reported tests")


if __name__ == "__main__":
    main()
