"""
E2.6/E2.8 -- eight reliability scorers on identical embeddings, primary
disentanglement ladder only.

Reads results/e2_6_scorer_comparison.csv, one row per (rung, seed, scorer):
mahalanobis, cosine, knn_k1, knn_k10, knn_k50 from extract_auroc_e2.py, and
energy, vim, density_kde from extract_e2_8_extra_scorers.py. Every formula
and hyperparameter was fixed in docs/experiment_contract.md before the
corresponding extraction code was written.

Three checks run before any statistic is computed, since both writers append
rather than overwrite and a re-run without clearing produces all three:

  1. exact duplicate rows are dropped, but reported rather than hidden;
  2. a duplicate (rung, seed, scorer) with differing auroc/fpr95 or
     checkpoint_path stops the script -- that is a conflict, not a re-run;
  3. all 104 rows (13 checkpoints x 8 scorers) must be present, or the
     script stops and lists what is missing.

Writes table_e2_6_scorer_summary.csv, e2_6_kendall_tau.csv,
e2_6_jonckheere_terpstra.csv and figures/figure_e2_6_scorer_comparison.pdf.
Kendall's tau and Jonckheere-Terpstra come from analyze_e1 by import.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_e1 import (
    COMMON_SEEDS,
    PRIMARY_RUNGS,
    RUNG_INDEX,
    RUNG_LAMBDA,
    jonckheere_terpstra,
    kendall_trend,
)

SCORERS = ["mahalanobis", "cosine", "knn_k1", "knn_k10", "knn_k50", "energy", "vim", "density_kde"]
PRIMARY_K_SCORER = "knn_k10"  # experiment_contract.md's headline k-NN value; knn_k1/knn_k50 are the robustness grid
RUNG_SEEDS = {"runA_grl": {42, 52, 62, 72, 82}, "runB_orth1": {42, 52, 62, 72, 82}, "runB": {42, 52, 62}}


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["rung"].isin(PRIMARY_RUNGS)].copy()

    unexpected_scorers = set(df["scorer"].unique()) - set(SCORERS)
    if unexpected_scorers:
        raise SystemExit(
            f"FATAL: {csv_path} contains unexpected scorer value(s) {sorted(unexpected_scorers)} -- "
            f"expected only {SCORERS} per experiment_contract.md's locked E2.6 design."
        )

    # 1. exact full-row duplicates -- drop, but loudly
    exact_dupe_mask = df.duplicated(keep="first")
    if exact_dupe_mask.any():
        dupe_keys = df.loc[exact_dupe_mask, ["rung", "seed", "scorer"]]
        print(
            f"[analyze_e2_6] WARNING: {exact_dupe_mask.sum()} exact duplicate row(s) found in {csv_path} "
            "and dropped -- e2_auroc.csv/distance_summary.csv/this file are append-only writers "
            "(extract_auroc_e2.py docstring); this means the file wasn't cleared before a re-run. "
            "Fix on the server side before the next rerun so this warning stops appearing:\n"
            f"{dupe_keys.to_string(index=False)}"
        )
        df = df[~exact_dupe_mask].copy()

    # 2. same (rung, seed, scorer), differing values -- NOT auto-resolved
    remaining_dupes = df[df.duplicated(subset=["rung", "seed", "scorer"], keep=False)]
    if len(remaining_dupes):
        raise SystemExit(
            f"FATAL: {csv_path} has (rung, seed, scorer) rows with DIFFERING auroc/fpr95/checkpoint_path -- "
            "not a safe idempotent duplicate, could indicate nondeterminism or a real conflict. "
            f"Resolve manually before proceeding:\n"
            f"{remaining_dupes.sort_values(['rung', 'seed', 'scorer']).to_string(index=False)}"
        )

    # 3. every (rung, seed, scorer) the manifest requires must be present
    expected = {
        (rung, seed, scorer)
        for rung, seeds in RUNG_SEEDS.items()
        for seed in seeds
        for scorer in SCORERS
    }
    actual = set(zip(df["rung"], df["seed"], df["scorer"]))
    missing = expected - actual
    if missing:
        missing_by_checkpoint: dict[tuple[str, int], list[str]] = {}
        for rung, seed, scorer in missing:
            missing_by_checkpoint.setdefault((rung, seed), []).append(scorer)
        lines = "\n".join(
            f"  {rung} seed={seed}: missing scorer(s) {sorted(scorers)}"
            for (rung, seed), scorers in sorted(missing_by_checkpoint.items())
        )
        n_expected = len(expected)
        raise SystemExit(
            f"FATAL: expected {n_expected} rows (13 checkpoints x {len(SCORERS)} scorers), "
            f"found {len(actual)} valid rows in {csv_path}. Missing:\n{lines}\n"
            "Stopping -- not analyzing a partial scorer comparison. Re-run "
            "scripts/extract_auroc_e2.py for exactly the checkpoint(s) listed above "
            "(the E2.5 Mahalanobis-only outputs for these checkpoints may already exist; "
            "this checks the E2.6 scorer-comparison file specifically)."
        )

    df["rung_index"] = df["rung"].map(RUNG_INDEX)
    print(f"[analyze_e2_6] Validated: {len(df)}/{len(expected)} rows, "
          f"{len(SCORERS)} scorers x 13 checkpoints, no unresolved duplicates.\n")
    return df


def table_scorer_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scorer in SCORERS:
        for rung in PRIMARY_RUNGS:
            sub = df[(df["scorer"] == scorer) & (df["rung"] == rung)]
            rows.append({
                "scorer": scorer, "rung": rung, "lambda_orth": RUNG_LAMBDA[rung], "n_seeds": len(sub),
                "auroc_mean": sub["auroc"].mean(), "auroc_sd": sub["auroc"].std(ddof=1),
                "fpr95_mean": sub["fpr95"].mean(), "fpr95_sd": sub["fpr95"].std(ddof=1),
            })
        pooled = df[df["scorer"] == scorer]
        rows.append({
            "scorer": scorer, "rung": "ALL (pooled)", "lambda_orth": np.nan, "n_seeds": len(pooled),
            "auroc_mean": pooled["auroc"].mean(), "auroc_sd": pooled["auroc"].std(ddof=1),
            "fpr95_mean": pooled["fpr95"].mean(), "fpr95_sd": pooled["fpr95"].std(ddof=1),
        })
    return pd.DataFrame(rows)


SCORER_LABELS = {
    "mahalanobis": "Mahalanobis", "cosine": "Cosine", "knn_k1": "$k$NN\n($k$=1)",
    "knn_k10": "$k$NN\n($k$=10)", "knn_k50": "$k$NN\n($k$=50)",
    "energy": "Energy", "vim": "ViM", "density_kde": "Density\n(KDE)",
}


def make_figure(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 5.5))
    x = np.arange(len(SCORERS))
    width = 0.8 / len(PRIMARY_RUNGS)
    colors = {"runA_grl": "tab:blue", "runB_orth1": "tab:orange", "runB": "tab:green"}

    for i, rung in enumerate(PRIMARY_RUNGS):
        means, sds = [], []
        for scorer in SCORERS:
            sub = df[(df["scorer"] == scorer) & (df["rung"] == rung)]["auroc"]
            means.append(sub.mean())
            sds.append(sub.std(ddof=1))
        ax.bar(x + i * width, means, width, yerr=sds, capsize=3,
               label=f"{rung} ($\\lambda$={RUNG_LAMBDA[rung]:g})", color=colors[rung])

    n_distance_scorers = 5  # mahalanobis, cosine, knn_k1/10/50 -- E2.6; the rest (energy/vim/density_kde) are E2.8
    group_width = width * len(PRIMARY_RUNGS)
    divider_x = (n_distance_scorers - 1) + group_width + (1 - group_width) / 2
    ax.axvline(divider_x, color="black", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUROC=0.5)")
    ax.set_xticks(x + width * (len(PRIMARY_RUNGS) - 1) / 2)
    ax.set_xticklabels([SCORER_LABELS[s] for s in SCORERS], fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel("AUROC (ISIC-test vs. PAD-UFES)", fontsize=20)
    ax.set_title("E2.6/E2.8 -- eight scorers on the identical embeddings", fontsize=17)
    ax.legend(fontsize=17)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_e2_6_scorer_comparison.png", dpi=300)
    fig.savefig(out_dir / "figure_e2_6_scorer_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--csv", default=str(results_dir / "e2_6_scorer_comparison.csv"))
    parser.add_argument("--output_dir", default=str(results_dir))
    parser.add_argument("--figures_dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = parser.parse_args()

    df = load_and_validate(Path(args.csv))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Table: per-scorer, per-rung + pooled AUROC/FPR95 ----
    summary = table_scorer_summary(df)
    summary.to_csv(out_dir / "table_e2_6_scorer_summary.csv", index=False)

    print("=== AUROC per scorer, pooled across the full primary ladder (13 checkpoints) ===")
    pooled_view = summary[summary["rung"] == "ALL (pooled)"].sort_values("auroc_mean", ascending=False)
    for _, row in pooled_view.iterrows():
        flag = "  <- headline k" if row["scorer"] == PRIMARY_K_SCORER else ""
        print(f"  {row['scorer']:12s} AUROC={row['auroc_mean']:.4f} +/- {row['auroc_sd']:.4f}  "
              f"FPR95={row['fpr95_mean']:.4f} +/- {row['fpr95_sd']:.4f}  (n={int(row['n_seeds'])}){flag}")

    print("\n=== AUROC per scorer, per rung ===")
    for rung in PRIMARY_RUNGS:
        print(f"\n  {rung} (lambda_orth={RUNG_LAMBDA[rung]:g}):")
        for scorer in SCORERS:
            row = summary[(summary["scorer"] == scorer) & (summary["rung"] == rung)].iloc[0]
            print(f"    {scorer:12s} AUROC={row['auroc_mean']:.4f} +/- {row['auroc_sd']:.4f}  "
                  f"FPR95={row['fpr95_mean']:.4f} +/- {row['fpr95_sd']:.4f}")

    best_scorer = pooled_view.iloc[0]
    print(
        f"\nReading this (descriptive, no pass/fail threshold per experiment_contract.md's E2.6 section): "
        f"the highest pooled AUROC belongs to '{best_scorer['scorer']}' at {best_scorer['auroc_mean']:.4f}. "
        "A materially higher AUROC from Cosine or k-NN than Mahalanobis's would point toward the scoring "
        "rule (covariance/Gaussian assumption) as the bottleneck; all scorers landing in the same ~0.35-0.45 "
        "range would point toward the representation itself, not any one scoring rule, as the bottleneck."
    )

    # ---- Kendall's tau: AUROC vs. rung order, per scorer ----
    print("\n=== Kendall's tau: AUROC vs. rung order, per scorer (EXACT permutation p-values) ===")
    tau_rows = []
    for scorer in SCORERS:
        sub = df[df["scorer"] == scorer]
        result = kendall_trend(sub, "auroc")
        result["scorer"] = scorer
        tau_rows.append(result)
        print(
            f"  {scorer:12s} tau={result['tau_full']:+.3f} (exact p={result['p_full_exact']:.6f}, n={result['n_full']})  |  "
            f"common-seed tau={result['tau_common_seed']:+.3f} (exact p={result['p_common_seed_exact']:.6f})  |  "
            f"sign_stable={result['sign_stable']}"
        )
    pd.DataFrame(tau_rows).to_csv(out_dir / "e2_6_kendall_tau.csv", index=False)

    # ---- Jonckheere-Terpstra: does AUROC trend across the ladder, per scorer ----
    print("\n=== Jonckheere-Terpstra: does AUROC itself trend across the ladder? (per scorer) ===")
    jt_rows = []
    for scorer in SCORERS:
        sub = df[df["scorer"] == scorer]
        groups = [sub[sub["rung"] == r]["auroc"].to_numpy() for r in PRIMARY_RUNGS]
        jt = jonckheere_terpstra(groups)
        jt["scorer"] = scorer
        jt_rows.append(jt)
        print(f"  {scorer:12s} J={jt['J']:.1f} (null mean={jt['mean_J_null']:.1f})  z={jt['z_normal_approx']:+.3f}  "
              f"p_normal_approx={jt['p_normal_approx']:.6f}  p_exact={jt['p_exact']:.6f} ({jt['n_arrangements']} arrangements)")
    pd.DataFrame(jt_rows).to_csv(out_dir / "e2_6_jonckheere_terpstra.csv", index=False)

    # ---- Figure ----
    make_figure(df, Path(args.figures_dir))
    print(f"\n[analyze_e2_6] Figure written to {args.figures_dir}/figure_e2_6_scorer_comparison.{{png,pdf}}")
    print(f"[analyze_e2_6] Tables written to {out_dir}/table_e2_6_scorer_summary.csv, "
          f"e2_6_kendall_tau.csv, e2_6_jonckheere_terpstra.csv")


if __name__ == "__main__":
    main()
