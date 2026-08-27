"""
E2 -- does the geometry measured in E1 track Mahalanobis AUROC?

Merges results/e1_geometry_metrics.csv with results/e2_auroc.csv on
(rung, seed, checkpoint_path) rather than (rung, seed), so that the same
nominal row pointing at two different checkpoints is a merge failure instead
of a silent join. This matters because CSG-SKin's find_checkpoint resolves a
directory to its newest file by mtime and is confirmed to pick a
non-best-val checkpoint for several runB/runB_orth1 seeds; without the
checkpoint in the key, E1 geometry and E2 AUROC could describe different
models and still merge cleanly.

Three checks run before any statistic, each stopping the script with the
offending keys listed: no duplicate (rung, seed) within either file; no
(rung, seed) present in both with differing checkpoint_path; exactly 13 rows
surviving the merge (5 runA_grl + 5 runB_orth1 + 3 runB).

Then computes Kendall's tau per geometry metric against AUROC, Jonckheere-
Terpstra for AUROC against rung order, Table 2 and Figure 2. Both variables
here are continuous and tie-free, which is the case scipy's method="exact"
is built for -- ties are checked for explicitly first, and the exact p-value
is cross-checked against a 100,000-draw Monte Carlo permutation test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from analyze_e1 import (
    CONTRACT_TAU_THRESHOLD,
    METRICS,
    PRIMARY_RUNGS,
    RUNG_INDEX,
    RUNG_LAMBDA,
    jonckheere_terpstra,
)

# The three E1a flagged as trend-bearing candidates -- experiment_contract.md
# E2a's success criterion is checked against these; the other two METRICS are
# still reported as companions, same pattern as E1a's CONTRACT_METRICS.
CONTRACT_METRICS = ["condition_number", "fisher_ratio_HL", "mardia_kurtosis_z"]


def load_and_merge(e1_path: Path, e2_path: Path) -> pd.DataFrame:
    e1 = pd.read_csv(e1_path)
    e2 = pd.read_csv(e2_path)

    e1 = e1[e1["rung"].isin(PRIMARY_RUNGS)].copy()
    e2 = e2[e2["rung"].isin(PRIMARY_RUNGS)].copy()

    # 1. duplicate (rung, seed) within either file (append-mode re-run risk)
    for name, df in [(str(e1_path), e1), (str(e2_path), e2)]:
        dupes = df[df.duplicated(subset=["rung", "seed"], keep=False)]
        if len(dupes):
            raise SystemExit(
                f"FATAL: {name} has duplicate (rung, seed) rows -- likely a checkpoint "
                "re-run appended a second row instead of the file being regenerated cleanly:\n"
                f"{dupes[['rung', 'seed', 'checkpoint_path']].to_string(index=False)}"
            )

    # 2. same (rung, seed), different checkpoint_path -- the literal Q3 failure mode
    key_check = e1[["rung", "seed", "checkpoint_path"]].merge(
        e2[["rung", "seed", "checkpoint_path"]], on=["rung", "seed"], suffixes=("_e1", "_e2")
    )
    mismatched = key_check[key_check["checkpoint_path_e1"] != key_check["checkpoint_path_e2"]]
    if len(mismatched):
        lines = "\n".join(
            f"  {r.rung} seed={r.seed}:\n    E1: {r.checkpoint_path_e1}\n    E2: {r.checkpoint_path_e2}"
            for r in mismatched.itertuples()
        )
        raise SystemExit(
            "FATAL: checkpoint_path mismatch between e1_geometry_metrics.csv and e2_auroc.csv "
            f"for the same (rung, seed) -- E1 and E2 would describe DIFFERENT checkpoints "
            f"(open_questions.md Q3):\n{lines}"
        )

    # 3. merge on the FULL key and require exactly 13 rows
    merged = e1.merge(e2, on=["rung", "seed", "checkpoint_path"], how="inner", suffixes=("_e1", "_e2"))

    e1_keys = set(zip(e1["rung"], e1["seed"]))
    e2_keys = set(zip(e2["rung"], e2["seed"]))
    only_e1 = e1_keys - e2_keys
    only_e2 = e2_keys - e1_keys
    if only_e1:
        print(f"[analyze_e2] (rung, seed) in {e1_path.name} but missing from {e2_path.name}: {sorted(only_e1)}")
    if only_e2:
        print(f"[analyze_e2] (rung, seed) in {e2_path.name} but missing from {e1_path.name}: {sorted(only_e2)}")

    if len(merged) != 13:
        raise SystemExit(
            f"FATAL: expected exactly 13 merged rows (5 runA_grl + 5 runB_orth1 + 3 runB), got {len(merged)}. "
            "See the missing-key lines above for which checkpoints are absent from one side. Stopping -- "
            "not analyzing a partial ladder."
        )

    merged["rung_index"] = merged["rung"].map(RUNG_INDEX)
    print(f"[analyze_e2] Merge validated: 13/13 rows, checkpoint_path identical between "
          f"{e1_path.name} and {e2_path.name} for every (rung, seed).\n")
    return merged


def kendall_metric_vs_auroc(merged: pd.DataFrame, metric: str, n_monte_carlo: int = 100_000, seed: int = 0) -> dict:
    """
    Kendall's tau between a geometry metric and AUROC, both continuous.
    Confirms no ties before trusting scipy's method="exact" (the lesson from
    analyze_e1.py's own Kendall's-tau bug: verify what the method actually
    does on this data's tie structure, don't assume) -- then cross-checks
    the exact p-value against a large Monte Carlo permutation test.
    """
    x = merged[metric].to_numpy()
    y = merged["auroc"].to_numpy()
    n = len(x)

    if len(np.unique(x)) < n or len(np.unique(y)) < n:
        raise ValueError(
            f"Unexpected ties in {metric} or auroc (n={n}, unique_x={len(np.unique(x))}, "
            f"unique_y={len(np.unique(y))}) -- scipy's exact Kendall's tau assumes none; "
            "would need the same full-enumeration treatment analyze_e1.py uses instead of "
            "trusting method='exact' here."
        )

    tau, p_exact = kendalltau(x, y, method="exact")

    rng = np.random.default_rng(seed)
    count_as_extreme = 0
    for _ in range(n_monte_carlo):
        perm_y = rng.permutation(y)
        t, _ = kendalltau(x, perm_y, method="asymptotic")  # method only affects p-value, not tau itself
        if abs(t) >= abs(tau) - 1e-9:
            count_as_extreme += 1
    p_monte_carlo = count_as_extreme / n_monte_carlo

    return {
        "metric": metric, "n": n, "tau": tau, "p_exact": p_exact,
        "p_monte_carlo": p_monte_carlo, "n_monte_carlo": n_monte_carlo,
    }


def table2_auroc_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rung in PRIMARY_RUNGS:
        sub = merged[merged["rung"] == rung]
        rows.append({
            "rung": rung, "lambda_orth": RUNG_LAMBDA[rung], "n_seeds": len(sub),
            "auroc_mean": sub["auroc"].mean(), "auroc_sd": sub["auroc"].std(ddof=1),
            "fpr95_mean": sub["fpr95"].mean(), "fpr95_sd": sub["fpr95"].std(ddof=1),
        })
    return pd.DataFrame(rows)


def make_figure2(merged: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.2))
    colors = {rung: c for rung, c in zip(PRIMARY_RUNGS, ["tab:blue", "tab:orange", "tab:green"])}

    for ax, metric in zip(axes, METRICS):
        for rung in PRIMARY_RUNGS:
            sub = merged[merged["rung"] == rung]
            ax.scatter(sub[metric], sub["auroc"], color=colors[rung], label=rung, s=40, alpha=0.85)
        ax.set_xlabel(metric, fontsize=9)
        ax.set_ylabel("Mahalanobis AUROC", fontsize=9)
        ax.set_title(metric, fontsize=10)

    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(
        "Figure 2 -- Geometry vs. Mahalanobis AUROC, primary disentanglement ladder\n"
        "(baseline_soft categorical reference not shown -- see experiment_contract.md E2b)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure2_e2_geometry_vs_auroc.png", dpi=300)
    fig.savefig(out_dir / "figure2_e2_geometry_vs_auroc.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--e1_csv", default=str(results_dir / "e1_geometry_metrics.csv"))
    parser.add_argument("--e2_csv", default=str(results_dir / "e2_auroc.csv"))
    parser.add_argument("--output_dir", default=str(results_dir))
    parser.add_argument("--figures_dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = parser.parse_args()

    merged = load_and_merge(Path(args.e1_csv), Path(args.e2_csv))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "e2_merged.csv", index=False)

    # ---- Table 2 ----
    t2 = table2_auroc_summary(merged)
    t2.to_csv(out_dir / "table2_e2_summary.csv", index=False)
    print("=== Table 2: Mahalanobis AUROC / FPR95 per rung (primary ladder) ===")
    for _, row in t2.iterrows():
        print(f"  {row['rung']:12s} (lambda={row['lambda_orth']:g}, n={int(row['n_seeds'])}): "
              f"AUROC={row['auroc_mean']:.4f} +/- {row['auroc_sd']:.4f}   "
              f"FPR95={row['fpr95_mean']:.4f} +/- {row['fpr95_sd']:.4f}")

    # ---- Kendall's tau: metric vs. AUROC (per experiment_contract.md E2a) ----
    print("\n=== Kendall's tau: geometry metric vs. Mahalanobis AUROC (EXACT p-value) ===")
    tau_results = []
    for metric in METRICS:
        result = kendall_metric_vs_auroc(merged, metric)
        tau_results.append(result)
        flag = "  [contract metric]" if metric in CONTRACT_METRICS else ""
        print(
            f"{metric:22s} tau={result['tau']:+.3f}  exact p={result['p_exact']:.6f}  "
            f"MC p={result['p_monte_carlo']:.6f} (n_mc={result['n_monte_carlo']}){flag}"
        )
    pd.DataFrame(tau_results).to_csv(out_dir / "e2_kendall_tau.csv", index=False)

    contract_pass_metrics = [
        r["metric"] for r in tau_results
        if r["metric"] in CONTRACT_METRICS and abs(r["tau"]) >= CONTRACT_TAU_THRESHOLD
    ]
    print(f"\n[experiment_contract.md E2a] |tau|>={CONTRACT_TAU_THRESHOLD} met by: {contract_pass_metrics or 'NONE'}")
    print(f"[experiment_contract.md E2a] Verdict: {'SUCCESS' if contract_pass_metrics else 'FAILURE'} "
          f"(mechanically applying the pre-registered criterion, not re-argued here)")

    # ---- Jonckheere-Terpstra on AUROC itself, across the ordered ladder ----
    print("\n=== Jonckheere-Terpstra: does AUROC itself trend across the ladder? ===")
    auroc_groups = [merged[merged["rung"] == r]["auroc"].to_numpy() for r in PRIMARY_RUNGS]
    jt = jonckheere_terpstra(auroc_groups)
    print(f"AUROC  J={jt['J']:.1f} (null mean={jt['mean_J_null']:.1f})  z={jt['z_normal_approx']:+.3f}  "
          f"p_normal_approx={jt['p_normal_approx']:.6f}  p_exact={jt['p_exact']:.6f} ({jt['n_arrangements']} arrangements)")
    pd.DataFrame([{**jt, "outcome": "auroc"}]).to_csv(out_dir / "e2_jonckheere_terpstra_auroc.csv", index=False)

    # ---- Figure 2 ----
    make_figure2(merged, Path(args.figures_dir))
    print(f"\n[analyze_e2] Figure 2 written to {args.figures_dir}/figure2_e2_geometry_vs_auroc.{{png,pdf}}")
    print(f"[analyze_e2] Table 2 written to {out_dir}/table2_e2_summary.csv")
    print(f"[analyze_e2] Merged data written to {out_dir}/e2_merged.csv")


if __name__ == "__main__":
    main()
