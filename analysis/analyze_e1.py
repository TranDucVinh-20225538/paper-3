"""
analyze_e1.py -- E1 primary ladder analysis: Table 1 + Figure 1.

Reads results/e1_geometry_metrics.csv (extract_embeddings_e1.py's output) and
produces the first real analysis artifacts of Paper 3's Results section:

  - Table 1: mean +/- SD per metric, for the primary disentanglement ladder
    (runA_grl -> runB_orth1 -> runB) and, separately, the baseline_soft
    categorical reference if present.
  - Kendall's tau (metric vs. rung order), with a TRUE exact permutation
    p-value: full enumeration of every distinct way to assign the observed
    values to rung labels matching the actual per-rung seed counts (e.g.
    72,072 = C(13,5)*C(8,5) arrangements for the 5/5/3 full ladder). This is
    NOT scipy's method="exact" (which raises ValueError given the ties rung
    labels always produce -- multiple seeds share a rung), and it is NOT
    scipy's method="auto" either: auto silently falls back to the asymptotic
    normal approximation whenever ties are present, with no warning. An
    earlier version of this script called that fallback "exact" by mistake;
    cross-checking against a from-scratch Monte Carlo permutation test
    caught the discrepancy (asymptotic p=0.0003 vs. the true exact p=0.000028
    for condition_number) before it went in the paper. Full enumeration, not
    an approximation, is what is reported now.
  - Jonckheere-Terpstra trend test (ordered-groups alternative to pooled
    Kendall's tau), using the SAME full-enumeration exact permutation null
    (shared implementation, see exact_permutation_pvalue below) rather than
    the standard normal approximation, which is not trusted at n=13-15 total
    without a check -- the same class of problem the Mardia-kurtosis
    null-calibration bug earlier in this project turned out to be. The
    normal-approximation p-value is still reported alongside it, labeled as
    such, for comparison against how this test is conventionally reported.
  - Figure 1: seed-level scatter + mean +/- SD across the three primary
    rungs, one panel per metric.

baseline_soft is deliberately NEVER plotted on the same axis as the primary
ladder and NEVER included in the Kendall/Jonckheere trend tests -- per
SPEC.md Sec 4 / open_questions.md Q4, it is a categorical reference, not a
fourth ordinal point, and plotting it on the same continuum would visually
reintroduce exactly the framing that design decision rejected. If present in
the CSV, it is reported only as a separate row in Table 1.

Does not implement checkpoint extraction or geometry computation -- purely a
downstream consumer of results/e1_geometry_metrics.csv, per this project's
scripts/ vs. analysis/ split (README.md layout).
"""

from __future__ import annotations

import argparse
import itertools
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm

PRIMARY_RUNGS = ["runA_grl", "runB_orth1", "runB"]  # ordered: lambda_orth = 0, 1, 5
RUNG_LAMBDA = {"runA_grl": 0.0, "runB_orth1": 1.0, "runB": 5.0}
RUNG_INDEX = {rung: i for i, rung in enumerate(PRIMARY_RUNGS)}
COMMON_SEEDS = {42, 52, 62}  # all three primary rungs have these; runB has only these

METRICS = [
    "condition_number",
    "fisher_ratio_HL",
    "fisher_ratio_scalar",
    "mardia_kurtosis_b",
    "mardia_kurtosis_z",
]
# The three experiment_contract.md E1a explicitly checks for the Success/Failure gate.
CONTRACT_METRICS = ["condition_number", "fisher_ratio_HL", "mardia_kurtosis_z"]
CONTRACT_TAU_THRESHOLD = 0.3


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (primary_ladder_df, baseline_reference_df). The latter may be empty."""
    df = pd.read_csv(csv_path)
    missing = set(METRICS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {sorted(missing)}")

    primary = df[df["rung"].isin(PRIMARY_RUNGS)].copy()
    baseline = df[df["rung"] == "baseline_soft"].copy()

    unexpected = set(df["rung"].unique()) - set(PRIMARY_RUNGS) - {"baseline_soft"}
    if unexpected:
        raise ValueError(
            f"Unexpected rung value(s) in {csv_path}: {sorted(unexpected)} -- "
            f"expected only {PRIMARY_RUNGS + ['baseline_soft']}"
        )

    primary["rung_index"] = primary["rung"].map(RUNG_INDEX)
    return primary, baseline


def table1(primary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SD per metric, primary ladder rows first (in ladder order), then baseline."""
    rows = []
    for rung in PRIMARY_RUNGS:
        sub = primary[primary["rung"] == rung]
        row = {"condition": rung, "role": "primary ladder", "n_seeds": len(sub), "lambda_orth": RUNG_LAMBDA[rung]}
        for m in METRICS:
            row[f"{m}_mean"] = sub[m].mean() if len(sub) else np.nan
            row[f"{m}_sd"] = sub[m].std(ddof=1) if len(sub) > 1 else np.nan
        rows.append(row)

    if len(baseline):
        row = {"condition": "baseline_soft", "role": "categorical reference (not on ladder)",
               "n_seeds": len(baseline), "lambda_orth": np.nan}
        for m in METRICS:
            row[f"{m}_mean"] = baseline[m].mean()
            row[f"{m}_sd"] = baseline[m].std(ddof=1) if len(baseline) > 1 else np.nan
        rows.append(row)
    else:
        print("[analyze_e1] Note: no baseline_soft rows in the CSV yet -- Table 1's "
              "categorical-reference row is omitted, not fabricated. E1b hasn't run yet.")

    return pd.DataFrame(rows)


MAX_EXACT_ARRANGEMENTS = 500_000  # safety cap; 5/5/3 -> 72,072, 3/3/3 -> 1,680, both well under this


def _enumerate_group_arrangements(group_sizes: tuple[int, ...]):
    """
    Yields every distinct way to partition range(sum(group_sizes)) positions
    into len(group_sizes) ordered groups of the given sizes -- i.e. every
    distinct rung-label arrangement consistent with the actual per-rung seed
    counts (so ties in the label variable are handled exactly, by
    construction, rather than approximated).
    """
    positions = tuple(range(sum(group_sizes)))

    def helper(remaining, sizes):
        if len(sizes) == 1:
            yield (tuple(remaining),)
            return
        for grp in itertools.combinations(remaining, sizes[0]):
            grp_set = set(grp)
            rest = tuple(p for p in remaining if p not in grp_set)
            for tail in helper(rest, sizes[1:]):
                yield (grp,) + tail

    yield from helper(positions, group_sizes)


def exact_permutation_pvalue(values: np.ndarray, group_sizes: tuple[int, ...], statistic_fn) -> dict:
    """
    TRUE exact permutation test: full enumeration of every distinct label
    arrangement consistent with group_sizes (see _enumerate_group_arrangements),
    recomputing statistic_fn(groups) for each, and reporting the two-sided
    exact p-value. Not scipy's method="exact" (refuses to run with ties) and
    not method="auto" (silently falls back to the asymptotic approximation
    when ties are present, with no warning -- confirmed directly against
    this project's own tied rung-label data before this function existed).

    statistic_fn(groups: list[np.ndarray]) -> float, where groups follows the
    same order as group_sizes.
    """
    n_arrangements = 1
    remaining = sum(group_sizes)
    for size in group_sizes:
        n_arrangements *= comb(remaining, size)
        remaining -= size
    if n_arrangements > MAX_EXACT_ARRANGEMENTS:
        raise ValueError(
            f"{n_arrangements} arrangements exceeds MAX_EXACT_ARRANGEMENTS={MAX_EXACT_ARRANGEMENTS}; "
            "full enumeration not attempted -- would need a Monte Carlo fallback instead."
        )

    values = np.asarray(values)
    obs_groups = []
    start = 0
    for size in group_sizes:
        obs_groups.append(values[start:start + size])
        start += size
    stat_obs = statistic_fn(obs_groups)

    count_as_extreme = 0
    for arrangement in _enumerate_group_arrangements(group_sizes):
        groups = [values[list(idx)] for idx in arrangement]
        stat_perm = statistic_fn(groups)
        if abs(stat_perm) >= abs(stat_obs) - 1e-9:
            count_as_extreme += 1

    return {
        "statistic": stat_obs,
        "p_exact": count_as_extreme / n_arrangements,
        "n_arrangements": n_arrangements,
        "n_as_extreme": count_as_extreme,
    }


def _kendall_tau_statistic(groups: list[np.ndarray]) -> float:
    """Kendall's tau-b between ordinal group index (0,1,2,...) and value, given as grouped arrays."""
    rung_index = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    values = np.concatenate(groups)
    tau, _ = kendalltau(rung_index, values, method="asymptotic")  # method only affects p-value, not tau itself
    return tau


def kendall_trend(primary: pd.DataFrame, metric: str) -> dict:
    """
    Kendall's tau between rung order and metric value, on the full ladder and
    the common-seed subset, per experiment_contract.md E1a. p-value is the
    TRUE exact permutation p-value (full enumeration, see
    exact_permutation_pvalue) -- NOT scipy's asymptotic fallback.
    """
    full = primary.sort_values("rung_index")[["rung_index", metric]].dropna()
    full_sizes = tuple(full.groupby("rung_index").size().sort_index().tolist())
    full_result = exact_permutation_pvalue(full[metric].to_numpy(), full_sizes, _kendall_tau_statistic)

    common = primary[primary["seed"].isin(COMMON_SEEDS)].sort_values("rung_index")[["rung_index", metric]].dropna()
    common_sizes = tuple(common.groupby("rung_index").size().sort_index().tolist())
    common_result = exact_permutation_pvalue(common[metric].to_numpy(), common_sizes, _kendall_tau_statistic)

    tau_full, tau_common = full_result["statistic"], common_result["statistic"]
    sign_stable = np.sign(tau_full) == np.sign(tau_common) if not (np.isnan(tau_full) or np.isnan(tau_common)) else False
    return {
        "metric": metric,
        "n_full": len(full),
        "tau_full": tau_full,
        "p_full_exact": full_result["p_exact"],
        "n_arrangements_full": full_result["n_arrangements"],
        "n_common_seed": len(common),
        "tau_common_seed": tau_common,
        "p_common_seed_exact": common_result["p_exact"],
        "n_arrangements_common": common_result["n_arrangements"],
        "sign_stable": bool(sign_stable),
        "passes_contract_threshold": bool(abs(tau_full) >= CONTRACT_TAU_THRESHOLD and sign_stable),
    }


def _jonckheere_J_statistic(groups: list[np.ndarray]) -> float:
    J = 0.0
    for i, j in itertools.combinations(range(len(groups)), 2):
        xi = np.asarray(groups[i])[:, None]
        yj = np.asarray(groups[j])[None, :]
        J += np.sum(xi < yj) + 0.5 * np.sum(xi == yj)
    return J


def jonckheere_terpstra(groups: list[np.ndarray]) -> dict:
    """
    J statistic (sum of pairwise Mann-Whitney counts, ties given 0.5 credit)
    for k ORDERED groups, testing for a monotonic trend across them.
    Verified against hand-computable ground truth: a perfectly-separated
    increasing 3-group/size-3 case gives exactly J=27, mean_J=13.5, z=3.0 as
    the closed-form formulas predict; perfect decreasing gives J=0, z=-3.0;
    a small tied example gives the expected 0.5-credit J=3.5.

    Reports both:
      - the standard normal approximation (Jonckheere 1954 / Terpstra 1952)
        -- the conventional way this test is reported, but its asymptotics
        assume group sizes larger than the n=3-5 per rung available here.
      - the TRUE exact permutation p-value (full enumeration, same method as
        exact_permutation_pvalue) -- trusted over the normal approximation at
        this sample size, for the same reason the Mardia-kurtosis
        closed-form null was replaced with a simulated one earlier in this
        project: small-n asymptotics are not assumed valid here without
        checking, and checking is cheap enough (n_arrangements <= 72,072 for
        this project's actual group sizes) to just do exactly rather than
        approximate.
    """
    ns = [len(g) for g in groups]
    N = sum(ns)

    J_obs = _jonckheere_J_statistic(groups)
    mean_J = (N**2 - sum(n**2 for n in ns)) / 4.0
    var_J = (N**2 * (2 * N + 3) - sum(n**2 * (2 * n + 3) for n in ns)) / 72.0
    z = (J_obs - mean_J) / np.sqrt(var_J) if var_J > 0 else np.nan
    p_normal = float(2 * (1 - norm.cdf(abs(z)))) if not np.isnan(z) else np.nan

    pooled = np.concatenate(groups)
    exact = exact_permutation_pvalue(pooled, tuple(ns), _jonckheere_J_statistic)

    return {
        "J": J_obs, "mean_J_null": mean_J, "z_normal_approx": z,
        "p_normal_approx": p_normal,
        "p_exact": exact["p_exact"], "n_arrangements": exact["n_arrangements"],
    }


def make_figure1(primary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.2))
    x_positions = [RUNG_INDEX[r] for r in PRIMARY_RUNGS]

    for ax, metric in zip(axes, METRICS):
        for rung in PRIMARY_RUNGS:
            sub = primary[primary["rung"] == rung]
            x = np.full(len(sub), RUNG_INDEX[rung], dtype=float)
            jitter = (np.random.default_rng(0).random(len(sub)) - 0.5) * 0.12
            ax.scatter(x + jitter, sub[metric], color="tab:gray", alpha=0.7, s=28, zorder=2)

        means = [primary[primary["rung"] == r][metric].mean() for r in PRIMARY_RUNGS]
        sds = [primary[primary["rung"] == r][metric].std(ddof=1) for r in PRIMARY_RUNGS]
        ax.errorbar(x_positions, means, yerr=sds, color="tab:blue", marker="o",
                    linewidth=2, capsize=4, zorder=3)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{r}\n($\\lambda$={RUNG_LAMBDA[r]:g})" for r in PRIMARY_RUNGS], fontsize=8)
        ax.set_title(metric, fontsize=10)
        ax.set_xlim(-0.5, len(PRIMARY_RUNGS) - 0.5)

    fig.suptitle(
        "Figure 1 -- Geometry across the primary disentanglement ladder\n"
        "(runA_grl -> runB_orth1 -> runB; baseline_soft categorical reference not shown -- see Table 1)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure1_e1_ladder_trend.png", dpi=300)
    fig.savefig(out_dir / "figure1_e1_ladder_trend.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(Path(__file__).resolve().parents[1] / "results" / "e1_geometry_metrics.csv"))
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--figures_dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = parser.parse_args()

    primary, baseline = load_data(Path(args.csv))
    print(f"[analyze_e1] Loaded {len(primary)} primary-ladder rows "
          f"({', '.join(f'{r}: n={len(primary[primary.rung==r])}' for r in PRIMARY_RUNGS)}) "
          f"and {len(baseline)} baseline_soft (categorical reference) rows.\n")

    # ---- Table 1 ----
    t1 = table1(primary, baseline)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t1.to_csv(out_dir / "table1_e1_summary.csv", index=False)
    print("=== Table 1: mean +/- SD (full precision in table1_e1_summary.csv) ===")
    for _, row in t1.iterrows():
        print(f"\n{row['condition']} ({row['role']}, n={int(row['n_seeds'])}):")
        for m in METRICS:
            print(f"    {m:22s} {row[f'{m}_mean']:>14.3f} +/- {row[f'{m}_sd']:<10.3f}")

    # ---- Kendall's tau (per experiment_contract.md E1a) ----
    print("\n=== Kendall's tau: metric vs. rung order (primary ladder only; EXACT permutation p-values) ===")
    contract_results = []
    for metric in METRICS:
        result = kendall_trend(primary, metric)
        contract_results.append(result)
        flag = "  [contract metric]" if metric in CONTRACT_METRICS else ""
        print(
            f"{metric:22s} tau={result['tau_full']:+.3f} (exact p={result['p_full_exact']:.6f}, "
            f"n={result['n_full']}, {result['n_arrangements_full']} arrangements)  |  "
            f"common-seed tau={result['tau_common_seed']:+.3f} (exact p={result['p_common_seed_exact']:.6f})  |  "
            f"sign_stable={result['sign_stable']}{flag}"
        )

    contract_pass = any(r["passes_contract_threshold"] for r in contract_results if r["metric"] in CONTRACT_METRICS)
    print(f"\n[experiment_contract.md E1a] Success threshold (|tau|>={CONTRACT_TAU_THRESHOLD}, sign-stable) "
          f"met by: {[r['metric'] for r in contract_results if r['metric'] in CONTRACT_METRICS and r['passes_contract_threshold']] or 'NONE'}")
    print(f"[experiment_contract.md E1a] Verdict: {'SUCCESS' if contract_pass else 'FAILURE'} "
          f"(mechanically applying the pre-registered criterion, not re-argued here)")

    pd.DataFrame(contract_results).to_csv(out_dir / "e1_kendall_tau.csv", index=False)

    # ---- Jonckheere-Terpstra ----
    print("\n=== Jonckheere-Terpstra trend test (ordered-groups alternative; EXACT permutation p-values) ===")
    jt_results = []
    for metric in METRICS:
        groups = [primary[primary["rung"] == r][metric].dropna().to_numpy() for r in PRIMARY_RUNGS]
        jt = jonckheere_terpstra(groups)
        jt["metric"] = metric
        jt_results.append(jt)
        print(
            f"{metric:22s} J={jt['J']:.1f} (null mean={jt['mean_J_null']:.1f})  "
            f"z={jt['z_normal_approx']:+.3f}  p_normal_approx={jt['p_normal_approx']:.6f}  "
            f"p_exact={jt['p_exact']:.6f}  ({jt['n_arrangements']} arrangements)"
        )
    pd.DataFrame(jt_results).to_csv(out_dir / "e1_jonckheere_terpstra.csv", index=False)
    print(
        "\nNote: p_exact (full enumeration) is what should be quoted, not p_normal_approx -- the normal "
        f"approximation is not trusted at n={sum(len(primary[primary.rung==r]) for r in PRIMARY_RUNGS)} total "
        "across group sizes "
        f"{', '.join(str(len(primary[primary.rung==r])) for r in PRIMARY_RUNGS)} without a check, for the same "
        "reason the Mardia kurtosis closed-form null was replaced with a simulated one earlier in this project. "
        "Both Kendall's tau's p-value above and this one use the identical full-enumeration method -- see "
        "exact_permutation_pvalue()."
    )

    # ---- Figure 1 ----
    make_figure1(primary, Path(args.figures_dir))
    print(f"\n[analyze_e1] Figure 1 written to {args.figures_dir}/figure1_e1_ladder_trend.{{png,pdf}}")
    print(f"[analyze_e1] Table 1 written to {out_dir}/table1_e1_summary.csv")


if __name__ == "__main__":
    main()
