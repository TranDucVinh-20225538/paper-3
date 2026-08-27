"""
Effect-size and power bounds for the null associations this paper reports.

For every Kendall's tau in the Results section, computes three things a
p-value on its own does not give:

    tau_crit   smallest |tau| that can reach exact-permutation p <= 0.05 at
               this design -- a function of n and the group sizes, not of
               the data
    power      rejection rate at alpha = 0.05 under an alternative whose
               expected statistic equals a target tau, by simulation
    CI         95% BCa bootstrap interval on the observed tau, resampled
               within rung

Two designs are handled separately, because the reported tau values are not
all the same statistic. Design A is continuous-continuous (E2: geometry
metric vs. Mahalanobis AUROC, n = 13, tie-free) and its null runs over all
13! orderings. Design B is an ordinal dose against a value (E1, E2.6, E2.7
vs. lambda_orth), where the dose is tied by construction and the null runs
over the 72,072 (or 1,680) distinct label arrangements -- the same full
enumeration analyze_e1.exact_permutation_pvalue does, imported from there.

Reads results/*.csv only; trains nothing and opens no checkpoint.

    python3 analysis/analyze_power_and_ci.py --n-boot 20000 --n-sim 20000

Writes power_design_summary.csv, power_curve.csv, kendall_tau_ci.csv,
jt_pvalue_conventions.csv and figures/figure_s_power.{pdf,png}.

Method, the design decisions behind the bootstrap, and two reporting
problems this analysis turned up are in docs/power_analysis.md.
"""

from __future__ import annotations

import argparse
import itertools
from math import comb, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import kendalltau, norm

from analyze_e1 import (
    COMMON_SEEDS,
    CONTRACT_TAU_THRESHOLD,
    METRICS,
    PRIMARY_RUNGS,
    RUNG_INDEX,
    _enumerate_group_arrangements,
)

ALPHA = 0.05
TOL = 1e-9  # same slack analyze_e1.exact_permutation_pvalue uses when counting "as extreme"


# ---- Design B: grouped (tied-x) Kendall tau-b and its exact null ----

def _pair_counts(group_sizes: tuple[int, ...]) -> tuple[int, int]:
    """(n0, M) = (total pairs, cross-group pairs) for the given group sizes."""
    N = sum(group_sizes)
    n0 = N * (N - 1) // 2
    tx = sum(n * (n - 1) // 2 for n in group_sizes)
    return n0, n0 - tx


def tau_b_from_J(J: np.ndarray | float, group_sizes: tuple[int, ...]) -> np.ndarray | float:
    """
    tau_b between the ordinal group index and a tie-free value variable, via
    the Jonckheere-Terpstra statistic. Exact when the values have no ties:
    C = J, D = M - C, so S = 2J - M and ty = 0.
    """
    n0, M = _pair_counts(group_sizes)
    return (2.0 * np.asarray(J, dtype=float) - M) / sqrt(M * n0)


def J_vectorised(y: np.ndarray, group_sizes: tuple[int, ...]) -> np.ndarray:
    """J for a batch of value vectors; y is (B, N), laid out in group order."""
    bounds = np.cumsum((0,) + tuple(group_sizes))
    blocks = [y[:, bounds[i]:bounds[i + 1]] for i in range(len(group_sizes))]
    J = np.zeros(y.shape[0])
    for i, j in itertools.combinations(range(len(group_sizes)), 2):
        lo = blocks[i][:, :, None]      # (B, n_i, 1)
        hi = blocks[j][:, None, :]      # (B, 1, n_j)
        J += (lo < hi).sum(axis=(1, 2)) + 0.5 * (lo == hi).sum(axis=(1, 2))
    return J


def grouped_null(group_sizes: tuple[int, ...]) -> dict:
    """
    Exact permutation null of J, hence of tau_b, by full enumeration of the
    label arrangements. It depends only on the ranks of the values, so one
    enumeration over 0..N-1 serves every tie-free metric at this design.
    """
    N = sum(group_sizes)
    ranks = np.arange(N)
    n_expected = 1
    remaining = N
    for size in group_sizes:
        n_expected *= comb(remaining, size)
        remaining -= size

    Js = []
    for arrangement in _enumerate_group_arrangements(group_sizes):
        groups = [ranks[list(idx)] for idx in arrangement]
        J = 0.0
        for i, j in itertools.combinations(range(len(groups)), 2):
            J += np.sum(groups[i][:, None] < groups[j][None, :])
        Js.append(J)
    Js = np.asarray(Js, dtype=float)
    if len(Js) != n_expected:
        raise SystemExit(
            f"Enumerated {len(Js)} arrangements for group sizes {group_sizes}, expected "
            f"{n_expected}; the exact null needs the complete set."
        )

    taus = tau_b_from_J(Js, group_sizes)
    n0, M = _pair_counts(group_sizes)
    return {
        "group_sizes": group_sizes, "n": N, "n_arrangements": len(Js),
        "J": Js, "tau": taus, "n0": n0, "M": M,
        "mean_J": M / 2.0, "max_abs_tau": float(np.abs(taus).max()),
    }


def critical_tau(null_taus: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """
    Smallest attainable |tau| whose two-sided exact p-value is <= alpha, and
    that p-value. (nan, nan) if nothing attainable reaches alpha at all.
    """
    levels = np.unique(np.abs(np.round(null_taus, 12)))
    for level in levels:
        p = float((np.abs(null_taus) >= level - TOL).mean())
        if p <= alpha:
            return float(level), p
    return float("nan"), float("nan")


def exact_p_grouped(tau_obs: float, null_taus: np.ndarray) -> float:
    return float((np.abs(null_taus) >= abs(tau_obs) - TOL).mean())


# ---- Design A: continuous-continuous Kendall tau, exact null by inversion DP ----

def untied_null(n: int) -> dict:
    """
    Exact null of Kendall's tau for n tie-free pairs. Discordant pairs under
    a random permutation are the inversion count, distributed by the Mann
    recursion; tau = (n0 - 2 * inversions) / n0. Same null scipy integrates
    under method="exact", checked against it in _verify_untied_null.
    """
    counts = np.array([1.0])
    for k in range(2, n + 1):
        new = np.zeros(len(counts) + k - 1)
        for i, c in enumerate(counts):
            new[i:i + k] += c
        counts = new
    n0 = n * (n - 1) // 2
    inversions = np.arange(len(counts))
    taus = (n0 - 2.0 * inversions) / n0
    probs = counts / counts.sum()
    return {"n": n, "tau": taus, "prob": probs, "n0": n0}


def exact_p_untied(tau_obs: float, null: dict) -> float:
    mask = np.abs(null["tau"]) >= abs(tau_obs) - TOL
    return float(null["prob"][mask].sum())


def critical_tau_untied(null: dict, alpha: float = ALPHA) -> tuple[float, float]:
    levels = np.unique(np.abs(np.round(null["tau"], 12)))
    for level in levels:
        p = float(null["prob"][np.abs(null["tau"]) >= level - TOL].sum())
        if p <= alpha:
            return float(level), p
    return float("nan"), float("nan")


def tau_untied_vectorised(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kendall's tau for a batch of tie-free pairs; x, y have shape (B, n)."""
    n = x.shape[1]
    iu, ju = np.triu_indices(n, k=1)
    s = np.sign(x[:, ju] - x[:, iu]) * np.sign(y[:, ju] - y[:, iu])
    return s.sum(axis=1) / (n * (n - 1) / 2.0)


# ---- Alternatives: expected-tau parameterisation ----
# Each design gets an alternative whose expected value of the reported
# statistic is available in closed form, so "power to detect tau = 0.3" means
# power when the reported statistic has expectation 0.3.

def expected_tau_grouped(theta: float, group_sizes: tuple[int, ...]) -> float:
    """
    E[tau_b] under the dose-shift alternative y_i = theta * d_i + eps_i,
    eps ~ N(0, 1), with d the rung index the test uses. A cross-rung pair
    i < j is concordant with probability Phi(theta * (d_j - d_i) / sqrt(2)),
    giving E[C] = sum_{i<j} n_i n_j Phi(...) and E[tau_b] = (2 E[C] - M) / sqrt(M * n0).
    """
    n0, M = _pair_counts(group_sizes)
    EC = 0.0
    for i, j in itertools.combinations(range(len(group_sizes)), 2):
        EC += group_sizes[i] * group_sizes[j] * norm.cdf(theta * (j - i) / sqrt(2.0))
    return (2.0 * EC - M) / sqrt(M * n0)


def theta_for_tau_grouped(target_tau: float, group_sizes: tuple[int, ...]) -> float:
    if target_tau <= 0:
        return 0.0
    hi = 1.0
    while expected_tau_grouped(hi, group_sizes) < target_tau:
        hi *= 2.0
        if hi > 1e4:
            return float("nan")  # target above the design's attainable ceiling
    return float(brentq(lambda t: expected_tau_grouped(t, group_sizes) - target_tau, 0.0, hi, xtol=1e-10))


def rho_for_tau_untied(target_tau: float) -> float:
    """Bivariate-normal alternative: E[tau] = (2/pi) arcsin(rho), so rho = sin(pi*tau/2)."""
    return float(np.sin(np.pi * target_tau / 2.0))


# ---- Power curves (common random numbers -> monotone, interpolable curves) ----

def power_curve_grouped(group_sizes, tau_grid, tau_crit, n_sim, rng) -> np.ndarray:
    N = sum(group_sizes)
    eps = rng.standard_normal((n_sim, N))                       # drawn ONCE: common random numbers
    dose = np.concatenate([np.full(s, i, dtype=float) for i, s in enumerate(group_sizes)])
    power = np.empty(len(tau_grid))
    for k, t in enumerate(tau_grid):
        theta = theta_for_tau_grouped(t, group_sizes)
        tau_sim = tau_b_from_J(J_vectorised(eps + theta * dose, group_sizes), group_sizes)
        power[k] = float((np.abs(tau_sim) >= tau_crit - TOL).mean()) if np.isfinite(tau_crit) else 0.0
    return power


def power_curve_untied(n, tau_grid, tau_crit, n_sim, rng) -> np.ndarray:
    x = rng.standard_normal((n_sim, n))                         # common random numbers
    z = rng.standard_normal((n_sim, n))
    power = np.empty(len(tau_grid))
    for k, t in enumerate(tau_grid):
        rho = rho_for_tau_untied(t)
        y = rho * x + sqrt(max(0.0, 1.0 - rho ** 2)) * z
        tau_sim = tau_untied_vectorised(x, y)
        power[k] = float((np.abs(tau_sim) >= tau_crit - TOL).mean()) if np.isfinite(tau_crit) else 0.0
    return power


def interpolate_crossing(tau_grid: np.ndarray, power: np.ndarray, target: float) -> float:
    """Smallest expected tau at which the power curve reaches target; NaN if never."""
    above = np.nonzero(power >= target)[0]
    if len(above) == 0:
        return float("nan")
    k = above[0]
    if k == 0:
        return float(tau_grid[0])
    x0, x1, y0, y1 = tau_grid[k - 1], tau_grid[k], power[k - 1], power[k]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


# ---- Bootstrap confidence intervals ----

def _bca_interval(theta_hat, boot, jack, alpha=ALPHA) -> dict:
    """
    BCa interval, with the percentile interval alongside. When all replicates
    fall on one side of the estimate, z0 is infinite and BCa is undefined; the
    percentile interval is returned instead and flagged as such.
    """
    boot = boot[np.isfinite(boot)]
    lo_pct, hi_pct = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    prop_less = float(np.mean(boot < theta_hat) + 0.5 * np.mean(boot == theta_hat))
    degenerate = not (0.0 < prop_less < 1.0)
    if degenerate:
        return {"ci_lo": float(lo_pct), "ci_hi": float(hi_pct), "ci_method": "percentile (BCa degenerate)",
                "ci_lo_pct": float(lo_pct), "ci_hi_pct": float(hi_pct), "bca_degenerate": True,
                "z0": float("nan"), "acceleration": float("nan"), "n_boot_valid": int(len(boot))}

    z0 = float(norm.ppf(prop_less))
    jack = jack[np.isfinite(jack)]
    dev = jack.mean() - jack
    denom = 6.0 * (np.sum(dev ** 2) ** 1.5)
    a = float(np.sum(dev ** 3) / denom) if denom > 0 else 0.0

    out = {}
    for name, z in (("lo", norm.ppf(alpha / 2)), ("hi", norm.ppf(1 - alpha / 2))):
        adj = z0 + (z0 + z) / (1.0 - a * (z0 + z))
        out[name] = float(np.percentile(boot, 100 * norm.cdf(adj)))
    return {"ci_lo": out["lo"], "ci_hi": out["hi"], "ci_method": "BCa",
            "ci_lo_pct": float(lo_pct), "ci_hi_pct": float(hi_pct), "bca_degenerate": False,
            "z0": z0, "acceleration": a, "n_boot_valid": int(len(boot))}


def bootstrap_tau_grouped(values_by_group: list[np.ndarray], n_boot: int, rng) -> dict:
    """
    Stratified bootstrap for a Design-B tau_b: resample within each rung,
    holding the 5/5/3 (or 3/3/3) design fixed. lambda_orth is a fixed design
    factor, so resampling across rungs would estimate something the
    experiment never measured.

    Resampling creates ties, and the two usual tie treatments scale the
    interval in opposite directions, so both are returned. scipy's tau_b
    shrinks its denominator as ties accumulate and inflates |tau|;
    (2J - M)/sqrt(M*n0) pins the denominator to the original design and
    cannot exceed its ceiling. They coincide on the observed tie-free data.

    Boundary case: an observed |tau| at the design maximum makes the
    resampling distribution one-sided, so no interval is valid. Those are
    flagged at_design_ceiling; see docs/power_analysis.md.
    """
    sizes = tuple(len(g) for g in values_by_group)
    dose = np.concatenate([np.full(s, i, dtype=float) for i, s in enumerate(sizes)])
    obs = np.concatenate(values_by_group)
    theta_hat = float(kendalltau(dose, obs, method="asymptotic")[0])

    boot_b = np.empty(n_boot)
    resamples = np.empty((n_boot, len(obs)))
    for b in range(n_boot):
        resampled = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in values_by_group])
        resamples[b] = resampled
        boot_b[b] = kendalltau(dose, resampled, method="asymptotic")[0]
    boot_fixed = tau_b_from_J(J_vectorised(resamples, sizes), sizes)

    jack_b, jack_fixed = [], []
    for drop in range(len(obs)):
        keep = np.ones(len(obs), dtype=bool)
        keep[drop] = False
        jack_b.append(kendalltau(dose[keep], obs[keep], method="asymptotic")[0])
        sizes_j = tuple(s - 1 if i == int(dose[drop]) else s for i, s in enumerate(sizes))
        jack_fixed.append(float(tau_b_from_J(J_vectorised(obs[keep][None, :], sizes_j)[0], sizes_j)))

    res = _bca_interval(theta_hat, boot_b, np.asarray(jack_b, dtype=float))
    alt = _bca_interval(theta_hat, boot_fixed, np.asarray(jack_fixed, dtype=float))
    res["tau"] = theta_hat
    res["ci_lo_fixed"], res["ci_hi_fixed"] = alt["ci_lo"], alt["ci_hi"]
    res["n_boot_nan"] = int(np.sum(~np.isfinite(boot_b)))
    return res


def bootstrap_tau_pairs(x: np.ndarray, y: np.ndarray, strata: np.ndarray, n_boot: int, rng) -> dict:
    """
    Stratified bootstrap for a Design-A tau: resample (x, y) pairs within
    rung, preserving the 5/5/3 composition. Pairs move together, so the
    resampled quantity stays the geometry-AUROC correlation across
    checkpoints. Same two-statistic treatment as bootstrap_tau_grouped, with
    S / n0 as the design-pinned variant.
    """
    n = len(x)
    n0 = n * (n - 1) / 2.0
    iu, ju = np.triu_indices(n, k=1)

    def tau_fixed(xv, yv):
        return float(np.sum(np.sign(xv[ju] - xv[iu]) * np.sign(yv[ju] - yv[iu])) / n0)

    theta_hat = float(kendalltau(x, y, method="asymptotic")[0])
    idx_by_stratum = [np.nonzero(strata == s)[0] for s in np.unique(strata)]

    boot_b = np.empty(n_boot)
    boot_fixed = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([idx[rng.integers(0, len(idx), len(idx))] for idx in idx_by_stratum])
        boot_b[b] = kendalltau(x[pick], y[pick], method="asymptotic")[0]
        boot_fixed[b] = tau_fixed(x[pick], y[pick])

    jack_b, jack_fixed = [], []
    for drop in range(n):
        keep = np.ones(n, dtype=bool)
        keep[drop] = False
        jack_b.append(kendalltau(x[keep], y[keep], method="asymptotic")[0])
        m = keep.sum()
        iu_k, ju_k = np.triu_indices(m, k=1)
        xk, yk = x[keep], y[keep]
        jack_fixed.append(float(np.sum(np.sign(xk[ju_k] - xk[iu_k]) * np.sign(yk[ju_k] - yk[iu_k])) / (m * (m - 1) / 2.0)))

    res = _bca_interval(theta_hat, boot_b, np.asarray(jack_b, dtype=float))
    alt = _bca_interval(theta_hat, boot_fixed, np.asarray(jack_fixed, dtype=float))
    res["tau"] = theta_hat
    res["ci_lo_fixed"], res["ci_hi_fixed"] = alt["ci_lo"], alt["ci_hi"]
    res["n_boot_nan"] = int(np.sum(~np.isfinite(boot_b)))
    return res


# ---- Verification against the machinery already in the repository ----

def _verify_tau_identity(values_by_group: list[np.ndarray], group_sizes: tuple[int, ...], label: str) -> None:
    """Check tau_b == (2J - M)/sqrt(M*n0) on the real data before relying on it."""
    y = np.concatenate(values_by_group)
    if len(np.unique(y)) < len(y):
        raise SystemExit(
            f"{label} has ties in the value variable, but the tau_b <-> J identity assumes "
            "none. Use the general tie-corrected form instead."
        )
    dose = np.concatenate([np.full(s, i, dtype=float) for i, s in enumerate(group_sizes)])
    tau_scipy = kendalltau(dose, y, method="asymptotic")[0]
    tau_ident = float(tau_b_from_J(J_vectorised(y[None, :], group_sizes)[0], group_sizes))
    if not np.isclose(tau_scipy, tau_ident, atol=1e-12):
        raise SystemExit(
            f"tau_b identity check failed on {label}: scipy={tau_scipy!r}, "
            f"identity={tau_ident!r}. Every power number here depends on it."
        )


def _verify_untied_null(null: dict) -> None:
    """Confirm the inversion-count null reproduces scipy's exact Kendall p-values."""
    n = null["n"]
    rng = np.random.default_rng(12345)
    for _ in range(5):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        tau, p_scipy = kendalltau(x, y, method="exact")
        p_here = exact_p_untied(tau, null)
        if not np.isclose(p_scipy, p_here, atol=1e-12):
            raise SystemExit(
                f"Inversion-count null disagrees with scipy's exact Kendall p at n={n}: "
                f"scipy={p_scipy!r}, here={p_here!r}."
            )


def _verify_against_published(new: pd.DataFrame, results_dir: Path) -> None:
    """
    Cross-check every recomputed tau and exact p against the CSVs the paper's
    numbers came from. A mismatch means this script is qualifying a different
    test than the one being reported, so it stops rather than continuing.
    """
    checks = [
        ("e1_kendall_tau.csv", "E1", "metric", "tau_full", "p_full_exact"),
        ("e2_kendall_tau.csv", "E2", "metric", "tau", "p_exact"),
        ("e2_6_kendall_tau.csv", "E2.6", "scorer", "tau_full", "p_full_exact"),
        ("e2_7_kendall_tau.csv", "E2.7", "probe", "tau_full", "p_full_exact"),
    ]
    problems = []
    for fname, family, key_col, tau_col, p_col in checks:
        path = results_dir / fname
        if not path.exists():
            problems.append(f"{fname}: missing, cannot cross-check {family}")
            continue
        published = pd.read_csv(path)
        sub = new[new["family"] == family]
        for row in sub.itertuples():
            match = published[published[key_col] == row.test]
            if len(match) != 1:
                problems.append(f"{fname}: expected exactly one row for {key_col}={row.test}, got {len(match)}")
                continue
            for col, mine, what in ((tau_col, row.tau, "tau"), (p_col, row.p_exact, "exact p")):
                theirs = float(match.iloc[0][col])
                if not np.isclose(theirs, mine, atol=1e-6):
                    problems.append(f"{fname} [{row.test}]: published {what}={theirs:.6g}, recomputed={mine:.6g}")
    if problems:
        raise SystemExit(
            "Recomputed statistics do not match the published results CSVs, so this would "
            "qualify a different test than the paper reports:\n  " + "\n  ".join(problems)
        )
    print("[power] Cross-check passed: recomputed tau and exact p match the published CSVs.\n")


# ---- Data assembly ----

def _grouped_values(df: pd.DataFrame, value_col: str) -> tuple[list[np.ndarray], tuple[int, ...]]:
    df = df[df["rung"].isin(PRIMARY_RUNGS)].copy()
    df["rung_index"] = df["rung"].map(RUNG_INDEX)
    df = df.sort_values(["rung_index", "seed"])
    groups = [df[df["rung_index"] == i][value_col].to_numpy(dtype=float) for i in range(len(PRIMARY_RUNGS))]
    return groups, tuple(len(g) for g in groups)


def collect_tests(results_dir: Path) -> list[dict]:
    """Every reported Kendall's tau, tagged with its design. Loads and groups only."""
    tests = []

    e1 = pd.read_csv(results_dir / "e1_geometry_metrics.csv")
    for metric in METRICS:
        groups, sizes = _grouped_values(e1, metric)
        tests.append({"family": "E1", "test": metric, "design": "B", "group_sizes": sizes,
                      "groups": groups, "label": f"E1: {metric} vs. $\\lambda_{{orth}}$"})

    e2 = pd.read_csv(results_dir / "e2_merged.csv")
    e2 = e2[e2["rung"].isin(PRIMARY_RUNGS)].copy()
    e2["rung_index"] = e2["rung"].map(RUNG_INDEX)
    e2 = e2.sort_values(["rung_index", "seed"])
    for metric in METRICS:
        tests.append({"family": "E2", "test": metric, "design": "A",
                      "x": e2[metric].to_numpy(dtype=float), "y": e2["auroc"].to_numpy(dtype=float),
                      "strata": e2["rung_index"].to_numpy(), "n": len(e2),
                      "label": f"E2: {metric} vs. AUROC"})

    e2_6 = pd.read_csv(results_dir / "e2_6_scorer_comparison.csv")
    for scorer in pd.read_csv(results_dir / "e2_6_kendall_tau.csv")["scorer"]:
        groups, sizes = _grouped_values(e2_6[e2_6["scorer"] == scorer], "auroc")
        tests.append({"family": "E2.6", "test": scorer, "design": "B", "group_sizes": sizes,
                      "groups": groups, "label": f"E2.6: {scorer} AUROC vs. $\\lambda_{{orth}}$"})

    e2_7 = pd.read_csv(results_dir / "e2_7_domain_probe.csv")
    for probe in pd.read_csv(results_dir / "e2_7_kendall_tau.csv")["probe"]:
        groups, sizes = _grouped_values(e2_7[e2_7["probe"] == probe], "domain_auroc")
        tests.append({"family": "E2.7", "test": probe, "design": "B", "group_sizes": sizes,
                      "groups": groups, "label": f"E2.7: {probe} AUROC vs. $\\lambda_{{orth}}$"})

    return tests


# ---- Jonckheere-Terpstra: both tail conventions, made explicit ----

def jt_conventions(tests: list[dict], nulls: dict) -> pd.DataFrame:
    rows = []
    for t in tests:
        if t["design"] != "B":
            continue
        sizes = t["group_sizes"]
        null = nulls[sizes]
        J_obs = float(J_vectorised(np.concatenate(t["groups"])[None, :], sizes)[0])
        mean_J = null["mean_J"]
        p_upper = float((null["J"] >= J_obs - TOL).mean())
        p_lower = float((null["J"] <= J_obs + TOL).mean())
        p_two_sided = float((np.abs(null["J"] - mean_J) >= abs(J_obs - mean_J) - TOL).mean())
        rows.append({
            "family": t["family"], "test": t["test"], "J": J_obs, "mean_J_null": mean_J,
            "p_exact_upper_tail": p_upper, "p_exact_lower_tail": p_lower,
            "p_exact_two_sided": p_two_sided,
            "p_as_currently_reported": p_upper,   # what exact_permutation_pvalue's |J| >= |J_obs| rule yields
            "convention_of_reported_p": "upper tail (one-sided)",
            "flips_at_alpha_0.05": bool((p_upper <= ALPHA) != (p_two_sided <= ALPHA)),
        })
    return pd.DataFrame(rows)


# ---- Figure ----

DESIGN_STYLE = {
    "A (n=13, continuous-continuous)": ("tab:blue", "-"),
    "B (5/5/3 ladder, n=13)": ("tab:orange", "-"),
    "B (3/3/3 common-seed, n=9)": ("tab:green", "--"),
}


def make_power_figure(curve: pd.DataFrame, summary: pd.DataFrame, ci: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax_pow, ax_ci) = plt.subplots(1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.0, 1.15]})

    # --- (A) power curves -------------------------------------------------
    for design, (color, ls) in DESIGN_STYLE.items():
        sub = curve[curve["design"] == design].sort_values("expected_tau")
        if not len(sub):
            continue
        ax_pow.plot(sub["expected_tau"], sub["power"], color=color, ls=ls, lw=2, label=design)
        row = summary[summary["design"] == design].iloc[0]
        if np.isfinite(row["tau_crit"]):
            ax_pow.axvline(row["tau_crit"], color=color, lw=0.9, alpha=0.45, ls=":")

    ax_pow.axhline(0.80, color="gray", lw=0.9, ls="--")
    ax_pow.axhline(ALPHA, color="gray", lw=0.9, ls=":")
    ax_pow.axvline(CONTRACT_TAU_THRESHOLD, color="crimson", lw=1.3, ls="--")
    ax_pow.text(CONTRACT_TAU_THRESHOLD + 0.012, 0.58, r"pre-registered $|\tau|\geq0.3$",
                color="crimson", fontsize=8, rotation=90, va="top")
    ax_pow.text(0.855, 0.815, "80% power", color="gray", fontsize=8, ha="right")

    ann = []
    for design in DESIGN_STYLE:
        row = summary[summary["design"] == design]
        if len(row):
            r = row.iloc[0]
            ann.append(f"{design}\n   power at $\\tau=0.3$: {r['power_at_tau_0.3']:.2f}"
                       f"   |   MDE$_{{80\\%}}$: {r['mde_power80']:.2f}"
                       f"   |   $\\tau_{{crit}}$: {r['tau_crit']:.2f}")
    ax_pow.text(0.02, 0.98, "\n".join(ann), transform=ax_pow.transAxes, fontsize=7.4,
                va="top", ha="left", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))

    ax_pow.set_xlabel(r"expected Kendall's $\tau$ under the alternative", fontsize=10)
    ax_pow.set_ylabel(r"power (exact permutation test, $\alpha=0.05$, two-sided)", fontsize=10)
    ax_pow.set_xlim(0, 0.86)
    ax_pow.set_ylim(0, 1.0)
    ax_pow.set_title("(A) What this design could have detected", fontsize=11)
    ax_pow.legend(fontsize=7.6, loc="lower right")
    ax_pow.grid(alpha=0.25)

    # --- (B) observed tau with bootstrap CIs -------------------------------
    ci = ci.sort_values(["family", "test"], ascending=[False, False]).reset_index(drop=True)
    ypos = np.arange(len(ci))
    fam_color = {"E1": "tab:purple", "E2": "tab:blue", "E2.6": "tab:orange", "E2.7": "tab:green"}
    for k, row in ci.iterrows():
        c = fam_color.get(row["family"], "k")
        if row["at_design_ceiling"]:
            # tau sits at the design's attainable maximum -> the bootstrap
            # distribution is one-sided by construction and no interval is
            # valid. Draw the point only, and say why.
            ax_ci.plot(row["tau"], k, "*", color=c, ms=11, mec="k", mew=0.5)
            ax_ci.annotate("at design ceiling:\nno valid bootstrap CI\n"
                           f"(exact $p$={row['p_exact']:.1e})", (row["tau"], k),
                           textcoords="offset points", xytext=(-10, -30), fontsize=6.4,
                           ha="right", va="center", color=c)
            continue
        ax_ci.plot([row["ci_lo"], row["ci_hi"]], [k, k], color=c, lw=1.8, alpha=0.85, solid_capstyle="butt")
        ax_ci.plot(row["tau"], k, "o", color=c, ms=5)

    crit_A = float(summary[summary["design"] == "A (n=13, continuous-continuous)"].iloc[0]["tau_crit"])
    crit_B = float(summary[summary["design"] == "B (5/5/3 ladder, n=13)"].iloc[0]["tau_crit"])
    ax_ci.axvspan(-crit_B, crit_B, color="crimson", alpha=0.055, zorder=0)
    for c in (-crit_B, crit_B):
        ax_ci.axvline(c, color="crimson", lw=1.0, ls="--", alpha=0.7, zorder=1)
    for c in (-crit_A, crit_A):
        ax_ci.axvline(c, color="tab:blue", lw=1.0, ls=":", alpha=0.7, zorder=1)
    ax_ci.axvline(0, color="k", lw=0.8, zorder=1)

    ax_ci.set_yticks(ypos)
    ax_ci.set_yticklabels([f"{r['family']}: {r['test']}" for _, r in ci.iterrows()], fontsize=7.2)
    ax_ci.set_xlabel(r"Kendall's $\tau$ (point estimate and 95% bootstrap CI)", fontsize=10)
    ax_ci.set_xlim(-1.02, 1.02)
    ax_ci.set_ylim(-1.5, len(ci) - 0.4)
    ax_ci.set_title("(B) What the data exclude", fontsize=11)
    ax_ci.text(0.0, -1.15, "shaded band: no $|\\tau|$ inside it can reach $p\\leq0.05$ at this design "
               "(dashed = 5/5/3 ladder, dotted = continuous-continuous)",
               fontsize=6.8, ha="center", va="center", color="crimson")
    ax_ci.grid(axis="x", alpha=0.25)

    fig.suptitle("Effect-size and power bounds on this study's null associations "
                 "(exact permutation tests, $\\alpha=0.05$)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"figure_s_power.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[power] Wrote {out_dir / 'figure_s_power.pdf'} (+ .png)")


# ---- Main ----

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--figures-dir", type=Path, default=Path("figures"))
    ap.add_argument("--n-boot", type=int, default=20000, help="bootstrap replicates per tau")
    ap.add_argument("--n-sim", type=int, default=20000, help="simulated datasets per power-curve point")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tests = collect_tests(args.results_dir)

    # ---- exact nulls, one per distinct design ----------------------------
    grouped_sizes = sorted({t["group_sizes"] for t in tests if t["design"] == "B"} | {(3, 3, 3)})
    nulls = {}
    for sizes in grouped_sizes:
        nulls[sizes] = grouped_null(sizes)
        print(f"[power] Design B {sizes}: enumerated {nulls[sizes]['n_arrangements']:,} arrangements, "
              f"max attainable |tau_b| = {nulls[sizes]['max_abs_tau']:.4f}")

    n_A = tests[[t["design"] for t in tests].index("A")]["n"]
    null_A = untied_null(n_A)
    _verify_untied_null(null_A)
    print(f"[power] Design A n={n_A}: inversion-count null verified against scipy's exact Kendall p-values.")

    for t in tests:
        if t["design"] == "B":
            _verify_tau_identity(t["groups"], t["group_sizes"], f"{t['family']}/{t['test']}")
    print("[power] tau_b <-> Jonckheere-J identity verified on every Design-B test's real data.\n")

    # ---- design-level critical values and power curves -------------------
    designs = {
        "A (n=13, continuous-continuous)": {"kind": "A", "n": n_A, "null_taus": None},
        "B (5/5/3 ladder, n=13)": {"kind": "B", "sizes": (5, 5, 3)},
        "B (3/3/3 common-seed, n=9)": {"kind": "B", "sizes": (3, 3, 3)},
    }
    tau_grid = np.round(np.arange(0.0, 0.861, 0.01), 4)
    curve_rows, summary_rows = [], []

    for name, spec in designs.items():
        if spec["kind"] == "A":
            tau_crit, p_crit = critical_tau_untied(null_A)
            max_tau, n_arr, n_here, sizes_str = 1.0, "13! orderings", n_A, "-"
            usable = tau_grid[tau_grid < 0.999]
            power = power_curve_untied(n_A, usable, tau_crit, args.n_sim,
                                       np.random.default_rng(args.seed + 1))
        else:
            sizes = spec["sizes"]
            if sizes not in nulls:
                nulls[sizes] = grouped_null(sizes)
            null = nulls[sizes]
            tau_crit, p_crit = critical_tau(null["tau"])
            max_tau, n_arr, n_here = null["max_abs_tau"], f"{null['n_arrangements']:,}", null["n"]
            sizes_str = "/".join(str(s) for s in sizes)
            usable = tau_grid[tau_grid < max_tau - 1e-6]
            power = power_curve_grouped(sizes, usable, tau_crit, args.n_sim,
                                        np.random.default_rng(args.seed + 1))

        for t, p in zip(usable, power):
            curve_rows.append({"design": name, "expected_tau": float(t), "power": float(p)})

        def power_at(target: float) -> float:
            if target > usable.max():
                return float("nan")
            return float(np.interp(target, usable, power))

        summary_rows.append({
            "design": name, "n": n_here, "group_sizes": sizes_str, "null_space": n_arr,
            "max_attainable_abs_tau": float(max_tau),
            "tau_crit": tau_crit, "p_at_tau_crit": p_crit,
            "contract_threshold_0.3_attainable": bool(np.isfinite(tau_crit) and tau_crit <= CONTRACT_TAU_THRESHOLD),
            "power_at_tau_0.3": power_at(0.30),
            "power_at_tau_0.5": power_at(0.50),
            "power_at_tau_crit": power_at(tau_crit) if np.isfinite(tau_crit) else float("nan"),
            "mde_power50": interpolate_crossing(usable, power, 0.50),
            "mde_power80": interpolate_crossing(usable, power, 0.80),
            "n_sim": args.n_sim, "alpha": ALPHA,
        })

    summary = pd.DataFrame(summary_rows)
    curve = pd.DataFrame(curve_rows)

    # ---- per-test observed tau, exact p, bootstrap CI --------------------
    ci_rows = []
    for t in tests:
        if t["design"] == "B":
            sizes = t["group_sizes"]
            boot = bootstrap_tau_grouped(t["groups"], args.n_boot, np.random.default_rng(args.seed + 7))
            p_exact = exact_p_grouped(boot["tau"], nulls[sizes]["tau"])
            design_name = f"B ({'/'.join(map(str, sizes))} ladder, n={sum(sizes)})"
            row = summary[summary["design"].str.startswith("B (" + "/".join(map(str, sizes)))].iloc[0]
        else:
            boot = bootstrap_tau_pairs(t["x"], t["y"], t["strata"], args.n_boot, np.random.default_rng(args.seed + 7))
            p_exact = exact_p_untied(boot["tau"], null_A)
            design_name = "A (n=13, continuous-continuous)"
            row = summary[summary["design"] == design_name].iloc[0]
        tau_crit = float(row["tau_crit"])
        max_tau = float(row["max_attainable_abs_tau"])
        at_ceiling = bool(abs(boot["tau"]) >= max_tau - 1e-9)

        ci_rows.append({
            "family": t["family"], "test": t["test"], "design": design_name,
            "tau": boot["tau"], "p_exact": p_exact,
            "at_design_ceiling": at_ceiling,
            "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "ci_method": boot["ci_method"],
            "ci_width": boot["ci_hi"] - boot["ci_lo"],
            "ci_lo_percentile": boot["ci_lo_pct"], "ci_hi_percentile": boot["ci_hi_pct"],
            "ci_lo_fixed_denominator": boot["ci_lo_fixed"], "ci_hi_fixed_denominator": boot["ci_hi_fixed"],
            "ci_contains_zero": bool(boot["ci_lo"] <= 0.0 <= boot["ci_hi"]),
            "ci_excludes_abs_tau_0.3": bool(max(abs(boot["ci_lo"]), abs(boot["ci_hi"])) < CONTRACT_TAU_THRESHOLD),
            "largest_effect_not_excluded": float(max(abs(boot["ci_lo"]), abs(boot["ci_hi"]))),
            "tau_crit_for_design": tau_crit,
            "bca_degenerate": boot["bca_degenerate"], "z0": boot["z0"], "acceleration": boot["acceleration"],
            "n_boot": args.n_boot, "n_boot_nan": boot["n_boot_nan"],
        })
    ci = pd.DataFrame(ci_rows)
    _verify_against_published(ci, args.results_dir)

    jt = jt_conventions(tests, nulls)

    # ---- write -----------------------------------------------------------
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for df, name in ((summary, "power_design_summary.csv"), (curve, "power_curve.csv"),
                     (ci, "kendall_tau_ci.csv"), (jt, "jt_pvalue_conventions.csv")):
        df.to_csv(args.results_dir / name, index=False)
        print(f"[power] Wrote {args.results_dir / name}  ({len(df)} rows)")

    make_power_figure(curve, summary, ci, args.figures_dir)

    # ---- console summary -------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n=== Design-level power (exact permutation, alpha=0.05, two-sided) ===")
    print(summary[["design", "n", "group_sizes", "max_attainable_abs_tau", "tau_crit",
                   "contract_threshold_0.3_attainable", "power_at_tau_0.3", "power_at_tau_0.5",
                   "mde_power80"]].to_string(index=False))

    print("\n=== Observed tau with 95% bootstrap CI ===")
    print(ci[["family", "test", "tau", "p_exact", "ci_lo", "ci_hi", "ci_lo_fixed_denominator",
              "ci_hi_fixed_denominator", "ci_contains_zero", "ci_excludes_abs_tau_0.3",
              "largest_effect_not_excluded", "at_design_ceiling"]].to_string(index=False))

    flips = jt[jt["flips_at_alpha_0.05"]]
    print(f"\n=== Jonckheere-Terpstra tail convention: {len(flips)} of {len(jt)} tests change "
          f"significance at alpha=0.05 between the upper-tail p currently reported and a two-sided p ===")
    if len(flips):
        print(flips[["family", "test", "J", "mean_J_null", "p_exact_upper_tail", "p_exact_two_sided"]].to_string(index=False))

    ci_valid = ci[~ci["at_design_ceiling"]]
    n_excl = int(ci_valid["ci_excludes_abs_tau_0.3"].sum())
    ci = ci  # (ci_valid used only for the summary line below)
    print(f"\n[power] Of the {len(ci_valid)} associations with a valid bootstrap interval "
          f"({len(ci) - len(ci_valid)} excluded as sitting at the design ceiling), {n_excl} have a 95% CI "
          f"that excludes |tau| = {CONTRACT_TAU_THRESHOLD}; the remaining {len(ci_valid) - n_excl} do not "
          "rule out an association of at least that size.")


if __name__ == "__main__":
    main()
