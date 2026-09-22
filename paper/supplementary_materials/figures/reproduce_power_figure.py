#!/usr/bin/env python3
"""Recompute design power and regenerate Figure 7 from disclosed inputs.

Panel A is recomputed by Monte Carlo with common random numbers. Panel B uses
the disclosed BCa output table. Recomputing the BCa endpoints themselves also
requires the original 13-row checkpoint output matrix; see README.md.
"""
from pathlib import Path
import csv, json, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "analysis/seed_manifest.json").read_text())
RNG = np.random.default_rng(CFG["power_simulation_seed"])
N_SIM = CFG["n_simulations_per_grid_point"]
GRID = np.arange(CFG["power_grid_start"], CFG["power_grid_stop"] + 1e-9,
                 CFG["power_grid_step"])

DESIGNS = {
    # Exact attainable thresholds; manuscript values are rounded to 3 decimals.
    "continuous_continuous": {"n": 13, "crit": 34/78},
    "ordered_5_5_3": {"dose": np.repeat([0., 1., 2.], [5, 5, 3]), "crit": 31/math.sqrt(55*78)},
    "ordered_3_3_3": {"dose": np.repeat([0., 1., 2.], [3, 3, 3]), "crit": 19/math.sqrt(27*36)},
}

def expected_tau_b(theta, dose):
    n = len(dose); n0 = n * (n - 1) / 2
    dx = dose[:, None] - dose[None, :]
    mask = np.triu(dx != 0, 1)
    m = mask.sum()
    expected_sign = 2 * norm.cdf(theta * np.abs(dx[mask]) / math.sqrt(2)) - 1
    return expected_sign.sum() / math.sqrt(m * n0)

def theta_for_tau(target, dose):
    if target == 0: return 0.0
    ceiling = expected_tau_b(100., dose)
    if target >= ceiling: return 100.
    return brentq(lambda z: expected_tau_b(z, dose) - target, 0., 100.)

def simulate_curve(name, spec):
    powers = []
    if name == "continuous_continuous":
        n = spec["n"]
        base = RNG.standard_normal((N_SIM, n, 2))
        ii, jj = np.triu_indices(n, 1)
        for target in GRID:
            rho = math.sin(math.pi * target / 2)
            x = base[:, :, 0]
            y = rho * x + math.sqrt(max(0., 1-rho*rho)) * base[:, :, 1]
            concord = np.sign(x[:, ii] - x[:, jj]) * np.sign(y[:, ii] - y[:, jj])
            vals = concord.mean(axis=1)
            powers.append(np.mean(np.abs(vals) >= spec["crit"] - 1e-12))
    else:
        dose = spec["dose"]
        eps = RNG.standard_normal((N_SIM, len(dose)))
        ceiling = expected_tau_b(100., dose)
        n = len(dose); n0 = n * (n - 1) / 2
        ii, jj = np.triu_indices(n, 1)
        cross = dose[ii] != dose[jj]
        ii, jj = ii[cross], jj[cross]
        xsign = np.sign(dose[ii] - dose[jj])
        denom = math.sqrt(len(ii) * n0)
        for target in GRID:
            theta = theta_for_tau(min(target, ceiling - 1e-9), dose)
            y = theta * dose + eps
            vals = (np.sign(y[:, ii] - y[:, jj]) * xsign).sum(axis=1) / denom
            powers.append(np.mean(np.abs(vals) >= spec["crit"] - 1e-12))
    return np.asarray(powers)

curves = {name: simulate_curve(name, spec) for name, spec in DESIGNS.items()}
with (ROOT / "results/power_curves_reproduced.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["expected_tau", *curves])
    for i, t in enumerate(GRID): w.writerow([f"{t:.2f}", *[f"{curves[k][i]:.5f}" for k in curves]])

rows = list(csv.DictReader((ROOT / "results/kendall_tau_ci.csv").open()))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.3))
labels = {"continuous_continuous":"continuous-continuous n=13",
          "ordered_5_5_3":"ordered 5/5/3 n=13", "ordered_3_3_3":"ordered 3/3/3 n=9"}
for k, v in curves.items(): ax1.plot(GRID, v, label=labels[k])
ax1.axhline(.8, color="0.5", ls="--", lw=1); ax1.axvline(.3, color="0.3", ls=":", lw=1)
ax1.set(xlabel="Expected Kendall tau", ylabel="Power", ylim=(0, 1.02), title="A. Exact-test power")
ax1.legend(frameon=False, fontsize=8)

y = np.arange(len(rows)); tau = np.array([float(r["tau"]) for r in rows])
lo = np.array([float(r["ci_low"]) if r["ci_low"] else np.nan for r in rows])
hi = np.array([float(r["ci_high"]) if r["ci_high"] else np.nan for r in rows])
ok = ~np.isnan(lo)
ax2.errorbar(tau[ok], y[ok], xerr=np.vstack((tau[ok]-lo[ok], hi[ok]-tau[ok])), fmt="o", ms=3, capsize=2)
ax2.plot(tau[~ok], y[~ok], "*", ms=9)
ax2.axvspan(-.473, .473, color="0.9"); ax2.axvline(0, color="0.4", lw=1)
ax2.set(xlabel="Observed Kendall tau (95% BCa CI)", yticks=y,
        yticklabels=[r["test"] for r in rows], title="B. Reported associations")
ax2.invert_yaxis(); ax2.tick_params(axis="y", labelsize=6)
fig.tight_layout()
fig.savefig(ROOT / "figs/fig7_power.pdf")
fig.savefig(ROOT / "figs/fig7_power.png", dpi=300)
print("Wrote results/power_curves_reproduced.csv and figs/fig7_power.{pdf,png}")
