#!/usr/bin/env python3
"""
Compare the ladder across however many stages it has been through.

compare_n13_n15.py handles two. The ladder has since passed through three that
matter and are not interchangeable:

  1. the published 13-checkpoint result
  2. 15 checkpoints, but trained across two software environments and, for two
     runB seeds, extracted from a superseded checkpoint
  3. 15 checkpoints trained and extracted in one environment

Stage 3 is not a correction of stage 2 in the way stage 2 corrected stage 1:
training is non-deterministic here, so re-running a seed produces a different
model. Stage 3 is an independent replication, and the comparison is between two
samples rather than between a value and its corrected self.

    python3 analysis/compare_stages.py LABEL=DIR [LABEL=DIR ...]
    # the last one may be "current" to read results/ directly
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNGS = ["runA_grl", "runB_orth1", "runB"]
GEO = ["condition_number", "fisher_ratio_HL", "fisher_ratio_scalar",
       "mardia_kurtosis_b", "mardia_kurtosis_z"]


def load(d: Path, name: str) -> pd.DataFrame | None:
    p = d / name
    return pd.read_csv(p) if p.is_file() else None


def fmt(m, s):
    if pd.isna(m):
        return "--"
    return f"{m:,.1f} ± {s:,.1f}" if abs(m) >= 100 else f"{m:.3f} ± {s:.3f}"


def main() -> None:
    stages = []
    for arg in sys.argv[1:]:
        label, _, path = arg.rpartition("=")   # nhãn có thể chứa "=", đường dẫn thì không
        stages.append((label, ROOT / "results" if path == "current" else Path(path)))
    if len(stages) < 2:
        sys.exit("cần ít nhất 2 mốc: LABEL=DIR LABEL=DIR ...")

    out = [f"# Ladder qua {len(stages)} mốc\n",
           "Sinh bởi `analysis/compare_stages.py`. Mốc cuối là bản hiện hành.\n"]

    out.append("\n## Số checkpoint\n")
    out.append("| nấc | " + " | ".join(l for l, _ in stages) + " |")
    out.append("|---" * (len(stages) + 1) + "|")
    for r in RUNGS:
        cells = []
        for _, d in stages:
            g = load(d, "e1_geometry_metrics.csv")
            cells.append("--" if g is None else str((g.rung == r).sum()))
        out.append(f"| `{r}` | " + " | ".join(cells) + " |")

    out.append("\n## Geometry theo nấc (mean ± SD)\n")
    out.append("| metric | nấc | " + " | ".join(l for l, _ in stages) + " |")
    out.append("|---" * (len(stages) + 2) + "|")
    for m in GEO:
        for r in RUNGS:
            cells = []
            for _, d in stages:
                g = load(d, "e1_geometry_metrics.csv")
                s = g[g.rung == r][m] if g is not None else pd.Series(dtype=float)
                cells.append("--" if not len(s) else fmt(s.mean(), s.std()))
            out.append(f"| `{m}` | `{r}` | " + " | ".join(cells) + " |")

    out.append("\n## Mahalanobis AUROC theo nấc\n")
    out.append("| nấc | " + " | ".join(l for l, _ in stages) + " |")
    out.append("|---" * (len(stages) + 1) + "|")
    for r in RUNGS:
        cells = []
        for _, d in stages:
            a = load(d, "e2_auroc.csv")
            s = a[a.rung == r]["auroc"] if a is not None else pd.Series(dtype=float)
            cells.append("--" if not len(s) else fmt(s.mean(), s.std()))
        out.append(f"| `{r}` | " + " | ".join(cells) + " |")

    for fn, key, title in [("e1_kendall_tau.csv", "metric", "E1 — geometry vs λ_orth"),
                           ("e2_kendall_tau.csv", "metric", "E2 — geometry vs AUROC"),
                           ("e2_6_kendall_tau.csv", "scorer", "E2.6 — scorer vs λ_orth"),
                           ("e2_7_kendall_tau.csv", "probe", "E2.7 — probe vs λ_orth")]:
        out.append(f"\n## {title} (τ, p)\n")
        out.append("| | " + " | ".join(l for l, _ in stages) + " |")
        out.append("|---" * (len(stages) + 1) + "|")
        rows = None
        frames = []
        for _, d in stages:
            f = load(d, fn)
            frames.append(None if f is None else f.set_index(key))
            if f is not None and rows is None:
                rows = list(f[key])
        for k in rows or []:
            cells = []
            for f in frames:
                if f is None or k not in f.index:
                    cells.append("--"); continue
                t = "tau_full" if "tau_full" in f.columns else "tau"
                p = ("p_full_exact" if "p_full_exact" in f.columns
                     else ("p_exact" if "p_exact" in f.columns else None))
                cells.append(f"{f.loc[k, t]:+.3f}, {f.loc[k, p]:.3g}" if p else f"{f.loc[k, t]:+.3f}")
            out.append(f"| `{k}` | " + " | ".join(cells) + " |")

    text = "\n".join(out) + "\n"
    (ROOT / "docs" / "ladder_stages.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
