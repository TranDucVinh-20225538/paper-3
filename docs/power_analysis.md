# Power, detectability, and interval estimates for Paper 3's null results

**Script**: [`analysis/analyze_power_and_ci.py`](../analysis/analyze_power_and_ci.py)
**Run**: `python3 analysis/analyze_power_and_ci.py --n-boot 20000 --n-sim 20000 --seed 0` (~55 s, CPU only)
**Inputs**: existing `results/*.csv` only — no checkpoint is touched, nothing is retrained.
**Outputs**: `results/power_design_summary.csv`, `results/power_curve.csv`, `results/kendall_tau_ci.csv`, `results/jt_pvalue_conventions.csv`, `figures/figure_s_power.{pdf,png}`.

## Why this exists

The Results section reports twenty non-significant Kendall's τ values and the Discussion bounded all of them with one phrase: *"the modest power available at n=13."* That is an admission without a number. This analysis replaces it with three quantities per test, computed on data already collected.

## What is computed

### 1. τ_crit — what the design can detect at all

The smallest |τ| attainable at a design whose two-sided exact permutation *p*-value reaches 0.05. A property of *n*, the group sizes, and the tie structure — **not** of the data. No dataset can produce a significant association below it.

| Design | n | groups | null space | max attainable \|τ\| | **τ_crit** |
|---|---|---|---|---|---|
| A — continuous/continuous (E2: geometry vs. AUROC) | 13 | — | 13! orderings | 1.000 | **0.436** |
| B — ordered ladder (E1, E2.6, E2.7 vs. λ_orth) | 13 | 5/5/3 | 72,072 | 0.840 | **0.473** |
| B — common-seed subset | 9 | 3/3/3 | 1,680 | 0.866 | **0.609** |

**The consequence worth stating in the paper**: `experiment_contract.md`'s pre-registered success criterion is `|τ| ≥ 0.3 AND significant at α=0.05` (`CONTRACT_TAU_THRESHOLD = 0.3` in `analyze_e1.py`). Since 0.3 < 0.436 ≤ τ_crit for every design used, **that conjunction was unsatisfiable before any data were collected**. An association of exactly the pre-registered magnitude could never have been declared significant. Better to state this ourselves than to have a reviewer derive it.

### 2. Power

Simulation, 2×10⁴ datasets per grid point, common random numbers across the grid (so the curve is monotone and interpolable). The alternative is parameterized so that the **expected value of the reported statistic** equals the target τ — not some latent parameter loosely corresponding to it:

- Design A: bivariate normal, `E[τ] = (2/π)·arcsin(ρ)` exactly, so `ρ = sin(πτ/2)`.
- Design B: linear dose shift `y_i = θ·d_i + ε_i`, `ε ~ N(0,1)`, `d = 0,1,2`. `E[τ_b] = (2·E[C] − M)/√(M·n₀)` with `E[C] = Σ_{i<j} n_i n_j Φ(θ(d_j−d_i)/√2)` — closed form, inverted by Brent for θ.

| Design | power at τ=0.3 | power at τ=0.5 | MDE at 50% | **MDE at 80%** |
|---|---|---|---|---|
| A (n=13) | **0.27** | 0.71 | 0.41 | **0.54** |
| B (5/5/3, n=13) | **0.23** | 0.63 | 0.44 | **0.57** |
| B (3/3/3, n=9) | **0.14** | 0.39 | 0.56 | **0.69** |

### 3. Bootstrap intervals — what the data exclude

95% BCa, 2×10⁴ resamples, **stratified within rung** (λ_orth is a fixed design factor, not a sampled one; resampling across rungs would bootstrap a quantity the experiment never estimated). Percentile intervals reported alongside; BCa degeneracy is flagged, never silently substituted.

Result: **19 of the 20 null associations have an interval wide enough to contain |τ| = 0.3.** The single exception is ViM (τ = 0.076, CI [−0.268, 0.270]) — the only null result in the study that genuinely excludes an association of the pre-registered size.

This is the honest outcome and it is *not* the flattering one: the intervals confirm the nulls are weak individually. The paper's claim survives because it rests on the consistency of near-zero estimates across twenty tests, not on any single interval — and the Limitations paragraph now says so, including the caveat that those twenty are not mutually independent.

## Design decisions worth knowing

**Two bootstrap statistics, both reported.** Resampling with replacement necessarily creates ties, and the two standard tie treatments pull the interval in *opposite* directions: scipy's τ_b shrinks its denominator `√((n₀−t_x)(n₀−t_y))` as ties accumulate and so *inflates* |τ| on tied resamples, while the design-normalized `(2J−M)/√(M·n₀)` pins the denominator to the original design and cannot exceed the design ceiling. They are identical on the observed tie-free data. Both intervals are in `kendall_tau_ci.csv` (`ci_lo/ci_hi` and `ci_lo_fixed_denominator/ci_hi_fixed_denominator`); endpoints differ by a median of 0.046 and at most 0.091, and the two agree on every |τ| = 0.3 exclusion verdict (ViM under both), so no conclusion depends on the choice.

**The design-ceiling case.** `condition_number` vs. λ_orth attains τ = 0.8397, which *is* the maximum the 5/5/3 design permits (a perfectly monotone ladder, J = 55 = M). Every resample therefore lies at or below the observed value, the resampling distribution is one-sided by construction, and **no bootstrap interval is valid**. Flagged as `at_design_ceiling` in the CSV, drawn as a point without an interval in the figure, and reported as such in the supplementary table. The exact permutation *p* (2.8×10⁻⁵) is the meaningful inference there. An earlier draft of this analysis produced the interval [0.851, 0.893] for it — which does not even contain its own point estimate — which is exactly the artifact this handling exists to prevent.

**The τ_b ↔ J identity.** With a tie-free response, `τ_b = (2J − M)/√(M·n₀)` exactly, where J is Jonckheere–Terpstra's statistic. This is verified numerically against scipy on every Design-B test's real data before any power number derived from it is used (`_verify_tau_identity`), and it is what makes the exact nulls and power simulations cheap enough to run by full enumeration rather than approximation.

**Verification.** Every recomputed τ and exact *p* is cross-checked against the published `results/e{1,2,2_6,2_7}_kendall_tau.csv` before anything is written (`_verify_against_published`); a mismatch is a hard stop, because it would mean this analysis is qualifying a different test than the paper reports. The Design-A inversion-count null is separately checked against `scipy.stats.kendalltau(method="exact")`.

## Two issues this analysis surfaced — decisions needed

### (a) Jonckheere–Terpstra *p*-values are one-sided, and were labeled two-sided

`analyze_e1.exact_permutation_pvalue` counts `|stat_perm| ≥ |stat_obs|`. For Kendall's τ that is a correct two-sided rule (τ is centered at 0 under the null). For J it is not: J is centered at `mean_J = 27.5` and is never negative, so the rule degenerates into an **upper-tail one-sided** test. The docstring calls the result two-sided, and the normal-approximation *p* printed in the same row genuinely is two-sided — the two columns are not comparable.

Full recomputation under both conventions is in `results/jt_pvalue_conventions.csv`. Exactly one test changes significance at α = 0.05:

| test | J | one-sided (as reported) | two-sided |
|---|---|---|---|
| E2.6 energy | 42.0 | **0.032** | **0.065** |

A one-sided JT test is the *conventional* reading (it is an ordered-alternative test), so the number is defensible — it just has to be **labeled** one-sided. What must change is the impression that it is independent corroboration: by the τ_b ↔ J identity, the two-sided JT *p* is **identical** to the Kendall τ *p* at every one of these designs (verify in the CSV: every `p_exact_two_sided` equals the matching Kendall exact *p*). Energy's "τ = 0.44, p = 0.065" and "JT p = 0.032" are one piece of evidence read two ways, not two.

**Applied**: `methods.tex` now states the tail convention and the identity; `discussion.tex`'s Energy sentence carries a clause saying the two are not independent. **Not applied**: nothing about the Energy result's substance was changed — it remains reported as a hypothesis-generating observation, as before.

### (b) Supplementary numbering drift — fixed

`supplementary.tex` typed its section headings literally (`S1.`…`S7.`) while letting LaTeX auto-number the figures and tables, and section S3 holds a table but no figure. The two had drifted apart: section "S4" contained **Figure S3**, "S5" contained **Figure S4**, "S6" contained **Figure S5**, and section "S7" contained **Table S4** — while `results.tex` referred to that last one, by hardcoded string, as "Supplementary Table~S7".

**Fixed by making the drift impossible rather than by renumbering once.** A new `\suppsec{<n>}{<title>}` macro opens section S*n* *and* pins the figure and table counters to *n*, so heading number and float numbers are driven by the same argument:

```latex
\newcommand{\suppsec}[2]{%
  \setcounter{figure}{#1}\addtocounter{figure}{-1}%
  \setcounter{table}{#1}\addtocounter{table}{-1}%
  \subsection*{S#1. #2}%
}
```

Rendered numbering is now Figures S1, S2, S4, S5, S6, S8 and Tables S1, S2, S3, S7, S8, S9 — each inside the section of the same number — with zero duplicate-destination or multiply-defined warnings from hyperref. All six hardcoded supplementary references in the main text (`results.tex` ×3, `methods.tex` ×1, `discussion.tex` ×2) now name items that exist and carry the number they claim.

The macro carries one invariant: **at most one figure and one table per section**. The material added by this analysis originally put a figure and *two* tables in a single section S8, which the scheme cannot number; it is therefore split into S8 (detectability and power — figure + design table) and S9 (interval estimates — the per-test table). Any future section needing a second float of the same kind must be split the same way.

**Why not `xr`.** The obvious alternative — `\usepackage{xr}` + `\externaldocument{supplementary}` in `paper.tex`, then real `\ref`s — was considered and rejected: it would make `paper.tex` require `supplementary.aux` at compile time, and an arXiv build of the main PDF alone (the supplementary is submitted as a separate file) would render every such reference as `??`. The dependency currently runs one way only, supplementary → paper, and that stays. Hardcoded strings in the main text are the deliberate cost; guaranteeing the numbers they name is the mitigation.

## Manuscript changes applied

| File | Change |
|---|---|
| `sections/methods.tex` | New `\subsection{Detectability, power, and interval estimates}` (`\label{sec:power}`); tail-convention + τ_b↔J identity paragraph added to `\subsection{Statistical testing}` |
| `sections/discussion.tex` | Limitations paragraph 3 replaced with the quantified version (τ_crit, unsatisfiable pre-registered criterion, power at 0.3, MDE₈₀); new paragraph on what the intervals exclude; Energy sentence gains the one-sided clause |
| `supplementary.tex` | New `\suppsec` macro pinning float counters to section numbers (fixes the pre-existing drift); all headings converted to it; new sections S8 (figure + power table) and S9 (interval table) |
| `tables/table_s8_power.tex` | New — design-level detectability |
| `tables/table_s9_tau_ci.tex` | New — all 21 τ with intervals |

Both `paper.tex` and `supplementary.tex` compile clean (no errors, no undefined references).

## Carried into the TMLR version

The `elsarticle`/BSPC manuscript in `paper/arxiv/` and the TMLR manuscript in `paper/tmlr/` are **different versions of the paper, not two builds of one source.** The TMLR revision reframes the finding as directional misalignment, and it cut the scorer set from eight to five variants of three families (Mahalanobis, cosine-to-centroid, pooled *k*-NN at *k* = 1, 10, 50). Energy, ViM and the KDE scorer are not in it. Three consequences for this analysis, all of which the TMLR text reflects:

- **Eighteen association tests, not twenty-one.** Seventeen non-significant, plus the condition-number effect at its design ceiling.
- **No interval excludes |τ| = 0.3 any more.** ViM was the single exclusion in the eight-scorer set, and ViM is gone. The TMLR Limitations says so directly — *every* one of the seventeen admits a moderate association, narrowest 0.40, widest 0.75 — rather than carrying over the "19 of 20, ViM the exception" sentence, which would be false at this scope.
- **The Energy tail-convention flip is moot.** Issue (a) below changed one test's significance at α = 0.05, and that test is not in this version. What does carry over is the labelling: the TMLR `Statistical testing` section states the one-sided upper-tail convention and the τ_b ↔ J identity, so the Jonckheere–Terpstra *p*-values it reports are not read as independent corroboration of the Kendall tests beside them.

Sixteen of the seventeen point estimates fall below |τ| = 0.3; the exception is cosine-to-centroid (τ = 0.32, non-significant), which the Results section now names rather than leaving for a reader to notice in the table.

| File | Role |
|---|---|
| `paper/tmlr/common_body.tex` | Abstract power-bound clause; `Statistical testing` gains the analysis-plan criterion, the tail convention and the τ_b↔J identity; new §`Detectability, power, and interval estimates` (`sec:power`); three Results null-result sentences quantified; Limitations power paragraph replaced and an interval paragraph added |
| `paper/tmlr/appendix.tex` | New — Appendix A, both subsections, and the figure |
| `paper/tmlr/tables/table_{power_design,tau_ci}.tex` | Generated, not hand-written |
| `paper/tmlr/figs/fig7_power.pdf` | Generated; panels stacked, not side by side, because the TMLR layout is single-column |
| `analysis/make_figure_power_tmlr.py` | Redraws the figure over the reported subset. Recomputes nothing |
| `analysis/make_tables_power_tmlr.py` | Emits both appendix tables from the CSVs |
| `paper/tmlr/references.bib` | `efron1987better` added for BCa |

Rebuild: `python3 analysis/make_figure_power_tmlr.py && python3 analysis/make_tables_power_tmlr.py`, then `pdflatex → bibtex → pdflatex ×2` in `paper/tmlr/`. Compiles with no warnings and no undefined references; 25 pages.

Appendix numbering needs no `\suppsec` equivalent — TMLR is one document, so `\ref` resolves and the hardcoded-string problem of issue (b) does not arise here.

## Still open

- **TMLR submission policy.** The anonymity and review-visibility rules are not recorded here and should be checked against the current TMLR author guide before submitting; `paper/tmlr/main_arxiv.tex` currently uses `\usepackage[preprint]{tmlr}` (de-anonymized, no "Under review" banner), which is a preprint build, not a submission build.
- **Risk–coverage / selective prediction is out of scope by declaration, not undone.** The Limitations section states that "reliability estimation" in this paper means OOD-detection AUROC specifically, and that calibration and selective prediction are not addressed. Risk–coverage belongs to Paper 2's roadmap, not here. Listing it as an open item for Paper 3 (as an earlier draft of this document did) misreads the paper's own scope statement.
- **Third-domain replication** stays blocked for the reason already in Limitations: no dataset without documented image overlap with the training data was available. Nothing in this analysis changes that.
