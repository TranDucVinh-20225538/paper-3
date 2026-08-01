# Fisher Ratio Under Cross-Examination

**Purpose**: `geometry_metric_audit.md` §5 chose the Fisher ratio, $J=\mathrm{tr}(\Sigma^{-1}S_B)$, as the Concern-2 (class-separation) metric with comparatively little scrutiny relative to condition number and Mardia's kurtosis — and `internal_review.md` then identified it as the metric most vulnerable to a specific, technical attack (entanglement with condition number via the shared precision matrix). This document is a direct cross-examination: can Fisher ratio be defended against a statistically literate reviewer, or should it be replaced before any experiment runs? It ends with a recommendation, not a rewrite of the audit — if the recommendation is accepted, `geometry_metric_audit.md` should be amended separately.

---

## 1. Why Fisher, and not the two obvious alternatives?

"Fisher ratio" as named in the audit is one member of a small, well-established family of classical multivariate separation statistics — the same family MANOVA is built from. Given between-class scatter $S_B$ and within-class (pooled) scatter $S_W$, the four classical statistics are:

| Statistic | Formula | What it aggregates |
|---|---|---|
| Hotelling–Lawley trace ("Fisher ratio" here) | $\mathrm{tr}(S_W^{-1}S_B)$ | Sum of separation across **all** discriminant directions, weighted by $S_W^{-1}$ |
| Wilks' Lambda | $\det(S_W)\,/\,\det(S_W+S_B)$ | Ratio of generalized variances (determinants) |
| Pillai's trace | $\mathrm{tr}\big((S_W+S_B)^{-1}S_B\big)$ | Similar to Hotelling–Lawley, normalized differently |
| Roy's largest root | $\lambda_{\max}(S_W^{-1}S_B)$ | Separation along the **single best** discriminant direction only |

The question "why Fisher, why not trace, why not log-det" turns out to be the question "why Hotelling–Lawley trace, and not Wilks' Lambda (the determinant-based member of the same family) or Roy's root (the single-eigenvalue member)." Answered in order:

**Why not a determinant-based separation statistic (the Concern-2 analogue of log-det)?** Wilks' Lambda requires $\det(S_W)$ and $\det(S_W+S_B)$. This is exactly the object §3 (A3) already rejected for Concern 1 — determinants are products of eigenvalues, and in this project's small-$n$-relative-to-$d$ regime (especially at $z_{context}$'s 64 dimensions and higher for backbone features), a single near-zero sample eigenvalue sends a determinant toward zero and a determinant *ratio* toward numerical garbage. Rejecting log-det for Concern 1 and then quietly accepting its structural twin for Concern 2 would be inconsistent. It is rejected here for the identical, already-stated reason.

**Why not an unratioed trace (the Concern-2 analogue of "just report trace(Σ)")?** Reporting $\mathrm{tr}(S_B)$ alone (or $\mathrm{tr}(S_W)$ alone) measures scale, not discriminability. It cannot distinguish "classes moved apart" from "classes got noisier" — if between-class scatter and within-class scatter both grow by the same factor, actual discriminability (in the sense that matters for $\min_c d_M(x,\mu_c)$) is unchanged, but an unratioed $\mathrm{tr}(S_B)$ would report "more separation." A ratio (or a properly whitened comparison) is structurally necessary to say anything about separation *relative to spread*, which is what Concern 2 is actually about. This is the same reasoning §3 (A2) already gave for why trace alone can't stand in for condition number on Concern 1.

**Why not Roy's largest root**, which is arguably the closest analogue to condition number's own "look at the extremes" logic? Because Roy's root is an *extremal* statistic — like $\lambda_{\min}$ in condition number, a single eigenvalue of a sample-estimated matrix is a high-variance, easily-perturbed quantity. Using an extremal statistic for *both* Concern 1 and Concern 2 would double down on the exact small-sample fragility the audit spent most of §3 worrying about, with no compensating benefit. Hotelling–Lawley trace is a sum over the whole discriminant spectrum, which is the more sample-robust choice — the same trade-off (sum vs. extremum) that motivated preferring condition number's ratio-of-extremes for Concern 1 *because that concern is genuinely about worst-case behavior*, while Concern 2 is genuinely about aggregate discriminability across however many directions carry class signal. The two concerns ask different questions; using an aggregate statistic for one and an extremal statistic for the other is not an inconsistency, it's matching the statistic's aggregation behavior to what each concern is actually asking. (This is worth stating explicitly, since a reviewer could otherwise read the asymmetry as arbitrary.)

So: among the four classical options, Hotelling–Lawley trace is the only one that is both (a) not a determinant (avoiding A3's instability) and (b) not a single extremal eigenvalue (avoiding condition number's own noted fragility repeated in a second metric). That is a real, defensible reason to prefer it over its two closest classical cousins — this part of the defense holds.

## 2. Why is it "the quantity closest to Mahalanobis"?

This is the strongest part of the original case, and it survives scrutiny largely intact: $S_W$ *is* $\Sigma$ up to the normalization constant $(N-K)$ — it is not analogous to the pooled within-class covariance CSG's `compute_mahalanobis_params_from_arrays` builds, it is the exact numerator of it (`cov = (centered.T @ centered) / denom`, i.e. $S_W = \text{centered}^\top\text{centered}$, `denom = N-K`). And $\mathrm{tr}(\Sigma^{-1}S_B)$ uses the *same* $\Sigma^{-1}$ — the identical `precision` array — that the Mahalanobis score itself multiplies against in `mahalanobis_min_squared_distances`. No other candidate audited in `geometry_metric_audit.md` shares an object this literally with the estimator being studied; Mardia's kurtosis comes close (built from $d_M^2$ values), but Fisher ratio is built from a *component* of the formula itself, not from its outputs. This claim was correct and remains correct after cross-examination.

## 3. The attack that actually lands: shared-matrix entanglement

`internal_review.md` raised a specific, checkable objection: because both condition number ($\kappa$) and Fisher ratio ($J$) are functions of the same `precision` matrix, they are not statistically independent measurements, even though the *concepts* they represent (conditioning vs. separation) are logically independent. If `precision`'s estimation quality degrades for a reason unrelated to true geometry (e.g. `reg_eps` dominating genuinely small eigenvalues at high feature dimensionality — exactly the failure mode §3/A1 spent several paragraphs on), that same degradation could move $\kappa$ and $J$ together across the ladder for a spurious, shared reason, undermining the audit's own claim that these three metrics test three independent hypotheses.

**Is this fatal, or survivable?** Three things are true simultaneously, and all three matter:

1. **It is a real risk, not a hypothetical one.** $\kappa$ and $J$ are both linear/rational functionals of the same random matrix. Claiming "conceptual independence" (a true statement about the underlying probabilistic assumptions) is not the same claim as "measurement independence" (a claim about the covariance of the two point-estimates), and the audit's "one estimation call, three read-outs" framing (§5) elided that distinction. This is a legitimate correction to the audit, not a rhetorical trick.
2. **It is not unique to Fisher ratio — it is unavoidable for *any* Concern-2 metric that uses $\Sigma^{-1}$ at all**, including Roy's root or Pillai's trace (both use $S_W^{-1}$ too). The only way to fully avoid it is to abandon the $\Sigma^{-1}$-based family entirely and use something that doesn't touch `precision` — which is exactly what the audit's own rejected alternatives (silhouette, Davies-Bouldin) do, at the cost of losing the tight mechanistic link to Mahalanobis established in §2 above. There is a genuine, irreducible trade-off here: mechanistic fidelity to the exact Mahalanobis formula versus statistical decoupling from its estimation noise. No single scalar satisfies both.
3. **There is a cheap, already-available way to *check* whether this risk is material, rather than merely assert it away or concede immediately.** The audit's own §3 (B1, "Robustness to sample size") already noted, in passing, a simpler proxy that was never adopted: $J_{\text{scalar}} = \mathrm{tr}(S_B)/\mathrm{tr}(S_W)$ — a ratio of two trace quantities, computed with **no matrix inverse at all**, hence **not built from `precision`** and not entangled with condition number's estimation noise by construction. It is a coarser measure (isotropic — it doesn't weight directions by how much variance they carry the way $\Sigma^{-1}$ does), but its coarseness is exactly what buys independence from $\kappa$.

## 4. A concrete, pre-experiment test of whether the entanglement is real

Rather than argue about this in the abstract, it is directly checkable with synthetic data, before touching any real checkpoint:

1. Fix a "true" class-conditional Gaussian generative model (means, a shared covariance) representing one rung.
2. Draw many bootstrap resamples of size $n$ (matching the real per-rung sample count) from this **fixed** model.
3. For each resample, fit `compute_mahalanobis_params_from_arrays` and compute both $\kappa$ and $J$.
4. Compute $\mathrm{Corr}(\kappa, J)$ across resamples, **at fixed true geometry**.

If this correlation is large, the concern is empirically confirmed for this project's actual sample sizes and dimensionalities: $\kappa$ and $J$ would move together across the real ladder partly (or mostly) because of shared estimation noise, not because conditioning and separation are truly moving together. If it is small, the two metrics are, in practice, capturing distinguishable information despite the shared input matrix, and the theoretical concern — while correct in principle — is not large enough at this project's actual $n$, $d$ to matter. This test costs nothing (synthetic data only, no checkpoints, no experiment run in the `SPEC.md` sense) and directly answers the question rather than leaving it as a philosophical objection.

## 5. Verdict

**Fisher ratio (Hotelling–Lawley trace form) survives as the primary Concern-2 metric.** Its justification — the tightest available algebraic connection to CSG's actual per-class Mahalanobis construction, and the only classical separation statistic that avoids both the determinant-instability problem (§1) and the extremal-statistic fragility problem (§1) — holds up under cross-examination better than either obvious alternative would.

It does **not** survive as an *unqualified*, standalone primary metric the way condition number and Mardia's kurtosis do. The entanglement objection is real, checkable, and currently unaddressed anywhere in this project's documents. The recommendation is therefore **keep, with two amendments**, not replace:

- Report the decoupled scalar ratio $\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$ **alongside** $J$ as a mandatory (not optional/secondary) companion — cheap, already fully specified, and it directly diagnoses whether an observed Fisher-ratio trend is separation or shared estimation noise. If $J$ and the scalar ratio tell the same story across the ladder, that convergence is itself evidence the entanglement isn't dominating. If they diverge, that divergence is the finding.
- Run the bootstrap check in §4 once real per-rung sample sizes are known, **before** trusting any E1/E2 result that uses $J$, and report the resulting $\mathrm{Corr}(\kappa,J)$-at-fixed-geometry number as a stated precondition for interpreting Fisher ratio's role in E2 at all.

If, after that bootstrap check, $\mathrm{Corr}(\kappa,J)$ turns out to be large at this project's actual sample sizes, that result — not a preemptive guess made now — is the point at which Fisher ratio should be demoted or replaced, with the check itself as the stated justification. Deciding that now, without the check, would be replacing a real metric with a guess about a metric.
