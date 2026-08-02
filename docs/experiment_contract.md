# Experiment Contract

**Purpose**: pre-registered Input / Output / Success / Failure for E1–E4, fixed *before* any experiment runs. Once an experiment executes, its outcome is read off against the criteria below — not re-argued afterward. If a criterion turns out to be wrong or ill-posed, that is a reason to amend this document explicitly (dated, with a stated reason), not to reinterpret a result to fit.

**One global rule, stated once**: if an experiment's Input preconditions cannot be met (a checkpoint can't be resolved to a verified file, a required deduplication can't be confirmed, a required upstream script/file doesn't exist), the experiment is marked **BLOCKED**, never **FAILED**. Blocked says nothing about H3. Failed is evidence against it. Conflating the two misrepresents what was actually tested.

**No experiment below predicts a *direction* for any geometry metric unless SPEC.md's H3 actually licenses one.** H3 (`SPEC.md` §3) claims shortcut-mitigation training changes feature geometry and that this *may* affect Mahalanobis validity — it does not claim which way any specific metric moves. A contract that quietly baked in an unearned directional prediction would be exactly the kind of unjustified leap `internal_review.md` already flagged elsewhere in this project.

---

## E1 — Geometry: primary disentanglement ladder + secondary categorical reference

**Design note** (`open_questions.md` Q4): `baseline_soft` is no longer a rung on the same ordinal ladder as the three CSG methods — it differs in architecture, representation dimensionality (2048-d vs. 16-d), and training recipe, and forcing it onto one 4-point trend conflated those confounds with the actual manipulated variable (λ_orth). Split below into a primary mechanistic test (E1a) and a secondary categorical reference (E1b), mirroring `SPEC.md` §4.

### E1a — Primary: the disentanglement ladder

**Input**
- 13 CSG-SKin checkpoints: `{runA_grl, runB_orth1}` × seeds `{42,52,62,72,82}` (5 each) + `runB` × seeds `{42,52,62}` (3 only) — each referenced by an explicit, individually-verified `.ckpt` file path recorded in a checkpoint manifest. Directory-resolved paths are not a valid input (see `REPOSITORY_MAP.md` risk #5).
- Per checkpoint: $z_{lesion}$ embeddings + ISIC class labels, extracted via `collect_z_lesion_labels_csg` on the **ISIC-train-under-eval-transform** loader — the same loader `eval_ood_benchmarks.py` uses, not `train_csg.py`'s augmented paired-CSG loader (avoiding the documented grayscale-double-conversion bug, per `threats_to_validity.md` #2).

**Expected output**
- `results/e1_geometry_metrics.csv`, one row per (rung, seed), `rung ∈ {runA_grl, runB_orth1, runB}`: `rung, method, seed, checkpoint_path, n_samples, feat_dim, condition_number, fisher_ratio_HL, fisher_ratio_scalar, mardia_kurtosis_b, mardia_kurtosis_z`.
  (`fisher_ratio_HL` = $\mathrm{tr}(\Sigma^{-1}S_B)$; `fisher_ratio_scalar` = $\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$, the decoupled companion required by `fisher_ratio_defense.md` §5.)

**Success**
- At least one of `{condition_number, fisher_ratio_HL, mardia_kurtosis_z}` has $|\tau_{\text{Kendall}}(\text{rung order}, \text{metric})| \geq 0.3$ across all available (rung, seed) points (13 points: 5+5+3), **and** the sign of $\tau$ is preserved when recomputed on the common-seed (42/52/62) subset alone. Kendall's τ, not Spearman's ρ or a fitted regression — with only 3 ordinal rung levels, a coefficient implying more precision than that would misrepresent the design.

**Failure**
- All three metrics have $|\tau| < 0.3$, **or** the sign of $\tau$ is not preserved between the full 13-point set and the common-seed 9-point subset — the latter means `runB`'s seed gap (3 vs. 5) is driving the apparent trend, not a genuine rung effect.

### E1b — Secondary: baseline_soft as categorical reference

**Input**
- 5 `baseline_soft` checkpoints (seeds 42/52/62/72/82), explicit file paths.
- Per checkpoint: the native 2048-d ResNet-50 backbone feature (`_extract_baseline`), ISIC class labels, same eval-transform loader convention as E1a.

**Expected output**
- Rows appended to `results/e1_geometry_metrics.csv` with `rung=baseline_soft`, same schema as E1a. `mardia_kurtosis_b`/`mardia_kurtosis_z` are computed and recorded but flagged exploratory in any downstream write-up — d=2048 is outside this test's validated regime (`geometry_metric_audit.md` §3, C1/C2) and is never used to support or refute H3.

**Success**
- No statistical criterion, matching E4's framing below. "Success" means: `condition_number`, `fisher_ratio_HL`, and `fisher_ratio_scalar` compute without error across all 5 seeds, and are reported descriptively alongside the E1a ladder's range (e.g. does baseline's condition number fall inside or outside the span `runA_grl`→`runB` covers).

**Failure**
- Any of the three load-bearing metrics fails to compute on any seed → reported as **not run** for that seed, not a null result. Mardia kurtosis failing/being slow at d=2048 is not a failure of E1b — it was never load-bearing here; if computed, report it, if not, note it as skipped and why.

---

## E2 — Geometry ↔ Mahalanobis AUROC association

### E2a — Primary: association within the disentanglement ladder

**Input**
- `results/e1_geometry_metrics.csv`, rows with `rung ∈ {runA_grl, runB_orth1, runB}` only.
- Per (rung, seed) Mahalanobis AUROC (ISIC-test vs. PAD-UFES), computed **exclusively** via `cbm_revision/scripts/eval_ood_benchmarks.py` (`threats_to_validity.md` #2). AUROC numbers from any other script (`train_csg.py`'s post-fit path, `eval_ood_scores.py`, a new reimplementation) are not a valid input to this experiment.

**Expected output**
- `results/e2_association.csv` (per-metric Kendall's τ + n) and `results/e2_association_report.md` stating the outcome against the criteria below in plain language, plus a seed-level scatter plot (rung on the ordinal axis, metric vs. AUROC).

**Success**
- For at least one metric E1a flagged as trend-bearing, $|\tau_{\text{Kendall}}(\text{metric}, \text{AUROC})| \geq 0.3$ across the same 13 points, with a sign consistent with both trending in the same rung direction (the direct test of H3's causal chain: representation change → geometry change → reliability change). This is reported as a descriptive association, not a hypothesis test with a formal p-value — three ordinal treatment levels do not support one.

**Failure**
- No E1a-trend-bearing metric reaches $|\tau| \geq 0.3$ against AUROC; **or** a metric trends with rung (per E1a) while AUROC itself shows no rung trend at all. The second case is recorded explicitly as **H3 not supported in this setting** — geometry moved, reliability didn't — not filed as merely inconclusive.

### E2b — Secondary: baseline reference vs. the ladder

**Input**
- E1b's `baseline_soft` geometry row(s) + its Mahalanobis AUROC (same canonical source as E2a).
- E2a's ladder-wide (geometry, AUROC) pattern, for context.

**Expected output**
- A descriptive paragraph in `results/e2_association_report.md`: does baseline's (geometry, AUROC) point sit inside, at the edge of, or outside the range the three-rung ladder spans — i.e. does having *any* disentanglement architecture (vs. none) look different in kind from varying its strength.

**Success / Failure**
- No statistical criterion, same reasoning as E1b/E4 — one categorical point cannot support a trend test. This section exists to contextualize E2a's magnitude, not to be judged pass/fail.

---

## E3 — Replication on HAM10000

**Input**
- The same 18 checkpoints from E1 (13 primary-ladder + 5 baseline-reference, no retraining).
- HAM10000 images/labels, **deduplicated against `isic_2018_binary_train.csv`'s image IDs** (per DST-Skin's orphaned `split.py` logic, which must be re-verified since its own output CSV does not currently exist — `REPOSITORY_MAP.md` §3.4/§6). Deduplication having been confirmed is a precondition of a valid input, not a post-hoc cleanup step — `REPOSITORY_MAP.md` §3.4 already documents that ISIC2018 and HAM10000 overlap heavily, so an unverified dedup makes this experiment untrustworthy by construction, not merely noisy.

**Expected output**
- `results/e3_geometry_metrics_ham10000.csv`, `results/e3_association_ham10000.csv` — same schemas as E1a/E2a for the primary ladder, plus the same E1b/E2b-style descriptive baseline row, HAM10000 substituted for PAD-UFES as the shifted-domain evaluation set.

**Success**
- E2a's trend-bearing association replicates on the primary ladder only: same sign, $|\tau_{\text{Kendall}}| \geq 0.2$ (a lower bar than E2a's 0.3 — this is a secondary confirmation set, smaller and not the primary test). `baseline_soft` replicates only as the same descriptive reference (E1b/E2b-style), never pooled into this criterion.

**Failure**
- Deduplication cannot be verified as complete → the experiment is **BLOCKED**, full stop; reporting an association computed on possibly-contaminated HAM10000 data as confirmatory would misstate what was tested, regardless of what number comes out.
- Deduplication succeeds but the sign reverses or $|\tau| < 0.2$ on the primary ladder → reported as **did not replicate on an independent domain**, named explicitly as a distinct outcome from "blocked."

---

## E4 — Supplementary check on DST-Skin (optional)

**Input**
- DST-Skin's 3 existing checkpoints (ResNet-18/50, EfficientNet-B3) — unseeded, single run each, no retraining.
- DST's own binary label taxonomy (not CSG's 8-class labels — no cross-repo label mapping is invented, per `threats_to_validity.md` #5).
- DST's `OODScorer` Mahalanobis (Ledoit-Wolf, single global mean) — a different estimator from E1–E3's, not interchangeable with it.
- Only `condition_number` and `mardia_kurtosis` are computed; Fisher ratio is not applicable here, since DST's `OODScorer.fit` never uses per-class labels (`geometry_metric_audit.md` §1, §6).

**Expected output**
- `results/e4_geometry_metrics_dst.csv` (3 rows, one per backbone) and `results/e4_notes.md` stating explicitly that this is a single-run, 3-point illustration, not a statistical test.

**Success**
- There is no statistical success criterion. With 3 backbones and zero seed replication, no correlation or regression is well-posed. "Success" here means only: both applicable metrics compute without error on all 3 backbones, and their values are reported descriptively alongside each backbone's Mahalanobis AUROC.

**Failure**
- Either metric fails to compute on any backbone (e.g. a shape/label mismatch in DST's pipeline) → reported as **not run** for that backbone, not as a null result.
- If E4 runs successfully, there is no further failure criterion, precisely because no statistical claim is being tested — any future write-up must state this explicitly so E4 is never cited later as though it had a pass/fail outcome it was never designed to have.
