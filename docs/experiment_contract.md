# Experiment Contract

**Purpose**: pre-registered Input / Output / Success / Failure for E1–E4, fixed *before* any experiment runs. Once an experiment executes, its outcome is read off against the criteria below — not re-argued afterward. If a criterion turns out to be wrong or ill-posed, that is a reason to amend this document explicitly (dated, with a stated reason), not to reinterpret a result to fit.

**One global rule, stated once**: if an experiment's Input preconditions cannot be met (a checkpoint can't be resolved to a verified file, a required deduplication can't be confirmed, a required upstream script/file doesn't exist), the experiment is marked **BLOCKED**, never **FAILED**. Blocked says nothing about H3. Failed is evidence against it. Conflating the two misrepresents what was actually tested.

**No experiment below predicts a *direction* for any geometry metric unless SPEC.md's H3 actually licenses one.** H3 (`SPEC.md` §3) claims shortcut-mitigation training changes feature geometry and that this *may* affect Mahalanobis validity — it does not claim which way any specific metric moves. A contract that quietly baked in an unearned directional prediction would be exactly the kind of unjustified leap `internal_review.md` already flagged elsewhere in this project.

---

## E1 — Geometry across the dose-response ladder

**Input**
- 18 CSG-SKin checkpoints: `{baseline_soft, runA_grl, runB_orth1}` × seeds `{42,52,62,72,82}` (5 each) + `runB` × seeds `{42,52,62}` (3 only) — each referenced by an explicit, individually-verified `.ckpt` file path recorded in a checkpoint manifest. Directory-resolved paths are not a valid input (see `REPOSITORY_MAP.md` risk #5).
- Per checkpoint: $z_{lesion}$ embeddings + ISIC class labels, extracted via `collect_z_lesion_labels_csg` on the **ISIC-train-under-eval-transform** loader — the same loader `eval_ood_benchmarks.py` uses, not `train_csg.py`'s augmented paired-CSG loader (avoiding the documented grayscale-double-conversion bug, per `threats_to_validity.md` #2).

**Expected output**
- `results/e1_geometry_metrics.csv`, one row per (rung, seed): `rung, method, seed, checkpoint_path, n_samples, feat_dim, condition_number, fisher_ratio_HL, fisher_ratio_scalar, mardia_kurtosis_b, mardia_kurtosis_z`.
  (`fisher_ratio_HL` = $\mathrm{tr}(\Sigma^{-1}S_B)$; `fisher_ratio_scalar` = $\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$, the decoupled companion required by `fisher_ratio_defense.md` §5.)

**Success**
- At least one of `{condition_number, fisher_ratio_HL, mardia_kurtosis_z}` has $|\rho_{\text{Spearman}}(\text{rung index}, \text{metric})| \geq 0.5$ across the common-seed (42/52/62) points, **and** the sign of $\rho$ is stable when recomputed on any 2-of-3 seed subset.

**Failure**
- All three metrics have $|\rho| < 0.5$, **or** the sign of $\rho$ flips depending on which 2-of-3 seeds are used — the latter means the apparent trend is a seed-selection artifact, not a rung effect, and must be reported as such, not as a weaker version of success.

---

## E2 — Geometry ↔ Mahalanobis AUROC association

**Input**
- `results/e1_geometry_metrics.csv`.
- Per (rung, seed) Mahalanobis AUROC (ISIC-test vs. PAD-UFES), computed **exclusively** via `cbm_revision/scripts/eval_ood_benchmarks.py` (`threats_to_validity.md` #2). AUROC numbers from any other script (`train_csg.py`'s post-fit path, `eval_ood_scores.py`, a new reimplementation) are not a valid input to this experiment.

**Expected output**
- `results/e2_association.csv` (per-metric correlation/regression coefficient + 95% CI + n) and `results/e2_association_report.md` stating the outcome against the criteria below in plain language.

**Success**
- For at least one metric E1 flagged as trend-bearing, $|\rho_{\text{Spearman}}(\text{metric}, \text{AUROC})| \geq 0.5$ across the same common-seed points, with a sign consistent with both trending in the same rung direction (the direct test of H3's causal chain: representation change → geometry change → reliability change).

**Failure**
- No E1-trend-bearing metric reaches $|\rho| \geq 0.5$ against AUROC; **or** a metric trends with rung (per E1) while AUROC itself shows no rung trend at all. The second case is recorded explicitly as **H3 not supported in this setting** — geometry moved, reliability didn't — not filed as merely inconclusive.
- Given $n \leq 12$: any CI wide enough to contain both a practically-large effect and zero is reported as **inconclusive**, a third outcome distinct from both success and failure — it must not be rounded into either.

---

## E3 — Replication on HAM10000

**Input**
- The same 18 checkpoints from E1 (no retraining).
- HAM10000 images/labels, **deduplicated against `isic_2018_binary_train.csv`'s image IDs** (per DST-Skin's orphaned `split.py` logic, which must be re-verified since its own output CSV does not currently exist — `REPOSITORY_MAP.md` §3.4/§6). Deduplication having been confirmed is a precondition of a valid input, not a post-hoc cleanup step — `REPOSITORY_MAP.md` §3.4 already documents that ISIC2018 and HAM10000 overlap heavily, so an unverified dedup makes this experiment untrustworthy by construction, not merely noisy.

**Expected output**
- `results/e3_geometry_metrics_ham10000.csv`, `results/e3_association_ham10000.csv` — same schemas as E1/E2, HAM10000 substituted for PAD-UFES as the shifted-domain evaluation set.

**Success**
- E2's trend-bearing association replicates: same sign, $|\rho| \geq 0.3$ (a lower bar than E2's 0.5 — this is a secondary confirmation set, not the primary test, and is smaller).

**Failure**
- Deduplication cannot be verified as complete → the experiment is **BLOCKED**, full stop; reporting an association computed on possibly-contaminated HAM10000 data as confirmatory would misstate what was tested, regardless of what number comes out.
- Deduplication succeeds but the sign reverses or $|\rho| < 0.3$ → reported as **did not replicate on an independent domain**, named explicitly as a distinct outcome from "blocked."

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
