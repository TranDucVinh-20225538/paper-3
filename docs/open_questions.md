# Open Questions

Log only. Nothing here gets acted on until its own Action item is explicitly executed as a separate task — this file exists so open concerns don't get silently redesigned around or silently forgotten.

---

## Q1. Dose-response ladder may contain multiple simultaneous interventions

**Status**: Largely resolved by Q4 — the specific rung this flagged (`baseline_soft`, rung 0) is no longer part of the primary dose-response trend at all. The remaining, much narrower question is whether `runA_grl`/`runB_orth1`/`runB` (the actual primary ladder now) share the same lr/transform recipe as each other — the CSG-guide table cited below suggests they do (all three are CSG runs, distinct from `baseline_soft`'s different recipe), but this should still be spot-checked before trusting E1a/E2a's trend.

**Context**: `internal_review.md` found that CSG-SKin's four ladder rungs (`baseline_soft → runA_grl → runB_orth1 → runB`) differ by more than `λ_orth` alone — per `REPOSITORY_MAP.md` §2.8, `baseline_soft` uses a different learning rate (2e-4 vs. 1e-4) and a different augmentation pipeline (light vs. robust transforms) than the three CSG runs. Rung 0 is not a controlled zero-dose condition.

**Action**: Verify actual per-rung training configs (learning rate, transform pipeline, epoch count) against `REPOSITORY_MAP.md`'s §2.8 table before interpreting any monotonic or rank-correlation trend from E1/E2 as caused specifically by shortcut-mitigation strength.

**Priority**: Medium.

---

## Q2. Fisher ratio's shared-precision-matrix entanglement with condition number

**Status**: Open, with a partial mitigation already specified.

**Context**: `fisher_ratio_defense.md` §5 requires a bootstrap check ($\mathrm{Corr}(\kappa, J)$ at fixed true geometry) before trusting any E1/E2 result that leans on Fisher ratio, and requires reporting the decoupled $\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$ companion alongside it.

**Action**: Run the bootstrap check specified in `fisher_ratio_defense.md` §4 once real per-rung sample sizes are known. Not blocking Task 1–2; blocking before E2 interprets any Fisher-ratio result.

**Priority**: Medium — has a defined trigger point (before E2), not before E1's implementation work.

---

## Q3. `eval_ood_benchmarks.py` — the canonical E2 Mahalanobis source — resolves checkpoints by directory, not explicit file

**Status**: Open.

**Context**: found while writing `extract_embeddings_e1.py` and reading `eval_ood_benchmarks.py` directly. Its `_find_ckpt(dir_path)` picks `sorted(dir_path.glob("best-*.ckpt"), key=mtime, reverse=True)[0]`, i.e. the same newest-by-mtime, directory-based resolution `REPOSITORY_MAP.md` risk #5 already flagged as confirmed-wrong for several `runB`/`runB_orth1` seed directories. `threats_to_validity.md` #2 designated this script as the canonical E1–E3 Mahalanobis source specifically to avoid `train_csg.py`'s transform-mismatch bug — but never checked it against the separate checkpoint-resolution bug. It has it too.

**Action**: before E2 runs, verify whether `eval_ood_benchmarks.py`'s per-seed checkpoint choice (for `runB` and `runB_orth1`) matches the explicit, manifest-verified paths `extract_embeddings_e1.py` uses for E1's geometry metrics on the same seeds. If they diverge, E1's geometry and E2's AUROC would be describing two different checkpoints for the same nominal (rung, seed) row.

**Priority**: High — this one directly threatens whether E1 and E2's outputs are even about the same model, not just a downstream statistical concern.

---

## Q4. `baseline_soft` is not a valid zero-dose point on the disentanglement ladder

**Status**: Resolved.

**Context**: originally treated as "λ_orth = 0," the fourth rung of one continuous dose-response trend alongside `runA_grl → runB_orth1 → runB`. It isn't a valid zero-dose condition: it differs from the three CSG rungs in architecture (ResNet-50 vs. EfficientNet-B3-based CSGLite), representation dimensionality (2048-d backbone feature vs. 16-d `z_lesion`), and training recipe (Q1, above). The dimensionality difference specifically surfaced as a concrete failure, not just a theoretical worry: computing Mardia kurtosis's bootstrap calibration on `baseline_soft`'s native 2048-d feature was the direct cause of the GPU-server performance crisis (`_simulate_mardia_null` dominated by repeated large-matrix operations) — exactly the regime `geometry_metric_audit.md` §3 (C1/C2) had already flagged as outside this test's validated range, before any checkpoint was ever run.

**Resolution**: E1/E2 are split into a **primary mechanistic test** (the three-rung disentanglement ladder — `runA_grl`, `runB_orth1`, `runB`, all sharing architecture and 16-d `z_lesion`) and a **secondary/categorical reference** (`baseline_soft`, evaluated on its own native 2048-d representation, never pooled into the primary trend statistic). The primary ladder's association test uses Kendall's τ across three ordinal points, not a 4-point regression or a Spearman threshold implying more precision than three treatment levels support. `baseline_soft`'s comparison against the ladder is reported descriptively (does its geometry/AUROC sit inside or outside the ladder's range), with no formal success/failure criterion — the same pattern already used for E4. Mardia's kurtosis remains load-bearing only for the primary ladder; for `baseline_soft` it is computed but explicitly exploratory, not used to support or refute H3.

**Updated**: `SPEC.md` §4 (E1/E2 experiment plan), `experiment_contract.md` (E1a/E1b, E2a/E2b split), `geometry_metric_audit.md` §6 and the C2 verdict note. No scripts modified.

**Priority**: Resolved — was High while open, since it bore on whether the primary claim H3 is tested against was even a coherent single comparison.
