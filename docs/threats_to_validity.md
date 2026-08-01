# Threats to Validity

Living document. Each entry is a concrete threat already surfaced by the repository audit (`REPOSITORY_MAP.md`), the specification (`SPEC.md`), or the geometry-metric audit (`geometry_metric_audit.md`) — not a hypothetical one. Intended to be copied near-verbatim into the paper's Discussion / Limitations section.

Format per entry: **Threat → Mitigation** (one line), then the four required fields.

---

## 1. PAD-UFES contamination

**PAD-UFES contamination → Mitigation → HAM10000 (E3)**

- **Threat**: CSG-SKin's "OOD test set" is, by construction, also its own auxiliary training-domain stream — `SkinDataModule.setup()` passes *all* of PAD-UFES into `CombinedTrainDataset` for CSG training, and the same all-PAD dataframe is what `test_dataloader()` hands back as the OOD test set (`REPOSITORY_MAP.md` §5, risk #2). Every CSG checkpoint has seen its own "OOD" evaluation images repeatedly during training.
- **Why it matters**: E1→E2 tests whether geometry changes across the dose-response ladder predict Mahalanobis AUROC. If that AUROC is measured on a domain the model was partly trained on, an observed association could reflect memorization of PAD-UFES specifically, not a generalizable relationship between representation geometry and reliability estimation. The result would not support the paper's actual claim.
- **How Paper 3 mitigates it**: E3 re-runs the same geometry (E1) and Mahalanobis-AUROC (E2) measurement pipeline on HAM10000, a domain neither CSG-SKin nor DST-Skin was trained or tuned on. Any E1↔E2 relationship that survives on HAM10000 cannot be attributed to PAD-UFES-specific memorization.
- **What remains unresolved**: HAM10000 data does not currently exist anywhere in either repository's checkout (`DST-Skin/split.py` references `data/raw/ham10000/`, which is absent — `REPOSITORY_MAP.md` §6). E3 has a hard external data-acquisition dependency that hasn't been resolved yet. Until then, any E1↔E2 finding is provisional and scoped to the contaminated ISIC/PAD-UFES setting.

---

## 2. Mahalanobis implementation mismatch

**Mahalanobis implementation mismatch → Mitigation → use `cbm_revision/scripts/eval_ood_benchmarks.py` only**

- **Threat**: CSG-SKin alone contains at least two divergent Mahalanobis-fitting code paths with a known, documented bug in one of them — `train_csg.py`'s post-fit path double-applies grayscale conversion to `z_lesion` inputs when fitting Mahalanobis parameters (a fix applied to train/val but not to this path), while `eval_ood_scores.py`/`cbm_revision/scripts/eval_ood_benchmarks.py` fit on a clean, single-conversion loader instead (`REPOSITORY_MAP.md` §5, risk #4; CSG guide §8 item 2). Separately, DST-Skin's `OODScorer` is a *different formula entirely* — Ledoit-Wolf shrinkage on a single global (non-per-class) mean/covariance, vs. CSG's raw pooled per-class-conditional covariance.
- **Why it matters**: if E1–E3's Mahalanobis AUROC numbers are drawn from whichever script happens to be run, "Mahalanobis AUROC" silently stops meaning one fixed thing across the ladder — a change in the observed AUROC could reflect which fitting bug was or wasn't present in a given run, not a real reliability change caused by representation geometry. This would invalidate any E2 association result without anyone noticing, since both numbers would still be labeled "Mahalanobis AUROC."
- **How Paper 3 mitigates it**: E1–E3 draw every Mahalanobis AUROC number from a single canonical source, `cbm_revision/scripts/eval_ood_benchmarks.py`, which already implements the clean (bug-free, consistent-transform) fitting path across all methods and seeds. `train_csg.py`'s own post-fit Mahalanobis numbers, `eval_ood_scores.py`'s (stdout-only, unarchived) numbers, and any new `paper-3/`-side reimplementation are explicitly not used as the E1–E3 dependent variable, to avoid adding a seventh divergent variant. DST-Skin's `OODScorer` is used only in E4, kept structurally separate and never pooled with E1–E3's numbers (see `geometry_metric_audit.md` §1's per-class-vs-global-mean note for why the two aren't the same estimator in the first place).
- **What remains unresolved**: this fixes *which script* produces the number, not the deeper fact that CSG-SKin's own Mahalanobis formula (raw pooled covariance) and DST-Skin's (Ledoit-Wolf shrinkage) are genuinely different estimators. E4, if pursued, reports a number computed by a different method than E1–E3's, and any comparison between them must say so explicitly rather than treating "Mahalanobis AUROC" as one comparable quantity across the whole paper.

---

## 3. Checkpoint resolution ambiguity

**Checkpoint resolution ambiguity → Mitigation → explicit file paths, never directories**

- **Threat**: CSG-SKin's `find_checkpoint` (and several re-implementations) resolves the newest-by-mtime `.ckpt` file in a directory, and this is confirmed — not hypothetical — to pick a non-best-val checkpoint in six specific `runB`/`runB_orth1` seed directories, due to stale duplicates left by repeated training re-runs into the same `dirpath` (`REPOSITORY_MAP.md` §5, risk #5, citing CSG guide §7.2's verified per-directory table). Three of the six affected seeds belong to `runB_orth1`, the rung the dose-response ladder's headline comparison depends on most.
- **Why it matters**: if E1's geometry measurements or E2/E3's Mahalanobis AUROC numbers are pulled by pointing a script at a run *directory*, they may silently describe a later-epoch, non-best checkpoint instead of the one training actually selected — corrupting the per-rung comparison at exactly the ladder position (`runB_orth1`) the paper's core claim rests on.
- **How Paper 3 mitigates it**: every checkpoint used by any `paper-3/` script is referenced by an explicit, individually-verified `.ckpt` file path, recorded in a checkpoint manifest committed alongside `results/` — never resolved from a directory at experiment-run time.
- **What remains unresolved**: verifying "which file is actually best" requires each run's logged validation metrics (`summary.json` / training logs), which live under `results/`/`lightning_logs/` — directories that are absent in at least some checkouts per `REPOSITORY_MAP.md`. Where those logs aren't available, the manifest's checkpoint choice for a stale-duplicate directory will need to be reconstructed or re-verified before it can be trusted, rather than assumed.

---

## 4. Uneven seed coverage across the dose-response ladder

**Uneven seed coverage → Mitigation → restrict cross-rung comparison to the common 3 seeds**

- **Threat**: `runB` (λ_orth=5.0) checkpoints exist for only 3 of 5 seeds (42, 52, 62); `baseline_soft`, `runA_grl`, and `runB_orth1` each have all 5 (`REPOSITORY_MAP.md` §5, risk #1, and `SPEC.md` §6). Note `runB` is distinct from `runB_orth1` — the ladder used for E1–E3 (`SPEC.md` §4) is `baseline_soft → runA_grl → runB_orth1 → runB`, and it is specifically the final rung that has the seed gap.
- **Why it matters**: if per-rung statistics are computed on whatever seeds happen to exist, the final rung's geometry/reliability measurement rests on a smaller, non-randomly-selected (whichever 3 happened to be trained) sample than the other three rungs — any across-ladder trend could partly reflect this sampling asymmetry rather than a real dose effect.
- **How Paper 3 mitigates it**: E1/E2's per-rung statistics are computed on the seeds common to all four rungs (42, 52, 62) as the primary comparison; results from the two extra `runB_orth1`/other-rung seeds (72, 82) are reported separately as supplementary, not pooled silently into the main 4-rung trend.
- **What remains unresolved**: even the common-seed comparison is n=3 per rung — a small sample for any statistical association claim in E2. The specific test statistic and how it handles this small n is deferred to E2's own design (`SPEC.md` §7), not resolved here.

---

## 5. Label-taxonomy mismatch (E4 scope only)

**Label-taxonomy mismatch → Mitigation → no cross-repo label mapping is invented**

- **Threat**: CSG-SKin uses the full 8-class ISIC label space; DST-Skin uses a binarized malignant/benign relabeling, and the two repos' own malignant-class definitions for PAD-UFES don't agree with each other's conventions (CSG's `PAD_DIAG_MAP` vs. DST's `{MEL,BCC,SCC,ACK}` malignant set — `REPOSITORY_MAP.md` §5, risk #1; `SPEC.md` §6).
- **Why it matters**: E4 uses DST-Skin's backbones as an independent validation path (`SPEC.md` §4). If E4 code silently reused CSG's label logic (or vice versa) on the assumption that "label" means the same thing in both repos' CSVs, any resulting metric would be computed against the wrong ground truth without an obvious symptom — the code would run without error and produce a plausible-looking but incorrect number.
- **How Paper 3 mitigates it**: E4, if implemented, uses DST-Skin's own binary taxonomy exactly as DST-Skin defines it, and does not import or re-derive labels from CSG-SKin's 8-class mapping. No implicit cross-repo label mapping is written; if one is ever needed, it will be an explicit, documented, separately-justified choice.
- **What remains unresolved**: since E4 is optional and scoped last (`SPEC.md` §4), this is a live open question, not a settled one — deferred along with the rest of E4's scoping.

---

## 6. Small-sample instability of covariance-derived geometry metrics

**Small-sample instability → Mitigation → measure the exact regularized object the estimator uses, not a raw proxy; scope Gaussianity testing to tractable dimensions**

- **Threat**: covariance-spectrum-based geometry metrics (condition number and related quantities) are known to be unstable when computed from a raw, unregularized sample covariance at high feature dimensionality relative to per-class sample size — a documented general property, and the reason CSG's own Mahalanobis fit already needs a `reg_eps·I` regularizer and DST's needs Ledoit-Wolf shrinkage in the first place (`geometry_metric_audit.md` §3, A1/A3). This project's representations span a wide dimensionality range: 16-d (`z_lesion`), 64-d (`z_context`), 1536–2048-d (backbone/baseline features).
- **Why it matters**: a geometry metric computed on a naively-estimated, unregularized covariance at $d=1536$–$2048$ could report "geometry got worse" or "better" purely as an artifact of sample-covariance estimation noise, unrelated to any real change induced by shortcut-mitigation training — undermining E1's core measurement before it reaches E2 at all.
- **How Paper 3 mitigates it**: per the final metric selection (`geometry_metric_audit.md` §5), the condition number and Fisher ratio are computed directly from the same regularized `precision` matrix CSG's existing Mahalanobis estimator already fits (`geometry_diagnostics.py`), not from an independently-estimated raw covariance — so the metric describes the exact object the reliability estimator uses, inheriting whatever regularization already stabilizes it. Mardia's kurtosis is explicitly scoped to dimensions where $n \gg d$ is plausible (primarily $z_{lesion}$, 16-d); it is not applied to `backbone_raw`/baseline features without a documented dimensionality-reduction step first.
- **What remains unresolved**: exact per-class sample counts for CSG-SKin's data are not verified in every checkout (`data/` is absent in at least one audited checkout, per `REPOSITORY_MAP.md`), so the $n \gg d$ assumption for $z_{context}$ (64-d) specifically has not been empirically checked yet — this needs verifying against real per-rung sample counts before Mardia's kurtosis is trusted at that dimensionality, if it's ever attempted there.

---

## 7. Geometry-metric selection as a researcher degree of freedom

**Metric-selection researcher degrees of freedom → Mitigation → audit and freeze the metric set before any result is computed**

- **Threat**: 13 candidate geometry metrics existed at the start of this design process; only 3 were kept. Any post-hoc metric selection — picking whichever metric happens to correlate with Mahalanobis AUROC after seeing the data — would be a form of p-hacking specific to geometry-metric choice, and would make any reported association unfalsifiable.
- **Why it matters**: a reader cannot distinguish "we found a real geometry-reliability relationship" from "we tried several geometry metrics and reported the one that worked" unless the metric-selection process is documented and dated *before* results exist.
- **How Paper 3 mitigates it**: the full 13-candidate audit, taxonomy, and rejection rationale for every non-selected metric is written down in `geometry_metric_audit.md`, and the 3-metric implementation in `geometry_diagnostics.py` deliberately excludes every rejected metric at the code level (not just in the writeup) — there is no code path that could silently reintroduce a rejected metric into a result table.
- **What remains unresolved**: this is a discipline commitment, not a technical guarantee — nothing stops a future revision from adding a 4th metric after seeing E1/E2 results. If that ever happens, it must be documented here as an amendment with a stated reason, not folded silently into the "final" set as if it had been there from the start.

---

## 8. Unseeded training in DST-Skin (E4 scope only)

**Unseeded DST-Skin training → Mitigation → treat existing checkpoints as a fixed single-run snapshot, not a multi-seed sample**

- **Threat**: none of DST-Skin's three training scripts (`train_{resnet18,resnet50,efficientnet_b3}_robust.py`) set a random seed (`REPOSITORY_MAP.md` §5, risk #8; DST guide §5) — the checkpoints currently in `data/models/` cannot be reproduced by re-running training, and there is no multi-seed sample to draw a confidence interval from the way CSG-SKin's ladder provides.
- **Why it matters**: if E4 is used as supplementary validation of the E1↔E2 relationship, any single-checkpoint-per-backbone result has no seed-to-seed variance estimate — a difference between backbones could reflect training-run luck rather than a genuine backbone-driven effect.
- **How Paper 3 mitigates it**: if E4 is implemented, its results are reported explicitly as a single-run supplementary check across DST-Skin's three backbones (a different-architecture generalization check), not as a multi-seed statistical claim on par with E1/E2's ladder-based comparison.
- **What remains unresolved**: whether E4 needs fresh, seeded DST-Skin training runs to be scientifically useful at all is an open scoping question, deferred along with the rest of E4 (`SPEC.md` §4, §7) — this entry does not resolve it, only states the constraint that applies if the existing unseeded checkpoints are reused as-is.
