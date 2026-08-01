# Paper 3 — Repository Map (CSG-Skin × DST-Skin)

**Purpose**: a pre-implementation orientation map across both source repositories, written before any Paper 3 code is touched. It answers: what exists, what depends on what, what's safe to reuse, what must not be touched, and what will bite us if we're not careful.

**Sources**: this map is a synthesis of `CSG-SKin/docs/REPOSITORY_GUIDE.md` and `DST-Skin/docs/REPOSITORY_GUIDE.md` (both pre-existing, exhaustive, file-by-file audits — read in full) plus direct verification of function signatures and line numbers in the actual source. No code was modified to produce this document.

**Working assumption about Paper 3's shape**: CSG-Skin contributes a *disentangled representation* (lesion-only `z_lesion` vs. context `z_context`, trained adversarially to reduce domain leakage) and DST-Skin contributes a *Mahalanobis-based OOD/triage scoring methodology* (Ledoit-Wolf shrinkage covariance, ReAct, ViM, risk-coverage analysis) built on plain (non-disentangled) backbones. The natural Paper 3 axis is combining them — e.g. asking whether CSG's leakage-reduced `z_lesion` gives better OOD/triage separation than DST's raw pooled features. Everything below is organized to support that direction, but the facts themselves (module locations, dependencies, risks) hold regardless of the exact hypothesis.

---

## 1. Important modules

### CSG-SKin (disentanglement / domain-leakage side)

| Module | Role | Notes |
|---|---|---|
| [`src/models/csg_lite.py`](CSG-SKin/src/models/csg_lite.py) | `CSGLite` — dual EfficientNet-B3 encoder (lesion + context branches), `GradientReversalLayer` | The core architecture. `lesion_latent_dim=16`, `context_latent_dim=64`. |
| [`src/models/csg_lightning.py`](CSG-SKin/src/models/csg_lightning.py) | `CSGLiteLightning` | Training wrapper: 5-term loss, GRL alpha schedule, 3-group optimizer |
| [`src/models/baseline.py`](CSG-SKin/src/models/baseline.py) | `BaselineNet` / `BaselineResNet50` | ResNet-50 reference (no disentanglement); 2048-d feature |
| [`src/models/effb3_single.py`](CSG-SKin/src/models/effb3_single.py) | `EffB3SingleNet` / `EffB3SingleLightning` | Single-encoder ablation, mirrors CSG's lesion branch exactly minus the context/adversary machinery |
| [`src/datasets/skin_dataset.py`](CSG-SKin/src/datasets/skin_dataset.py) | `SkinDataModule`, `SkinDataset`, `CombinedTrainDataset`, transforms | The datamodule everything trains from |
| [`src/datasets/splits.py`](CSG-SKin/src/datasets/splits.py) | `load_filtered_master`, `build_id_ood_test_dataloaders` | Independent re-implementation of the same split logic as `skin_dataset.py` |
| [`src/datasets/preprocess_metadata.py`](CSG-SKin/src/datasets/preprocess_metadata.py) | ISIC+PAD → `master_metadata.csv` | First stage of the data pipeline |
| [`scripts/build_lesion_only_metadata.py`](CSG-SKin/scripts/build_lesion_only_metadata.py) | Otsu-crop → `master_metadata_lesion_only_soft.csv` | The metadata actually used for training |
| [`src/utils/ood_metrics.py`](CSG-SKin/src/utils/ood_metrics.py) | Mahalanobis fit/score, `find_checkpoint`, feature collectors | CSG's own (separate, non-shrinkage) Mahalanobis implementation |
| [`scripts/check_leakage.py`](CSG-SKin/scripts/check_leakage.py) | Domain linear-probe on `z_lesion`/`z_context`/`backbone_raw` | The paper's central metric; richest feature collector (`collect_test_features_csg/baseline`) |
| [`scripts/train_csg.py`](CSG-SKin/scripts/train_csg.py) / [`scripts/train_baseline.py`](CSG-SKin/scripts/train_baseline.py) | Training entry points | Each does its own post-fit Mahalanobis + writes `summary.json` |
| `checkpoints/` (symlink → `/Users/cubo/ResearchArtifacts/checkpoints`) | 67 `.ckpt` files, 5 methods × up to 5 seeds | See §7 of the CSG guide for stale-duplicate caveats before pointing anything at a *directory* rather than a file |

### DST-Skin (Mahalanobis / OOD-triage side)

| Module | Role | Notes |
|---|---|---|
| [`src/utils/scoring.py`](DST-Skin/src/utils/scoring.py) | `OODScorer` (`fit`, `get_all_scores`) | Ledoit-Wolf-shrinkage Mahalanobis + MSP/Energy/k-NN, all normalized to "higher = more ID" |
| [`src/utils/ood_vim_react.py`](DST-Skin/src/utils/ood_vim_react.py) | `fit_react_on_fc`, `react_energy_score`, `fit_vim`, `vim_score` | Standalone ReAct/ViM, usable independent of `OODScorer` |
| [`src/utils/feature_extractor.py`](DST-Skin/src/utils/feature_extractor.py) | `extract_features_and_logits` | Generic forward-hook extractor; works on any torchvision-style CNN with `avgpool`/`features` |
| [`src/utils/calibration.py`](DST-Skin/src/utils/calibration.py) | `compute_ece` | 15-bin ECE, duplicated by hand elsewhere in this repo too |
| [`src/datasets/isic_dataset.py`](DST-Skin/src/datasets/isic_dataset.py) | `ISICDataset` | Generic CSV+image-folder binary dataset (not ISIC-specific despite the name) |
| [`src/models/{resnet18,resnet50,efficientnet_b3}.py`](DST-Skin/src/models/) | `get_resnet18/50`, `get_efficientnet_b3` | Thin torchvision factory functions, 2-class head |
| [`scripts/extract_once_{resnet18,resnet50,efficientnet_b3}.py`](DST-Skin/scripts/) | Feature caching | Only place `outputs/features/*.pt` is produced; **overwrites on every run** |
| [`scripts/analyze_benchmark_{resnet18,resnet50,efficientnet_b3}.py`](DST-Skin/scripts/) | Current scoring pipeline | Reads cached `.pt` features, fits `OODScorer` w/ ReAct+ViM, writes the score-comparison/ablation/risk-coverage CSVs |
| [`scripts/select_reader_cases.py`](DST-Skin/scripts/select_reader_cases.py) | 96-case balanced reader-study set | `build_behavior_groups`, `danger_score`, `fill_to_target` — reusable triage-sampling pattern |
| [`scripts/grad_cam.py`](DST-Skin/scripts/grad_cam.py) | Hook-based Grad-CAM | Portable to any CNN with a nameable target conv layer |
| `data/models/*.pth` | 6 checkpoints (3 backbones × best/last) | Plain `state_dict()` saves, loaded via each model's factory function |

---

## 2. Dependency graph

### CSG-SKin (linear pipeline, two generations of orchestration layered on the same building blocks)

```
ISIC GroundTruth + PAD metadata.csv
        │
        ▼
preprocess_metadata.py ─────► data/master_metadata.csv
        │
        ▼
build_lesion_only_metadata.py (Otsu crop) ─► data/master_metadata_lesion_only_soft.csv
        │
        ▼
SkinDataModule (skin_dataset.py) / splits.py  ◄── read by every train_*.py and eval script
        │
        ├──► train_baseline.py ──► checkpoints/baseline/<run>/*.ckpt ──► summary.json
        │                                              │
        ├──► train_csg.py ────────► checkpoints/csg_lite/<run>/*.ckpt ──► summary.json
        │                                              │
        └──► run_effb3_control.py ► checkpoints/effb3_single/<run>/*.ckpt
                                                       │
                        ┌──────────────────────────────┴───────────────────────────┐
                        ▼                                                          ▼
              check_leakage.py (domain probe)                        ood_metrics.py Mahalanobis
                        │  leakage.json                                            │
                        ▼                                                          ▼
              aggregate_results.py ──► results_final_v1.csv / RESULTS_TABLES.csv
                        │
                        ▼
        generate_paper_figures.py, plot_confusion_matrix_runb.py,
        plot_reliability_diagram.py, eval_ood_benchmarks.py (cbm_revision) ──► figures

cbm_revision/scripts/*  = second-generation 5-seed re-run of the exact same train_*.py /
                           check_leakage.py / aggregate_results.py building blocks
                           (run_one_shot_cbm.sh orchestrates; not a separate codebase)
```

Key coupling points relevant to Paper 3:
- Every feature/latent (`z_lesion`, `z_context`, `backbone_raw`) is **recomputed in memory per script invocation** — nothing is ever persisted to disk (verified: no `torch.save`/`np.save` of embeddings anywhere in this repo). Any Paper 3 work that wants CSG embeddings on disk (e.g. to feed DST's `OODScorer`) has to add that persistence step itself.
- `configs/*.yaml` are dead — not loaded by any script. Don't treat them as a source of truth for hyperparameters; read `train_*.py` argparse defaults + the runner shell scripts instead.

### DST-Skin (script-driven, two coexisting benchmark generations, one hidden cross-dependency)

```
data/raw/isic2018/, data/raw/pad_ufes20/
        │
        ▼
create_isic_split.py, (hand-run "x" scripts) ──► data/processed/{isic2018_binary,pad_ufes20_binary}/*.csv
        │
        ▼
train_{resnet18,resnet50,efficientnet_b3}_robust.py ──► data/models/*.pth
        │
        ▼
extract_once_{resnet18,resnet50,efficientnet_b3}.py ──► outputs/features/*.pt   [OVERWRITES]
        │
        ▼
analyze_benchmark_{resnet18,resnet50,efficientnet_b3}.py   (current pipeline)
        │  uses src/utils/scoring.py::OODScorer (+ ood_vim_react.py)
        ▼
outputs/reports/{backbone}_{score_comparison,mahalanobis_ablation_N,risk_coverage_*}.csv
        │
        ▼
plot_ablation.py, plot_umap_visual.py ──► outputs/figures/*

── separate, legacy branch, still load-bearing ──
run_journal_benchmark_efficientnet_b3.py (does its OWN feature extraction, ignores outputs/features/*.pt)
        │
        ▼
outputs/efficientnet_b3_per_sample.csv   ◄── ONLY producer of this file
        │
        ▼
select_reader_cases.py ──► data/reader_study/reader_cases_selected_96.csv
        │
        ▼
collect_reader_images.py ──► reader_images_96/
grad_cam.py ──► outputs/reports/figure_gradcam_reader_study_main.*
```

**Critical hidden dependency**: `select_reader_cases.py` requires `outputs/efficientnet_b3_per_sample.csv`, which only `run_journal_benchmark_efficientnet_b3.py` (the *legacy* pipeline) produces. The README's documented pipeline (`analyze_benchmark_*.py`) never generates it. If Paper 3 needs the reader-study/triage artifacts, this legacy script must be run first — this is undocumented anywhere in DST-Skin itself.

### Cross-repo (Paper 3's actual integration surface)

```
CSG-SKin checkpoints (.ckpt, PyTorch Lightning)          DST-Skin checkpoints (.pth, plain state_dict)
        │                                                        │
        ▼                                                        ▼
CSGLiteLightning.load_from_checkpoint / z_lesion (16-d)   get_{resnet18,50,efb3}() + load_state_dict / pooled feature (512/2048/1536-d)
        │                                                        │
        └──────────────────► [NO EXISTING BRIDGE] ◄──────────────┘
                    any script combining the two must be net-new;
                    neither repo imports from or knows about the other
```

There is currently **zero code coupling** between the two repos — different frameworks (Lightning vs. plain PyTorch), different checkpoint formats, different label taxonomies (CSG: 8-class ISIC label space; DST: binarized malignant/benign), different dataset splits (CSG: ISIC2019 train/test 80/20 stratified; DST: official ISIC2018 Task 3 train/val split), and different Mahalanobis implementations (CSG: pooled within-class covariance, raw; DST: Ledoit-Wolf shrinkage on L2-normalized features). Any Paper 3 script that uses both will need to reconcile these explicitly — see §5 (risks).

---

## 3. Reusable functions (verified, safe to call as-is)

**From CSG-SKin:**
- `src/utils/ood_metrics.py::compute_mahalanobis_params_from_arrays` (line 29) and `mahalanobis_min_squared_distances` (line 53) — correct, but an O(N·K) pure-Python loop; vectorize before scaling up.
- `src/utils/ood_metrics.py::fpr_at_95_tpr` (line 18) — standalone, no dependency on anything CSG-specific.
- `src/datasets/skin_dataset.py::SkinDataModule` — deterministic stratified ISIC/PAD split (`random_state=42`, decoupled from `--seed`), directly reusable for any new model that needs the same ID/OOD split.
- `scripts/check_leakage.py`'s linear-probe methodology (domain-leakage-via-logistic-probe with a shuffle-label sanity control) — a solid, dataset-agnostic evaluation pattern.
- `scripts/build_lesion_only_metadata.py`'s Otsu-crop pipeline — dependency-free (no cv2/skimage), reusable for other lesion-image preprocessing.

**From DST-Skin:**
- `src/utils/scoring.py::OODScorer.fit` / `.get_all_scores` — generic over any `(features, logits)` pair, not skin-imagery-specific. **This is the most likely direct reuse point for Paper 3**: it can be fit on CSG's `z_lesion` embeddings just as easily as on DST's pooled features, provided the embeddings are extracted and handed to it as plain numpy arrays.
- `src/utils/ood_vim_react.py::fit_react_on_fc` / `fit_vim` — standalone, usable independent of `OODScorer`.
- `src/utils/feature_extractor.py::extract_features_and_logits` — generic hook-based extractor; would need a CSG-side equivalent (or direct calls to `CSGLite.encode_z_lesion`) since CSG's models aren't plain `torchvision` backbones.
- `scripts/select_reader_cases.py`'s `build_behavior_groups` / `danger_score` / `fill_to_target` — a reusable "stratified case sampling by (correctness × confidence × reliability)" pattern for any future human-reader-study design.
- `scripts/grad_cam.py`'s `generate_gradcam` / `overlay_heatmap_on_image` — portable to any CNN with a nameable target conv layer, including CSG's lesion/context backbones.

**Reusable but needs a fix first:**
- CSG's `scripts/aggregate_results.py` — fix the `leakage.json`-wins merge-order bug (baseline gets silently re-scored under the wrong transform) before trusting any aggregate it produces.
- DST's `analyze_benchmark_*.py`'s `calc_auroc`/`calc_fpr95`/`risk_coverage` helpers — currently triplicated per-backbone; worth consolidating into `src/utils/` before extending to a fourth (CSG-derived) feature source.

---

## 4. Files we should never modify

These are either upstream-of-everything artifacts, frozen experimental records, or files whose exact current behavior (including its bugs) is load-bearing for reproducing the existing papers' numbers:

- **CSG-SKin `results_final_v1.csv` and `RESULTS_TABLES.csv`** — frozen snapshots of a completed n=3 run; `RESULTS_TABLES.csv` in particular has no generator script in the checkout, so it cannot be regenerated if altered.
- **CSG-SKin `checkpoints/` contents** — read-only artifacts (16 GB, symlinked from outside the repo). Never overwrite or retrain into the same `dirpath` without a new run name; several directories already have stale duplicate checkpoints (see CSG guide §7.2) precisely from this happening before.
- **DST-Skin `data/models/*.pth`** — the only trained weights in this repo; training is unseeded, so these cannot be regenerated to bit-identical files. Treat as frozen unless a fresh Paper-3-specific training run is explicitly intended.
- **DST-Skin `outputs/features/*.pt`** — large cached tensors; re-running `extract_once_*.py` **silently overwrites** these with no versioning by checkpoint. If Paper 3 needs to compare against the numbers currently in `outputs/reports/*_score_comparison.csv`, don't re-run extraction against a different checkpoint first without renaming the old `.pt` files.
- **DST-Skin `outputs/reports/final_predictions_for_triage.csv`, `full_triage_data.csv`, and `outputs/gradcam_examples/*`** — orphaned, no generator script exists in either repo; if deleted or altered they cannot be reconstructed from code.
- **Both `docs/REPOSITORY_GUIDE.md` files** — treat as reference documentation, not code to refactor; they're the ground truth this map was built from.
- **CSG-SKin `data/`, `results/`, `lightning_logs/` and DST-Skin `data/raw/`, `data/processed/`** — raw/derived data directories; regenerating them changes every downstream number silently. Any Paper 3 preprocessing should write to new, clearly-named outputs rather than overwriting these in place.

## 5. Files we should extend (rather than fork or rewrite)

- **`DST-Skin/src/utils/scoring.py`** — the natural home for a "fit `OODScorer` on CSG `z_lesion`/`z_context` instead of a plain pooled feature" code path, since `OODScorer` is already feature-source-agnostic. Extend, don't fork: forking risks the same near-duplicate-implementation drift both guides flag repeatedly (CSG has 6+ divergent feature collectors already; DST has two divergent benchmark generations).
- **`DST-Skin/src/utils/feature_extractor.py`** — add a CSG-specific extraction function here (or a sibling module) that calls `CSGLite.encode_z_lesion`/`extract_context_features` directly, following the existing `(features, logits)`-tuple return convention so it plugs into `OODScorer` unchanged.
- **`CSG-SKin/src/utils/ood_metrics.py`** — if Paper 3 needs CSG-side Mahalanobis with DST's shrinkage-covariance approach instead of CSG's current pooled-covariance approach, extend this module with a Ledoit-Wolf variant rather than duplicating DST's `OODScorer` inline in a new script.
- **`DST-Skin/scripts/analyze_benchmark_*.py`** — the `calc_auroc`/`calc_fpr95`/`risk_coverage` trio here is the template for whatever "analyze_benchmark_csg.py"-equivalent Paper 3 adds; promote these three functions to a shared module first (per §3) rather than copy-pasting a fourth near-duplicate.
- **`scripts/aggregate_results.py` (CSG)** and the reporting CSV schema in DST's `outputs/reports/` — if Paper 3 needs a combined results table, extend one of these existing schemas rather than inventing a third, to keep downstream `stat_tests_cbm.py`/`plot_cbm_figures.py`-style consumers compatible.

## 6. Potential risks

1. **Silent transform/label mismatches across repos.** CSG's label space is the full 8-class ISIC taxonomy; DST's is a hand-picked binary malignant/benign relabeling, and the two repos' malignant-class definitions for PAD-UFES don't even agree with each other's ISIC mapping conventions (CSG: `PAD_DIAG_MAP` maps 6 keys to the 8-class space; DST: malignant iff `{MEL,BCC,SCC,ACK}`). Any Paper 3 script comparing the two must pick one taxonomy explicitly and re-derive labels consistently — do not assume a shared `label` column means the same thing in both repos' CSVs.
2. **Different splits, different "ID"/"OOD" definitions.** CSG evaluates against ISIC2019 with an 80/20 split (`random_state=42`, independent of training seed) and, critically, its "OOD test set" (all PAD-UFES) is also the model's own auxiliary training-domain stream (CSG guide §8 item 1) — so any OOD-detection number from CSG checkpoints is not evaluated on genuinely unseen data. DST uses ISIC2018's *official* train/val split and treats PAD-UFES as a clean, never-trained-on OOD set. **If Paper 3 puts these two OOD numbers side by side, they are not measuring the same thing** — CSG's is contaminated by design, DST's is not.
3. **Two different Mahalanobis estimators with the same name.** CSG's `ood_metrics.py` uses a raw pooled within-class covariance (`+ reg_eps·I`, no shrinkage); DST's `OODScorer` uses Ledoit-Wolf shrinkage on L2-normalized features. A Paper 3 comparison of "Mahalanobis AUROC" between the two codebases is comparing two different formulas unless explicitly reconciled — pick one and use it consistently, or clearly label results by estimator.
4. **CSG's Mahalanobis fit/score transform bug is real and unfixed** (CSG guide §8 item 2): `train_csg.py`'s post-fit path double-applies grayscale conversion to `z_lesion` inputs used for Mahalanobis fitting, while training itself avoids exactly this. Don't reuse `train_csg.py`'s post-fit Mahalanobis numbers as ground truth without checking which code path produced them; prefer `eval_ood_scores.py`'s clean-transform fitting if precision matters.
5. **Checkpoint directory resolution silently picks the wrong file.** `find_checkpoint` (CSG) picks `max(mtime)` over all `.ckpt` files in a directory; confirmed to misresolve to a non-best, later-epoch checkpoint in at least 6 `runB`/`runB_orth1` seed directories (CSG guide §7.2), including seeds for the paper's headline method. **Always pass an explicit `.ckpt` file path, never a directory**, when loading CSG checkpoints for Paper 3 work.
6. **Feature extraction is either never cached (CSG) or eagerly overwritten (DST).** CSG recomputes every latent per script run with no disk cache at all; DST's `extract_once_*.py` overwrites `outputs/features/*.pt` with no checkpoint-provenance tagging. Combining both into one Paper 3 pipeline needs an explicit, versioned embedding-cache step that neither repo currently has — build this once, in one place, rather than letting it accrete per-script like the 6+ duplicate collectors already in CSG.
7. **DST's reader-study/triage pipeline has an undocumented hard dependency on a "legacy" script** (`run_journal_benchmark_efficientnet_b3.py`) that the README doesn't mention needs to run first. If Paper 3 reuses the reader-study/triage machinery, budget for running this legacy step, or better, port its one useful output (`efficientnet_b3_per_sample.csv`'s schema) into the current `analyze_benchmark_*.py` pipeline so the dependency isn't silently inherited into Paper 3's own pipeline.
8. **Unseeded training in DST-Skin.** None of DST's three training scripts set a random seed — the checkpoints in `data/models/` cannot be reproduced by re-running training. If Paper 3 needs a fresh DST-side checkpoint (e.g. retrained under a protocol matching CSG's), seed it explicitly; don't assume re-running `train_*_robust.py` reproduces current numbers.
9. **Dependency/version drift in both repos.** CSG's `requirements.txt` is missing `torchmetrics`/`scipy`/`seaborn` (all directly imported) and lists an unused `opencv-python`; DST's `requirements.txt` is entirely unpinned (`>=` everywhere, no lockfile). Building a Paper 3 environment that imports from both repos should pin versions explicitly rather than trusting either `requirements.txt` as sufficient.
10. **Config files are decorative in CSG-SKin.** `configs/baseline.yaml`/`configs/csg.yaml` are never loaded by any script and actively disagree with the actual run presets (§8 item 8 of the CSG guide). Don't use them as a starting point for a Paper 3 config system without first verifying against the real argparse defaults in `train_*.py`.
11. **Empty scaffold directories signal unfinished intent, not available infrastructure.** DST's `configs/`, `experiments/`, `logs/`, `src/losses/` are all empty — there is no config system, experiment tracker, or logging framework to plug into despite the directory names suggesting otherwise. Paper 3 tooling needs to bring its own if it wants either.

---

## 7. Open questions to resolve before writing any Paper 3 code

- Which Mahalanobis formulation (CSG's raw pooled covariance vs. DST's Ledoit-Wolf shrinkage) is the one Paper 3 reports — or is comparing the two formulations itself part of the contribution?
- Which label taxonomy (CSG's 8-class vs. DST's binary malignant/benign) does Paper 3 target, and how should the PAD-UFES malignant-class definitions be reconciled between the two repos' differing mappings?
- Does Paper 3 need a genuinely held-out PAD-UFES OOD split for CSG (fixing risk #2 above), or is it explicitly scoped to DST's checkpoints/splits only, with CSG contributing architecture/methodology rather than its own trained checkpoints?
- Where should the new cross-repo glue code live — a third top-level directory (e.g. this `paper-3/`), or a subpackage inside one of the two existing repos? (Given zero existing coupling between them, a third location is the lower-risk default.)
