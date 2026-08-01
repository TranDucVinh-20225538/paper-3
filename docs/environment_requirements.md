# Environment Requirements

**Scope**: what `paper-3/` actually needs to run, determined by reading real import statements (not by copying either source repo's `requirements.txt` verbatim — both are already documented elsewhere as incomplete). No package is installed by this document; it is a specification to build an environment from, once explicitly authorized.

**Topology this project actually runs across** (per direct instruction, not inferred):

| Environment | CSG-SKin checkpoints | ISIC2018 / PAD-UFES data | DST-Skin | Role |
|---|---|---|---|---|
| Local (this Mac) | Present (`CSG-SKin/checkpoints` → `/Users/cubo/ResearchArtifacts/checkpoints`, verified) | Absent | Present, with its own checkpoints + data | `paper-3/` code development; DST-Skin-path (E4) work is the one thing actually runnable here today |
| GPU server | Present (assumed) | Present (assumed: ISIC2018 + PAD-UFES) | **Not present** | E1–E3 execution |

This asymmetry matters for what follows: **E1–E3 (CSG-SKin path) can only run on the server; E4 (DST-Skin path), if implemented, can only run locally**, since DST-Skin isn't on the server at all. A single unified environment spec covering both paths doesn't correspond to any single machine this project actually uses — see §4.

---

## 1. Direct dependencies of `paper-3/scripts/*.py`

From `geometry_diagnostics.py`: `numpy` only (stdlib `sys`, `pathlib`, `typing` aside), plus the one CSG-SKin import below.

From `extract_embeddings_e1.py`: `numpy`, `torch`, `scikit-learn` (`sklearn.model_selection.train_test_split`), `torch.utils.data.DataLoader` — plus the CSG-SKin imports below.

## 2. Transitive dependencies, via the exact CSG-SKin modules these scripts import

Traced by grepping the actual `import`/`from` lines in every CSG-SKin file reached (`src/utils/ood_metrics.py`, `src/datasets/skin_dataset.py`, `src/datasets/splits.py`, `src/datasets/constants.py`, `src/models/baseline.py`, `src/models/csg_lightning.py`, `src/models/csg_lite.py`, `src/models/effb3_single.py`, `src/losses/csg_losses.py`), not assumed from `CSG-SKin/requirements.txt`:

| Package | Why needed | Evidence |
|---|---|---|
| `torch` | Everywhere | all model/dataset files |
| `torchvision` | `resnet50`, `efficientnet_b3` backbones, `transforms` | `baseline.py`, `csg_lite.py`, `effb3_single.py`, `skin_dataset.py` |
| `pytorch-lightning` (import name `pytorch_lightning`) | `CSGLiteLightning`, `BaselineResNet50`, `EffB3SingleLightning`, `SkinDataModule` all subclass `pl.LightningModule`/`pl.LightningDataModule` | `csg_lightning.py`, `baseline.py`, `effb3_single.py`, `skin_dataset.py` |
| `torchmetrics` | `MulticlassAccuracy`, `MulticlassConfusionMatrix`, `MulticlassF1Score` | `csg_lightning.py`, `baseline.py`, `effb3_single.py` — **not listed in `CSG-SKin/requirements.txt` at all** (already flagged in `REPOSITORY_MAP.md` §8 item 8; independently re-confirmed here by direct import grep) |
| `pandas` | metadata CSV handling | `skin_dataset.py`, `splits.py` |
| `scikit-learn` | `train_test_split`, `roc_auc_score`, `roc_curve` | `skin_dataset.py`, `splits.py`, `ood_metrics.py`, `extract_embeddings_e1.py` directly |
| `Pillow` (import name `PIL`) | `Image` loading | `skin_dataset.py` |

**`opencv-python` is in `CSG-SKin/requirements.txt` but is not imported by anything `paper-3/` reaches** (consistent with `REPOSITORY_MAP.md`'s finding that CSG-SKin's own Otsu-crop code is hand-rolled NumPy, not `cv2`). Not included in the spec below — installing it would be harmless but is not a real requirement of this pipeline.

## 3. Version constraints determinable from API usage (not guesses — specific code evidence)

No lockfile exists anywhere in CSG-SKin or DST-Skin (`REPOSITORY_MAP.md` already flags this — "no version pins at all"). The following are the constraints actually inferable by reading which APIs are called:

- **`torchvision >= 0.13`**: `baseline.py`/`effb3_single.py` use the `Weights` enum API (`ResNet50_Weights.IMAGENET1K_V1`, `EfficientNet_B3_Weights.IMAGENET1K_V1`), introduced in torchvision 0.13, replacing the older `pretrained=True` boolean.
- **`torchmetrics >= 0.11`**: the task-specific `torchmetrics.classification.Multiclass*` class names (`MulticlassAccuracy`, etc.) were introduced in that redesign; earlier versions used a differently-shaped `Accuracy(num_classes=..., average=...)` API that would not match these imports.
- **`pytorch-lightning`**: only stable, long-supported APIs are used (`LightningModule`, `LightningDataModule`, `.load_from_checkpoint()`, `.save_hyperparameters()`, `Trainer`/`ModelCheckpoint` in the training scripts paper-3 doesn't touch) — no version-specific feature observed that narrows this further from source reading alone. **This is exactly the case where API-compatibility reasoning runs out and only matching the checkpoint-producing environment's actual version is reliable** — pytorch-lightning has had breaking changes to checkpoint internals across 1.x→2.x; a version mismatch here fails at `load_from_checkpoint()` time, not at import time, which is a worse failure mode (later, harder to diagnose) than a missing-package `ImportError`.

**Everything else (`torch`, `numpy`, `pandas`, `scikit-learn`, `Pillow` exact versions) cannot be determined from source reading alone** — no version-gated API is used for any of them in the code paths paper-3 reaches.

## 4. What this means for pinning — do not invent versions here

The checkpoints in `CSG-SKin/checkpoints/` were produced by *some* concrete environment (presumably the GPU server, or an earlier version of it) that is not recorded anywhere in either repo. Picking fresh/latest versions now and hoping they load 16GB of existing `.ckpt` files correctly is exactly the "works on my machine" risk already raised. The reliable sequence, in order:

1. **On the GPU server** (where CSG-SKin training/eval scripts are already known to work, since the checkpoints exist there): run `pip freeze` and capture that as the reference lock for the CSG-SKin path (E1–E3). This machine already proved it can load these exact checkpoints — nothing here needs to be inferred.
2. **Diff that lock against this document's package list** (§2) — flag anything present on the server but not accounted for here (an unnoticed transitive dependency this audit missed) or vice versa (something listed here as needed that the server doesn't actually have, which would mean either the server environment is itself incomplete or one of these code paths is dead).
3. Only pin exact versions in a `paper-3/requirements-e1.txt` (or equivalent) once step 1–2 are done — not from this document alone.
4. **For DST-Skin (E4)**, `DST-Skin/requirements.txt` already gives loose (`>=`) pins; since DST-Skin has no `pytorch-lightning` dependency at all (confirmed — no `pytorch_lightning`/`lightning` import anywhere in `DST-Skin/src` or `DST-Skin/scripts`), its own environment is simpler and can plausibly be built from that file directly on the local machine, where DST-Skin's data and checkpoints already are. This is a separate environment from E1–E3's, not a shared one — see §6.

## 5. This local machine's current state (informational only — not a target to match)

Installed: `numpy 2.4.2`, `torch 2.10.0`, `torchvision 0.25.0`, `pandas 3.0.1`, `scikit-learn 1.8.0`, `scipy 1.17.1`, `Pillow 12.1.1`, `opencv-python 4.13.0.92`. Python 3.13.12.

**Missing for the CSG-SKin path**: `pytorch-lightning` (confirmed via `pip3 list` and a direct `ModuleNotFoundError`), `torchmetrics` (not checked directly yet — should be verified alongside `pytorch-lightning` before any install, since both are needed together for the same import chain).

This machine's versions (`torch 2.10.0`, `torchvision 0.25.0`) are almost certainly *newer* than whatever produced the existing checkpoints (both are very recent releases relative to a project with checkpoints already sitting at ~250MB×dozens of files, i.e. not created yesterday) — installing `pytorch-lightning` here and declaring the environment "ready" without checking this would risk exactly the version-drift problem this whole document exists to avoid. This machine is not necessarily where E1–E3 should run at all, per §"Topology" above — it's where `paper-3/` code gets written and, for E4, where DST-Skin's local checkpoints/data already are.

## 6. Two separate environments, not one shared spec

- **`paper-3-e1e3.env`** (server-side, CSG-SKin path): `torch`, `torchvision>=0.13`, `pytorch-lightning` (version = whatever the server's existing working environment has — see §4), `torchmetrics>=0.11`, `pandas`, `scikit-learn`, `Pillow`, `numpy`.
- **`paper-3-e4.env`** (local, DST-Skin path, if/when E4 is implemented): per `DST-Skin/requirements.txt` — `torch>=2.0`, `torchvision>=0.15`, `numpy>=1.23`, `pandas>=1.5`, `scikit-learn>=1.1`, `scipy>=1.10`, `Pillow>=9.0` (plus `umap-learn`/`seaborn`/`matplotlib`/`tqdm`/`opencv-python` only if E4 ever touches DST-Skin's plotting/UMAP scripts, which the current E4 contract scope — condition number + Mardia kurtosis only — does not require).

These should not be conflated into one requirements file: E1–E3's environment needs `pytorch-lightning`/`torchmetrics` that E4 never uses, and E4's environment needs to exist somewhere `pytorch-lightning`'s presence or absence is irrelevant, on a machine that has no CSG-SKin checkpoints at all.

## 7. Action items (not executed by this document)

- [ ] On the GPU server: confirm CSG-SKin's own scripts (e.g. `scripts/check_leakage.py` or `cbm_revision/scripts/eval_ood_benchmarks.py`) already run successfully there, and `pip freeze` that environment.
- [ ] Diff that freeze against §2's package list.
- [ ] Produce `paper-3/requirements-e1.txt` pinned from that freeze (not from this document's inferred lower bounds).
- [ ] Only then: create the environment and proceed to Task 3 (one checkpoint).
