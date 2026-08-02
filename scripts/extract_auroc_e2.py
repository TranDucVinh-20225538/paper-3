"""
extract_auroc_e2.py -- E2 input: one checkpoint in, one Mahalanobis AUROC/FPR95 row out.

Refactored from CSG-SKin's own cbm_revision/scripts/eval_ood_benchmarks.py --
the canonical E1-E3 Mahalanobis source per threats_to_validity.md #2 -- into a
single-checkpoint, explicit-path script matching extract_embeddings_e1.py's
design exactly (Task 3 pattern: one checkpoint per invocation; a batch driver
looping over all 13 primary-ladder checkpoints is a separate, later,
not-yet-authorized step, same as it was for E1).

Two departures from the canonical script's own control flow, both REQUIRED
by this task's explicit-checkpoint-path requirement, not optional style
choices:

  1. No _find_ckpt() / _method_ckpt_dir(), anywhere. The canonical script
     resolves each seed's checkpoint by globbing a directory for the
     newest-by-mtime best-*.ckpt file. open_questions.md Q3 confirms this is
     the SAME class of bug REPOSITORY_MAP.md risk #5 already found in
     find_checkpoint -- silently wrong for several runB/runB_orth1 seed
     directories. This script takes --checkpoint as an explicit,
     individually-verified file path instead; --rung/--seed are metadata
     recorded on the output row, never used to locate anything.

  2. Method dispatch is by --method (baseline/csg/effb3_single), not a
     hardcoded per-rung checkpoint-directory table. The canonical script's
     `methods` list only covers "Baseline Soft" / "Run A / GRL" /
     "Run B (orth=1.0)" -- it has NO case for `runB` (orth=5.0) at all, so it
     could never have produced an AUROC number for 3 of E1a's 13
     primary-ladder checkpoints. Since --method here only selects an
     ARCHITECTURE (identical across runA_grl/runB_orth1/runB -- all
     CSGLiteLightning; only lambda_orth differed during training, which
     doesn't change how features are extracted from an already-trained
     checkpoint), runB is supported with no special-casing needed. This was
     a genuine gap in the canonical script for this project's purposes, not
     something this refactor introduces by choice.

Everything else is mirrored, not reinvented:
  - _build_isic_train_loader: byte-for-byte the same construction as
    extract_embeddings_e1.py's (which itself already mirrors this canonical
    script) -- same split_config, same eval_transform, same SkinDataset.
  - build_id_ood_test_dataloaders(dm) for the ISIC-test (ID) / PAD-UFES (OOD)
    split -- imported from src.datasets.splits, not reimplemented. Requires
    dm.setup() first (extract_embeddings_e1.py deliberately never calls
    dm.setup() because it doesn't need this loader; this script does).
  - The Mahalanobis fit + score + AUROC/FPR95 computation: reuses
    src.utils.ood_metrics.compute_mahalanobis_params_from_arrays,
    mahalanobis_min_squared_distances, and fpr_at_95_tpr directly (imported,
    not reimplemented), with the same y_ood/score concatenation convention as
    eval_ood_benchmarks.py::_compute_scores's Mahalanobis branch. reg_eps is
    the one deliberate exception to "mirrored exactly" -- see below.
  - seed_everything(42) + pl.seed_everything(42, workers=True): the
    canonical script's own fixed global seed (NOT the checkpoint's own
    training seed) -- reproduced exactly, once per invocation.

REG_EPS DELIBERATELY DIVERGES FROM eval_ood_benchmarks.py, one line, by
decision (open_questions.md Q5): the canonical script fits Mahalanobis with
reg_eps=1e-3, an undocumented override with no comment anywhere in CSG-SKin
justifying it over compute_mahalanobis_params_from_arrays's own default,
reg_eps=1e-5 -- confirmed by reading every occurrence of reg_eps in the
codebase before making this change, not assumed. This script uses 1e-5
instead, matching extract_embeddings_e1.py's geometry metrics, so E1's
precision matrix and E2's precision matrix for the same nominal (rung, seed)
are the literal same fitted object -- required for the paper's actual claim
(geometry of THIS precision matrix explains AUROC) to be about one thing,
not two differently-regularized approximations of it. Every other part of
eval_ood_benchmarks.py's methodology (loader, transform, split, scoring
convention) is still reproduced exactly -- this is the one intentional,
reasoned departure, not an oversight.

Does not implement batching or E2's Kendall/Jonckheere analysis -- this
script's whole job is one checkpoint, one row, matching Task 3's scope for
E1 before Task 4 looped it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# CSG-SKin's root is located via _repo_paths (marker-file search,
# layout-agnostic) -- see _repo_paths.py / geometry_diagnostics.py's own use
# of it after the server's nested-layout deployment broke a hardcoded
# parent-count assumption.
from _repo_paths import find_csg_skin_root  # noqa: E402

_CSG_ROOT = find_csg_skin_root(__file__)
if str(_CSG_ROOT) not in sys.path:
    sys.path.insert(0, str(_CSG_ROOT))

from src.datasets.skin_dataset import SkinDataModule, SkinDataset  # noqa: E402
from src.datasets.splits import build_id_ood_test_dataloaders, load_filtered_master  # noqa: E402
from src.models.baseline import BaselineResNet50  # noqa: E402
from src.models.csg_lightning import CSGLiteLightning  # noqa: E402
from src.models.effb3_single import EffB3SingleLightning  # noqa: E402
from src.utils import ood_metrics  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

NUM_CLASSES = 8  # CSG-SKin's fixed 8-class ISIC label space (src/datasets/constants.py)
REG_EPS = 1e-5  # deliberately NOT eval_ood_benchmarks.py's 1e-3 -- matches E1's geometry metrics instead, per open_questions.md Q5
GLOBAL_SEED = 42  # eval_ood_benchmarks.py's own fixed seed, independent of the checkpoint's training seed
METHODS = {"baseline", "csg", "effb3_single"}


def _require_explicit_file(path_str: str) -> Path:
    """Refuse directories outright -- REPOSITORY_MAP.md risk #5 / open_questions.md Q3."""
    p = Path(path_str)
    if not p.is_file():
        raise ValueError(
            f"--checkpoint must be an explicit file, got: {path_str!r} "
            f"(is_dir={p.is_dir()}, exists={p.exists()}). "
            "Directory-based checkpoint resolution is explicitly disallowed for this project "
            "(REPOSITORY_MAP.md risk #5: find_checkpoint is confirmed to pick non-best checkpoints "
            "for several runB/runB_orth1 seed directories; open_questions.md Q3: "
            "eval_ood_benchmarks.py's own _find_ckpt has the identical bug)."
        )
    return p


def _build_isic_train_loader(dm: SkinDataModule, batch_size: int, num_workers: int) -> DataLoader:
    """Byte-for-byte the same construction as extract_embeddings_e1.py's / eval_ood_benchmarks.py's."""
    df = load_filtered_master(dm.metadata_csv)
    isic_df = df[df["domain"] == "isic"].copy()
    isic_train_val, _ = train_test_split(
        isic_df,
        test_size=dm.split_config.isic_test_fraction,
        stratify=isic_df["label_idx"],
        random_state=dm.split_config.random_state,
    )
    isic_train, _ = train_test_split(
        isic_train_val,
        test_size=dm.split_config.val_fraction,
        stratify=isic_train_val["label_idx"],
        random_state=dm.split_config.random_state,
    )
    dataset = SkinDataset(isic_train, transform=dm.eval_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def _extract_baseline(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        _logits, f = model(images, return_features=True)
        feats.append(f.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


@torch.no_grad()
def _extract_csg(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        _logits, _d_ctx, _d_adv, z_lesion, _z_ctx = model(images, x_lesion=None, return_latents=True)
        feats.append(z_lesion.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


@torch.no_grad()
def _extract_effb3(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        z = model.extract_embedding(images)
        feats.append(z.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def load_model(checkpoint: Path, method: str, device):
    """
    Loads the checkpoint ONCE and returns (model, extract_fn) so the caller can
    reuse the same in-memory model across the train/id/ood loaders -- matching
    eval_ood_benchmarks.py's own efficiency (it loads each checkpoint once and
    calls its collector three times), not extract_embeddings_e1.py's per-loader
    load_and_extract (E1 only ever needed one loader per invocation, so it
    never had this redundant-reload consideration).
    """
    if method == "baseline":
        lit = BaselineResNet50.load_from_checkpoint(str(checkpoint))
        return lit.to(device), _extract_baseline
    if method == "csg":
        lit = CSGLiteLightning.load_from_checkpoint(str(checkpoint), strict=False)
        return lit.model.to(device), _extract_csg
    if method == "effb3_single":
        # No strict=False here, matching extract_embeddings_e1.py's precedent
        # (cbm_revision/scripts/run_effb3_control.py:229 is the one real
        # reference call for this class, and it uses no such flag).
        lit = EffB3SingleLightning.load_from_checkpoint(str(checkpoint))
        return lit.net.to(device), _extract_effb3
    raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")


def compute_mahalanobis_auroc_fpr95(
    z_train: np.ndarray, y_train: np.ndarray, z_id: np.ndarray, z_ood: np.ndarray,
    num_classes: int, reg_eps: float,
) -> tuple[float, float]:
    """
    Mirrors eval_ood_benchmarks.py::_compute_scores's Mahalanobis branch
    exactly: fit on (z_train, y_train), score z_id/z_ood by min squared
    Mahalanobis distance to the nearest class mean, y=1 for OOD (PAD-UFES),
    higher score = more OOD-like -- same label/score convention, not
    reinterpreted.
    """
    means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(
        z_train, y_train, num_classes=num_classes, reg_eps=reg_eps
    )
    s_id = ood_metrics.mahalanobis_min_squared_distances(z_id, means, precision)
    s_ood = ood_metrics.mahalanobis_min_squared_distances(z_ood, means, precision)

    y_ood_binary = np.concatenate([np.zeros(len(s_id), dtype=np.int64), np.ones(len(s_ood), dtype=np.int64)])
    scores = np.concatenate([s_id, s_ood])

    auroc = float(roc_auc_score(y_ood_binary, scores))
    fpr95 = ood_metrics.fpr_at_95_tpr(y_ood_binary, scores)
    return auroc, fpr95


def append_auroc_row(csv_path: Path, row: dict) -> None:
    fieldnames = ["rung", "method", "seed", "checkpoint_path", "auroc", "fpr95"]
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Explicit .ckpt file path. Directories are rejected.")
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--rung", required=True, help="e.g. baseline_soft, runA_grl, runB_orth1, runB")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--metadata_csv", default=str(_CSG_ROOT / "data" / "master_metadata_lesion_only_soft.csv"))
    parser.add_argument("--batch_size", type=int, default=128)  # matches eval_ood_benchmarks.py's own default
    parser.add_argument("--num_workers", type=int, default=8)   # matches eval_ood_benchmarks.py's own default (E1 used 4)
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = parser.parse_args()

    checkpoint = _require_explicit_file(args.checkpoint)

    seed_everything(GLOBAL_SEED)
    pl.seed_everything(GLOBAL_SEED, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = SkinDataModule(metadata_csv=args.metadata_csv, batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup()  # REQUIRED here -- build_id_ood_test_dataloaders needs dm.pad_df_all etc. populated
    id_loader, ood_loader = build_id_ood_test_dataloaders(dm)
    train_loader = _build_isic_train_loader(dm, args.batch_size, args.num_workers)

    model, extract_fn = load_model(checkpoint, args.method, device)
    z_train, y_train = extract_fn(model, train_loader, device)
    z_id, _y_id = extract_fn(model, id_loader, device)
    z_ood, _y_ood = extract_fn(model, ood_loader, device)

    auroc, fpr95 = compute_mahalanobis_auroc_fpr95(z_train, y_train, z_id, z_ood, NUM_CLASSES, REG_EPS)

    row = {
        "rung": args.rung,
        "method": args.method,
        "seed": args.seed,
        "checkpoint_path": str(checkpoint),
        "auroc": auroc,
        "fpr95": fpr95,
    }
    output_dir = Path(args.output_dir)
    append_auroc_row(output_dir / "e2_auroc.csv", row)

    print(f"AUROC={auroc:.4f} FPR95={fpr95:.4f}")
    print(row)


if __name__ == "__main__":
    main()
