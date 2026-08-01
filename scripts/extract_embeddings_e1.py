"""
E1 embedding extraction: one checkpoint in, one geometry-metrics row out.

Per paper-3/docs/experiment_contract.md (E1): takes a single, explicitly-named
checkpoint file (never a directory — no find_checkpoint-style resolution
anywhere in this script), extracts (features, labels) on the ISIC-train-under-
eval-transform loader, then computes the approved 3-metric geometry set via
geometry_diagnostics.compute_geometry_diagnostics.

Loader construction mirrors CSG-SKin's own
cbm_revision/scripts/eval_ood_benchmarks.py::_build_isic_train_loader exactly
(same split_config, same eval_transform, same SkinDataset) — this project
committed to that script as the canonical Mahalanobis-fitting source
(paper-3/docs/threats_to_validity.md #2), so the embeddings E1 measures
geometry on must come from the identical loader, not a re-derived one.

The three per-architecture extraction helpers below intentionally mirror
eval_ood_benchmarks.py's _collect_baseline_features / _collect_csg_features
(and add an EffB3 equivalent it doesn't have) rather than importing them,
because they are private, non-exported functions in a script, not in src/ --
there is nothing in src/ to import here. They are ~10-line forward-pass loops
with no independent logic to diverge on; the risk this reuse rule normally
guards against does not apply the same way it does to estimation code
(Mahalanobis fitting, splits) that IS reused via import throughout this file.

Does not implement checkpoint-manifest resolution or looping over multiple
checkpoints -- that is the batch driver, a separate, not-yet-authorized step.
This script handles exactly one checkpoint per invocation, matching Task 3's
scope ("run ONE checkpoint only") before Task 4 loops it.

TEMPORARY INSTRUMENTATION: heavily annotated with `_debug(...)` progress
prints (tagged "[E1-DEBUG t+SSSs]") to localize a GPU-server hang that occurs
before any GPU activity and persists with --num_workers=0 (ruling out a
multiprocessing worker deadlock). No scientific logic was changed to add
these -- the batch-extraction loops were rewritten from `for x in loader` to
an explicit `iter()`/`next()` form solely so "about to fetch a batch" and
"received a batch" can be logged as distinct steps; the data read, transform,
model calls, and accumulation are identical to before. Safe to strip all
`_debug(...)` calls and revert the loops to plain `for` once the hanging
statement is identified.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_START_TIME = time.time()


def _debug(msg: str) -> None:
    """Temporary instrumentation for the GPU-server hang investigation -- not
    part of E1's scientific logic. Remove once the hang is diagnosed."""
    elapsed = time.time() - _START_TIME
    print(f"[E1-DEBUG t+{elapsed:7.2f}s] {msg}", flush=True)


_debug("module execution started (stdlib imports done)")

import numpy as np  # noqa: E402

_debug("numpy imported")

import torch  # noqa: E402

_debug(f"torch imported (version={torch.__version__})")

from sklearn.model_selection import train_test_split  # noqa: E402

_debug("scikit-learn imported")

from torch.utils.data import DataLoader  # noqa: E402

# CSG-SKin's root is located via _repo_paths (marker-file search,
# layout-agnostic) rather than a hardcoded parent-count -- paper-3/ has been
# deployed both as a CSG-SKin sibling and nested inside it (see _repo_paths.py).
from _repo_paths import find_csg_skin_root  # noqa: E402

_debug("_repo_paths imported")

_CSG_ROOT = find_csg_skin_root(__file__)
_debug(f"CSG-SKin root resolved: {_CSG_ROOT}")
if str(_CSG_ROOT) not in sys.path:
    sys.path.insert(0, str(_CSG_ROOT))

_debug("importing CSG-SKin src modules (this pulls in pytorch_lightning, torchmetrics)...")

from src.datasets.skin_dataset import SkinDataModule, SkinDataset  # noqa: E402
from src.datasets.splits import load_filtered_master  # noqa: E402
from src.models.baseline import BaselineResNet50  # noqa: E402
from src.models.csg_lightning import CSGLiteLightning  # noqa: E402
from src.models.effb3_single import EffB3SingleLightning  # noqa: E402

_debug("CSG-SKin src modules imported OK")

from geometry_diagnostics import compute_geometry_diagnostics  # noqa: E402

_debug("geometry_diagnostics imported OK -- all imports complete")

NUM_CLASSES = 8  # CSG-SKin's fixed 8-class ISIC label space (src/datasets/constants.py)
REG_EPS = 1e-5  # matches compute_mahalanobis_params_from_arrays' own default, reused not re-chosen
METHODS = {"baseline", "csg", "effb3_single"}


def _require_explicit_file(path_str: str) -> Path:
    """Refuse directories outright -- REPOSITORY_MAP.md risk #5 / experiment_contract.md E1."""
    p = Path(path_str)
    if not p.is_file():
        raise ValueError(
            f"--checkpoint must be an explicit file, got: {path_str!r} "
            f"(is_dir={p.is_dir()}, exists={p.exists()}). "
            "Directory-based checkpoint resolution is explicitly disallowed for this project "
            "(REPOSITORY_MAP.md risk #5: find_checkpoint/_find_ckpt is confirmed to pick non-best "
            "checkpoints for several runB/runB_orth1 seed directories)."
        )
    return p


def _build_isic_train_loader(dm: SkinDataModule, batch_size: int, num_workers: int) -> DataLoader:
    """Mirrors eval_ood_benchmarks.py::_build_isic_train_loader exactly."""
    _debug(f"reading metadata CSV: {dm.metadata_csv}")
    df = load_filtered_master(dm.metadata_csv)
    _debug(f"metadata CSV loaded: {len(df)} total rows")

    isic_df = df[df["domain"] == "isic"].copy()
    _debug(f"filtered to ISIC domain: {len(isic_df)} rows")

    _debug("splitting ISIC rows: train_val vs test")
    isic_train_val, _ = train_test_split(
        isic_df,
        test_size=dm.split_config.isic_test_fraction,
        stratify=isic_df["label_idx"],
        random_state=dm.split_config.random_state,
    )
    _debug(f"train_val/test split done: {len(isic_train_val)} rows in train_val")

    _debug("splitting ISIC rows: train vs val")
    isic_train, _ = train_test_split(
        isic_train_val,
        test_size=dm.split_config.val_fraction,
        stratify=isic_train_val["label_idx"],
        random_state=dm.split_config.random_state,
    )
    _debug(f"train/val split done: {len(isic_train)} rows in train (this is E1's fitting set)")

    _debug("constructing SkinDataset")
    dataset = SkinDataset(isic_train, transform=dm.eval_transform)
    _debug(f"SkinDataset constructed: len={len(dataset)}")

    _debug(f"constructing DataLoader (batch_size={batch_size}, num_workers={num_workers})")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    _debug("DataLoader object constructed (lazy -- no data touched yet, no worker processes spawned yet)")
    return loader


def _log_first_batch_progress(batch_idx: int, images, y, stage: str) -> None:
    """Temporary instrumentation helper shared by the three _extract_* loops below."""
    if batch_idx == 0:
        _debug(f"received first batch: images.shape={tuple(images.shape)} labels.shape={tuple(y.shape)}")
    elif batch_idx % 20 == 0:
        _debug(f"received batch {batch_idx} ({stage})")


@torch.no_grad()
def _extract_baseline(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    _debug("entering inference loop (baseline)")
    loader_iter = iter(loader)
    batch_idx = 0
    while True:
        if batch_idx == 0 or batch_idx % 20 == 0:
            _debug("requesting first batch" if batch_idx == 0 else f"requesting batch {batch_idx}")
        try:
            images, y = next(loader_iter)
        except StopIteration:
            break
        _log_first_batch_progress(batch_idx, images, y, "baseline")
        images = images.to(device, non_blocking=True)
        if batch_idx == 0:
            _debug("first batch moved to device, running model forward")
        _logits, f = model(images, return_features=True)
        if batch_idx == 0:
            _debug("first batch processed through model")
        feats.append(f.cpu())
        labels.append(y.cpu())
        batch_idx += 1
    _debug(f"inference loop complete (baseline): {batch_idx} batches processed")
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


@torch.no_grad()
def _extract_csg(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    _debug("entering inference loop (csg)")
    loader_iter = iter(loader)
    batch_idx = 0
    while True:
        if batch_idx == 0 or batch_idx % 20 == 0:
            _debug("requesting first batch" if batch_idx == 0 else f"requesting batch {batch_idx}")
        try:
            images, y = next(loader_iter)
        except StopIteration:
            break
        _log_first_batch_progress(batch_idx, images, y, "csg")
        images = images.to(device, non_blocking=True)
        if batch_idx == 0:
            _debug("first batch moved to device, running model forward")
        _logits, _d_ctx, _d_adv, z_lesion, _z_ctx = model(images, x_lesion=None, return_latents=True)
        if batch_idx == 0:
            _debug("first batch processed through model")
        feats.append(z_lesion.cpu())
        labels.append(y.cpu())
        batch_idx += 1
    _debug(f"inference loop complete (csg): {batch_idx} batches processed")
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


@torch.no_grad()
def _extract_effb3(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    _debug("entering inference loop (effb3_single)")
    loader_iter = iter(loader)
    batch_idx = 0
    while True:
        if batch_idx == 0 or batch_idx % 20 == 0:
            _debug("requesting first batch" if batch_idx == 0 else f"requesting batch {batch_idx}")
        try:
            images, y = next(loader_iter)
        except StopIteration:
            break
        _log_first_batch_progress(batch_idx, images, y, "effb3_single")
        images = images.to(device, non_blocking=True)
        if batch_idx == 0:
            _debug("first batch moved to device, running model forward")
        z = model.extract_embedding(images)
        if batch_idx == 0:
            _debug("first batch processed through model")
        feats.append(z.cpu())
        labels.append(y.cpu())
        batch_idx += 1
    _debug(f"inference loop complete (effb3_single): {batch_idx} batches processed")
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def load_and_extract(checkpoint: Path, method: str, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    _debug(f"load_and_extract: method={method}, checkpoint={checkpoint}")
    if method == "baseline":
        _debug("loading BaselineResNet50.load_from_checkpoint (torch.load under the hood)")
        lit = BaselineResNet50.load_from_checkpoint(str(checkpoint))
        _debug("checkpoint loaded into LightningModule")
        _debug(f"moving model to device={device}")
        model = lit.to(device)
        _debug("model moved to device")
        return _extract_baseline(model, loader, device)
    if method == "csg":
        _debug("loading CSGLiteLightning.load_from_checkpoint (strict=False)")
        lit = CSGLiteLightning.load_from_checkpoint(str(checkpoint), strict=False)
        _debug("checkpoint loaded into LightningModule")
        _debug(f"moving underlying CSGLite model to device={device}")
        model = lit.model.to(device)
        _debug("model moved to device")
        return _extract_csg(model, loader, device)
    if method == "effb3_single":
        # No strict=False here, matching the one real reference call
        # (cbm_revision/scripts/run_effb3_control.py:229) -- unlike CSGLiteLightning,
        # nothing documents a state-dict mismatch reason for this class.
        _debug("loading EffB3SingleLightning.load_from_checkpoint")
        lit = EffB3SingleLightning.load_from_checkpoint(str(checkpoint))
        _debug("checkpoint loaded into LightningModule")
        _debug(f"moving underlying net to device={device}")
        model = lit.net.to(device)
        _debug("model moved to device")
        return _extract_effb3(model, loader, device)
    raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")


def append_geometry_row(csv_path: Path, row: dict) -> None:
    fieldnames = [
        "rung", "method", "seed", "checkpoint_path", "n_samples", "feat_dim",
        "condition_number", "fisher_ratio_HL", "fisher_ratio_scalar",
        "mardia_kurtosis_b", "mardia_kurtosis_z",
    ]
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    _debug("main() started")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Explicit .ckpt file path. Directories are rejected.")
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--rung", required=True, help="e.g. baseline_soft, runA_grl, runB_orth1, runB")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--metadata_csv", default=str(_CSG_ROOT / "data" / "master_metadata_lesion_only_soft.csv"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    _debug("parsing arguments")
    args = parser.parse_args()
    _debug(
        f"arguments parsed: checkpoint={args.checkpoint} method={args.method} rung={args.rung} "
        f"seed={args.seed} metadata_csv={args.metadata_csv} batch_size={args.batch_size} "
        f"num_workers={args.num_workers} output_dir={args.output_dir}"
    )

    _debug("validating checkpoint path (must be an explicit file, not a directory)")
    checkpoint = _require_explicit_file(args.checkpoint)
    _debug(f"checkpoint path OK: {checkpoint} (size={checkpoint.stat().st_size} bytes)")

    _debug("checking CUDA availability (torch.cuda.is_available())")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _debug(f"device resolved: {device}")

    _debug("constructing SkinDataModule")
    dm = SkinDataModule(metadata_csv=args.metadata_csv, batch_size=args.batch_size, num_workers=args.num_workers)
    _debug("SkinDataModule constructed (note: .setup() is intentionally never called -- "
           "_build_isic_train_loader reads the metadata CSV itself and does not need it)")

    _debug("building ISIC-train-under-eval-transform DataLoader")
    loader = _build_isic_train_loader(dm, args.batch_size, args.num_workers)
    _debug(f"DataLoader ready (dataset size={len(loader.dataset)})")

    _debug(f"loading checkpoint and extracting embeddings (method={args.method})")
    features, labels = load_and_extract(checkpoint, args.method, loader, device)
    _debug(f"extraction complete: features.shape={features.shape} labels.shape={labels.shape}")

    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "e1_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = embeddings_dir / f"{args.rung}_s{args.seed}.npz"
    _debug(f"saving embeddings to {embeddings_path}")
    np.savez(
        embeddings_path,
        features=features,
        labels=labels,
        checkpoint_path=str(checkpoint),
        method=args.method,
        rung=args.rung,
        seed=args.seed,
    )
    _debug("embeddings saved")

    _debug("computing geometry diagnostics")
    diagnostics = compute_geometry_diagnostics(features, labels, num_classes=NUM_CLASSES, reg_eps=REG_EPS)
    _debug("geometry diagnostics computed")

    row = {
        "rung": args.rung,
        "method": args.method,
        "seed": args.seed,
        "checkpoint_path": str(checkpoint),
        "n_samples": diagnostics.n_samples,
        "feat_dim": diagnostics.feat_dim,
        "condition_number": diagnostics.condition_number,
        "fisher_ratio_HL": diagnostics.fisher_ratio,
        "fisher_ratio_scalar": diagnostics.fisher_ratio_scalar,
        "mardia_kurtosis_b": diagnostics.mardia_kurtosis_b,
        "mardia_kurtosis_z": diagnostics.mardia_kurtosis_z,
    }
    _debug(f"appending geometry row to {output_dir / 'e1_geometry_metrics.csv'}")
    append_geometry_row(output_dir / "e1_geometry_metrics.csv", row)
    _debug("CSV row appended -- done")

    print(f"Embeddings saved: {embeddings_path}")
    print(f"Geometry row appended: {output_dir / 'e1_geometry_metrics.csv'}")
    print(row)


if __name__ == "__main__":
    main()
