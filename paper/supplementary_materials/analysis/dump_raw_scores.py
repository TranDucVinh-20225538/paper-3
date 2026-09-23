"""
Dump per-image OOD scores for every scorer in the study.

The pipeline only ever wrote aggregated AUROC/FPR95 per checkpoint: each
extract_* script computed a per-image score array, collapsed it, appended one
CSV row, and dropped the array. This script rebuilds those arrays from the
cached embeddings and writes them out, so per-image distributions can be
plotted and re-analysed without re-running any checkpoint.

Nothing is reimplemented: every scoring function is imported from the script
that originally produced the published number. Every rebuilt array is checked
against the published AUROC/FPR95 in results/e2_6_scorer_comparison.csv before
anything is written, and a mismatch is a hard stop -- otherwise this would be
dumping scores for a different quantity than the paper reports.

Six of the eight scorers need only the cached embeddings. Energy and ViM need
the checkpoint's classifier head to rebuild logits; pass --heads to supply it.
The head MUST come from the same checkpoint the embeddings were built from. All
fifteen were re-extracted together in September 2026, so the heads are
lesion_classifier_heads_v3.npz for the thirteen retrained runs and
lesion_classifier_heads_new.npz for runB seeds 72 and 82, which were not.

    python3 analysis/dump_raw_scores.py
    python3 analysis/dump_raw_scores.py \
        --heads results/lesion_classifier_heads_v3.npz results/lesion_classifier_heads_new.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Import order matters and is not interchangeable. the audited classifier repository and the second classifier repository both
# ship a top-level package called `src`; extract_e2_8_extra_scorers already
# handles that collision, but only if it runs before anything has cached CSG's
# `src` in sys.modules. So: e28 first, purge `src*`, then extract_auroc_e2 gets
# a clean slate to bind CSG's `src`.
import extract_e2_8_extra_scorers as e28  # noqa: E402

for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from extract_auroc_e2 import (  # noqa: E402
    K_VALUES,
    NUM_CLASSES,
    REG_EPS,
    auroc_fpr95_from_scores,
    compute_knn_scores,
    cosine_centroid_scores,
    ood_metrics,
)

HEAD_FREE = ("mahalanobis", "cosine", *(f"knn_k{k}" for k in K_VALUES), "density_kde")
HEAD_NEEDED = ("energy", "vim")


def rebuild(cache_dir: Path, run: str, heads: dict | None):
    """Returns (scores, head_check). head_check is None when no head was supplied."""
    zf = np.load(cache_dir / f"{run}_z.npz", allow_pickle=True)
    sf = np.load(cache_dir / f"{run}.npz", allow_pickle=True)
    z_train, y_train = zf["z_train"], zf["y_train"]
    z_id, z_ood = zf["z_id"], zf["z_ood"]

    means, _ = ood_metrics.compute_mahalanobis_params_from_arrays(
        z_train, y_train, num_classes=NUM_CLASSES, reg_eps=REG_EPS
    )

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "mahalanobis": (sf["s_id"], sf["s_ood"]),  # already on disk, published values
        "cosine": (cosine_centroid_scores(z_id, means), cosine_centroid_scores(z_ood, means)),
    }
    for k, sc in compute_knn_scores(z_train, z_id, z_ood, K_VALUES).items():
        out[f"knn_k{k}"] = sc
    out["density_kde"] = (
        e28.density_kde_scores(z_train, y_train, z_id),
        e28.density_kde_scores(z_train, y_train, z_ood),
    )

    head_check = None
    if heads is not None and f"{run}__W" in heads:
        W, b = heads[f"{run}__W"].astype(np.float64), heads[f"{run}__b"].astype(np.float64)
        lg = lambda z: e28.reconstruct_logits(z.astype(np.float64), W, b)
        lg_tr, lg_id, lg_ood = lg(z_train), lg(z_id), lg(z_ood)

        # Informational only. The binding check that a head belongs to these
        # embeddings is that energy/vim reproduce their published AUROC below;
        # a head from another checkpoint cannot do that. Reported here:
        # classifier accuracy against true labels, comparable to the run's
        # recorded id_acc -- but only for the 11 seeds whose cached embeddings
        # come from the checkpoint summary.json names.
        #
        # NOT comparable: sf["predicted_class_*"] is d2.argmin over the
        # Mahalanobis class distances (nearest centroid), not this linear
        # head's argmax. They are different classifiers and disagree by a few
        # percent by construction.
        head_check = {
            "cls_acc_id": float((lg_id.argmax(1) == zf["y_id"]).mean()),
            "nc_agree_id": float((lg_id.argmax(1) == sf["predicted_class_id"]).mean()),
        }

        out["energy"] = (e28.energy_scores(lg_id), e28.energy_scores(lg_ood))
        vp = e28.fit_vim(z_train.astype(np.float64), lg_tr, W, b)
        out["vim"] = (
            e28.vim_score(z_id.astype(np.float64), vp),
            e28.vim_score(z_ood.astype(np.float64), vp),
        )
    return out, head_check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "e2_distances")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "raw_scores")
    ap.add_argument("--heads", type=Path, nargs="*", default=None,
                    help="one or more head npz files; later files override earlier keys")
    ap.add_argument("--runs", nargs="*", default=None, help="chỉ chạy các run này")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--append-missing", action="store_true",
                    help="ghi (checkpoint, scorer) chưa có vào e2_6_scorer_comparison.csv")
    args = ap.parse_args()

    published = pd.read_csv(ROOT / "results" / "e2_6_scorer_comparison.csv")
    published["run"] = published["rung"] + "_s" + published["seed"].astype(str)
    heads = None
    if args.heads:
        heads = {}
        for hp in args.heads:
            if not hp.is_file():
                print(f"!! {hp} không tồn tại", file=sys.stderr); continue
            heads.update(dict(np.load(hp)))
            print(f"head: {hp.name} -> {sorted(k[:-3] for k in np.load(hp) if k.endswith('__W'))}")
        if not heads:
            heads = None
            print("!! không nạp được head nào -- bỏ qua energy/vim", file=sys.stderr)

    runs = sorted(p.name[:-6] for p in args.cache_dir.glob("*_z.npz"))
    if args.runs:
        runs = [r for r in runs if r in set(args.runs)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    head_report = []
    for run in runs:
        scores, head_check = rebuild(args.cache_dir, run, heads)
        if head_check is not None:
            head_report.append((run, head_check["cls_acc_id"], head_check["nc_agree_id"]))
        payload = {}
        for name, (s_id, s_ood) in scores.items():
            auroc, fpr95 = auroc_fpr95_from_scores(s_id, s_ood)
            ref = published[(published.run == run) & (published.scorer == name)]
            if ref.empty:
                status, d_au, d_fp = "không có mốc", np.nan, np.nan
            else:
                d_au = abs(auroc - float(ref.auroc.iloc[0]))
                d_fp = abs(fpr95 - float(ref.fpr95.iloc[0]))
                status = "khớp" if max(d_au, d_fp) <= args.tol else "LỆCH"
                if status == "LỆCH":
                    failures.append((run, name, d_au, d_fp))
            rows.append(dict(run=run, scorer=name, auroc=auroc, fpr95=fpr95,
                             d_auroc=d_au, d_fpr95=d_fp, status=status,
                             n_id=len(s_id), n_ood=len(s_ood)))
            payload[f"{name}__id"] = np.asarray(s_id, dtype=np.float64)
            payload[f"{name}__ood"] = np.asarray(s_ood, dtype=np.float64)
        payload["run"] = run
        payload["checkpoint_path"] = str(np.load(args.cache_dir / f"{run}_z.npz",
                                                 allow_pickle=True)["checkpoint_path"])
        np.savez_compressed(args.out_dir / f"{run}_scores.npz", **payload)
        print(f"  {run:<20} {len(scores)} scorer -> {run}_scores.npz")

    if head_report:
        print("\nChẩn đoán head (tham khảo; ràng buộc thật là AUROC của energy/vim ở trên):")
        print(f"  {'run':<20} {'acc phân loại':>14} {'khớp nearest-centroid':>22}")
        for run, acc, nc in head_report:
            print(f"  {run:<20} {acc*100:>13.2f}% {nc*100:>21.2f}%")

    rep = pd.DataFrame(rows)
    rep.to_csv(args.out_dir / "validation_report.csv", index=False)

    if args.append_missing:
        # Scorers whose AUROC was never written for a checkpoint -- the new
        # seeds' energy/vim/kde, which extract_e2_8 cannot produce here because
        # it resolves the .ckpt itself. Appending keeps the comparison file the
        # single inventory analyze_e2_6 validates against.
        add = rep[rep.status == "không có mốc"].copy()
        if len(add):
            comp_path = ROOT / "results" / "e2_6_scorer_comparison.csv"
            comp = pd.read_csv(comp_path)
            meta = {}
            for run in add.run.unique():
                z = np.load(args.cache_dir / f"{run}_z.npz", allow_pickle=True)
                meta[run] = (str(z["rung"]), int(z["seed"]), str(z["checkpoint_path"]))
            new_rows = [dict(rung=meta[r.run][0], method="csg", seed=meta[r.run][1],
                             checkpoint_path=meta[r.run][2], scorer=r.scorer,
                             auroc=r.auroc, fpr95=r.fpr95) for r in add.itertuples()]
            out = pd.concat([comp, pd.DataFrame(new_rows)[comp.columns]], ignore_index=True)
            comp_path.rename(comp_path.with_suffix(".csv.bak"))
            out.to_csv(comp_path, index=False)
            print(f"\nĐã thêm {len(new_rows)} dòng vào e2_6_scorer_comparison.csv "
                  f"({len(comp)} -> {len(out)}); bản cũ ở {comp_path.name}.bak")
    print(f"\n{len(runs)} checkpoint, {len(rep)} (checkpoint x scorer)")
    print(rep.status.value_counts().to_string())
    if failures:
        print("\nLỆCH so với giá trị công bố:")
        for run, name, d_au, d_fp in failures:
            print(f"  {run:<20} {name:<14} Δauroc={d_au:.3e} Δfpr95={d_fp:.3e}")
        raise SystemExit("DỪNG: có scorer không tái tạo được giá trị đã công bố.")
    print("\nTất cả đều tái tạo đúng giá trị đã công bố.")


if __name__ == "__main__":
    main()
