"""
E2.7 -- how much domain information the representation retains, measured
independently of any trained discriminator.

E2.6 and E2.8 found all eight scorers landing near 0.40-0.42 AUROC, which
points away from the scoring rule and toward the representation. The obvious
next reading -- the trained domain discriminator's own accuracy -- is not
used here: its output is what the training objective optimised, so a null
there would not separate "no domain information in z" from "discriminator
undertrained or miscalibrated".

Instead, three fresh classifiers with different inductive biases
(LogisticRegression, linear SVM, RandomForest) are fit to predict domain
(ISIC-test = 0, PAD-UFES = 1) from the same z_id/z_ood embeddings the
scorers used. No hyperparameter tuning, since tuning to maximise domain
AUROC would bias the question. 5-fold stratified cross-validation,
out-of-fold AUROC, because d = 16 makes fit-and-evaluate-on-the-same-data
leakage a real risk.

Reads results/e2_distances/{rung}_s{seed}_z.npz, z_id and z_ood only; needs
no GPU and no checkpoint reload. Writes e2_7_domain_probe.csv,
table_e2_7_domain_vs_distance.csv, e2_7_kendall_tau.csv and
figures/figure_e2_7_domain_probe.pdf.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analyze_e1 import PRIMARY_RUNGS, RUNG_INDEX, RUNG_LAMBDA, kendall_trend

RUNG_SEEDS = {"runA_grl": {42, 52, 62, 72, 82}, "runB_orth1": {42, 52, 62, 72, 82}, "runB": {42, 52, 62}}
N_FOLDS = 5
CV_SEED = 0


def make_probes() -> dict:
    """Library defaults only -- no tuning, since tuning to maximize domain AUROC would bias the question."""
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "linear_svm": make_pipeline(StandardScaler(), LinearSVC()),
        "random_forest": RandomForestClassifier(random_state=CV_SEED),
    }


def domain_probe_auroc(z_id: np.ndarray, z_ood: np.ndarray, probe) -> float:
    """
    5-fold stratified CV, out-of-fold scores, AUROC on the pooled out-of-fold
    predictions -- avoids the same-data-fit-and-eval leakage that would
    inflate domain AUROC at d=16 with a few thousand points per class.
    LinearSVC has no predict_proba; decision_function is monotonic in the
    same direction and AUROC is rank-based, so it's used directly for that
    probe instead of forcing a probability calibration step nothing else needs.
    """
    z = np.concatenate([z_id, z_ood], axis=0)
    domain = np.concatenate([np.zeros(len(z_id), dtype=np.int64), np.ones(len(z_ood), dtype=np.int64)])
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    if hasattr(probe, "predict_proba") or (hasattr(probe, "steps") and hasattr(probe.steps[-1][1], "predict_proba")):
        scores = cross_val_predict(probe, z, domain, cv=cv, method="predict_proba")[:, 1]
    else:
        scores = cross_val_predict(probe, z, domain, cv=cv, method="decision_function")

    return float(roc_auc_score(domain, scores))


def load_z(npz_dir: Path, rung: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_dir / f"{rung}_s{seed}_z.npz")
    return data["z_id"], data["z_ood"]


def check_inputs_present(npz_dir: Path) -> None:
    missing = [
        (rung, seed) for rung, seeds in RUNG_SEEDS.items() for seed in seeds
        if not (npz_dir / f"{rung}_s{seed}_z.npz").is_file()
    ]
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)}/13 raw-embedding files missing from {npz_dir} "
            f"(expected '{{rung}}_s{{seed}}_z.npz', written by extract_auroc_e2.py's "
            "save_raw_embeddings -- see experiment_contract.md E2.7's Input section):\n"
            + "\n".join(f"  {rung}_s{seed}_z.npz" for rung, seed in sorted(missing))
            + "\nSync these from the server before running this script -- E2.7 needs no rerun and no "
            "GPU, only these already-computed files."
        )


def discover_baseline_seeds(npz_dir: Path) -> list[int]:
    """
    baseline_soft is a categorical reference (open_questions.md Q4 / experiment_contract.md E1b/E2b),
    never part of the primary ladder's ordinal trend -- auto-discovered from whichever
    baseline_soft_s{seed}_z.npz files happen to be present (as few as one checkpoint is enough to
    know the direction, per the user's own scoping for this comparison), rather than hardcoding an
    expected seed set the way RUNG_SEEDS does for the primary ladder.
    """
    seeds = []
    for path in sorted(npz_dir.glob("baseline_soft_s*_z.npz")):
        seed_str = path.stem[len("baseline_soft_s"):-len("_z")]
        seeds.append(int(seed_str))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--npz_dir", default=str(results_dir / "e2_distances"))
    parser.add_argument("--scorer_csv", default=str(results_dir / "e2_6_scorer_comparison.csv"))
    parser.add_argument("--output_dir", default=str(results_dir))
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    check_inputs_present(npz_dir)

    probes = make_probes()
    rows = []
    for rung, seeds in RUNG_SEEDS.items():
        for seed in sorted(seeds):
            z_id, z_ood = load_z(npz_dir, rung, seed)
            for probe_name, probe in probes.items():
                auroc = domain_probe_auroc(z_id, z_ood, probe)
                rows.append({
                    "rung": rung, "seed": seed, "probe": probe_name,
                    "domain_auroc": auroc, "n_id": len(z_id), "n_ood": len(z_ood),
                })
                print(f"  {rung:12s} seed={seed:3d} {probe_name:20s} domain_AUROC={auroc:.4f}")

    ladder_df = pd.DataFrame(rows)
    ladder_df["rung_index"] = ladder_df["rung"].map(RUNG_INDEX)

    # baseline_soft: categorical reference (open_questions.md Q4), computed the identical way but
    # NEVER folded into the ladder's Kendall's tau -- same separation analyze_e1.load_data uses.
    baseline_rows = []
    baseline_seeds = discover_baseline_seeds(npz_dir)
    for seed in baseline_seeds:
        z_id, z_ood = load_z(npz_dir, "baseline_soft", seed)
        for probe_name, probe in probes.items():
            auroc = domain_probe_auroc(z_id, z_ood, probe)
            baseline_rows.append({
                "rung": "baseline_soft", "seed": seed, "probe": probe_name,
                "domain_auroc": auroc, "n_id": len(z_id), "n_ood": len(z_ood),
            })
            print(f"  baseline_soft seed={seed:3d} {probe_name:20s} domain_AUROC={auroc:.4f}")
    baseline_df = pd.DataFrame(baseline_rows)
    if len(baseline_rows):
        baseline_df["rung_index"] = np.nan

    df = pd.concat([ladder_df, baseline_df], ignore_index=True) if len(baseline_rows) else ladder_df
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "e2_7_domain_probe.csv", index=False)

    print("\n=== Domain AUROC (ISIC-test vs. PAD-UFES), pooled across the primary ladder, per probe ===")
    for probe_name in probes:
        sub = ladder_df[ladder_df["probe"] == probe_name]
        print(f"  {probe_name:20s} domain_AUROC={sub['domain_auroc'].mean():.4f} +/- {sub['domain_auroc'].std(ddof=1):.4f}")

    if len(baseline_rows):
        print("\n=== Domain AUROC, baseline_soft (categorical reference, descriptive only -- not on the ladder) ===")
        for probe_name in probes:
            sub = baseline_df[baseline_df["probe"] == probe_name]
            print(f"  {probe_name:20s} domain_AUROC={sub['domain_auroc'].mean():.4f}"
                  + (f" +/- {sub['domain_auroc'].std(ddof=1):.4f}" if len(sub) > 1 else "  (n=1, no SD)"))
    else:
        print("\n[analyze_e2_7] No baseline_soft_s*_z.npz found -- baseline_soft comparison skipped "
              "(sync one from the server to include it; not required for the ladder-only result above).")

    # ---- Joint comparison against E2.6's distance-scorer AUROC ----
    scorer_path = Path(args.scorer_csv)
    if scorer_path.is_file():
        scorer_df_all = pd.read_csv(scorer_path)
        scorer_df = scorer_df_all[scorer_df_all["rung"].isin(PRIMARY_RUNGS)]
        joint_rows = []
        for rung in PRIMARY_RUNGS:
            distance_mean = scorer_df[scorer_df["rung"] == rung]["auroc"].mean()
            for probe_name in probes:
                probe_mean = ladder_df[(ladder_df["rung"] == rung) & (ladder_df["probe"] == probe_name)]["domain_auroc"].mean()
                joint_rows.append({
                    "rung": rung, "lambda_orth": RUNG_LAMBDA[rung], "role": "primary ladder",
                    "nonprobing_scorer_auroc_mean": distance_mean,
                    "probe": probe_name, "domain_probe_auroc_mean": probe_mean,
                })

        if len(baseline_rows):
            baseline_scorer_df = scorer_df_all[scorer_df_all["rung"] == "baseline_soft"]
            baseline_distance_mean = baseline_scorer_df["auroc"].mean()
            for probe_name in probes:
                probe_mean = baseline_df[baseline_df["probe"] == probe_name]["domain_auroc"].mean()
                joint_rows.append({
                    "rung": "baseline_soft", "lambda_orth": np.nan, "role": "categorical reference",
                    "nonprobing_scorer_auroc_mean": baseline_distance_mean,
                    "probe": probe_name, "domain_probe_auroc_mean": probe_mean,
                })

        joint = pd.DataFrame(joint_rows)
        joint.to_csv(out_dir / "table_e2_7_domain_vs_distance.csv", index=False)

        print("\n=== The actual E2.7 question: non-probing-scorer AUROC vs. domain-probe AUROC, per rung ===")
        n_scorers_ladder = scorer_df["scorer"].nunique()  # primary ladder: 8 as of E2.8 (baseline_soft excluded from E2.8, see its docstring)
        for rung in PRIMARY_RUNGS + (["baseline_soft"] if len(baseline_rows) else []):
            sub = joint[joint["rung"] == rung]
            label = f"lambda_orth={RUNG_LAMBDA[rung]:g}" if rung in RUNG_LAMBDA else "categorical reference"
            n_scorers = n_scorers_ladder if rung != "baseline_soft" else baseline_scorer_df["scorer"].nunique()
            pooled_label = "E2.6+E2.8 pooled" if rung != "baseline_soft" else "E2.6 only, E2.8 excludes baseline_soft"
            print(f"\n  {rung} ({label}): "
                  f"non-probing scorers (all {n_scorers}, {pooled_label}) AUROC={sub['nonprobing_scorer_auroc_mean'].iloc[0]:.4f}")
            for _, r in sub.iterrows():
                print(f"    domain probe [{r['probe']:20s}] AUROC={r['domain_probe_auroc_mean']:.4f}")

        overall_probe_mean = ladder_df["domain_auroc"].mean()
        print(
            f"\nReading this (primary ladder only): non-probing scorers sit at ~0.40 (E2.6+E2.8). Domain probes sit "
            f"at {overall_probe_mean:.3f} on average. "
            + ("Domain information is close to absent from z -- consistent with 'the representation "
               "genuinely lost domain-discriminative structure.'" if overall_probe_mean < 0.6 else
               "Domain information clearly SURVIVES in z (well above chance) despite every non-probing "
               "scorer failing on it -- the deeper reading: domain information is present but not in a form "
               "any non-probing scorer tested here (Mahalanobis, cosine, k-NN, Energy, ViM, KDE density) can exploit.")
        )
        if len(baseline_rows):
            baseline_probe_mean = baseline_df["domain_auroc"].mean()
            print(
                f"\nbaseline_soft (categorical reference, descriptive only): non-probing scorers AUROC="
                f"{baseline_distance_mean:.4f}, domain probes AUROC={baseline_probe_mean:.4f} on average -- "
                + ("both non-probing scorers AND domain probes succeed here, unlike the CSG ladder: consistent "
                   "with the failure being specific to the disentangled representation, not a general property "
                   "of ISIC-vs-PAD-UFES domain shift." if baseline_distance_mean > 0.6 and baseline_probe_mean > 0.6
                   else "reported without further interpretation here -- see the printed numbers above.")
            )
    else:
        print(f"\n[analyze_e2_7] NOTE: {scorer_path} not found -- skipping the joint distance-vs-probe "
              "comparison table; domain-probe-only results above are still valid and saved.")

    # ---- Kendall's tau: domain_auroc vs. rung order, per probe (primary ladder only -- baseline_soft
    # is a categorical reference and is never folded into this trend statistic, same rule as E1a/E2a) ----
    print("\n=== Kendall's tau: domain AUROC vs. rung order, per probe (EXACT permutation p-values) ===")
    tau_rows = []
    for probe_name in probes:
        sub = ladder_df[ladder_df["probe"] == probe_name]
        result = kendall_trend(sub, "domain_auroc")
        result["probe"] = probe_name
        tau_rows.append(result)
        print(
            f"  {probe_name:20s} tau={result['tau_full']:+.3f} (exact p={result['p_full_exact']:.6f}, n={result['n_full']})  |  "
            f"common-seed tau={result['tau_common_seed']:+.3f} (exact p={result['p_common_seed_exact']:.6f})  |  "
            f"sign_stable={result['sign_stable']}"
        )
    pd.DataFrame(tau_rows).to_csv(out_dir / "e2_7_kendall_tau.csv", index=False)

    print(f"\n[analyze_e2_7] Written: {out_dir}/e2_7_domain_probe.csv, "
          f"table_e2_7_domain_vs_distance.csv, e2_7_kendall_tau.csv")


if __name__ == "__main__":
    main()
