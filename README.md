# Decodable but Directionally Misaligned

Code and derived results for *Decodable but Directionally Misaligned: Auditing
Distance-Based OOD Ranking on Domain-Adversarial Skin-Lesion Representations*.

Duc-Vinh Tran, Quyet-Thang Huynh
School of Information and Communication Technology, Hanoi University of Science
and Technology.

## What the study does

Distance-based OOD scoring assumes two separate things: that a representation
separates source from shifted inputs, and that the induced ordering points the
right way, so that a larger distance means more out-of-distribution. This audit
holds architecture and datasets fixed and varies orthogonality-regularization
strength (lambda_orth = 0, 1, 5) across 13 checkpoints, then asks what happens
to each assumption.

Covariance conditioning changes by about two orders of magnitude across that
ladder while the other four geometry summaries show no significant trend. Five
distance-based scorers from three structurally distinct families stay near 0.40
directed AUROC at every level; their orientation-free separability is only
0.58-0.60, meaning the scores retain weak domain signal but rank PAD-UFES
samples as *more* source-typical than ISIC-test samples. Supervised probes
recover domain membership from the identical embeddings at 0.72-0.81 AUROC.

## Reproducing the analysis

Everything in the paper's tables and figures is regenerated from the CSVs in
`results/`. No GPU, no checkpoints, and no dataset download.

```bash
pip install -r requirements.txt

python3 analysis/analyze_e1.py                    # Table 1, Figure 2, geometry trends
python3 analysis/analyze_e2.py                    # geometry vs. Mahalanobis AUROC
python3 analysis/analyze_e2_distances.py          # ID/OOD distance distributions
python3 analysis/analyze_e2_6.py                  # all scorers vs. lambda_orth
python3 analysis/analyze_e2_7_domain_probe.py     # domain probes
python3 analysis/analyze_power_and_ci.py          # detectability, power, bootstrap CIs
```

`analyze_e2_6.py` enumerates 72,072 label arrangements per scorer and takes a
few minutes; the rest finish in seconds, except `analyze_power_and_ci.py` at
about one minute for the default 2x10^4 bootstrap resamples and simulations.

Statistical choices, and two reporting problems this analysis turned up, are
written up in [`docs/power_analysis.md`](docs/power_analysis.md). The analysis
plan fixed before the experiments ran is
[`docs/experiment_contract.md`](docs/experiment_contract.md), committed in this
repository roughly five hours before the first result file.

## Re-extracting from checkpoints

Only needed to rebuild `results/` from scratch. Requires a local checkout of
CSG-SKin (Paper 1), its trained checkpoints, and a GPU. Uncomment the torch
lines in `requirements.txt` first.

```bash
bash run_e2_all.sh          # all 13 checkpoints, explicit paths
```

Checkpoints are always named by explicit file path, never resolved by scanning
a directory: CSG-SKin's own `find_checkpoint` picks the newest file by mtime
and is confirmed to select a non-best-validation checkpoint for several
`runB`/`runB_orth1` seeds.

## Layout

```
analysis/    reads results/, writes tables and figures. No side effects on checkpoints.
scripts/     reads checkpoints, writes results/. Needs CSG-SKin and a GPU.
results/     per-checkpoint metrics and per-test statistics, as CSV.
figures/     generated figures.
docs/        experiment contract, metric audit, power analysis, open questions.
paper/tmlr/  manuscript source (TMLR format).
```

## Data availability

Both datasets are public:

- ISIC 2018: <https://challenge.isic-archive.com/data/>
- PAD-UFES-20: <https://data.mendeley.com/datasets/zr7vgbcyr2>

`results/*.csv` holds every number the paper reports and is included here.

The intermediate arrays are not: `results/e1_embeddings/` and
`results/e2_distances/` together come to 349 MB, and two individual files
exceed GitHub's 100 MB limit. They are byte-reproducible from the checkpoints
via `scripts/extract_embeddings_e1.py` and `scripts/extract_auroc_e2.py`.

**Open item:** for the camera-ready these should be deposited in an archive
that issues a DOI (Zenodo accepts 50 GB per record), and the DOI added here.

## Not included

Trained checkpoints live with Paper 1 (CSG-SKin) and Paper 2 (DST-Skin), which
this project reads from and never modifies. Both must be public for the
extraction path above to be reproducible by a third party.

## License

Not yet chosen. This needs to be settled before the repository is made public;
without a license file, default copyright leaves readers no right to reuse the
code.

## Citation

To be added on acceptance.
