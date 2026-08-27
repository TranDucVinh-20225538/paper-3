# Figure Captions — Draft

Written against `manuscript_blueprint.md` §3 (figure order) and the locked `results_draft.md`/`discussion_draft.md` numbers — nothing here introduces a new claim or number not already in the locked body text. Each caption: bold lead sentence (the takeaway, readable without the main text), then panel/axis description, then the specific statistic backing the takeaway.

**Build status, so this isn't confused with "figures done"**: Figures 4 (`figure_e2_6_scorer_comparison.png`) exists as final. Figures 2, 3, 6 exist as source images but need trimming/recomposing per the blueprint's main-vs-supplementary split (full 5-metric panels → supplementary, condensed 1–2-panel versions → main text). Figures 1 (conceptual schematic) and 5 (the headline dumbbell plot) do not exist as image files yet — captions below are written to the locked spec so image-building can target them directly, not from a finished image.

---

## Main text

**Figure 1. Study design.** An implicit assumption connects representation geometry to distance-based reliability estimation: a training intervention that reshapes geometry should also reshape estimation quality. This is tested using a disentanglement dose-response ladder — three checkpoint families sharing one architecture and one 16-dimensional representation, differing only in orthogonality strength ($\lambda_{orth}=0,1,5$). For each of 13 checkpoints, geometry metrics, three distance-based scorers, and three domain-information probes are computed on the identical embeddings (§2.0). *[Image: schematic, not yet built.]*

**Figure 2. Representation geometry changes with disentanglement strength.** Condition number (left) and the decoupled Fisher-ratio scalar $\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$ (right), one gray point per checkpoint ($n=13$), mean $\pm$ SD in blue, across $\lambda_{orth}=0,1,5$. Condition number increases by approximately two orders of magnitude with $\lambda_{orth}$ (Kendall's $\tau=0.84$, exact $p=2.8\times10^{-5}$; Table 2); the Fisher-ratio scalar shows no significant trend. The remaining three geometry metrics (Fisher ratio, Hotelling–Lawley form; Mardia's kurtosis $b$ and $z$) are in Supplementary Figure S1. *[Image: trim `figure1_e1_ladder_trend.png` from 5 panels to these 2.]*

**Figure 3. Representation geometry does not predict Mahalanobis AUROC.** Condition number vs. Mahalanobis-distance AUROC (ISIC-test vs. PAD-UFES), one point per checkpoint, colored by rung. No association is apparent (Kendall's $\tau=-0.13$, $p=0.59$; Table 3); the remaining four geometry metrics show the same pattern (Supplementary Figure S2). *[Image: trim `figure2_e2_geometry_vs_auroc.png` from 5 panels to this 1.]*

**Figure 4. Three distance-based scorers fail alike.** AUROC (ISIC-test vs. PAD-UFES) for Mahalanobis distance, cosine-to-centroid similarity, and pooled $k$-nearest-neighbor distance ($k=1,10,50$), computed on the identical embeddings, grouped by rung (mean $\pm$ SD across seeds). Dashed line: chance (AUROC = 0.5). All five scorer variants fall within a 0.40–0.42 range; none shows a significant association with $\lambda_{orth}$ (Table 4). *[Image: `figure_e2_6_scorer_comparison.png`, final as-is.]*

**Figure 5. Domain information is decodable but not distance-accessible.** For each rung, mean AUROC across the five distance-based scorers (Figure 4) is connected by a line to mean AUROC across three independent domain probes — logistic regression, linear SVM, random forest (§2.5) — computed on the identical embeddings. Distance-based scorers cluster at or below chance ($\approx 0.40$); probes recover domain membership at 0.72–0.81 AUROC at every $\lambda_{orth}$ level (Table 5). `baseline_soft` (§3.6) is not plotted on this axis — see Supplementary Figure S3 and its caption for why. *[Image: dumbbell plot, not yet built — this is the paper's single most important figure; build and review before anything else in this list.]*

**Figure 6. Neither feature-norm difference nor majority-class attraction fully accounts for the raw ID/OOD distance gap.** (A) Pooled feature-norm distributions, ISIC-test vs. PAD-UFES, per rung. (B) Nearest-centroid assignment of ISIC-test and PAD-UFES samples to the majority ISIC class (Nevus), per rung. The norm difference is modest and not consistent in direction at the level of means (§3.3); PAD-UFES's excess attraction to the majority class (8.7–10.5 percentage points above the ISIC-test baseline) is real but does not, on its own, account for the full distance separation in Figure 4. *[Image: recompose from `figure_norm_distributions.png`/`figure_id_confusion_matrix.png` into a 2-panel figure.]*

## Supplementary

**Figure S1.** All five geometry metrics across the ladder (extends Figure 2: condition number, Fisher ratio HL, Fisher ratio scalar, Mardia's kurtosis $b$, Mardia's kurtosis $z$). Source: `figure1_e1_ladder_trend.png`, unmodified.

**Figure S2.** All five geometry metrics vs. Mahalanobis AUROC (extends Figure 3). Source: `figure2_e2_geometry_vs_auroc.png`, unmodified.

**Figure S3. `baseline_soft` descriptive reference.** Distance-scorer AUROC (pooled across the five scorers, §2.4) and probe AUROC (§2.5) for a single, non-disentangled ResNet-50 checkpoint. Reported for reference only — this checkpoint differs from the disentanglement ladder in architecture, representation dimensionality, and training recipe simultaneously (§2.2), and is not plotted against the ladder on a shared axis anywhere in the main text (cf. Figure 5).

**Figure S4.** ID/OOD Mahalanobis squared-distance distributions (histogram, KDE, ECDF), all three rungs, pooled across seeds. Source: `figure_e2_5_distance_distributions.png`, unmodified. Supports §3.3's distance-gap numbers with the full distributional shape — visibly overlapping, not cleanly separated.

**Figure S5.** Full nearest-centroid confusion matrix (8×8, all ISIC classes), ISIC-test, per rung. Source: `figure_id_confusion_matrix.png`, unmodified. Supports Figure 6B's majority-class-only summary with the complete per-class breakdown.

**Figure S6.** Full predicted-class distribution for PAD-UFES samples (all 8 classes, not majority-class-only). Source: `figure_predicted_class_ood.png`, unmodified.

---

## Open item

Figure 5 (the dumbbell plot) does not exist yet and is the highest-priority image to build — it is the figure the paper's central claim depends on being legible in five seconds, per the original design discussion. Figures 2, 3, and 6 need recomposing from existing source images (crop/select panels, not re-run any analysis — the underlying numbers don't change). Say which one to build first.
