# Results — Draft

Written under one constraint, per direct instruction: **evidence only, in the order E1 → E2 → E2.5 → E2.6 → E2.7, no interpretation, no mechanism, no discussion.** Every subsection ends on a plain statement of what was measured, not what it means. All "why" is deferred to Discussion. Numbers are copied from `results/*.csv`; none are recomputed here.

Table/figure numbering follows `manuscript_blueprint.md` §3–4 (Table 1 = ladder design, in Methods §2.2; Figures 2–6 as specified there).

---

## 3.1 Representation geometry changes with disentanglement strength

Condition number increased monotonically with $\lambda_{orth}$: 75.5±7.8 (`runA_grl`), 559.8±125.9 (`runB_orth1`), 5329.9±2672.2 (`runB`) — a two-order-of-magnitude change across the ladder (Table 2, Figure 2). This trend was significant by exact-permutation Kendall's $\tau$ ($\tau=0.84$, $p=2.8\times10^{-5}$, $n=13$) and sign-stable on the common-seed subset ($\tau=0.87$, $p=0.0012$, $n=9$), and confirmed by Jonckheere–Terpstra trend testing ($J=55.0$ against a null mean of 27.5, exact $p=1.4\times10^{-5}$).

The remaining four geometry metrics showed no significant trend: Fisher ratio (Hotelling–Lawley), $\tau=0.17$, $p=0.52$; Fisher ratio (decoupled scalar), $\tau=0.08$, $p=0.80$; Mardia's kurtosis ($b$), $\tau=-0.14$, $p=0.61$; Mardia's kurtosis ($z$), $\tau=-0.17$, $p=0.52$ (Table 2).

**Increasing $\lambda_{orth}$ is associated with a large, statistically significant change in condition number; the other four geometry metrics tested show no significant association with $\lambda_{orth}$.**

## 3.2 Representation geometry shows no measurable association with Mahalanobis AUROC

Mahalanobis AUROC (ISIC-test vs. PAD-UFES) was 0.401±0.028 (`runA_grl`), 0.408±0.022 (`runB_orth1`), and 0.393±0.025 (`runB`) — below the chance value of 0.5 at every rung (Table 3).

None of the five geometry metrics, including condition number, showed a significant association with Mahalanobis AUROC: condition number, $\tau=-0.13$, $p=0.59$; Fisher ratio (HL), $\tau=-0.28$, $p=0.20$; Fisher ratio (scalar), $\tau=-0.15$, $p=0.51$; Mardia's kurtosis ($b$), $\tau=-0.13$, $p=0.59$; Mardia's kurtosis ($z$), $\tau=-0.13$, $p=0.59$ (Figure 3). AUROC itself showed no significant trend across the ladder (Jonckheere–Terpstra, $J=25.0$ against a null mean of 27.5, exact $p=0.65$).

**An approximately two-orders-of-magnitude change in condition number (§3.1) is not accompanied by a significant change in Mahalanobis AUROC; no geometry metric tested shows a significant association with AUROC.**

## 3.3 Distance-based scores are systematically lower for the target domain

Median Mahalanobis squared distance was lower for PAD-UFES than for ISIC-test at every one of the 13 checkpoints. Pooled across seeds within each rung: `runA_grl`, ISIC-test median 13.50 vs. PAD-UFES median 8.57 ($n_{id}=25335$, $n_{ood}=11490$); `runB_orth1`, 14.10 vs. 8.99 ($n_{id}=25335$, $n_{ood}=11490$); `runB`, 13.20 vs. 8.08 ($n_{id}=15201$, $n_{ood}=6894$). **Median distance was lower for the target domain than the source domain at all 13 checkpoints.**

Feature-norm distributions differed modestly between domains. Pooled median $\|z\|$: `runA_grl`, ISIC-test 6.31 vs. PAD-UFES 6.21; `runB_orth1`, 6.38 vs. 6.28; `runB`, 6.28 vs. 5.97. At the level of pooled means, the direction was not consistent across rungs (`runA_grl` mean: 6.72 ISIC-test vs. 6.83 PAD-UFES). **The direction of this difference was not consistent at the level of means.**

Nearest-centroid classification assigned PAD-UFES samples to the majority ISIC class (Nevus) at 61.6% (`runA_grl`), 61.9% (`runB_orth1`), and 63.9% (`runB`), compared to 53.0%, 53.1%, and 53.4% respectively for ISIC-test samples classified against their own nearest centroid. **This was 8.7–10.5 percentage points higher than the ISIC-test baseline rate (ratio 1.16–1.20).**

## 3.4 The failure generalizes across three distance-based scorer families

Pooled across all 13 checkpoints, AUROC was 0.402±0.024 (Mahalanobis), 0.418±0.022 (cosine-to-centroid), 0.424±0.023 ($k$-NN, $k=1$), 0.414±0.024 ($k$-NN, $k=10$), and 0.411±0.023 ($k$-NN, $k=50$) (Table 4, Figure 4). All five values fall within a 0.40–0.42 range and below the chance value of 0.5. Despite relying on substantially different scoring formulations — parametric with covariance, parametric without covariance, and non-parametric — all scorer families produced highly similar AUROC values.

None of the five scorer variants showed a significant association with $\lambda_{orth}$: Mahalanobis, $\tau=-0.08$, $p=0.80$; cosine, $\tau=0.32$, $p=0.19$; $k$-NN ($k=1$), $\tau=-0.05$, $p=0.90$; $k$-NN ($k=10$), $\tau=0.02$, $p=1.00$; $k$-NN ($k=50$), $\tau=0.08$, $p=0.80$. Jonckheere–Terpstra trend testing gave the same result for all five (exact $p$ between 0.10 and 0.65).

**Three structurally distinct distance-based scorers — parametric (Mahalanobis), semi-parametric (cosine-to-centroid), and non-parametric ($k$-NN at three values of $k$) — report AUROC in the same 0.40–0.42 range, all below chance, none associated with $\lambda_{orth}$.**

## 3.5 Domain information remains decodable despite distance-based inaccessibility

On the identical embeddings scored in §3.4, three supervised probes recovered domain membership (ISIC-test vs. PAD-UFES) at: `runA_grl`, 0.719 (logistic regression), 0.717 (linear SVM), 0.807 (random forest); `runB_orth1`, 0.741, 0.740, 0.815; `runB`, 0.750, 0.751, 0.800 (Table 5, Figure 5). AUROC exceeded 0.70 for all three probe families at every rung.

None of the three probes showed a significant association with $\lambda_{orth}$: logistic regression, $\tau=0.26$, $p=0.30$; linear SVM, $\tau=0.26$, $p=0.30$; random forest, $\tau=-0.11$, $p=0.70$.

**Three independent classifiers, applied to the same embeddings scored by the distance-based scorers in §3.4, recover domain membership at 0.72–0.81 AUROC at every $\lambda_{orth}$ level tested; probe AUROC shows no significant association with $\lambda_{orth}$.**

## 3.6 Descriptive reference: a non-disentangled baseline

For a single ResNet-50 checkpoint (`baseline_soft`; one seed, architecture and representation dimensionality differing from the ladder, §2.2), pooled distance-scorer AUROC was 0.828 and probe AUROC was 0.998 (logistic regression), 0.997 (linear SVM), and 0.977 (random forest) (Table 5).

---

## Notes for the writing session (not manuscript text)

- Every "no significant association" statement above is an exact-permutation Kendall's $\tau$/Jonckheere–Terpstra result at $n=13$ (or $n=9$ common-seed), reported per §2.6's disclosure — absence of significance is not asserted as evidence of absence anywhere above; that interpretive step belongs in Discussion/Limitations.
- §3.3 deliberately stops at reporting the norm and NV-attractor numbers, with no "rules out" or "not large enough to explain" language — even a ruling-out claim is an inference. Discussion draws that conclusion, if it is drawn at all.
- §3.6 has no comparative language against the ladder anywhere — no "unlike," no "in contrast," no juxtaposition sentence. The reader can compare Table 5's rows themselves; the paper does not do it for them in Results.
- Figure/table numbers assume the full set from `manuscript_blueprint.md` §3–4 gets built; none of the actual figures (2–6) exist as final artifacts yet — this draft cites them by their planned number so the prose and the figure-build task stay in sync.
- **For Discussion**: this section reads as five results (§3.1–3.5) but is structurally two. Result 1 is §3.1 alone (geometry changes enormously). Result 2 is §3.2–3.5 together, one evidentiary chain (distance-based accessibility does not change, tested four independent ways: association with geometry, direction and magnitude of the raw distance gap, generalization across scorer families, and decodability by a non-distance-based readout). Discussion's structure should follow this two-result shape, not re-narrate five separate findings — §3.2–3.5 support one claim, they are not four claims.
