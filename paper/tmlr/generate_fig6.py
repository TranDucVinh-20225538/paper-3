import matplotlib.pyplot as plt
import numpy as np


labels = [
    "runA_grl\n($\\lambda_{orth}=0$)",
    "runB_orth1\n($\\lambda_{orth}=1$)",
    "runB\n($\\lambda_{orth}=5$)",
]
x = np.arange(len(labels))
directed = np.array([0.412, 0.414, 0.413])
separability = 1.0 - directed
probe = np.array([
    np.mean([0.719, 0.717, 0.807]),
    np.mean([0.741, 0.740, 0.815]),
    np.mean([0.750, 0.751, 0.800]),
])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0.9,
})

fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)

for i in x:
    ax.plot([i, i], [directed[i], separability[i]], color="0.55", lw=1.4, zorder=1)
    ax.plot([i, i], [separability[i], probe[i]], color="0.25", lw=1.4, ls="--", zorder=1)

ax.scatter(x, directed, s=82, marker="o", facecolor="white", edgecolor="black",
           linewidth=1.5, label=r"Directed distance $\mathrm{AUROC}_{dir}$", zorder=3)
ax.scatter(x, separability, s=90, marker="s", facecolor="0.70", edgecolor="black",
           linewidth=1.2, label=r"Orientation-free $\mathrm{AUROC}_{sep}$", zorder=3)
ax.scatter(x, probe, s=92, marker="^", facecolor="0.15", edgecolor="black",
           linewidth=1.0, label="Supervised probes (mean of 3)", zorder=3)

ax.axhline(0.5, color="0.50", lw=1.1, ls="--")
ax.text(2.03, 0.505, "chance", color="0.40", va="bottom", ha="left")
ax.set_xticks(x, labels)
ax.set_xlim(-0.35, 2.60)
ax.set_ylim(0.35, 0.82)
ax.set_ylabel("AUROC (ISIC-test vs. PAD-UFES)")
ax.legend(loc="upper left", frameon=True, edgecolor="0.70", fontsize=9.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="0.90", lw=0.7)

fig.savefig("figs/fig6_decodable_vs_accessible.png", dpi=300, bbox_inches="tight")
fig.savefig("figs/fig6_directional_misalignment.pdf", bbox_inches="tight")
