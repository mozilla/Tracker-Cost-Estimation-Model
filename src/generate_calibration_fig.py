"""
Generate calibration figure for Cycle 3.

Plots predicted mean vs actual mean by predicted-value bin (log-log),
bubble-sized by n. Saves to figures/fig4_calibration.pdf and .png.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "models" / "per_request" / "advanced_analysis_results.json"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

with open(DATA) as f:
    results = json.load(f)

cal = results["calibration"]

bins = [d["bin"] for d in cal]
pred_means = [d["pred_mean"] for d in cal]
actual_means = [d["actual_mean"] for d in cal]
ns = [d["n"] for d in cal]
within25 = [d["within_25pct"] * 100 for d in cal]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- Panel (a): Calibration scatter (log-log) ---
ax = axes[0]
sizes = [max(n / 30, 30) for n in ns]
colors = [w / 100 for w in within25]
sc = ax.scatter(pred_means, actual_means, s=sizes, c=colors, cmap="RdYlGn",
                vmin=0.3, vmax=0.9, alpha=0.85, zorder=5, edgecolors="gray", linewidths=0.5)

all_vals = pred_means + actual_means
lo = min(v for v in all_vals if v > 0) * 0.5
hi = max(all_vals) * 2.0
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect calibration", alpha=0.6)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Predicted mean (bytes)", fontsize=10)
ax.set_ylabel("Actual mean (bytes)", fontsize=10)
ax.set_title("(a) Calibration by predicted-value bin", fontsize=10)
ax.legend(fontsize=8)

# Annotate key bins
highlight = {"0-100", "500-1K", "50K-100K", "250K+"}
for i, label in enumerate(bins):
    if label in highlight:
        ax.annotate(
            label,
            (pred_means[i], actual_means[i]),
            fontsize=7.5,
            xytext=(6, 2),
            textcoords="offset points",
            color="#333333",
        )

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Frac. within 25%", fontsize=8)

# --- Panel (b): % within 25% by bin ---
ax = axes[1]
x = np.arange(len(bins))
bar_colors = [plt.cm.RdYlGn(w / 100) for w in within25]
bars = ax.bar(x, within25, color=bar_colors, edgecolor="gray", linewidth=0.5)
ax.axhline(50, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.7, label="50%")
ax.set_xticks(x)
ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("% predictions within 25% of actual", fontsize=10)
ax.set_title("(b) Per-bin accuracy", fontsize=10)
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
ax.set_yticks([0, 25, 50, 75, 100])

plt.tight_layout()
for ext in ("pdf", "png"):
    path = OUT / f"fig4_calibration.{ext}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")

plt.close()
print("Done.")
