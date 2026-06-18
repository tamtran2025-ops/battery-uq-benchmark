"""Regenerate Figure 1: MAE bar chart with all 14 methods at n_early = 150."""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Manuscript Table 2 numbers @ n_early = 150 (14 methods after V5 revision)
DATA = [
    # (method, MAE, std, family)
    ("Deep Ensemble PINN-Knee", 111.6, 38.2, "physics"),
    ("Combined UQ PINN-Knee", 111.4, 38.3, "physics"),
    ("Bootstrap Ensemble PINN-Knee", 112.8, 39.9, "physics"),
    ("Jackknife+ PINN-Knee", 113.8, 40.3, "physics"),
    ("Hyper-Deep Ensemble", 112.9, 36.8, "physics"),  # NEW
    ("SNGP [46]", 113.6, 40.4, "non-physics-good"),
    ("NGBoost", 123.4, 36.5, "non-physics-good"),  # NEW
    ("Gaussian Process", 125.3, 34.4, "non-physics-good"),
    ("Heteroscedastic MLP (v2)", 137.1, 30.8, "variance-mlp"),
    ("CQR-MLP (v2)", 140.9, 29.9, "variance-mlp"),
    ("CQR-PINN-Knee", 153.7, 25.7, "variance-mlp"),
    ("Bayesian LSTM", 184.6, 33.5, "weak"),
    ("Ensemble NN", 191.7, 30.1, "weak"),
    ("Last-Layer Laplace [47]", 510.2, 583.5, "unstable"),
]

COLOR_MAP = {
    "physics": "#2ca02c",          # green
    "non-physics-good": "#1f77b4", # blue
    "variance-mlp": "#ff7f0e",     # orange
    "weak": "#9467bd",             # purple
    "unstable": "#7f7f7f",         # grey
}

# Sort by MAE (ascending)
DATA.sort(key=lambda x: x[1])

methods = [d[0] for d in DATA]
maes = [d[1] for d in DATA]
stds = [d[2] for d in DATA]
colors = [COLOR_MAP[d[3]] for d in DATA]

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(methods))
ax.barh(y_pos, maes, xerr=stds, color=colors, alpha=0.85,
        edgecolor='black', linewidth=0.6, capsize=3,
        error_kw={'ecolor': 'black', 'elinewidth': 0.8})
ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=10)
ax.set_xlabel(r'MAE at $n_{\mathrm{early}}=150$ (cycles)', fontsize=11)
ax.invert_yaxis()  # best at top
ax.set_xlim(0, 1300)

# Add MAE values at end of each bar
for i, (m, s) in enumerate(zip(maes, stds)):
    ax.text(m + s + 8, i, f'{m:.1f}', va='center', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor=COLOR_MAP["physics"], label="Physics-informed ensembles"),
    Patch(facecolor=COLOR_MAP["non-physics-good"], label="Distance-aware / GP baselines"),
    Patch(facecolor=COLOR_MAP["variance-mlp"], label="Variance-aware MLP variants"),
    Patch(facecolor=COLOR_MAP["weak"], label="Plain ensemble / sequential"),
    Patch(facecolor=COLOR_MAP["unstable"], label="Unstable on this dataset size"),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=8.5, frameon=True)

ax.grid(axis='x', alpha=0.3, linestyle=':')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

OUT_PNG = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\JPS_Submission_Files\figures\Figure_1.png"
plt.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.savefig(OUT_PNG.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {OUT_PNG}")
print(f"Methods: {len(methods)}")
