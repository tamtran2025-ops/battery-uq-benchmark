"""Regenerate Figure 2: PICP raw vs conformal at n_early = 150."""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# From Table 3 @ n_early = 150 (14 methods after V5 revision)
DATA = [
    ("Deep Ensemble PINN-Knee", 0.38, 0.94),
    ("Combined UQ PINN-Knee", 0.39, 0.94),
    ("Bootstrap Ens. PINN-Knee", 0.56, 0.94),
    ("Jackknife+ PINN-Knee", 0.46, 0.93),
    ("Hyper-Deep Ensemble", 0.21, 0.95),  # NEW
    ("CQR-PINN-Knee", 0.55, 0.92),
    ("Ensemble NN", 0.08, 0.95),
    ("Bayesian LSTM", 0.70, 0.92),
    ("Gaussian Process", 0.00, 0.92),
    ("NGBoost", 0.25, 0.97),  # NEW
    ("Heteroscedastic MLP (v2)", 0.89, 0.94),
    ("CQR-MLP (v2)", 0.28, 0.95),
    ("SNGP", 0.45, 0.92),
    ("Last-Layer Laplace", 0.70, 0.94),
]

methods = [d[0] for d in DATA]
picp_raw = [d[1] for d in DATA]
picp_conf = [d[2] for d in DATA]

fig, ax = plt.subplots(figsize=(10, 5.5))
y_pos = np.arange(len(methods))
width = 0.4

bars_raw = ax.barh(y_pos - width/2, picp_raw, width, color='#d62728', alpha=0.7,
                    label='Raw posterior', edgecolor='black', linewidth=0.5)
bars_conf = ax.barh(y_pos + width/2, picp_conf, width, color='#2ca02c', alpha=0.85,
                     label='Split-conformal', edgecolor='black', linewidth=0.5)

ax.axvline(0.95, color='black', linestyle='--', linewidth=1.2, label='Nominal 95% target')
ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=9.5)
ax.set_xlabel('Prediction Interval Coverage Probability (PICP)', fontsize=11)
ax.set_xlim(0, 1.05)
ax.invert_yaxis()
ax.legend(loc='lower right', fontsize=9)

# Add values at bar ends
for i, (r, c) in enumerate(zip(picp_raw, picp_conf)):
    ax.text(r + 0.01, i - width/2, f'{r:.2f}', va='center', fontsize=8)
    ax.text(c + 0.01, i + width/2, f'{c:.2f}', va='center', fontsize=8)

ax.grid(axis='x', alpha=0.3, linestyle=':')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

OUT_PNG = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\JPS_Submission_Files\figures\Figure_2.png"
plt.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.savefig(OUT_PNG.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {OUT_PNG}")
print(f"Methods: {len(methods)}")
