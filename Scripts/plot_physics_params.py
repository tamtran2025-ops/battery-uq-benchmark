"""Plot physics parameter analysis as Supp Figure S4.

Per-cell scatter of (a, b, c, d, s) vs true knee, colored by tertile.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = r"D:\Project Python\PythonProject9\Paper 7\Paper_Knee\results\physics_params_per_cell_agg.csv"
OUT = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\JPS_Submission_Files\figures\Supp_Figure_S4.png"

if not os.path.exists(CSV):
    print(f"CSV not found yet: {CSV}")
    sys.exit(1)

df = pd.read_csv(CSV)
df["tertile"] = pd.qcut(df["y_true"], q=3, labels=["low", "mid", "high"])
colors = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#d62728"}

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharey=True)
params = ["a", "b", "c", "d", "s"]
labels = ["a (initial cap.)", "b (linear-fade)", "c (super-linear)",
          "d (knee scale)", "s (transition sharp.)"]

for ax, p, lab in zip(axes, params, labels):
    for ter, col in colors.items():
        sub = df[df["tertile"] == ter]
        ax.scatter(sub[f"{p}_mean"], sub["y_true"], c=col, alpha=0.7,
                   s=22, label=ter, edgecolors='k', linewidth=0.3)
    corr = float(np.corrcoef(df["y_true"], df[f"{p}_mean"])[0, 1])
    ax.set_xlabel(f'{lab}\n(r = {corr:+.2f})', fontsize=10)
    ax.grid(alpha=0.3, linestyle=':')

axes[0].set_ylabel('True knee-cycle (cycles)', fontsize=11)
axes[-1].legend(fontsize=8, title='Knee tertile', loc='lower right')
plt.suptitle(r"PINN-Knee physics parameters $(a,b,c,d,s)$ vs true knee-cycle, $n_{\mathrm{early}} = 150$",
             fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig(OUT, dpi=600, bbox_inches='tight')
print(f"Saved: {OUT}")
