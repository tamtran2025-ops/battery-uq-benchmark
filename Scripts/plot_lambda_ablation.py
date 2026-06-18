"""Plot lambda_phys ablation results as Supp Figure S3.

x-axis: lambda_scale (log scale, 0.0 → 10.0)
y-axis: MAE with error bars (mean ± std across 25 networks per lambda)
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = r"D:\Project Python\PythonProject9\Paper 7\Paper_Knee\results\lambda_phys_ablation.csv"
OUT = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\JPS_Submission_Files\figures\Supp_Figure_S3.png"

if not os.path.exists(CSV):
    print(f"CSV not found yet: {CSV}")
    sys.exit(1)

df = pd.read_csv(CSV)
df_finite = df[~df["diverged"]].copy()

agg = df_finite.groupby("lambda_scale").agg(
    MAE_mean=("MAE", "mean"),
    MAE_std=("MAE", "std"),
    PICP_conf_mean=("PICP_conformal", "mean"),
    n_runs=("MAE", "count"),
).reset_index()

print(agg.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Use log scale; shift 0 → 1e-3 for plotting
plot_x = agg["lambda_scale"].copy()
plot_x[plot_x == 0] = 1e-3
ax1.errorbar(plot_x, agg["MAE_mean"], yerr=agg["MAE_std"],
             fmt='o-', color='#2ca02c', capsize=4, linewidth=2, markersize=8,
             label='MAE')
ax1.set_xscale('log')
ax1.set_xlabel(r'$\lambda_{\mathrm{phys}}$ scale', fontsize=11)
ax1.set_ylabel('MAE (cycles)', fontsize=11)
ax1.set_title(r'Lambda ablation: MAE at $n_{\mathrm{early}} = 150$', fontsize=11)
ax1.grid(alpha=0.3, linestyle=':')
ax1.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5,
            label='Default $\\lambda = 1.0$')
ax1.legend(fontsize=9)

ax2.errorbar(plot_x, agg["PICP_conf_mean"], fmt='s-', color='#d62728',
             capsize=4, linewidth=2, markersize=8)
ax2.set_xscale('log')
ax2.set_xlabel(r'$\lambda_{\mathrm{phys}}$ scale', fontsize=11)
ax2.set_ylabel('Conformal PICP', fontsize=11)
ax2.set_title(r'Conformal coverage vs $\lambda_{\mathrm{phys}}$', fontsize=11)
ax2.axhline(0.95, color='black', linestyle='--', linewidth=1, alpha=0.5,
            label='Nominal 0.95')
ax2.set_ylim(0.7, 1.0)
ax2.grid(alpha=0.3, linestyle=':')
ax2.legend(fontsize=9, loc='lower right')

plt.suptitle(r"Physics-loss ablation: PINN-Knee Deep Ensemble at $n_{\mathrm{early}} = 150$",
             fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig(OUT, dpi=600, bbox_inches='tight')
print(f"\nSaved: {OUT}")
