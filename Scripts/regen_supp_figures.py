"""Regenerate Supplementary Figures S1, S2, S5 from real prediction data.

Bug-report fixes:
  - S1 (was: cost/safety chart) -> 14-method reliability diagrams, per-method panels
       at n_early=100, after split-conformal post-processing.
  - S2 (was: ensemble-size ablation) -> per-budget reliability overlay
       (n_early ∈ {50, 100, 150}), 14 methods, after split-conformal.
  - S5 (in-image title was 'Figure S7') -> coverage vs n_cal with correct title
       'Figure S5' (data from REVISION_Package_RESS/results/R4_coverage_vs_ncal.csv).
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_OLD = r"D:\Project Python\PythonProject9\00_Battery_Paper1_UQ-Benchmark_(RESS)\Paper4_UQ_Comparison\Predictions"
ROOT_NEW = r"D:\Project Python\PythonProject9\00_Battery_Paper2_PINN-Knee_(JES-submitted)\Paper_Knee\results"
OUT = r"D:\Project Python\PythonProject9\00_Battery_Paper1_UQ-Benchmark_(RESS)\REVISION_Package_RESS\RESS_figures"
os.makedirs(OUT, exist_ok=True)

METHODS = [
    ("Deep Ensemble PINN-Knee", ROOT_OLD, "deep_ensemble_preds", "#2ca02c"),
    ("Combined UQ PINN-Knee",   ROOT_NEW, "combined_uq_preds",   "#228b22"),
    ("Bootstrap Ens. PINN-Knee", ROOT_NEW, "bootstrap_preds",    "#006400"),
    ("Jackknife+ PINN-Knee",    ROOT_NEW, "jackknife_plus_preds", "#8fbc8f"),
    ("Hyper-Deep Ensemble",     ROOT_NEW, "hyper_deep_ensemble_preds", "#3cb371"),
    ("CQR-PINN-Knee",           ROOT_OLD, "cqr_pinn_preds",      "#ff7f0e"),
    ("Ensemble NN",             ROOT_OLD, "ensemble_nn_preds",   "#9467bd"),
    ("Bayesian LSTM",           ROOT_OLD, "bayesian_lstm_preds", "#d62728"),
    ("Gaussian Process",        ROOT_OLD, "gp_preds",            "#1f77b4"),
    ("NGBoost",                 ROOT_NEW, "ngboost_preds",       "#5f9ea0"),
    ("Hetero MLP (v2)",         ROOT_NEW, "hetero_preds_v2",     "#ff8c00"),
    ("CQR-MLP (v2)",            ROOT_NEW, "cqr_preds_v2",        "#ffa500"),
    ("SNGP",                    ROOT_NEW, "sngp_preds",          "#4682b4"),
    ("Laplace",                 ROOT_NEW, "laplace_preds",       "#7f7f7f"),
]


def pool_residuals(root, dname, n_early):
    """Pool test residuals + calibration residuals across the 5 folds.
    Returns (resid_test, cal_pool) or None if no data."""
    yt, pt, cal = [], [], []
    for fold in range(5):
        path = os.path.join(root, dname, f"preds_ne{n_early}_f{fold}.npz")
        if not os.path.exists(path):
            continue
        d = dict(np.load(path, allow_pickle=True))
        y_true = d["y_true"]
        preds = d["preds_all"]
        point = np.clip(preds.mean(axis=0) if preds.ndim == 2 else preds, 0, 1e6)
        yt.append(y_true); pt.append(point)
        if "y_cal" in d and "cal_preds_all" in d:
            cp = np.clip(d["cal_preds_all"].mean(axis=0), 0, 1e6)
            cal.append(np.abs(d["y_cal"] - cp))
        elif "loo_residuals" in d:
            cal.append(np.abs(d["loo_residuals"]))
    if not yt:
        return None
    resid_test = np.abs(np.concatenate(yt) - np.concatenate(pt))
    cal_pool = np.concatenate(cal) if cal else resid_test
    return resid_test, cal_pool


def reliability_curve(resid_test, cal_pool, alphas):
    """Conformal reliability: for each nominal 1-alpha, q_hat from cal,
    empirical coverage on test."""
    n = len(cal_pool); cs = np.sort(cal_pool)
    emp = []
    for a in alphas:
        k = int(np.ceil((n + 1) * (1 - a)))
        q = float(cs[min(k - 1, n - 1)])
        emp.append(float(np.mean(resid_test <= q)))
    return np.array(emp)


# ============================================================
# SUPP S1 — 14-panel per-method reliability at n_early=100
# ============================================================
print("Generating Supp Figure S1 (per-method reliability, n_early=100)...")
alphas = np.linspace(0.05, 0.95, 19); nominals = 1 - alphas

fig, axes = plt.subplots(4, 4, figsize=(13, 13), sharex=True, sharey=True)
axes = axes.flatten()
n_drawn = 0
for ax, (label, root, dname, color) in zip(axes, METHODS):
    r = pool_residuals(root, dname, 100)
    if r is None:
        ax.set_visible(False); continue
    resid_test, cal_pool = r
    emp = reliability_curve(resid_test, cal_pool, alphas)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6)
    ax.plot(nominals, emp, color=color, lw=2)
    ax.fill_between(nominals, nominals - 0.05, nominals + 0.05, color='gray', alpha=0.12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(label, fontsize=9)
    ax.grid(alpha=0.25, linestyle=':')
    n_drawn += 1
for ax in axes[n_drawn:]:
    ax.set_visible(False)
fig.supxlabel('Nominal coverage  $1-\\alpha$', fontsize=11)
fig.supylabel('Empirical coverage (split-conformal)', fontsize=11)
fig.suptitle('Figure S1. Per-method reliability diagrams after split-conformal post-processing ($n_{\\mathrm{early}}=100$)',
             fontsize=12, y=0.995)
plt.tight_layout(rect=(0.02, 0.02, 1, 0.985))
plt.savefig(os.path.join(OUT, 'Supp_Figure_S1.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved Supp_Figure_S1.png  ({n_drawn} methods plotted)')


# ============================================================
# SUPP S2 — 3-panel per-budget reliability overlay
# ============================================================
print("\nGenerating Supp Figure S2 (per-budget reliability overlay, 3 panels)...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, ne in zip(axes, (50, 100, 150)):
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Perfect calibration')
    n_drawn = 0
    for label, root, dname, color in METHODS:
        r = pool_residuals(root, dname, ne)
        if r is None: continue
        emp = reliability_curve(r[0], r[1], alphas)
        ax.plot(nominals, emp, color=color, lw=1.4, alpha=0.85, label=label)
        n_drawn += 1
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel('Nominal coverage  $1-\\alpha$', fontsize=10)
    ax.set_title(f'$n_{{\\mathrm{{early}}}} = {ne}$ ({n_drawn} methods)', fontsize=11)
    ax.grid(alpha=0.3, linestyle=':')
axes[0].set_ylabel('Empirical coverage (split-conformal)', fontsize=10)
axes[-1].legend(fontsize=7, loc='lower right', ncol=2, framealpha=0.9)
fig.suptitle('Figure S2. Per-budget reliability diagrams after split-conformal post-processing',
             fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'Supp_Figure_S2.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved Supp_Figure_S2.png')


# ============================================================
# SUPP S5 — coverage vs n_cal, title corrected from S7 -> S5
# ============================================================
print("\nGenerating Supp Figure S5 (coverage vs n_cal)...")
csv_path = r"D:\Project Python\PythonProject9\00_Battery_Paper1_UQ-Benchmark_(RESS)\REVISION_Package_RESS\results\R4_coverage_vs_ncal.csv"
df = pd.read_csv(csv_path)
d = df[df.n_early == 150].sort_values('n_cal')

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.errorbar(d.n_cal, d.PICP_mean, yerr=d.PICP_std, fmt='o-', color='#1f77b4',
            capsize=4, lw=2, ms=7, label='Empirical PICP (mean ± std)', zorder=3)
ax.plot(d.n_cal, d.theoretical_lower_bound, 's--', color='#d62728', lw=1.6, ms=5,
        label=r'Theoretical lower bound  $1-\alpha-\frac{1}{n_{cal}+1}$')
ax.axhline(0.95, ls=':', color='gray', lw=1.4, label='Target 0.95')
ax.axvline(22, ls='-', color='green', alpha=0.35, lw=8, zorder=0)
ax.annotate('operating point\n n_cal=22, PICP=0.957', xy=(22, 0.957), xytext=(28, 0.86),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green')
ax.set_xlabel('Calibration set size  $n_{cal}$'); ax.set_ylabel('PICP (95% target)')
ax.set_title('Figure S5. Split-conformal coverage vs calibration-set size\n'
             '(Deep Ensemble PINN-Knee, $n_{\\mathrm{early}}=150$, 207 pooled cells)', fontsize=10)
ax.set_ylim(0.78, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'Supp_Figure_S5.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved Supp_Figure_S5.png')

print('\nAll supplementary figures regenerated.')
