"""Regenerate Figures 3, 5, 6, 7 from actual prediction data.

Fig 3: Reliability diagrams (12 methods at n_early=100, after conformal)
Fig 5: Epistemic vs aleatoric scatter (Combined UQ PINN-Knee, n_early=100)
Fig 6: Sigma vs |error| scatter (multi-method)
Fig 7: Selective prediction curve (multi-method)
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob

ROOT_OLD = r"D:\Project Python\PythonProject9\00_Battery_Paper1_UQ-Benchmark_(RESS)\Paper4_UQ_Comparison\Predictions"
ROOT_NEW = r"D:\Project Python\PythonProject9\00_Battery_Paper2_PINN-Knee_(JES-submitted)\Paper_Knee\results"
OUT = r"D:\Project Python\PythonProject9\00_Battery_Paper1_UQ-Benchmark_(RESS)\REVISION_Package_RESS\RESS_figures"
os.makedirs(OUT, exist_ok=True)

# Method registry: (label, dir_root, dir_name, color, linestyle)
METHODS = [
    ("Deep Ensemble PINN-Knee",       ROOT_OLD, "deep_ensemble_preds", "#2ca02c", "-"),
    ("Combined UQ PINN-Knee",         ROOT_NEW, "combined_uq_preds",   "#228b22", "-"),
    ("Bootstrap Ens. PINN-Knee",      ROOT_NEW, "bootstrap_preds",     "#006400", "-"),
    ("Jackknife+ PINN-Knee",          ROOT_NEW, "jackknife_plus_preds", "#8fbc8f", "-"),
    ("Hyper-Deep Ensemble",           ROOT_NEW, "hyper_deep_ensemble_preds", "#3cb371", "-"),  # NEW
    ("CQR-PINN-Knee",                 ROOT_OLD, "cqr_pinn_preds",      "#ff7f0e", "-"),
    ("Ensemble NN",                   ROOT_OLD, "ensemble_nn_preds",   "#9467bd", "--"),
    ("Bayesian LSTM",                 ROOT_OLD, "bayesian_lstm_preds", "#d62728", "--"),
    ("Gaussian Process",              ROOT_OLD, "gp_preds",            "#1f77b4", "--"),
    ("NGBoost",                       ROOT_NEW, "ngboost_preds",       "#5f9ea0", "--"),  # NEW
    ("Hetero MLP (v2)",               ROOT_NEW, "hetero_preds_v2",     "#ff8c00", "--"),
    ("CQR-MLP (v2)",                  ROOT_NEW, "cqr_preds_v2",        "#ffa500", "--"),
    ("SNGP",                          ROOT_NEW, "sngp_preds",          "#4682b4", ":"),
    ("Laplace",                       ROOT_NEW, "laplace_preds",       "#7f7f7f", ":"),
]


def conformal_q_hat(y_cal, pred_cal, alpha=0.05):
    resid = np.abs(y_cal - pred_cal)
    n = len(resid)
    if n == 0:
        return 100.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(resid)[min(k - 1, n - 1)])


def load_method(root, dir_name, n_early, fold):
    path = os.path.join(root, dir_name, f"preds_ne{n_early}_f{fold}.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def get_point_pred(d):
    """Method-aware point prediction (clip overflow)."""
    preds = d["preds_all"]
    point = preds.mean(axis=0) if preds.ndim == 2 else preds
    return np.clip(point, 0, 1e6)


def get_sigma(d, method_label):
    """Method-aware sigma estimate."""
    if "sigmas_all" in d:
        return np.clip(d["sigmas_all"].mean(axis=0), 1e-3, 1e6)
    if "stds_all" in d:
        return np.clip(d["stds_all"].mean(axis=0), 1e-3, 1e6)
    # Ensemble: between-seed std
    preds = d["preds_all"]
    return np.clip(preds.std(axis=0), 1e-3, 1e6)


def collect_method(label, root, dname, n_early, alpha=0.05):
    """Return (y_true_all, point_all, sigma_all, lower_c_all, upper_c_all) across folds."""
    yt, pp, ss, lo_c, up_c = [], [], [], [], []
    for fold in range(5):
        d = load_method(root, dname, n_early, fold)
        if d is None:
            continue
        y_true = d["y_true"]
        point = get_point_pred(d)
        sigma = get_sigma(d, label)
        # Jackknife+ has built-in conformal via loo_residuals
        if "y_cal" not in d:
            # Use ensemble std × 1.96 as proxy interval if we have residuals
            if "loo_residuals" in d:
                q_hat = float(np.quantile(np.abs(d["loo_residuals"]), 1 - alpha))
            else:
                q_hat = float(np.median(np.abs(y_true - point))) * 2
        else:
            y_cal = d["y_cal"]
            cal_point = np.clip(d["cal_preds_all"].mean(axis=0), 0, 1e6)
            q_hat = conformal_q_hat(y_cal, cal_point, alpha)
        lower = point - q_hat
        upper = point + q_hat
        yt.append(y_true); pp.append(point); ss.append(sigma)
        lo_c.append(lower); up_c.append(upper)
    if not yt:
        return None
    return (np.concatenate(yt), np.concatenate(pp), np.concatenate(ss),
            np.concatenate(lo_c), np.concatenate(up_c))


# ============== FIGURE 3: Reliability diagrams ==============
print("Generating Figure 3 (Reliability diagrams)...")
fig, ax = plt.subplots(figsize=(8, 7))

alphas = np.linspace(0.05, 0.95, 19)
nominals = 1 - alphas
for label, root, dname, color, linestyle in METHODS:
    # Pool test residuals AND calibration residuals across folds (for true split-conformal)
    yt_all, pt_all, cal_resid_all = [], [], []
    for fold in range(5):
        d = load_method(root, dname, n_early=100, fold=fold)
        if d is None:
            continue
        y_true = d["y_true"]
        point = get_point_pred(d)
        yt_all.append(y_true); pt_all.append(point)
        if "y_cal" in d and "cal_preds_all" in d:
            cp = np.clip(d["cal_preds_all"].mean(axis=0), 0, 1e6)
            cal_resid_all.append(np.abs(d["y_cal"] - cp))
        elif "loo_residuals" in d:
            cal_resid_all.append(np.abs(d["loo_residuals"]))
    if not yt_all:
        continue
    yt = np.concatenate(yt_all); pt = np.concatenate(pt_all)
    resid_test = np.abs(yt - pt)
    cal_pool = np.concatenate(cal_resid_all) if cal_resid_all else resid_test
    # Empirical coverage at each nominal 1-alpha via split-conformal q_hat
    empirical = []
    n_cal = len(cal_pool); cal_sorted = np.sort(cal_pool)
    for a in alphas:
        k = int(np.ceil((n_cal + 1) * (1 - a)))
        q_hat = float(cal_sorted[min(k - 1, n_cal - 1)])
        empirical.append(float(np.mean(resid_test <= q_hat)))

    ax.plot(nominals, empirical, color=color, linestyle=linestyle, linewidth=1.6,
            label=label, alpha=0.85)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration', alpha=0.7)
ax.set_xlabel('Nominal coverage', fontsize=11)
ax.set_ylabel('Empirical coverage', fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.grid(alpha=0.3, linestyle=':')
ax.legend(fontsize=7.5, loc='lower right', ncol=2)
ax.set_title(r'Reliability diagrams at $n_{\mathrm{early}} = 100$', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Figure_3.png"), dpi=600, bbox_inches='tight')
plt.close()
print("  Saved Figure_3.png")

# ============== FIGURE 5: Epistemic vs Aleatoric (Combined UQ) ==============
print("\nGenerating Figure 5 (Epistemic/Aleatoric scatter)...")

# Combined UQ has mc_test_all shape (n_seeds=10, n_mc=30, n_test)
# Epistemic = between-seed variance of seed means
# Aleatoric = mean of within-seed MC variance
fig, ax = plt.subplots(figsize=(8, 6))
all_ep, all_al, all_err, all_ratio = [], [], [], []
for fold in range(5):
    d = load_method(ROOT_NEW, "combined_uq_preds", 100, fold)
    if d is None:
        continue
    y_true = d["y_true"]
    mc = d["mc_test_all"]  # (10 seeds, 30 mc, n_test)
    seed_means = mc.mean(axis=1)  # (10 seeds, n_test)
    point = seed_means.mean(axis=0)  # (n_test,)
    point = np.clip(point, 0, 1e6)
    err = np.abs(point - y_true)

    # Epistemic: variance of seed means across seeds
    sigma_ep = seed_means.std(axis=0)
    # Aleatoric: average within-seed MC std
    sigma_al = mc.std(axis=1).mean(axis=0)
    sigma_total2 = sigma_ep**2 + sigma_al**2

    sigma_ep = np.clip(sigma_ep, 1, 1e4)
    sigma_al = np.clip(sigma_al, 1, 1e4)

    all_ep.append(sigma_ep); all_al.append(sigma_al); all_err.append(err)
    ratio = sigma_ep**2 / np.maximum(sigma_total2, 1e-9)
    all_ratio.append(ratio)

if all_ep:
    ep = np.concatenate(all_ep)
    al = np.concatenate(all_al)
    err = np.concatenate(all_err)
    ratio = np.concatenate(all_ratio)

    sc = ax.scatter(ep, al, c=ratio, cmap='RdYlGn_r', s=20 + err/4,
                    edgecolors='k', linewidth=0.4, alpha=0.85, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label=r'$\sigma_{\mathrm{ep}}^2 / \sigma_{\mathrm{total}}^2$')
    ax.set_xlabel(r'Epistemic $\sigma_{\mathrm{ep}}$ (cycles)', fontsize=11)
    ax.set_ylabel(r'Aleatoric $\sigma_{\mathrm{al}}$ (cycles)', fontsize=11)
    ax.set_title(r'Combined UQ PINN-Knee at $n_{\mathrm{early}} = 100$ (marker size $\propto$ error)', fontsize=11)
    ax.grid(alpha=0.3, linestyle=':')
    ax.set_xscale('log'); ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Figure_5.png"), dpi=600, bbox_inches='tight')
plt.close()
print("  Saved Figure_5.png")

# ============== FIGURE 6: Sigma vs |error| scatter ==============
print("\nGenerating Figure 6 (Sigma-error scatter)...")
fig, ax = plt.subplots(figsize=(9, 6))

panel_methods = [m for m in METHODS if m[0] in {
    "Deep Ensemble PINN-Knee", "Combined UQ PINN-Knee", "Gaussian Process",
    "Bayesian LSTM", "SNGP", "Hetero MLP (v2)"
}]

for label, root, dname, color, linestyle in panel_methods:
    res = collect_method(label, root, dname, n_early=100)
    if res is None:
        continue
    y_true, point, sigma, _, _ = res
    err = np.abs(y_true - point)
    sigma = np.clip(sigma, 1, 5000)
    err = np.clip(err, 1, 5000)
    ax.scatter(sigma, err, color=color, s=18, alpha=0.55, label=label, edgecolors='none')

ax.set_xlabel(r'Predicted $\sigma$ (cycles)', fontsize=11)
ax.set_ylabel(r'$|y_{\mathrm{true}} - \hat{n}|$ (cycles)', fontsize=11)
ax.set_xscale('log'); ax.set_yscale('log')
ax.grid(alpha=0.3, linestyle=':')
ax.legend(fontsize=9, loc='upper left')
ax.set_title(r'Per-cell $\sigma$ vs absolute error at $n_{\mathrm{early}} = 100$', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Figure_6.png"), dpi=600, bbox_inches='tight')
plt.close()
print("  Saved Figure_6.png")

# ============== FIGURE 7: Selective prediction ==============
print("\nGenerating Figure 7 (Selective prediction)...")
fig, ax = plt.subplots(figsize=(8, 6))

for label, root, dname, color, linestyle in panel_methods:
    res = collect_method(label, root, dname, n_early=100)
    if res is None:
        continue
    y_true, point, sigma, _, _ = res
    err = np.abs(y_true - point)
    # Sort by sigma ascending; report MAE on most-confident rho fraction
    order = np.argsort(sigma)
    err_sorted = err[order]
    rhos = np.linspace(0.05, 1.0, 20)
    maes = []
    for rho in rhos:
        n = max(1, int(np.ceil(rho * len(err_sorted))))
        maes.append(float(err_sorted[:n].mean()))
    ax.plot(rhos, maes, color=color, linestyle=linestyle, linewidth=1.7,
            label=label, alpha=0.85, marker='o', markersize=3)

ax.set_xlabel(r'Coverage fraction $\rho$ (most-confident cells retained)', fontsize=11)
ax.set_ylabel(r'MAE on retained cells (cycles)', fontsize=11)
ax.grid(alpha=0.3, linestyle=':')
ax.legend(fontsize=9, loc='upper left')
ax.set_title(r'Selective prediction at $n_{\mathrm{early}} = 100$', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Figure_7.png"), dpi=600, bbox_inches='tight')
plt.close()
print("  Saved Figure_7.png")

print("\nAll figures regenerated.")
