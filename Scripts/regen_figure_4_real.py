"""Regenerate Figure 4 (ECE comparison) with REAL ECE values computed from predictions."""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT_OLD = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\Predictions"
ROOT_NEW = r"D:\Project Python\PythonProject9\Paper 7\Paper_Knee\results"
OUT = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\JPS_Submission_Files\figures"

METHODS = [
    ("Deep Ens. PINN-Knee",         ROOT_OLD, "deep_ensemble_preds"),
    ("Combined UQ PINN-Knee",        ROOT_NEW, "combined_uq_preds"),
    ("Bootstrap Ens. PINN-Knee",     ROOT_NEW, "bootstrap_preds"),
    ("Jackknife+ PINN-Knee",         ROOT_NEW, "jackknife_plus_preds"),
    ("CQR-PINN-Knee",                ROOT_OLD, "cqr_pinn_preds"),
    ("Ensemble NN",                  ROOT_OLD, "ensemble_nn_preds"),
    ("Bayesian LSTM",                ROOT_OLD, "bayesian_lstm_preds"),
    ("Gaussian Process",             ROOT_OLD, "gp_preds"),
    ("Hetero MLP (v2)",              ROOT_NEW, "hetero_preds_v2"),
    ("CQR-MLP (v2)",                 ROOT_NEW, "cqr_preds_v2"),
    ("SNGP",                         ROOT_NEW, "sngp_preds"),
    ("Laplace",                      ROOT_NEW, "laplace_preds"),
]


def conformal_q_hat(y_cal, pred_cal, alpha=0.05):
    resid = np.abs(y_cal - pred_cal)
    n = len(resid)
    if n == 0: return 100.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(resid)[min(k - 1, n - 1)])


def get_sigma(d):
    if "sigmas_all" in d:
        return np.clip(d["sigmas_all"].mean(axis=0), 1e-3, 1e6)
    if "stds_all" in d:
        return np.clip(d["stds_all"].mean(axis=0), 1e-3, 1e6)
    preds = d["preds_all"]
    return np.clip(preds.std(axis=0), 1e-3, 1e6)


def compute_ece(y_true, point, sigma_or_qhat, mode="gauss", target=0.95):
    """ECE = |empirical PICP - target|.

    mode='gauss': raw ECE using point ± 1.96σ.
    mode='conformal': conformal ECE using point ± q_hat (already at 95%).
    """
    if mode == "gauss":
        # Raw 95% Gaussian interval: point ± 1.96 σ
        z = norm.ppf(0.5 + target/2)
        lower = point - z * sigma_or_qhat
        upper = point + z * sigma_or_qhat
    else:
        # Conformal: q_hat is already the 95% quantile
        q_hat = sigma_or_qhat
        lower = point - q_hat
        upper = point + q_hat
    empirical = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return abs(empirical - target)


# Compute ECE for each method
print("Computing real ECE values...")
results = {}  # label -> {n_early: (raw_ece, conf_ece)}
for label, root, dname in METHODS:
    print(f"  {label}...")
    rows = {}
    for n_early in [50, 100, 150]:
        all_y = []; all_p = []; all_s = []; all_q_per_fold = []
        for fold in range(5):
            path = os.path.join(root, dname, f"preds_ne{n_early}_f{fold}.npz")
            if not os.path.exists(path): continue
            d = dict(np.load(path, allow_pickle=True))
            y_true = d["y_true"]
            preds = d["preds_all"]
            point = preds.mean(axis=0) if preds.ndim == 2 else preds
            point = np.clip(point, 0, 1e6)
            sigma = get_sigma(d)

            # Conformal q_hat per fold
            if "y_cal" in d:
                cal_point = np.clip(d["cal_preds_all"].mean(axis=0), 0, 1e6)
                q_hat = conformal_q_hat(d["y_cal"], cal_point, alpha=0.05)
            elif "loo_residuals" in d:
                q_hat = float(np.quantile(np.abs(d["loo_residuals"]), 0.95))
            else:
                q_hat = float(np.median(np.abs(y_true - point))) * 2

            all_y.append(y_true); all_p.append(point); all_s.append(sigma)
            all_q_per_fold.append((y_true, point, q_hat))

        if not all_y: continue
        y_all = np.concatenate(all_y)
        p_all = np.concatenate(all_p)
        s_all = np.concatenate(all_s)

        ece_raw = compute_ece(y_all, p_all, s_all, mode="gauss")
        # Conformal ECE: average per-fold
        ece_conf_folds = []
        for yf, pf, qhf in all_q_per_fold:
            ece_conf_folds.append(compute_ece(yf, pf, qhf, mode="conformal"))
        ece_conf = float(np.mean(ece_conf_folds))

        rows[n_early] = (ece_raw, ece_conf)
    results[label] = rows


# Plot at n_early = 150
fig, ax = plt.subplots(figsize=(10, 5))
methods_plot = [m[0] for m in METHODS if 150 in results[m[0]]]
ece_raw = [results[m][150][0] for m in methods_plot]
ece_conf = [results[m][150][1] for m in methods_plot]

x_pos = np.arange(len(methods_plot))
width = 0.4

ax.bar(x_pos - width/2, ece_raw, width, color='#d62728', alpha=0.75,
       label='Raw posterior', edgecolor='black', linewidth=0.5)
ax.bar(x_pos + width/2, ece_conf, width, color='#2ca02c', alpha=0.85,
       label='Split-conformal', edgecolor='black', linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(methods_plot, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=11)
ax.set_ylim(0, max(max(ece_raw), max(ece_conf)) * 1.15)
ax.legend(loc='upper left', fontsize=10)

# Annotate values
for i, (r, c) in enumerate(zip(ece_raw, ece_conf)):
    ax.text(i - width/2, r + 0.005, f'{r:.2f}', ha='center', fontsize=8)
    ax.text(i + width/2, c + 0.005, f'{c:.2f}', ha='center', fontsize=8)

ax.grid(axis='y', alpha=0.3, linestyle=':')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title(r'Expected Calibration Error at $n_{\mathrm{early}} = 150$ (computed from predictions)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Figure_4.png"), dpi=600, bbox_inches='tight')
print(f"\nSaved: {os.path.join(OUT, 'Figure_4.png')}")

# Print computed values
print("\n=== Computed ECE values @ n_early=150 ===")
for m, r, c in zip(methods_plot, ece_raw, ece_conf):
    print(f"  {m:30s}: raw={r:.3f}, conformal={c:.3f}")
