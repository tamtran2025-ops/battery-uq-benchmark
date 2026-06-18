"""#2 — Batch-holdout: train on 2 batches, test on 3rd. All 12 methods.

Severson 2019 collected cells in 3 separate batches. Each batch was cycled
under a (different) family of fast-charge protocols, so leave-one-batch-out
is a meaningful distribution-shift test, less stringent than full
protocol-holdout but more stringent than random-split CV.

We compute PICP raw vs PICP conformal across all 12 methods to test the
"split-conformal mandatory" claim under shift.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

ROOT_OLD = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\Predictions"
ROOT_NEW = r"D:\Project Python\PythonProject9\Paper 7\Paper_Knee\results"
OUT = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\Metrics\revision"
os.makedirs(OUT, exist_ok=True)

# (label, root, dirname) — 14 methods
METHODS = [
    ("Deep Ensemble PINN-Knee",      ROOT_OLD, "deep_ensemble_preds"),
    ("Combined UQ PINN-Knee",         ROOT_NEW, "combined_uq_preds"),
    ("Bootstrap Ens. PINN-Knee",      ROOT_NEW, "bootstrap_preds"),
    ("Jackknife+ PINN-Knee",          ROOT_NEW, "jackknife_plus_preds"),
    ("Hyper-Deep Ensemble",           ROOT_NEW, "hyper_deep_ensemble_preds"),
    ("CQR-PINN-Knee",                 ROOT_OLD, "cqr_pinn_preds"),
    ("Ensemble NN",                   ROOT_OLD, "ensemble_nn_preds"),
    ("Bayesian LSTM",                 ROOT_OLD, "bayesian_lstm_preds"),
    ("Gaussian Process",              ROOT_OLD, "gp_preds"),
    ("NGBoost",                       ROOT_NEW, "ngboost_preds"),
    ("Hetero MLP (v2)",               ROOT_NEW, "hetero_preds_v2"),
    ("CQR-MLP (v2)",                  ROOT_NEW, "cqr_preds_v2"),
    ("SNGP",                          ROOT_NEW, "sngp_preds"),
    ("Laplace",                       ROOT_NEW, "laplace_preds"),
]


def get_batch(cell_name):
    s = str(cell_name)
    if "batch1" in s: return "batch1"
    if "batch2" in s: return "batch2"
    if "batch3" in s: return "batch3"
    return "unknown"


def conformal_q_hat(scores, alpha=0.05):
    n = len(scores)
    if n == 0: return 100.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k - 1, n - 1)])


def get_sigma(d):
    if "sigmas_all" in d: return np.clip(d["sigmas_all"].mean(axis=0), 1e-3, 1e6)
    if "stds_all" in d:   return np.clip(d["stds_all"].mean(axis=0), 1e-3, 1e6)
    return np.clip(d["preds_all"].std(axis=0), 1e-3, 1e6)


# Aggregate per-cell preds across folds, tagged with batch
print("Computing batch-holdout PICP for all 12 methods...")
rows = []
for label, root, dname in METHODS:
    # Pool all (cell, y_true, y_pred, sigma, q_hat) across folds at n_early=150
    cell_data = {}  # cell_name -> dict
    for fold in range(5):
        path = os.path.join(root, dname, f"preds_ne150_f{fold}.npz")
        if not os.path.exists(path): continue
        d = dict(np.load(path, allow_pickle=True))
        y_true = d["y_true"]
        names = [str(n) for n in d["cell_names"]]
        preds = d["preds_all"]
        point = preds.mean(axis=0) if preds.ndim == 2 else preds
        point = np.clip(point, 0, 1e6)
        sigma = get_sigma(d)

        # Cal residuals (this fold) for absolute-residual conformal q_hat
        if "y_cal" in d:
            cal_pt = np.clip(d["cal_preds_all"].mean(axis=0), 0, 1e6)
            q_hat = conformal_q_hat(np.abs(d["y_cal"] - cal_pt))
        elif "loo_residuals" in d:
            q_hat = conformal_q_hat(np.abs(d["loo_residuals"]))
        else:
            q_hat = float(np.median(np.abs(y_true - point))) * 2

        for i, name in enumerate(names):
            cell_data[name] = {
                "y_true": float(y_true[i]),
                "point": float(point[i]),
                "sigma": float(sigma[i]),
                "q_hat_fold": q_hat,
                "batch": get_batch(name),
            }

    if not cell_data: continue

    # For each batch as held-out:
    for hold_batch in ["batch1", "batch2", "batch3"]:
        # cells in held-out
        held = {n: v for n, v in cell_data.items() if v["batch"] == hold_batch}
        not_held = {n: v for n, v in cell_data.items() if v["batch"] != hold_batch}
        if not held or not not_held: continue

        # raw PICP on held: |y - point| <= 1.96 * sigma
        y_h = np.array([v["y_true"] for v in held.values()])
        p_h = np.array([v["point"] for v in held.values()])
        s_h = np.array([v["sigma"] for v in held.values()])
        raw_picp = float(((y_h >= p_h - 1.96 * s_h) & (y_h <= p_h + 1.96 * s_h)).mean())
        raw_mpiw = float(2 * 1.96 * s_h.mean())

        # conformal q_hat from non-held cells
        y_nh = np.array([v["y_true"] for v in not_held.values()])
        p_nh = np.array([v["point"] for v in not_held.values()])
        cal_resid = np.abs(y_nh - p_nh)
        q_hat_pool = conformal_q_hat(cal_resid, alpha=0.05)

        conf_picp = float(((y_h >= p_h - q_hat_pool) & (y_h <= p_h + q_hat_pool)).mean())
        conf_mpiw = float(2 * q_hat_pool)

        # MAE on held set
        mae_held = float(np.mean(np.abs(y_h - p_h)))

        rows.append({
            "method": label,
            "held_batch": hold_batch,
            "n_held": len(held),
            "MAE_held": mae_held,
            "PICP_raw": raw_picp,
            "MPIW_raw": raw_mpiw,
            "PICP_conformal": conf_picp,
            "MPIW_conformal": conf_mpiw,
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "batch_holdout_all_methods.csv"), index=False)

print(f"\n=== Per-method per-batch results ===")
print(df.to_string(index=False))

# Summary: average over the 3 held-out batches per method
print(f"\n=== Method-level summary (averaged over 3 held-out batches) ===")
summ = df.groupby("method").agg(
    MAE_mean=("MAE_held", "mean"),
    MAE_std=("MAE_held", "std"),
    PICP_raw_mean=("PICP_raw", "mean"),
    PICP_conf_mean=("PICP_conformal", "mean"),
    MPIW_conf_mean=("MPIW_conformal", "mean"),
).reset_index().sort_values("MAE_mean")
print(summ.to_string(index=False))

summ.to_csv(os.path.join(OUT, "batch_holdout_summary.csv"), index=False)
print(f"\nSaved to {OUT}/batch_holdout_*.csv")
