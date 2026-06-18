"""Compute final metrics from v2/SNGP/Laplace runs and emit Markdown rows
to backfill into the manuscript tables.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

ROOT = r"D:\Project Python\PythonProject9\Paper 7"
PRED_DIR_OLD = os.path.join(ROOT, "Paper4_UQ_Comparison", "Predictions")
PRED_DIR_NEW = os.path.join(ROOT, "Paper_Knee", "results")
OUT_DIR = os.path.join(ROOT, "Paper4_UQ_Comparison", "Metrics", "revision")
os.makedirs(OUT_DIR, exist_ok=True)

NEW_METHODS = {
    "hetero_preds_v2": ("Heteroscedastic MLP (v2)", "het_v2"),
    "cqr_preds_v2": ("CQR-MLP (v2)", "cqr_v2"),
    "sngp_preds": ("SNGP", "sngp"),
    "laplace_preds": ("Laplace last-layer", "laplace"),
}

BUDGETS = [50, 100, 150]
FOLDS = list(range(5))


def conformal_q_hat(y_cal, pred_cal, alpha=0.05):
    resid = np.abs(y_cal - pred_cal)
    n = len(resid)
    if n == 0: return 100.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(resid)[min(k - 1, n - 1)]


def compute(method_dir, label):
    rows = []
    for n_early in BUDGETS:
        fold_metrics = []
        for fold in FOLDS:
            path = os.path.join(PRED_DIR_NEW, method_dir, f"preds_ne{n_early}_f{fold}.npz")
            if not os.path.exists(path):
                continue
            d = dict(np.load(path, allow_pickle=True))
            y_true = d["y_true"]
            y_cal = d["y_cal"]
            preds = d["preds_all"]
            cal_preds = d["cal_preds_all"]

            point = preds.mean(axis=0)
            cal_point = cal_preds.mean(axis=0)
            point = np.clip(point, 0, 1e6)
            cal_point = np.clip(cal_point, 0, 1e6)

            err = point - y_true
            abs_err = np.abs(err)
            ape = abs_err / np.maximum(y_true, 1.0) * 100

            mae = float(np.nanmean(abs_err))
            medae = float(np.nanmedian(abs_err))
            rmse = float(np.sqrt(np.nanmean(err ** 2)))
            mape = float(np.nanmean(ape))

            # Sigma (method-specific)
            if "sigmas_all" in d:
                sigma = d["sigmas_all"].mean(axis=0)
            elif "stds_all" in d:
                sigma = d["stds_all"].mean(axis=0)
            else:
                sigma = preds.std(axis=0)
            sigma = np.clip(sigma, 1e-3, 1e6)

            # Raw PICP/MPIW
            lower_raw = point - 1.96 * sigma
            upper_raw = point + 1.96 * sigma
            picp_raw = float(((y_true >= lower_raw) & (y_true <= upper_raw)).mean())
            mpiw_raw = float((upper_raw - lower_raw).mean())

            # Conformal PICP/MPIW
            q_hat = conformal_q_hat(y_cal, cal_point, alpha=0.05)
            lower_c = point - q_hat
            upper_c = point + q_hat
            picp_c = float(((y_true >= lower_c) & (y_true <= upper_c)).mean())
            mpiw_c = float((upper_c - lower_c).mean())

            fold_metrics.append({
                "MAE": mae, "MedAE": medae, "RMSE": rmse, "MAPE": mape,
                "PICP_raw": picp_raw, "MPIW_raw": mpiw_raw,
                "PICP_conf": picp_c, "MPIW_conf": mpiw_c,
            })

        if not fold_metrics:
            continue
        arr = pd.DataFrame(fold_metrics)
        rows.append({
            "method": label,
            "n_early": n_early,
            "MAE": arr["MAE"].mean(),
            "MAE_std": arr["MAE"].std(),
            "MedAE": arr["MedAE"].mean(),
            "RMSE": arr["RMSE"].mean(),
            "MAPE": arr["MAPE"].mean(),
            "PICP_raw": arr["PICP_raw"].mean(),
            "MPIW_raw": arr["MPIW_raw"].mean(),
            "PICP_conf": arr["PICP_conf"].mean(),
            "MPIW_conf": arr["MPIW_conf"].mean(),
            "n_folds": len(fold_metrics),
        })
    return rows


all_rows = []
for method_dir, (label, _) in NEW_METHODS.items():
    print(f"Processing {method_dir}...")
    rows = compute(method_dir, label)
    all_rows.extend(rows)

if not all_rows:
    print("No new results yet — GPU runs not complete.")
    sys.exit(0)

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(OUT_DIR, "v2_methods_metrics.csv"), index=False)

print(f"\n=== Final metrics for v2 + new baselines ===")
print(df.to_string(index=False))

print(f"\n=== Markdown rows (Table 2) ===")
for _, r in df.iterrows():
    print(f"| {r['method']} | {int(r['n_early'])} | {r['MAE']:.1f} ({r['MAE_std']:.1f}) | {r['MedAE']:.1f} | {r['RMSE']:.1f} | {r['MAPE']:.1f} |")

print(f"\n=== Markdown rows (Table 3) ===")
for _, r in df.iterrows():
    if r['n_early'] == 50:
        print(f"| {r['method']} | 50 | {r['PICP_raw']:.2f} | {r['MPIW_raw']:.1f} | **{r['PICP_conf']:.2f}** | {r['MPIW_conf']:.1f} |")
        prev_label = r['method']
    elif r['method'] == prev_label:
        print(f"|  | {int(r['n_early'])} | {r['PICP_raw']:.2f} | {r['MPIW_raw']:.1f} | **{r['PICP_conf']:.2f}** | {r['MPIW_conf']:.1f} |")

print(f"\nSaved: {os.path.join(OUT_DIR, 'v2_methods_metrics.csv')}")
