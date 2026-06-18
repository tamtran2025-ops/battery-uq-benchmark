"""Revision analysis: MAPE, failure case, stratified coverage, Bonferroni.

Computes per-method-budget-fold:
  - MAPE (% error per cell, then mean)
  - Top-5 worst-case cells (with knee-cycle, predicted, error)
  - Coverage stratified by knee-cycle tertile (low/mid/high)
  - Bonferroni-corrected pairwise Wilcoxon p-values
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import wilcoxon, norm
from itertools import combinations

ROOT = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison"
PRED_DIR = os.path.join(ROOT, "Predictions")
OUT_DIR = os.path.join(ROOT, "Metrics", "revision")
os.makedirs(OUT_DIR, exist_ok=True)

# Map method dir -> display name + has_uncertainty info
METHOD_MAP = {
    "deep_ensemble_preds":   ("Deep_Ensemble_PINN_Knee", "ensemble_K"),
    "cqr_pinn_preds":        ("CQR_PINN_Knee",            "cqr"),
    "cqr_preds":             ("CQR_MLP",                  "cqr"),
    "ensemble_nn_preds":     ("Ensemble_NN",              "ensemble_K"),
    "bayesian_lstm_preds":   ("Bayesian_LSTM",            "mc_dropout"),
    "gp_preds":              ("Gaussian_Process",         "gp_sigma"),
    "hetero_preds":          ("Heteroscedastic_MLP",      "het_sigma"),
}

# Identify budgets and folds in each method
BUDGETS = [50, 100, 150]
FOLDS = list(range(5))


def load_predictions(method_dir, n_early, fold):
    path = os.path.join(PRED_DIR, method_dir, f"preds_ne{n_early}_f{fold}.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def per_seed_run_mae(d, method_type):
    """Return list of per-seed MAEs given predictions dict and method type."""
    y_true = d["y_true"]
    preds = d["preds_all"]
    if method_type == "ensemble_K":
        # preds shape (K=10, n_test) — use ensemble mean
        ensemble_mean = preds.mean(axis=0)
        return [np.mean(np.abs(y_true - ensemble_mean))], [ensemble_mean]
    elif method_type == "cqr":
        # preds shape (K=10, n_test) where K is seeds; preds_all is the median
        # Use each seed separately
        return [np.mean(np.abs(y_true - preds[k])) for k in range(preds.shape[0])], [preds[k] for k in range(preds.shape[0])]
    elif method_type == "mc_dropout":
        # preds shape (300=N_mc*30, n_test); use mean over MC samples
        # Actually 300 might mean 30 MC samples per 10 seeds OR 300 MC samples × 1 seed
        # We treat as "all MC samples" and report mean as point pred
        ensemble_mean = preds.mean(axis=0)
        return [np.mean(np.abs(y_true - ensemble_mean))], [ensemble_mean]
    elif method_type == "gp_sigma":
        # preds (K=10 seeds, n_test)
        return [np.mean(np.abs(y_true - preds[k])) for k in range(preds.shape[0])], [preds[k] for k in range(preds.shape[0])]
    elif method_type == "het_sigma":
        # preds (K=10 seeds, n_test)
        return [np.mean(np.abs(y_true - preds[k])) for k in range(preds.shape[0])], [preds[k] for k in range(preds.shape[0])]


def main():
    # ================= TASK 1.4 — MAPE =================
    print("=" * 80)
    print("TASK 1.4 — MAPE per method/budget/fold")
    print("=" * 80)
    rows = []
    failure_records = []  # cell-level worst predictions

    for method_dir, (label, mtype) in METHOD_MAP.items():
        for n_early in BUDGETS:
            for fold in FOLDS:
                d = load_predictions(method_dir, n_early, fold)
                if d is None:
                    continue
                y_true = d["y_true"]
                cell_names = d["cell_names"]

                if mtype == "ensemble_K":
                    pred = d["preds_all"].mean(axis=0)
                elif mtype == "mc_dropout":
                    pred = d["preds_all"].mean(axis=0)
                elif mtype in ("cqr", "gp_sigma", "het_sigma"):
                    pred = d["preds_all"].mean(axis=0)
                else:
                    pred = d["preds_all"].mean(axis=0)

                # Sanitize: clip extreme overflow values to avoid NaN/Inf in stats
                pred_clean = np.clip(pred, 0, 1e6)
                err = pred_clean - y_true
                abs_err = np.abs(err)
                ape = abs_err / np.maximum(y_true, 1.0) * 100  # avoid /0

                rows.append({
                    "method": label,
                    "n_early": n_early,
                    "fold": fold,
                    "MAE": float(np.nanmean(abs_err)),
                    "MAPE_%": float(np.nanmean(ape)),
                    "MedAPE_%": float(np.nanmedian(ape)),
                    "RMSE": float(np.sqrt(np.nanmean(err**2))),
                    "n_cells": len(y_true),
                    "diverged": bool(np.any(np.abs(pred) > 1e8)),
                })

                # ============ TASK 1.5 — Failure cases ============
                for i, cn in enumerate(cell_names):
                    failure_records.append({
                        "method": label,
                        "n_early": n_early,
                        "fold": fold,
                        "cell": str(cn),
                        "y_true": float(y_true[i]),
                        "y_pred": float(pred_clean[i]),
                        "abs_err": float(abs_err[i]),
                        "ape_%": float(ape[i]),
                    })

    df_mape = pd.DataFrame(rows)
    df_mape.to_csv(os.path.join(OUT_DIR, "mape_per_fold.csv"), index=False)
    print(f"  Saved {len(df_mape)} per-fold MAPE rows")

    # Aggregate
    agg = df_mape.groupby(["method", "n_early"]).agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        MAPE_mean=("MAPE_%", "mean"),
        MAPE_std=("MAPE_%", "std"),
        MedAPE_mean=("MedAPE_%", "mean"),
        n_runs=("MAE", "count"),
        any_diverged=("diverged", "any"),
    ).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, "mape_aggregate.csv"), index=False)
    print("\n=== Aggregated MAPE per method × budget ===")
    print(agg.to_string(index=False))

    # ============= TASK 1.5 — Failure case analysis =============
    print("\n\n" + "=" * 80)
    print("TASK 1.5 — Top-5 worst-prediction cells (Deep Ensemble PINN-Knee, n=150)")
    print("=" * 80)
    df_fail = pd.DataFrame(failure_records)
    df_fail.to_csv(os.path.join(OUT_DIR, "all_predictions_per_cell.csv"), index=False)

    # For headline method, find cells with high error across folds
    de = df_fail[(df_fail["method"] == "Deep_Ensemble_PINN_Knee") & (df_fail["n_early"] == 150)]
    cell_agg = de.groupby("cell").agg(
        mean_y_true=("y_true", "mean"),
        mean_pred=("y_pred", "mean"),
        mean_abs_err=("abs_err", "mean"),
        mean_ape=("ape_%", "mean"),
        n_appear=("abs_err", "count"),
    ).reset_index().sort_values("mean_abs_err", ascending=False)
    cell_agg.to_csv(os.path.join(OUT_DIR, "deep_ensemble_per_cell.csv"), index=False)
    print(f"\nTop-5 worst cells (Deep Ensemble PINN-Knee, n_early=150):")
    print(cell_agg.head(10).to_string(index=False))

    # ============ TASK 2.3 — Stratified coverage ============
    print("\n\n" + "=" * 80)
    print("TASK 2.3 — Coverage stratified by knee-cycle tertile")
    print("=" * 80)
    strat_rows = []
    for method_dir, (label, mtype) in METHOD_MAP.items():
        for n_early in BUDGETS:
            for fold in FOLDS:
                d = load_predictions(method_dir, n_early, fold)
                if d is None:
                    continue
                y_true = d["y_true"]
                if len(y_true) < 6:
                    continue

                # Compute conformal interval (use deep_ensemble or method-specific)
                preds = d["preds_all"]
                if mtype == "ensemble_K":
                    point = preds.mean(axis=0)
                    sigma_ep = preds.std(axis=0)
                    sigma = sigma_ep
                elif mtype == "het_sigma":
                    point = preds.mean(axis=0)
                    sigma = d["sigmas_all"].mean(axis=0)
                elif mtype == "gp_sigma":
                    point = preds.mean(axis=0)
                    sigma = d["stds_all"].mean(axis=0)
                elif mtype == "mc_dropout":
                    point = preds.mean(axis=0)
                    sigma = preds.std(axis=0)
                elif mtype == "cqr":
                    # use median q + half-width for interval
                    q = d["q_test_all"]  # (K, n, 3)
                    point = q.mean(axis=0)[:, 1]
                    half = (q.mean(axis=0)[:, 2] - q.mean(axis=0)[:, 0]) / 2
                    sigma = half / 1.96
                else:
                    continue

                # Sanitize
                point = np.clip(point, 0, 1e6)
                sigma = np.clip(sigma, 1e-6, 1e6)

                # Conformal calibration (split-conformal absolute residual)
                if "y_cal" in d and "cal_preds_all" in d:
                    y_cal = d["y_cal"]
                    cal_pred_seeds = d["cal_preds_all"]
                    if mtype == "ensemble_K" or mtype == "mc_dropout":
                        cal_point = cal_pred_seeds.mean(axis=0)
                    else:
                        cal_point = cal_pred_seeds.mean(axis=0)
                    cal_resid = np.abs(y_cal - np.clip(cal_point, 0, 1e6))
                    n_cal = len(y_cal)
                    q_hat = np.quantile(cal_resid, np.ceil((n_cal+1)*0.95)/n_cal) if n_cal > 0 else np.median(cal_resid)
                    lower = point - q_hat
                    upper = point + q_hat
                else:
                    lower = point - 1.96 * sigma
                    upper = point + 1.96 * sigma

                in_interval = (y_true >= lower) & (y_true <= upper)

                # Stratify by knee tertile
                terts = np.percentile(y_true, [33.33, 66.67])
                strat_low = y_true <= terts[0]
                strat_mid = (y_true > terts[0]) & (y_true <= terts[1])
                strat_high = y_true > terts[1]

                for stratum_label, mask in [("low", strat_low), ("mid", strat_mid), ("high", strat_high)]:
                    if mask.sum() > 0:
                        strat_rows.append({
                            "method": label,
                            "n_early": n_early,
                            "fold": fold,
                            "stratum": stratum_label,
                            "n_cells": int(mask.sum()),
                            "coverage": float(in_interval[mask].mean()),
                            "mean_y_true": float(y_true[mask].mean()),
                        })

    df_strat = pd.DataFrame(strat_rows)
    df_strat.to_csv(os.path.join(OUT_DIR, "stratified_coverage.csv"), index=False)
    strat_agg = df_strat.groupby(["method", "n_early", "stratum"]).agg(
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        n_folds=("coverage", "count"),
    ).reset_index()
    strat_agg.to_csv(os.path.join(OUT_DIR, "stratified_coverage_aggregate.csv"), index=False)
    print(f"\nStratified coverage (mean across folds × seeds):")
    print(strat_agg[strat_agg["n_early"] == 150].to_string(index=False))

    # ============ TASK 2.1 — Bonferroni-corrected pairwise Wilcoxon ============
    print("\n\n" + "=" * 80)
    print("TASK 2.1 — Bonferroni-corrected Wilcoxon (Deep Ensemble vs each, n=150)")
    print("=" * 80)
    n_early = 150
    method_maes = {}  # method -> per-fold MAE list
    for label in METHOD_MAP.values():
        m = label[0]
        sub = df_mape[(df_mape["method"] == m) & (df_mape["n_early"] == n_early)]
        if len(sub) > 0:
            method_maes[m] = sub["MAE"].values

    # Pairwise Wilcoxon (DE vs others)
    de_label = "Deep_Ensemble_PINN_Knee"
    if de_label in method_maes:
        de_maes = method_maes[de_label]
        wilcoxon_rows = []
        n_methods = len(method_maes) - 1  # pairs against DE
        bonf_alpha = 0.05 / n_methods
        for label, maes in method_maes.items():
            if label == de_label or len(maes) != len(de_maes):
                continue
            try:
                stat, p = wilcoxon(de_maes, maes)
                wilcoxon_rows.append({
                    "comparison": f"DE vs {label}",
                    "n": len(de_maes),
                    "DE_mean": float(de_maes.mean()),
                    "other_mean": float(maes.mean()),
                    "wilcoxon_stat": float(stat),
                    "p_uncorrected": float(p),
                    "p_bonferroni_threshold": float(bonf_alpha),
                    "significant_uncorr": bool(p < 0.05),
                    "significant_bonferroni": bool(p < bonf_alpha),
                })
            except Exception as e:
                wilcoxon_rows.append({"comparison": f"DE vs {label}", "error": str(e)})

        df_w = pd.DataFrame(wilcoxon_rows)
        df_w.to_csv(os.path.join(OUT_DIR, "wilcoxon_bonferroni.csv"), index=False)
        print(f"\n{df_w.to_string(index=False)}")
        print(f"\nBonferroni threshold (alpha=0.05/{n_methods} comparisons) = {bonf_alpha:.5f}")

    print("\n\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
