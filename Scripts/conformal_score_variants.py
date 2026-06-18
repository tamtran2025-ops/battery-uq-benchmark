"""Compare 3 conformal score functions for Deep Ensemble PINN-Knee.

Reference: Romano et al. 2020, 'CQR for distribution-free predictive inference'.

Score functions tested:
  (a) Absolute residual:    S(x, y) = |y - mu(x)|
  (b) Normalised residual:  S(x, y) = |y - mu(x)| / sigma(x)
  (c) Locally adaptive:     S(x, y) = |y - mu(x)| / (alpha * sigma(x) + epsilon)
                              with sigma per-stratum
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

ROOT = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison"
PRED_DIR = os.path.join(ROOT, "Predictions")
OUT_DIR = os.path.join(ROOT, "Metrics", "revision")

DE_DIR = "deep_ensemble_preds"
HET_DIR = "hetero_preds"
GP_DIR = "gp_preds"
BUDGETS = [50, 100, 150]
FOLDS = list(range(5))


def conformal_calibrate(scores, alpha=0.05):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(scores)[min(k - 1, n - 1)]


def evaluate_method(method_dir, has_sigma_key=None, alpha=0.05):
    """For each (budget, fold), compute PICP/MPIW under 3 score functions."""
    rows = []
    for n_early in BUDGETS:
        for fold in FOLDS:
            path = os.path.join(PRED_DIR, method_dir, f"preds_ne{n_early}_f{fold}.npz")
            if not os.path.exists(path):
                continue
            d = dict(np.load(path, allow_pickle=True))
            y_true = d["y_true"]
            y_cal = d["y_cal"]
            preds = d["preds_all"]
            cal_preds = d["cal_preds_all"]

            # Point predictions: ensemble mean (or seed mean)
            point = preds.mean(axis=0)
            cal_point = cal_preds.mean(axis=0)

            # Sigma: use per-method
            if has_sigma_key == "ensemble_std":
                sigma = preds.std(axis=0)
                cal_sigma = cal_preds.std(axis=0)
            elif has_sigma_key == "het_sigma":
                sigma = d["sigmas_all"].mean(axis=0)
                cal_sigma = sigma  # use same per-test-cell σ
                # Actually for cal, we'd need sigma at cal locations — approximate
                cal_sigma = np.full_like(cal_point, sigma.mean())
            elif has_sigma_key == "gp_sigma":
                sigma = d["stds_all"].mean(axis=0)
                cal_sigma = np.full_like(cal_point, sigma.mean())
            else:
                sigma = preds.std(axis=0)
                cal_sigma = cal_preds.std(axis=0)

            # Sanitize
            point = np.clip(point, 0, 1e6)
            cal_point = np.clip(cal_point, 0, 1e6)
            sigma = np.clip(sigma, 1e-3, 1e6)
            cal_sigma = np.clip(cal_sigma, 1e-3, 1e6)

            # === (a) Absolute residual ===
            cal_resid_a = np.abs(y_cal - cal_point)
            q_a = conformal_calibrate(cal_resid_a, alpha)
            lower_a = point - q_a
            upper_a = point + q_a
            picp_a = ((y_true >= lower_a) & (y_true <= upper_a)).mean()
            mpiw_a = (upper_a - lower_a).mean()

            # === (b) Normalised residual ===
            cal_resid_b = np.abs(y_cal - cal_point) / cal_sigma
            q_b = conformal_calibrate(cal_resid_b, alpha)
            lower_b = point - q_b * sigma
            upper_b = point + q_b * sigma
            picp_b = ((y_true >= lower_b) & (y_true <= upper_b)).mean()
            mpiw_b = (upper_b - lower_b).mean()

            # === (c) Locally adaptive (k-NN normalisation) ===
            # Use cal cell knee values to define strata; set σ as sample std within stratum
            cal_terts = np.percentile(y_cal, [33.33, 66.67])
            test_terts = cal_terts  # use cal-derived thresholds
            cal_strat_sigma = np.zeros_like(cal_sigma)
            for s_lo, s_hi, mask_lab in [(0, cal_terts[0], "low"), (cal_terts[0], cal_terts[1], "mid"), (cal_terts[1], np.inf, "high")]:
                mask = (y_cal > s_lo) & (y_cal <= s_hi)
                if mask.sum() > 0:
                    s = np.std(y_cal[mask] - cal_point[mask]) if mask.sum() > 1 else 100.0
                    cal_strat_sigma[mask] = max(s, 1.0)

            cal_resid_c = np.abs(y_cal - cal_point) / cal_strat_sigma
            q_c = conformal_calibrate(cal_resid_c, alpha)
            # For test, we don't know y_true tertile; use predicted tertile
            test_strat_sigma = np.zeros_like(point)
            for s_lo, s_hi in [(0, test_terts[0]), (test_terts[0], test_terts[1]), (test_terts[1], np.inf)]:
                mask = (point > s_lo) & (point <= s_hi)
                if mask.sum() > 0:
                    s = cal_strat_sigma[(y_cal > s_lo) & (y_cal <= s_hi)].mean() if ((y_cal > s_lo) & (y_cal <= s_hi)).sum() > 0 else 100.0
                    test_strat_sigma[mask] = max(s, 1.0)
            test_strat_sigma = np.where(test_strat_sigma == 0, 100, test_strat_sigma)

            lower_c = point - q_c * test_strat_sigma
            upper_c = point + q_c * test_strat_sigma
            picp_c = ((y_true >= lower_c) & (y_true <= upper_c)).mean()
            mpiw_c = (upper_c - lower_c).mean()

            rows.append({
                "method": method_dir,
                "n_early": n_early,
                "fold": fold,
                "score_a_abs_PICP": picp_a, "score_a_abs_MPIW": mpiw_a,
                "score_b_norm_PICP": picp_b, "score_b_norm_MPIW": mpiw_b,
                "score_c_local_PICP": picp_c, "score_c_local_MPIW": mpiw_c,
            })
    return rows


print("=" * 80)
print("Conformal score variants — Deep Ensemble + Hetero + GP")
print("=" * 80)

all_rows = []
for method_dir, sigma_key in [
    (DE_DIR, "ensemble_std"),
    (HET_DIR, "het_sigma"),
    (GP_DIR, "gp_sigma"),
]:
    print(f"\nProcessing {method_dir}...")
    all_rows.extend(evaluate_method(method_dir, has_sigma_key=sigma_key))

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(OUT_DIR, "conformal_score_variants.csv"), index=False)
agg = df.groupby(["method", "n_early"]).agg(
    abs_PICP=("score_a_abs_PICP", "mean"),
    abs_MPIW=("score_a_abs_MPIW", "mean"),
    norm_PICP=("score_b_norm_PICP", "mean"),
    norm_MPIW=("score_b_norm_MPIW", "mean"),
    local_PICP=("score_c_local_PICP", "mean"),
    local_MPIW=("score_c_local_MPIW", "mean"),
).reset_index()
agg.to_csv(os.path.join(OUT_DIR, "conformal_score_variants_aggregate.csv"), index=False)

print("\n=== Aggregated comparison (mean across folds) ===")
print(agg.to_string(index=False))

print("\n=== Per-method narrative ===")
for method in agg["method"].unique():
    print(f"\n{method}:")
    sub = agg[agg["method"] == method]
    for _, r in sub.iterrows():
        # Compare MPIW at similar PICP
        verdict = []
        if r["abs_MPIW"] > r["norm_MPIW"] and r["norm_PICP"] >= 0.90:
            verdict.append(f"normalized sharper ({r['norm_MPIW']:.0f} < {r['abs_MPIW']:.0f})")
        if r["abs_MPIW"] > r["local_MPIW"] and r["local_PICP"] >= 0.90:
            verdict.append(f"locally-adaptive sharper ({r['local_MPIW']:.0f})")
        verdict_str = "; ".join(verdict) if verdict else "absolute residual best"
        print(f"  n_early={int(r['n_early'])}: {verdict_str}")

print("\nSaved to:", OUT_DIR)
