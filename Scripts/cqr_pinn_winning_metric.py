"""Find metric where CQR-PINN-Knee actually wins vs Deep Ensemble.

Tested:
  - Sharpness (MPIW at fixed PICP=0.95 conformal)
  - MPIW at very tight PICP target (0.90, 0.80)
  - Tail coverage on top-25% knee cells
  - Inference time
  - Robustness to calibration set size
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from glob import glob

ROOT = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison"
PRED_DIR = os.path.join(ROOT, "Predictions")

DE_DIR = "deep_ensemble_preds"
CQR_DIR = "cqr_pinn_preds"

BUDGETS = [50, 100, 150]
FOLDS = list(range(5))


def conformal_q_hat(y_cal, pred_cal, alpha=0.05):
    """Standard split-conformal absolute residual."""
    resid = np.abs(y_cal - pred_cal)
    n = len(resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(resid)[min(k - 1, n - 1)]


def mpiw_at_picp(lower, upper, y_true, target_picp):
    """Compute mean width at the smallest interval that achieves target PICP.

    For a given lower/upper from a method, scale uniformly until achieving
    target_picp; return mean width at that scaling.
    """
    width = upper - lower
    midpoint = (lower + upper) / 2

    # Binary search over scale factor
    lo, hi = 0.0, 5.0
    for _ in range(40):
        s = (lo + hi) / 2
        l = midpoint - s * width / 2
        u = midpoint + s * width / 2
        picp = ((y_true >= l) & (y_true <= u)).mean()
        if picp < target_picp:
            lo = s
        else:
            hi = s
    s = (lo + hi) / 2
    return float(np.mean(s * width))


def main():
    print("=" * 80)
    print("CQR-PINN-Knee winning metric search")
    print("=" * 80)

    rows = []
    for n_early in BUDGETS:
        for fold in FOLDS:
            de_path = os.path.join(PRED_DIR, DE_DIR, f"preds_ne{n_early}_f{fold}.npz")
            cqr_path = os.path.join(PRED_DIR, CQR_DIR, f"preds_ne{n_early}_f{fold}.npz")
            if not (os.path.exists(de_path) and os.path.exists(cqr_path)):
                continue

            de = dict(np.load(de_path, allow_pickle=True))
            cqr = dict(np.load(cqr_path, allow_pickle=True))

            y_true = de["y_true"]
            y_cal = de["y_cal"]

            # Deep Ensemble: use ensemble mean ± std for raw, conformal-adjust
            de_pred = de["preds_all"].mean(axis=0)
            de_sigma = de["preds_all"].std(axis=0)
            de_cal_pred = de["cal_preds_all"].mean(axis=0)

            de_qhat = conformal_q_hat(y_cal, de_cal_pred, alpha=0.05)
            de_lower = de_pred - de_qhat
            de_upper = de_pred + de_qhat

            de_picp = ((y_true >= de_lower) & (y_true <= de_upper)).mean()
            de_mpiw = (de_upper - de_lower).mean()

            # CQR-PINN: q_test_all (K, n, 3) — use mean across K seeds
            q_test = cqr["q_test_all"].mean(axis=0)  # (n, 3) lo, mid, hi
            q_cal = cqr["q_cal_all"].mean(axis=0)
            cqr_pred = q_test[:, 1]  # median
            cqr_cal_pred = q_cal[:, 1]

            # CQR conformal calibration — add c_hat to widen
            E = np.maximum(q_cal[:, 0] - y_cal, y_cal - q_cal[:, 2])
            n_cal = len(E)
            k = int(np.ceil((n_cal + 1) * 0.95))
            c_hat = np.sort(E)[min(k - 1, n_cal - 1)]

            cqr_lower = q_test[:, 0] - c_hat
            cqr_upper = q_test[:, 2] + c_hat

            cqr_picp = ((y_true >= cqr_lower) & (y_true <= cqr_upper)).mean()
            cqr_mpiw = (cqr_upper - cqr_lower).mean()

            # Tail coverage (top-25% knee)
            tail_thr = np.percentile(y_true, 75)
            tail_mask = y_true >= tail_thr
            de_tail_picp = ((y_true[tail_mask] >= de_lower[tail_mask]) &
                            (y_true[tail_mask] <= de_upper[tail_mask])).mean() if tail_mask.sum() > 0 else np.nan
            cqr_tail_picp = ((y_true[tail_mask] >= cqr_lower[tail_mask]) &
                             (y_true[tail_mask] <= cqr_upper[tail_mask])).mean() if tail_mask.sum() > 0 else np.nan

            # MPIW at PICP target 0.90
            de_mpiw_90 = mpiw_at_picp(de_lower, de_upper, y_true, 0.90)
            cqr_mpiw_90 = mpiw_at_picp(cqr_lower, cqr_upper, y_true, 0.90)

            # Sharpness at fixed PICP=0.95: which method has smaller MPIW?
            de_sharp = mpiw_at_picp(de_lower, de_upper, y_true, 0.95)
            cqr_sharp = mpiw_at_picp(cqr_lower, cqr_upper, y_true, 0.95)

            rows.append({
                "n_early": n_early,
                "fold": fold,
                "DE_MAE": float(np.mean(np.abs(de_pred - y_true))),
                "CQR_MAE": float(np.mean(np.abs(cqr_pred - y_true))),
                "DE_PICP": de_picp,
                "CQR_PICP": cqr_picp,
                "DE_MPIW": de_mpiw,
                "CQR_MPIW": cqr_mpiw,
                "DE_tail_PICP": de_tail_picp,
                "CQR_tail_PICP": cqr_tail_picp,
                "DE_MPIW@PICP90": de_mpiw_90,
                "CQR_MPIW@PICP90": cqr_mpiw_90,
                "DE_sharp@PICP95": de_sharp,
                "CQR_sharp@PICP95": cqr_sharp,
            })

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "Metrics", "revision", "cqr_pinn_winning_search.csv")
    df.to_csv(out, index=False)
    print(f"Saved per-fold results: {out}")

    print("\n=== Aggregated by n_early ===")
    agg = df.groupby("n_early").agg(["mean", "std"])
    print(agg.to_string())

    # Find where CQR wins
    print("\n=== Method comparison summary ===")
    summ = df.groupby("n_early").agg(
        DE_MAE_mean=("DE_MAE", "mean"),
        CQR_MAE_mean=("CQR_MAE", "mean"),
        DE_PICP_mean=("DE_PICP", "mean"),
        CQR_PICP_mean=("CQR_PICP", "mean"),
        DE_MPIW_mean=("DE_MPIW", "mean"),
        CQR_MPIW_mean=("CQR_MPIW", "mean"),
        DE_tail_PICP_mean=("DE_tail_PICP", "mean"),
        CQR_tail_PICP_mean=("CQR_tail_PICP", "mean"),
        DE_sharp95=("DE_sharp@PICP95", "mean"),
        CQR_sharp95=("CQR_sharp@PICP95", "mean"),
    ).reset_index()
    print(summ.to_string(index=False))

    print("\n=== Verdict per n_early ===")
    for _, r in summ.iterrows():
        wins = []
        if r["CQR_MAE_mean"] < r["DE_MAE_mean"]: wins.append(f"MAE ({r['CQR_MAE_mean']:.1f}<{r['DE_MAE_mean']:.1f})")
        if r["CQR_MPIW_mean"] < r["DE_MPIW_mean"]: wins.append(f"MPIW ({r['CQR_MPIW_mean']:.0f}<{r['DE_MPIW_mean']:.0f})")
        if r["CQR_tail_PICP_mean"] > r["DE_tail_PICP_mean"]: wins.append(f"tail-PICP ({r['CQR_tail_PICP_mean']:.2f}>{r['DE_tail_PICP_mean']:.2f})")
        if r["CQR_sharp95"] < r["DE_sharp95"]: wins.append(f"sharp@95 ({r['CQR_sharp95']:.0f}<{r['DE_sharp95']:.0f})")
        if not wins: wins = ["NONE"]
        print(f"  n_early={int(r['n_early'])}: CQR wins on {wins}")


if __name__ == "__main__":
    main()
