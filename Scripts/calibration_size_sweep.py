"""#5 — Calibration set size sweep for Deep Ensemble PINN-Knee.

Sweep cal_fraction ∈ {0.15, 0.25 (current), 0.33, 0.50}.
Use existing predictions; resample (cal, test) splits within each fold.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from glob import glob

ROOT_OLD = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\Predictions\deep_ensemble_preds"
OUT = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\Metrics\revision"

CAL_FRACTIONS = [0.15, 0.25, 0.33, 0.50]
N_RESAMPLES = 50  # bootstrap resamples per fraction


def conformal_q_hat(scores, alpha=0.05):
    n = len(scores)
    if n == 0: return 100.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k - 1, n - 1)])


rows = []
for n_early in [50, 100, 150]:
    # Aggregate all fold predictions
    all_y, all_pred = [], []
    for fold in range(5):
        path = os.path.join(ROOT_OLD, f"preds_ne{n_early}_f{fold}.npz")
        if not os.path.exists(path): continue
        d = dict(np.load(path, allow_pickle=True))
        # Combine cal + test (we'll resample)
        y_pool = np.concatenate([d["y_cal"], d["y_true"]])
        p_pool = np.concatenate([d["cal_preds_all"].mean(axis=0),
                                 d["preds_all"].mean(axis=0)])
        p_pool = np.clip(p_pool, 0, 1e6)
        all_y.append(y_pool); all_pred.append(p_pool)

    for cal_frac in CAL_FRACTIONS:
        picps, mpiws = [], []
        for fold_idx, (y, p) in enumerate(zip(all_y, all_pred)):
            n_total = len(y)
            n_cal = max(2, int(n_total * cal_frac))

            for resample in range(N_RESAMPLES):
                rng = np.random.RandomState(fold_idx * 1000 + resample)
                idx = rng.permutation(n_total)
                cal_idx, test_idx = idx[:n_cal], idx[n_cal:]
                if len(test_idx) < 5: continue

                resid_cal = np.abs(y[cal_idx] - p[cal_idx])
                q_hat = conformal_q_hat(resid_cal, alpha=0.05)
                lower = p[test_idx] - q_hat
                upper = p[test_idx] + q_hat
                picp = float(((y[test_idx] >= lower) & (y[test_idx] <= upper)).mean())
                mpiw = float(2 * q_hat)
                picps.append(picp); mpiws.append(mpiw)

        rows.append({
            "n_early": n_early,
            "cal_fraction": cal_frac,
            "n_cal_per_fold": int(len(all_y[0]) * cal_frac),
            "PICP_mean": np.mean(picps),
            "PICP_std": np.std(picps),
            "MPIW_mean": np.mean(mpiws),
            "MPIW_std": np.std(mpiws),
            "n_resamples": len(picps),
        })

df = pd.DataFrame(rows)
out_csv = os.path.join(OUT, "calibration_size_sweep.csv")
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}\n")
print(df.to_string(index=False))

print("\n=== Summary at n_early = 150 ===")
sub = df[df["n_early"] == 150]
for _, r in sub.iterrows():
    print(f"  cal_fraction={r['cal_fraction']:.2f} (n_cal≈{int(r['n_cal_per_fold'])}): "
          f"PICP={r['PICP_mean']:.3f}±{r['PICP_std']:.3f}, "
          f"MPIW={r['MPIW_mean']:.0f}±{r['MPIW_std']:.0f}")
