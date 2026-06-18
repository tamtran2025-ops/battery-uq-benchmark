"""
multi_partition_cv.py - Partition-variance check for the knee-point UQ benchmark.

Reviewer concern (R1): the headline 5-fold CV uses a SINGLE partition (seed 42),
so CV-partition variance is unquantified. This script repeats the full 5-fold CV
over P independent PARTITION seeds (distinct from model-init seeds) and reports
partition-level mean +/- SD and 95% CI for MAE and the split-conformal coverage
(PICP) and interval width (MPIW).

Representative method: the Gaussian Process baseline (CPU, ~1.5 s/fold) - the
exact GP configuration of gp_baseline_preds.py. Run from 2_CODE/Scripts/ once the
feature cache (_severson_cache.pkl) has been built:

    python multi_partition_cv.py --partitions 10 --alpha 0.05

(alpha = 0.05 matches the paper's CONFORMAL_ALPHA / 95% intervals; do NOT use 0.1.)
GP needs no GPU and finishes in a few minutes; it gives a real partition-variance
number for the headline conclusions. To add the GPU methods (Deep Ensemble,
CQR-PINN-Knee, ...), see EXTENDING TO OTHER METHODS at the bottom.
"""
import os, sys, argparse, pickle, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from config import RESULTS_DIR, EARLY_CYCLE_COUNTS                 # noqa: E402
from features import build_feature_matrix, normalize_features      # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor      # noqa: E402
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel  # noqa: E402

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')


def kfold_split(cells, n_folds, seed):
    """Identical structure to gp_baseline_preds.py, but the seed is a parameter."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(cells))
    fold_size = len(cells) // n_folds
    splits = []
    for f in range(n_folds):
        test_idx = idx[f * fold_size:(f + 1) * fold_size]
        remaining = np.concatenate([idx[:f * fold_size], idx[(f + 1) * fold_size:]])
        n_cal = max(1, len(remaining) // 4)
        splits.append(([cells[i] for i in remaining[n_cal:]],
                       [cells[i] for i in remaining[:n_cal]],
                       [cells[i] for i in test_idx]))
    return splits


def gp_predict(X_tr, y_tr, X_te, X_ca, model_seeds=(0, 1, 2)):
    """Mean GP prediction over model-init seeds (log-target, same kernel as paper)."""
    te = np.zeros((len(model_seeds), X_te.shape[0]))
    ca = np.zeros((len(model_seeds), X_ca.shape[0]))
    y_tr_log = np.log1p(y_tr)
    for i, s in enumerate(model_seeds):
        kernel = (ConstantKernel(1.0, (0.1, 10)) * RBF(1.0, (0.1, 100))
                  + WhiteKernel(1e-3, (1e-5, 1.0)))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                      random_state=s, alpha=1e-6)
        gp.fit(X_tr, y_tr_log)
        te[i] = np.expm1(gp.predict(X_te))
        ca[i] = np.expm1(gp.predict(X_ca))
    return te.mean(0), ca.mean(0)


def run_partition(cells, ne, partition_seed, n_folds, alpha):
    """One full 5-fold CV at a given partition seed -> pooled MAE, PICP, MPIW."""
    folds = kfold_split(cells, n_folds, partition_seed)
    abs_err, widths, covered, ntot = [], [], 0, 0
    for tr, ca, te in folds:
        X_tr, y_tr, _, _ = build_feature_matrix(tr, ne)
        X_ca, y_ca, _, _ = build_feature_matrix(ca, ne)
        X_te, y_te, _, _ = build_feature_matrix(te, ne)
        if len(y_te) == 0:
            continue
        X_trn, X_ten, X_can, _ = normalize_features(X_tr, X_te, X_ca)
        pred_te, pred_ca = gp_predict(X_trn, y_tr, X_ten, X_can)
        # split conformal, absolute-residual score (paper default)
        scores = np.abs(y_ca - pred_ca); n = len(scores)
        k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        q = np.sort(scores)[k - 1]
        lo, hi = pred_te - q, pred_te + q
        covered += int(np.sum((y_te >= lo) & (y_te <= hi))); ntot += len(y_te)
        widths.extend((hi - lo).tolist()); abs_err.extend(np.abs(pred_te - y_te).tolist())
    return float(np.mean(abs_err)), covered / max(ntot, 1), float(np.mean(widths))


def ci95(x):
    from scipy.stats import t as _t
    x = np.asarray(x, float); n = len(x); m = x.mean(); sd = x.std(ddof=1)
    crit = float(_t.ppf(0.975, n - 1)) if n > 1 else 0.0   # Student-t (n=10 -> 2.262), -> 1.96 as n grows
    half = crit * sd / np.sqrt(n)
    return m, sd, m - half, m + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partitions', type=int, default=10)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=0.05)  # 95% intervals, matches paper CONFORMAL_ALPHA
    a = ap.parse_args()

    with open(CACHE_PATH, 'rb') as f:
        cells = pickle.load(f)
    cells = [c for c in cells if c.get('has_knee_point', False)
             and c.get('knee_cycle') is not None]
    print(f'cells={len(cells)}  partitions={a.partitions}  folds={a.folds}  alpha={a.alpha}')

    rows = []
    for ne in EARLY_CYCLE_COUNTS:
        per = []
        for P in range(a.partitions):
            mae, picp, mpiw = run_partition(cells, ne, P, a.folds, a.alpha)
            per.append((mae, picp, mpiw))
            print(f'  ne={ne} partition={P:2d}: MAE={mae:6.1f}  PICP={picp:.3f}  MPIW={mpiw:6.0f}')
        for j, name in enumerate(['MAE', 'PICP', 'MPIW']):
            arr = np.array([p[j] for p in per])
            m, sd, lo, hi = ci95(arr)
            print(f'  >> ne={ne} {name}: mean={m:.3f}  SD={sd:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]  range=[{arr.min():.3f}, {arr.max():.3f}]')
            rows.append([ne, name, round(m, 3), round(sd, 3), round(lo, 3), round(hi, 3), round(float(arr.min()), 3), round(float(arr.max()), 3)])

    out = os.path.join(RESULTS_DIR, 'multi_partition_cv_GP.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['budget', 'metric', 'mean', 'sd', 'ci_lo', 'ci_hi', 'min', 'max']); w.writerows(rows)
    print('wrote', out)


# =====================================================================
# EXTENDING TO OTHER METHODS (GPU: Deep Ensemble, CQR-PINN-Knee, ...)
# ---------------------------------------------------------------------
# Each method script hardcodes kfold_split(cells, N_FOLDS, seed=42). To get
# partition variance:
#   1. Parameterise that call:  seed = int(os.environ.get('PARTITION_SEED', 42))
#   2. Re-run the method for PARTITION_SEED in 0..9, writing predictions to
#      gp_preds_p{P}/ , deep_ensemble_preds_p{P}/ , etc.
#   3. For each method+partition, load the per-fold .npz (y_true, preds_all,
#      cal_preds_all, y_cal), apply the same absolute-residual split-conformal
#      block as run_partition() above, and collect (MAE, PICP, MPIW).
#   4. Feed the P values of each metric through ci95() to get mean/SD/95% CI.
# The GP result this script produces is already a valid, reviewer-facing
# partition-variance number for the headline conclusions.
# =====================================================================
if __name__ == '__main__':
    main()
