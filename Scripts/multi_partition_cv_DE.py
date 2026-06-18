"""
multi_partition_cv_DE.py - Partition-variance check for the HEADLINE method
(Deep Ensemble PINN-Knee), GPU.

Companion to multi_partition_cv.py (GP baseline). Same protocol: repeat the full
5-fold CV over P independent PARTITION seeds (distinct from model-init seeds) and
report partition-level mean +/- SD and 95% CI (Student-t) for MAE, split-conformal
PICP and MPIW. Reuses the exact training pipeline of deep_ensemble_with_preds.py
(create_model('PINN_Knee') + train_pinn_knee, log-target, ensemble mean,
absolute-residual split-conformal).

Run from anywhere with the Paper_Knee scripts on PYTHONPATH:

    python multi_partition_cv_DE.py --partitions 5 --seeds 5 --alpha 0.05

ETA on GTX 970M: ~10 s per (seed,fold) training -> P=5, K=5 ~= 1 hour.
K=5 matches the paper's recommended ensemble size (Sec 5.4: K=5 ~ K=10 within noise).
Output: results/multi_partition_cv_DE.csv
"""
import os, sys, argparse, pickle, csv, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from config import DEVICE, RESULTS_DIR, EARLY_CYCLE_COUNTS   # noqa: E402
from features import build_feature_matrix, normalize_features  # noqa: E402
from models import create_model                                # noqa: E402
from train import train_pinn_knee                              # noqa: E402

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')


def kfold_split(cells, n_folds, seed):
    """Identical structure to deep_ensemble_with_preds.py, seed is a parameter."""
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


def _predict_raw(model, X):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return np.expm1(model(Xt).cpu().numpy().squeeze())


def de_predict(X_tr, y_tr, train_cells, ne, X_te, X_ca, y_ca, k_seeds):
    """Deep Ensemble PINN-Knee mean prediction over k_seeds (same as
    deep_ensemble_with_preds.py: cal set used as early-stopping val set)."""
    te = np.zeros((k_seeds, X_te.shape[0]))
    ca = np.zeros((k_seeds, X_ca.shape[0]))
    yv = y_ca if len(y_ca) > 0 else None
    for si in range(k_seeds):
        torch.manual_seed(si); np.random.seed(si)
        model = create_model('PINN_Knee', Q0=1.1, n_features=X_tr.shape[1], device=DEVICE)
        model, _ = train_pinn_knee(model, X_tr, y_tr, train_cells, ne,
                                   X_val=X_ca, y_val=yv,
                                   use_log_target=True, verbose=False)
        te[si] = _predict_raw(model, X_te)
        ca[si] = _predict_raw(model, X_ca)
    return te.mean(0), ca.mean(0)


def run_partition(cells, ne, partition_seed, n_folds, alpha, k_seeds, tag=''):
    folds = kfold_split(cells, n_folds, partition_seed)
    abs_err, widths, covered, ntot = [], [], 0, 0
    for fi, (tr, ca, te) in enumerate(folds):
        X_tr, y_tr, _, _ = build_feature_matrix(tr, ne)
        X_ca, y_ca, _, _ = build_feature_matrix(ca, ne)
        X_te, y_te, _, _ = build_feature_matrix(te, ne)
        if len(y_te) == 0 or len(y_tr) == 0:
            continue
        X_trn, X_ten, X_can, _ = normalize_features(X_tr, X_te, X_ca)
        pred_te, pred_ca = de_predict(X_trn, y_tr, tr, ne, X_ten, X_can, y_ca, k_seeds)
        scores = np.abs(y_ca - pred_ca); n = len(scores)
        k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        q = np.sort(scores)[k - 1]
        lo, hi = pred_te - q, pred_te + q
        covered += int(np.sum((y_te >= lo) & (y_te <= hi))); ntot += len(y_te)
        widths.extend((hi - lo).tolist()); abs_err.extend(np.abs(pred_te - y_te).tolist())
        print(f'    {tag} fold {fi}: fold_MAE={np.mean(np.abs(pred_te-y_te)):.0f}', flush=True)
    return float(np.mean(abs_err)), covered / max(ntot, 1), float(np.mean(widths))


def ci95(x):
    from scipy.stats import t as _t
    x = np.asarray(x, float); n = len(x); m = x.mean(); sd = x.std(ddof=1)
    crit = float(_t.ppf(0.975, n - 1)) if n > 1 else 0.0
    half = crit * sd / np.sqrt(n)
    return m, sd, m - half, m + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partitions', type=int, default=5)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=5)       # ensemble size K
    ap.add_argument('--alpha', type=float, default=0.05)  # 95% intervals, matches paper
    a = ap.parse_args()

    with open(CACHE_PATH, 'rb') as f:
        cells = pickle.load(f)
    cells = [c for c in cells if c.get('has_knee_point', False)
             and c.get('knee_cycle') is not None]
    print(f'cells={len(cells)}  partitions={a.partitions}  folds={a.folds}  K={a.seeds}  '
          f'alpha={a.alpha}  device={DEVICE}', flush=True)

    t0 = time.time(); rows = []
    for ne in EARLY_CYCLE_COUNTS:
        per = []
        for P in range(a.partitions):
            mae, picp, mpiw = run_partition(cells, ne, P, a.folds, a.alpha, a.seeds,
                                            tag=f'ne={ne} P={P}')
            per.append((mae, picp, mpiw))
            print(f'  ne={ne} partition={P:2d}: MAE={mae:6.1f}  PICP={picp:.3f}  '
                  f'MPIW={mpiw:6.0f}  (elapsed {(time.time()-t0)/60:.1f}m)', flush=True)
        for j, name in enumerate(['MAE', 'PICP', 'MPIW']):
            arr = np.array([p[j] for p in per])
            m, sd, lo, hi = ci95(arr)
            print(f'  >> ne={ne} {name}: mean={m:.3f}  SD={sd:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]  '
                  f'range=[{arr.min():.3f}, {arr.max():.3f}]', flush=True)
            rows.append([ne, name, a.partitions, round(m, 3), round(sd, 3),
                         round(lo, 3), round(hi, 3), round(float(arr.min()), 3),
                         round(float(arr.max()), 3)])

    out = os.path.join(RESULTS_DIR, 'multi_partition_cv_DE.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['budget', 'metric', 'n_partitions', 'mean', 'sd', 'ci_lo', 'ci_hi', 'min', 'max'])
        w.writerows(rows)
    print('wrote', out, f'  total {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
