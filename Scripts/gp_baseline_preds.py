"""
Option C: Gaussian Process baseline with predictive variance.

Classical UQ benchmark — GP gives predictive mean + std natively.
Scope: 10 seeds × 3 n_early × 5 folds = 150 runs (fast on CPU).
"""
import os, sys, time, pickle, traceback
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

from config import RESULTS_DIR, EARLY_CYCLE_COUNTS
from features import build_feature_matrix, normalize_features

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')
OUT_DIR = os.path.join(RESULTS_DIR, 'gp_preds')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(10))
N_FOLDS = 5


def kfold_split(cells, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(cells))
    fold_size = len(cells) // n_folds
    splits = []
    for f in range(n_folds):
        test_idx = idx[f*fold_size:(f+1)*fold_size]
        remaining = np.concatenate([idx[:f*fold_size], idx[(f+1)*fold_size:]])
        n_cal = max(1, len(remaining) // 4)
        splits.append(([cells[i] for i in remaining[n_cal:]],
                       [cells[i] for i in remaining[:n_cal]],
                       [cells[i] for i in test_idx]))
    return splits


def main():
    print('=' * 70)
    print('  GAUSSIAN PROCESS baseline — Option C for Paper 4')
    print('=' * 70)

    with open(CACHE_PATH, 'rb') as f:
        cells = pickle.load(f)
    cells = [c for c in cells if c.get('has_knee_point', False)
             and c.get('knee_cycle') is not None]
    print(f'  Cells: {len(cells)}')

    folds = kfold_split(cells, N_FOLDS, seed=42)
    total = len(EARLY_CYCLE_COUNTS) * len(SEEDS) * N_FOLDS
    print(f'  Runs: {total}')
    print('-' * 70)

    t0 = time.time()
    run_i = 0
    for ne in EARLY_CYCLE_COUNTS:
        for fold_idx in range(N_FOLDS):
            out_file = os.path.join(OUT_DIR, f'preds_ne{ne}_f{fold_idx}.npz')
            if os.path.exists(out_file):
                run_i += len(SEEDS)
                print(f'[SKIP] ne={ne} f={fold_idx}')
                continue

            train_cells, cal_cells, test_cells = folds[fold_idx]
            X_tr, y_tr, _, _ = build_feature_matrix(train_cells, ne)
            X_ca, y_ca, _, _ = build_feature_matrix(cal_cells, ne)
            X_te, y_te, _, te_valid = build_feature_matrix(test_cells, ne)
            if len(y_te) == 0: continue
            X_tr_n, X_te_n, X_ca_n, _ = normalize_features(X_tr, X_te, X_ca)
            test_names = [test_cells[i]['name'] for i in te_valid]

            # GP predictions across seeds (different kernel init)
            preds_all = np.zeros((len(SEEDS), len(y_te)))
            stds_all = np.zeros((len(SEEDS), len(y_te)))
            cal_preds_all = np.zeros((len(SEEDS), len(y_ca)))

            for si, seed in enumerate(SEEDS):
                run_i += 1
                elapsed = time.time() - t0
                eta = (elapsed / max(run_i - 1, 1)) * (total - run_i + 1) if run_i > 1 else 0
                print(f'[{run_i:3d}/{total}] ne={ne} f={fold_idx} s={seed}  '
                      f'ETA={int(eta//60):d}m{int(eta%60):02d}s', end=' ', flush=True)
                t1 = time.time()
                try:
                    # Log-transform targets for stability
                    y_tr_log = np.log1p(y_tr)
                    # Kernel: constant * RBF + white noise
                    kernel = ConstantKernel(1.0, (0.1, 10)) * RBF(1.0, (0.1, 100)) + \
                             WhiteKernel(1e-3, (1e-5, 1.0))
                    gp = GaussianProcessRegressor(
                        kernel=kernel, n_restarts_optimizer=2,
                        random_state=seed, alpha=1e-6,
                    )
                    gp.fit(X_tr_n, y_tr_log)
                    # Predict mean + std
                    mu_log, std_log = gp.predict(X_te_n, return_std=True)
                    cal_mu_log = gp.predict(X_ca_n)
                    preds_all[si] = np.expm1(mu_log)
                    stds_all[si] = np.expm1(mu_log + std_log) - np.expm1(mu_log)
                    cal_preds_all[si] = np.expm1(cal_mu_log)
                    mae = np.mean(np.abs(preds_all[si] - y_te))
                    dt = time.time() - t1
                    print(f' MAE={mae:5.0f}  σ_mean={np.mean(stds_all[si]):5.0f} ({dt:.0f}s)')
                except Exception as e:
                    print(f' ERROR: {e}')
                    traceback.print_exc()

            np.savez_compressed(
                out_file,
                y_true=y_te,
                preds_all=preds_all,
                stds_all=stds_all,   # GP native predictive std
                cell_names=np.array(test_names),
                y_cal=y_ca,
                cal_preds_all=cal_preds_all,
            )

    total_min = (time.time() - t0) / 60
    print('-' * 70)
    print(f'DONE in {total_min:.1f} min')


if __name__ == '__main__':
    main()
