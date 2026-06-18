"""
Re-train PINN-Knee 10 seeds AND save per-cell predictions + conformal intervals.

Uses the proper train_pinn_knee() from train.py (composite loss, log-space,
early stopping).  Saves per-cell predictions per (n_early, fold) aggregated
across 10 seeds.

Scope: 10 seeds × 3 n_early × 5 folds = 150 runs
ETA: ~25 min on GTX 970M
Output: results/deep_ensemble_preds/preds_ne{NE}_f{F}.npz
"""
import os
import sys
import time
import pickle
import traceback
import numpy as np
import torch

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, '_analysis'))

from config import DEVICE, RESULTS_DIR, EARLY_CYCLE_COUNTS
from features import build_feature_matrix, normalize_features
from models import create_model
from train import train_pinn_knee

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')
OUT_DIR = os.path.join(RESULTS_DIR, 'deep_ensemble_preds')
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
        cal_idx = remaining[:n_cal]
        train_idx = remaining[n_cal:]
        splits.append(([cells[i] for i in train_idx],
                       [cells[i] for i in cal_idx],
                       [cells[i] for i in test_idx]))
    return splits


def predict_raw(model, X):
    """Predict raw cycle (model outputs log-space, convert back)."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_log = model(X_t).cpu().numpy().squeeze()
        return np.expm1(y_log)


def main():
    print('=' * 70)
    print('  DEEP ENSEMBLE WITH PER-CELL PREDICTIONS — for Paper 4 UQ metrics')
    print('=' * 70)
    print(f'  Device: {DEVICE}')
    print(f'  Seeds: {SEEDS}')
    print(f'  n_early: {EARLY_CYCLE_COUNTS}')
    print(f'  Folds: {N_FOLDS}')

    with open(CACHE_PATH, 'rb') as f:
        cells = pickle.load(f)
    cells = [c for c in cells if c.get('has_knee_point', False)
             and c.get('knee_cycle') is not None]
    print(f'  Cells: {len(cells)}')

    folds = kfold_split(cells, N_FOLDS, seed=42)
    total = len(EARLY_CYCLE_COUNTS) * len(SEEDS) * N_FOLDS
    print(f'  Total runs: {total}')
    print('-' * 70)

    t0 = time.time()
    run_i = 0
    for ne in EARLY_CYCLE_COUNTS:
        for fold_idx in range(N_FOLDS):
            train_cells, cal_cells, test_cells = folds[fold_idx]
            X_tr, y_tr, _, _ = build_feature_matrix(train_cells, ne)
            X_ca, y_ca, _, _ = build_feature_matrix(cal_cells, ne)
            X_te, y_te, _, te_valid = build_feature_matrix(test_cells, ne)
            if len(y_te) == 0 or len(y_tr) == 0:
                continue
            X_tr_n, X_te_n, X_ca_n, _ = normalize_features(X_tr, X_te, X_ca)
            test_cell_names = [test_cells[i]['name'] for i in te_valid]

            preds_all = np.zeros((len(SEEDS), len(y_te)))
            cal_preds_all = np.zeros((len(SEEDS), len(y_ca))) if len(y_ca) > 0 else None

            for si, seed in enumerate(SEEDS):
                run_i += 1
                elapsed = time.time() - t0
                avg = elapsed / max(run_i - 1, 1) if run_i > 1 else 0
                eta = avg * (total - run_i + 1)
                print(f'[{run_i:3d}/{total}] ne={ne} fold={fold_idx} seed={seed}  '
                      f'ETA={int(eta//60):d}m{int(eta%60):02d}s', end=' ', flush=True)
                t1 = time.time()
                try:
                    torch.manual_seed(seed); np.random.seed(seed)
                    model = create_model('PINN_Knee', Q0=1.1,
                                         n_features=X_tr_n.shape[1], device=DEVICE)
                    # train_pinn_knee expects RAW y (not log); does log internally
                    model, _ = train_pinn_knee(
                        model, X_tr_n, y_tr, train_cells, ne,
                        X_val=X_ca_n, y_val=y_ca if len(y_ca) > 0 else None,
                        use_log_target=True, verbose=False,
                    )
                    preds_all[si] = predict_raw(model, X_te_n)
                    if cal_preds_all is not None:
                        cal_preds_all[si] = predict_raw(model, X_ca_n)
                    dt = time.time() - t1
                    mae = np.mean(np.abs(preds_all[si] - y_te))
                    print(f' MAE={mae:6.1f} ({dt:.0f}s)')
                except Exception as e:
                    print(f' ERROR: {e}')
                    traceback.print_exc()

            out_file = os.path.join(OUT_DIR, f'preds_ne{ne}_f{fold_idx}.npz')
            save_kwargs = dict(
                y_true=y_te,
                preds_all=preds_all,
                cell_names=np.array(test_cell_names),
                y_cal=y_ca if len(y_ca) > 0 else np.array([]),
            )
            if cal_preds_all is not None:
                save_kwargs['cal_preds_all'] = cal_preds_all
            np.savez_compressed(out_file, **save_kwargs)

    total_min = (time.time() - t0) / 60
    print('-' * 70)
    print(f'DONE in {total_min:.1f} min')
    print(f'Predictions saved to: {OUT_DIR}/preds_ne*_f*.npz')


if __name__ == '__main__':
    main()
