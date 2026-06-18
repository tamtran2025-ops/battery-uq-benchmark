"""
Train Ensemble_NN (5x Pure NN) with 10 seeds, save per-cell predictions.

For each (n_early, fold), run 10 seeds. Each seed trains 5-member ensemble.
Save per-cell predictions from each member for proper UQ analysis.

Scope: 10 seeds × 3 n_early × 5 folds × 5 members = 750 sub-trainings
ETA: ~15 min on GTX 970M (MLP is fast)
"""
import os, sys, time, pickle, traceback
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, '_analysis'))
sys.path.insert(0, os.path.join(SCRIPTS, '_baselines'))
EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENTS)

from config import DEVICE, RESULTS_DIR, EARLY_CYCLE_COUNTS
from features import build_feature_matrix, normalize_features
from models import Ensemble_NN_Member
from run_experiments import train_nn_model

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')
OUT_DIR = os.path.join(RESULTS_DIR, 'ensemble_nn_preds')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(10))
N_FOLDS = 5
ENSEMBLE_SIZE = 5  # 5 members per ensemble


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


def predict_nn(model, X, use_log_target=True):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_log = model(X_t).cpu().numpy().squeeze()
        return np.expm1(y_log) if use_log_target else y_log


def main():
    print('=' * 70)
    print('  ENSEMBLE_NN WITH PER-CELL PREDICTIONS — for Paper 4 UQ')
    print('=' * 70)
    print(f'  Seeds: {SEEDS}, Members per ensemble: {ENSEMBLE_SIZE}')

    with open(CACHE_PATH, 'rb') as f:
        cells = pickle.load(f)
    cells = [c for c in cells if c.get('has_knee_point', False)
             and c.get('knee_cycle') is not None]
    print(f'  Cells: {len(cells)}')

    folds = kfold_split(cells, N_FOLDS, seed=42)
    total = len(EARLY_CYCLE_COUNTS) * len(SEEDS) * N_FOLDS
    print(f'  Total (seeds × ne × folds): {total}')
    print('-' * 70)

    t0 = time.time()
    run_i = 0
    for ne in EARLY_CYCLE_COUNTS:
        for fold_idx in range(N_FOLDS):
            out_file = os.path.join(OUT_DIR, f'preds_ne{ne}_f{fold_idx}.npz')
            if os.path.exists(out_file):
                run_i += len(SEEDS)
                print(f'[SKIP] ne={ne} f={fold_idx} already done ({out_file})')
                continue
            train_cells, cal_cells, test_cells = folds[fold_idx]
            X_tr, y_tr, _, _ = build_feature_matrix(train_cells, ne)
            X_ca, y_ca, _, _ = build_feature_matrix(cal_cells, ne)
            X_te, y_te, _, te_valid = build_feature_matrix(test_cells, ne)
            if len(y_te) == 0: continue
            X_tr_n, X_te_n, X_ca_n, _ = normalize_features(X_tr, X_te, X_ca)
            test_names = [test_cells[i]['name'] for i in te_valid]

            # Log targets for NN
            y_tr_log = np.log1p(y_tr)
            y_ca_log = np.log1p(y_ca) if len(y_ca) > 0 else None

            # For each seed, train 5-member ensemble and average
            all_preds = np.zeros((len(SEEDS), len(y_te)))  # seed-averaged ensemble mean
            all_member_preds = np.zeros((len(SEEDS), ENSEMBLE_SIZE, len(y_te)))  # per-member
            all_cal_preds = np.zeros((len(SEEDS), len(y_ca))) if len(y_ca) > 0 else None

            for si, seed in enumerate(SEEDS):
                run_i += 1
                elapsed = time.time() - t0
                avg = elapsed / max(run_i - 1, 1) if run_i > 1 else 0
                eta = avg * (total - run_i + 1)
                print(f'[{run_i:3d}/{total}] ne={ne} f={fold_idx} s={seed}  '
                      f'ETA={int(eta//60):d}m{int(eta%60):02d}s', end=' ', flush=True)
                t1 = time.time()
                try:
                    member_preds = np.zeros((ENSEMBLE_SIZE, len(y_te)))
                    member_cal_preds = np.zeros((ENSEMBLE_SIZE, len(y_ca))) if len(y_ca) > 0 else None
                    for m in range(ENSEMBLE_SIZE):
                        member_seed = seed * 1000 + m * 17 + 3
                        torch.manual_seed(member_seed); np.random.seed(member_seed)
                        member = Ensemble_NN_Member(n_features=X_tr_n.shape[1]).to(DEVICE)
                        member, _ = train_nn_model(
                            member, X_tr_n, y_tr_log,
                            X_val=X_ca_n, y_val=y_ca_log,
                            verbose=False,
                        )
                        member_preds[m] = predict_nn(member, X_te_n, use_log_target=True)
                        if member_cal_preds is not None:
                            member_cal_preds[m] = predict_nn(member, X_ca_n, use_log_target=True)
                    all_member_preds[si] = member_preds
                    all_preds[si] = member_preds.mean(axis=0)
                    if all_cal_preds is not None:
                        all_cal_preds[si] = member_cal_preds.mean(axis=0)
                    dt = time.time() - t1
                    mae = np.mean(np.abs(all_preds[si] - y_te))
                    print(f' MAE={mae:6.1f} ({dt:.0f}s)')
                except Exception as e:
                    print(f' ERROR: {e}')
                    traceback.print_exc()

            # Save
            np.savez_compressed(
                out_file,
                y_true=y_te,
                preds_all=all_preds,           # (n_seeds, n_test) - seed-averaged ensemble means
                member_preds=all_member_preds, # (n_seeds, n_members, n_test) - per-member
                cell_names=np.array(test_names),
                y_cal=y_ca if len(y_ca) > 0 else np.array([]),
                **({'cal_preds_all': all_cal_preds} if all_cal_preds is not None else {}),
            )

    total_min = (time.time() - t0) / 60
    print('-' * 70)
    print(f'DONE in {total_min:.1f} min')
    print(f'Predictions: {OUT_DIR}/preds_ne*_f*.npz')


if __name__ == '__main__':
    main()
