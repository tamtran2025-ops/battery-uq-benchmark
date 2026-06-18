"""
Bayesian_LSTM with per-cell predictions (MC Dropout for UQ).

Uses LSTM backbone with permanent dropout (both training AND inference).
At inference, MC-sample predictions K times → aleatoric+epistemic uncertainty.

Scope: 10 seeds × 3 n_early × 5 folds = 150 training runs
Each seed: train once, then MC-sample K=30 predictions per cell.
ETA: ~50 min (LSTM is slow on 970M).
"""
import os, sys, time, pickle, traceback
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, '_analysis'))
EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENTS)

from config import DEVICE, RESULTS_DIR, EARLY_CYCLE_COUNTS
from features import build_feature_matrix, normalize_features
from models import create_model
from run_experiments import train_nn_model

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')
OUT_DIR = os.path.join(RESULTS_DIR, 'bayesian_lstm_preds')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(10))
N_FOLDS = 5
MC_SAMPLES = 30  # MC Dropout samples at inference


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


def mc_dropout_predict(model, X, K=30):
    """Enable dropout at inference, sample K predictions.

    Returns (K, N) array of log-space predictions.
    """
    model.train()  # Keep dropout active!
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    preds = []
    with torch.no_grad():
        for _ in range(K):
            p = model(X_t).cpu().numpy().squeeze()
            preds.append(p)
    return np.array(preds)


def main():
    print('=' * 70)
    print('  BAYESIAN_LSTM per-cell predictions — for Paper 4 UQ')
    print('=' * 70)
    print(f'  Seeds: {SEEDS}, MC samples: {MC_SAMPLES}')

    with open(CACHE_PATH, 'rb') as f:
        cells = pickle.load(f)
    cells = [c for c in cells if c.get('has_knee_point', False)
             and c.get('knee_cycle') is not None]
    print(f'  Cells: {len(cells)}')

    folds = kfold_split(cells, N_FOLDS, seed=42)
    total = len(EARLY_CYCLE_COUNTS) * len(SEEDS) * N_FOLDS
    print(f'  Total: {total}')
    print('-' * 70)

    t0 = time.time()
    run_i = 0
    for ne in EARLY_CYCLE_COUNTS:
        for fold_idx in range(N_FOLDS):
            train_cells, cal_cells, test_cells = folds[fold_idx]
            X_tr, y_tr, _, _ = build_feature_matrix(train_cells, ne)
            X_ca, y_ca, _, _ = build_feature_matrix(cal_cells, ne)
            X_te, y_te, _, te_valid = build_feature_matrix(test_cells, ne)
            if len(y_te) == 0: continue
            X_tr_n, X_te_n, X_ca_n, _ = normalize_features(X_tr, X_te, X_ca)
            test_names = [test_cells[i]['name'] for i in te_valid]

            y_tr_log = np.log1p(y_tr)
            y_ca_log = np.log1p(y_ca) if len(y_ca) > 0 else None

            # Store per-seed × per-MC-sample predictions
            # Shape: (seeds × MC, N) = K_total samples per cell
            all_preds = np.zeros((len(SEEDS) * MC_SAMPLES, len(y_te)))
            all_cal_preds = np.zeros((len(SEEDS) * MC_SAMPLES, len(y_ca))) if len(y_ca) > 0 else None

            for si, seed in enumerate(SEEDS):
                run_i += 1
                elapsed = time.time() - t0
                avg = elapsed / max(run_i - 1, 1) if run_i > 1 else 0
                eta = avg * (total - run_i + 1)
                print(f'[{run_i:3d}/{total}] ne={ne} f={fold_idx} s={seed}  '
                      f'ETA={int(eta//60):d}m{int(eta%60):02d}s', end=' ', flush=True)
                t1 = time.time()
                try:
                    torch.manual_seed(seed); np.random.seed(seed)
                    model = create_model('Bayesian_LSTM', Q0=1.1,
                                         n_features=X_tr_n.shape[1], device=DEVICE)
                    model, _ = train_nn_model(
                        model, X_tr_n, y_tr_log,
                        X_val=X_ca_n, y_val=y_ca_log,
                        verbose=False,
                    )
                    # MC Dropout sampling
                    mc_te = mc_dropout_predict(model, X_te_n, K=MC_SAMPLES)
                    mc_ca = mc_dropout_predict(model, X_ca_n, K=MC_SAMPLES) if len(y_ca) > 0 else None
                    # Convert back to raw cycles
                    mc_te_raw = np.expm1(mc_te)
                    mc_ca_raw = np.expm1(mc_ca) if mc_ca is not None else None
                    # Store in the (seed block × MC) layout
                    for k in range(MC_SAMPLES):
                        all_preds[si * MC_SAMPLES + k] = mc_te_raw[k]
                        if mc_ca_raw is not None:
                            all_cal_preds[si * MC_SAMPLES + k] = mc_ca_raw[k]
                    dt = time.time() - t1
                    mae = np.mean(np.abs(mc_te_raw.mean(axis=0) - y_te))
                    print(f' MAE={mae:5.0f} ({dt:.0f}s)')
                except Exception as e:
                    print(f' ERROR: {e}')
                    traceback.print_exc()

            # Save
            np.savez_compressed(
                os.path.join(OUT_DIR, f'preds_ne{ne}_f{fold_idx}.npz'),
                y_true=y_te,
                preds_all=all_preds,                   # (seeds × MC, N_test)
                cell_names=np.array(test_names),
                y_cal=y_ca if len(y_ca) > 0 else np.array([]),
                **({'cal_preds_all': all_cal_preds} if all_cal_preds is not None else {}),
            )

    total_min = (time.time() - t0) / 60
    print('-' * 70)
    print(f'DONE in {total_min:.1f} min')


if __name__ == '__main__':
    main()
