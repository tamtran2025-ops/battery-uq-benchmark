"""
Option F: Ensemble Size Ablation — does UQ quality improve with more seeds?

Uses existing Deep Ensemble PINN-Knee predictions (10 seeds).
Computes UQ metrics for ensemble sizes {2, 3, 5, 8, 10} per fold.
Offline only — no training needed.
"""
import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.stats import norm

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
from config import RESULTS_DIR, FIGURES_DIR, EARLY_CYCLE_COUNTS

PREDS_DIR = os.path.join(RESULTS_DIR, 'deep_ensemble_preds')
OUT_CSV = os.path.join(RESULTS_DIR, 'ensemble_size_ablation.csv')
ENSEMBLE_SIZES = [2, 3, 5, 8, 10]


def compute_metrics(y_true, preds_subset, y_cal=None, cal_preds=None):
    mu = preds_subset.mean(axis=0)
    sigma = np.maximum(preds_subset.std(axis=0, ddof=1), 1.0)
    z = 1.96
    lo = mu - z*sigma; hi = mu + z*sigma
    mae = float(np.mean(np.abs(y_true - mu)))
    picp = float(np.mean((y_true >= lo) & (y_true <= hi)))
    mpiw = float(np.mean(hi - lo))
    if y_cal is not None and cal_preds is not None:
        cal_mu = cal_preds.mean(axis=0)
        residuals = np.abs(y_cal - cal_mu)
        n = len(residuals)
        k = int(np.ceil((n+1) * 0.95))
        q = np.sort(residuals)[min(k-1, n-1)]
        picp_c = float(np.mean((y_true >= mu - q) & (y_true <= mu + q)))
        mpiw_c = float(2 * q)
        return mae, picp, mpiw, picp_c, mpiw_c
    return mae, picp, mpiw, np.nan, np.nan


def main():
    print('=' * 70)
    print('  ENSEMBLE SIZE ABLATION — Deep Ensemble PINN-Knee')
    print('=' * 70)
    print(f'  Sizes: {ENSEMBLE_SIZES}')

    rows = []
    for ne in EARLY_CYCLE_COUNTS:
        fold_files = sorted(glob.glob(os.path.join(PREDS_DIR, f'preds_ne{ne}_f*.npz')))
        fold_files = [f for f in fold_files if '_s' not in os.path.basename(f).replace('preds_', '')]
        for fold_idx, fp in enumerate(fold_files):
            d = np.load(fp)
            y = d['y_true']
            preds = d['preds_all']  # (10, N)
            y_cal = d['y_cal'] if 'y_cal' in d.files else None
            cal_preds = d.get('cal_preds_all', None)

            for size in ENSEMBLE_SIZES:
                if size > preds.shape[0]:
                    continue
                # Use first N seeds
                mae, picp, mpiw, picp_c, mpiw_c = compute_metrics(
                    y, preds[:size], y_cal, cal_preds[:size] if cal_preds is not None else None)
                rows.append({
                    'n_early': ne, 'fold': fold_idx, 'ensemble_size': size,
                    'MAE': mae, 'PICP_gauss': picp, 'MPIW_gauss': mpiw,
                    'PICP_conformal': picp_c, 'MPIW_conformal': mpiw_c,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')
    print()
    print('=' * 70)
    print('  ABLATION SUMMARY (mean over 5 folds)')
    print('=' * 70)
    for ne in EARLY_CYCLE_COUNTS:
        print(f'\nn_early={ne}:')
        print(f'  {"size":>6s} {"MAE":>7s} {"PICP":>7s} {"MPIW":>7s} {"PICP_c":>8s} {"MPIW_c":>8s}')
        for size in ENSEMBLE_SIZES:
            sub = df[(df.n_early == ne) & (df.ensemble_size == size)]
            if len(sub) == 0: continue
            print(f'  {size:>6d} {sub.MAE.mean():>7.1f} {sub.PICP_gauss.mean():>7.3f} '
                  f'{sub.MPIW_gauss.mean():>7.0f} {sub.PICP_conformal.mean():>8.3f} '
                  f'{sub.MPIW_conformal.mean():>8.0f}')


if __name__ == '__main__':
    main()
