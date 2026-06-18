"""
Compute UQ metrics for ALL methods with per-cell predictions.

Scans results/{method}_preds/ for .npz files, computes PICP/MPIW/NLL/CRPS per fold,
aggregates by (method, n_early). Output: one unified CSV.
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
from config import RESULTS_DIR, EARLY_CYCLE_COUNTS

METHODS = {
    'Deep_Ensemble_PINN_Knee': 'deep_ensemble_preds',
    'Ensemble_NN_5x_MLP': 'ensemble_nn_preds',
    'Bayesian_LSTM': 'bayesian_lstm_preds',
    'CQR_MLP': 'cqr_preds',
    'CQR_PINN_Knee_FLAGSHIP': 'cqr_pinn_preds',
    'Gaussian_Process': 'gp_preds',
    'Heteroscedastic_MLP': 'hetero_preds',
}

ALPHA = 0.05
OUT_CSV = os.path.join(RESULTS_DIR, 'all_uq_metrics.csv')


def gaussian_nll(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    return np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((y - mu) / sigma) ** 2)


def crps_gaussian(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))


def picp_mpiw(y, lo, hi):
    return np.mean((y >= lo) & (y <= hi)), np.mean(hi - lo)


def conformal_quantile(residuals, alpha):
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(np.abs(residuals))[min(k - 1, n - 1)]


def fold_metrics(npz_path):
    d = np.load(npz_path)
    y_true = d['y_true']
    preds_all = d['preds_all']  # (seeds, n_test)
    n_seeds = preds_all.shape[0]

    mu = preds_all.mean(axis=0)
    sigma = np.maximum(preds_all.std(axis=0, ddof=1), 1.0)
    err = np.abs(y_true - mu)

    out = {
        'n_seeds': n_seeds,
        'n_test': len(y_true),
        'MAE': float(np.mean(err)),
        'MedAE': float(np.median(err)),
        'RMSE': float(np.sqrt(np.mean((y_true - mu) ** 2))),
        'mean_sigma': float(np.mean(sigma)),
    }

    # Gaussian intervals
    z = 1.96
    gauss_picp, gauss_mpiw = picp_mpiw(y_true, mu - z * sigma, mu + z * sigma)
    out['PICP_gauss'] = float(gauss_picp)
    out['MPIW_gauss'] = float(gauss_mpiw)
    out['NLL_gauss'] = float(gaussian_nll(y_true, mu, sigma))
    out['CRPS_gauss'] = float(crps_gaussian(y_true, mu, sigma))

    # Conformal
    if 'cal_preds_all' in d.files and 'y_cal' in d.files and len(d['y_cal']) > 0:
        y_cal = d['y_cal']
        cal_mu = d['cal_preds_all'].mean(axis=0)
        q = conformal_quantile(y_cal - cal_mu, alpha=ALPHA)
        c_picp, c_mpiw = picp_mpiw(y_true, mu - q, mu + q)
        out['PICP_conformal'] = float(c_picp)
        out['MPIW_conformal'] = float(c_mpiw)
        out['q_hat'] = float(q)
    return out


def main():
    print('=' * 75)
    print('  ALL UQ METRICS — Deep Ensemble PINN-Knee vs Ensemble_NN (vs Bayesian_LSTM)')
    print('=' * 75)

    rows = []
    for method, folder in METHODS.items():
        path = os.path.join(RESULTS_DIR, folder)
        if not os.path.exists(path):
            print(f'  SKIP {method}: {path} not found')
            continue
        print(f'\n[{method}]')
        for ne in EARLY_CYCLE_COUNTS:
            fold_files = sorted(glob.glob(os.path.join(path, f'preds_ne{ne}_f*.npz')))
            # Skip seed-specific files like preds_s3_ne50_f0.npz
            fold_files = [f for f in fold_files
                          if '_s' not in os.path.basename(f).replace('preds_', '')]
            if not fold_files:
                continue
            fold_metrics_list = []
            for f in fold_files:
                try:
                    m = fold_metrics(f)
                    m['method'] = method
                    m['n_early'] = ne
                    m['fold'] = int(os.path.basename(f).split('_f')[-1].split('.')[0])
                    fold_metrics_list.append(m)
                    rows.append(m)
                except Exception as e:
                    print(f'  ERROR {f}: {e}')

            if fold_metrics_list:
                # Aggregate
                keys = ['MAE', 'MedAE', 'RMSE', 'mean_sigma',
                        'PICP_gauss', 'MPIW_gauss', 'NLL_gauss', 'CRPS_gauss',
                        'PICP_conformal', 'MPIW_conformal', 'q_hat']
                avg = {k: np.mean([m.get(k, np.nan) for m in fold_metrics_list])
                       for k in keys if k in fold_metrics_list[0]}
                print(f'  n_early={ne} ({len(fold_metrics_list)} folds):')
                print(f'    MAE={avg.get("MAE", 0):.1f}  σ={avg.get("mean_sigma", 0):.1f}')
                print(f'    Gauss:     PICP={avg.get("PICP_gauss", 0):.3f}  '
                      f'MPIW={avg.get("MPIW_gauss", 0):.0f}  '
                      f'NLL={avg.get("NLL_gauss", 0):.1f}  '
                      f'CRPS={avg.get("CRPS_gauss", 0):.1f}')
                if 'PICP_conformal' in avg and not np.isnan(avg['PICP_conformal']):
                    print(f'    Conformal: PICP={avg["PICP_conformal"]:.3f}  '
                          f'MPIW={avg["MPIW_conformal"]:.0f}  q̂={avg["q_hat"]:.0f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')


if __name__ == '__main__':
    main()
