"""
Compute proper UQ metrics from Deep Ensemble per-cell predictions.

Reads results/deep_ensemble_preds/preds_ne{NE}_f{F}.npz and computes:
  - Point accuracy: MAE, MedAE, RMSE
  - Calibration: PICP at 95% (empirical coverage)
  - Sharpness: MPIW (mean prediction interval width)
  - Scoring rules: NLL (Gaussian), CRPS (Gaussian)

Two UQ methods compared:
  1. Ensemble variance (naive Gaussian) — σ² from seed disagreement
  2. Split conformal adjustment — calibration-set-based non-conformity

Output: results/uq_metrics_proper.csv + printed table.
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

from config import RESULTS_DIR, EARLY_CYCLE_COUNTS

PREDS_DIR = os.path.join(RESULTS_DIR, 'deep_ensemble_preds')
OUT_CSV = os.path.join(RESULTS_DIR, 'uq_metrics_proper.csv')
OUT_SUMMARY = os.path.join(RESULTS_DIR, 'uq_metrics_summary.txt')
ALPHA = 0.05  # 95% coverage


def gaussian_nll(y_true, mu, sigma):
    """Per-point Gaussian NLL. Returns mean NLL."""
    sigma = np.maximum(sigma, 1e-6)
    return np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) +
                    0.5 * ((y_true - mu) / sigma) ** 2)


def crps_gaussian(y_true, mu, sigma):
    """CRPS under Gaussian predictive distribution."""
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-6)
    z = (y_true - mu) / sigma
    return np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))


def picp(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))


def mpiw(lower, upper):
    return np.mean(upper - lower)


def conformal_split_quantile(residuals, alpha=ALPHA):
    """Compute conformal quantile q_hat from calibration residuals."""
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(np.abs(residuals))[min(k - 1, n - 1)]


def process_fold(preds_file):
    """Compute UQ metrics for one fold."""
    d = np.load(preds_file)
    y_true = d['y_true']
    preds_all = d['preds_all']  # (n_seeds, n_test)
    y_cal = d['y_cal'] if 'y_cal' in d.files else np.array([])
    cal_preds_all = d.get('cal_preds_all', None)

    n_seeds, n_test = preds_all.shape
    mu = preds_all.mean(axis=0)
    sigma = preds_all.std(axis=0, ddof=1)
    sigma = np.maximum(sigma, 1.0)  # min std 1 cycle to avoid degenerate

    # 1. Gaussian intervals (z_α/2 = 1.96 for 95%)
    z = 1.96
    lo_gauss = mu - z * sigma
    hi_gauss = mu + z * sigma

    # 2. Split conformal on ensemble mean predictions
    cal_intervals = None
    if cal_preds_all is not None and len(y_cal) > 0:
        cal_mu = cal_preds_all.mean(axis=0)
        cal_residuals = np.abs(y_cal - cal_mu)
        q_hat = conformal_split_quantile(cal_residuals, alpha=ALPHA)
        lo_conf = mu - q_hat
        hi_conf = mu + q_hat
        cal_intervals = (lo_conf, hi_conf, q_hat)

    # 3. Metrics
    abs_err = np.abs(y_true - mu)
    metrics = {
        'n_test': n_test,
        'n_seeds': n_seeds,
        'MAE': float(np.mean(abs_err)),
        'MedAE': float(np.median(abs_err)),
        'RMSE': float(np.sqrt(np.mean((y_true - mu) ** 2))),
        'mean_sigma': float(np.mean(sigma)),
        'PICP_gauss': float(picp(y_true, lo_gauss, hi_gauss)),
        'MPIW_gauss': float(mpiw(lo_gauss, hi_gauss)),
        'NLL_gauss': float(gaussian_nll(y_true, mu, sigma)),
        'CRPS_gauss': float(crps_gaussian(y_true, mu, sigma)),
    }
    if cal_intervals is not None:
        lo_conf, hi_conf, q_hat = cal_intervals
        metrics.update({
            'q_hat': float(q_hat),
            'PICP_conformal': float(picp(y_true, lo_conf, hi_conf)),
            'MPIW_conformal': float(mpiw(lo_conf, hi_conf)),
        })
    return metrics


def main():
    print('=' * 75)
    print('  PROPER UQ METRICS from Deep Ensemble per-cell predictions')
    print('=' * 75)

    rows = []
    for ne in EARLY_CYCLE_COUNTS:
        fold_rows = []
        for f in range(5):
            fp = os.path.join(PREDS_DIR, f'preds_ne{ne}_f{f}.npz')
            if not os.path.exists(fp):
                print(f'  MISSING: {fp}')
                continue
            m = process_fold(fp)
            m['n_early'] = ne
            m['fold'] = f
            fold_rows.append(m)
            rows.append(m)

        if fold_rows:
            # Aggregate across folds
            agg = {k: np.mean([r[k] for r in fold_rows if k in r])
                   for k in fold_rows[0] if isinstance(fold_rows[0][k], (int, float))}
            print(f'\nn_early={ne} (avg across {len(fold_rows)} folds):')
            print(f'  MAE      = {agg.get("MAE", 0):.1f} cycles')
            print(f'  MedAE    = {agg.get("MedAE", 0):.1f} cycles')
            print(f'  RMSE     = {agg.get("RMSE", 0):.1f} cycles')
            print(f'  σ (seed) = {agg.get("mean_sigma", 0):.1f} cycles')
            print(f'  Gaussian intervals (ensemble variance):')
            print(f'    PICP   = {agg.get("PICP_gauss", 0):.3f} (target 0.95)')
            print(f'    MPIW   = {agg.get("MPIW_gauss", 0):.1f} cycles')
            print(f'    NLL    = {agg.get("NLL_gauss", 0):.3f}')
            print(f'    CRPS   = {agg.get("CRPS_gauss", 0):.1f} cycles')
            if 'PICP_conformal' in agg:
                print(f'  Conformal intervals (calibration-based):')
                print(f'    PICP   = {agg.get("PICP_conformal", 0):.3f} (target 0.95)')
                print(f'    MPIW   = {agg.get("MPIW_conformal", 0):.1f} cycles')
                print(f'    q̂    = {agg.get("q_hat", 0):.1f} cycles')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved per-fold metrics: {OUT_CSV}')

    # Summary text
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write('UQ METRICS SUMMARY — Paper 4 Table 1 source\n')
        f.write('=' * 70 + '\n\n')
        for ne in EARLY_CYCLE_COUNTS:
            sub = df[df['n_early'] == ne]
            if len(sub) == 0: continue
            f.write(f'n_early={ne}:\n')
            for col in ['MAE', 'MedAE', 'RMSE', 'PICP_gauss', 'MPIW_gauss',
                        'NLL_gauss', 'CRPS_gauss', 'PICP_conformal', 'MPIW_conformal']:
                if col in sub.columns:
                    v = sub[col].mean()
                    s = sub[col].std()
                    f.write(f'  {col:<20}: {v:.3f} ± {s:.3f}\n')
            f.write('\n')
    print(f'Saved summary text: {OUT_SUMMARY}')


if __name__ == '__main__':
    main()
