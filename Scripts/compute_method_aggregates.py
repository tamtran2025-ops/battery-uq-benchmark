"""Compute MAE, MAPE, RMSE per method × budget for all 12 methods."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path

PAPER4 = Path(__file__).resolve().parent.parent
PER_CELL = PAPER4 / 'Metrics' / 'revision' / 'all_predictions_per_cell_v3.csv'
OUT_AGG = PAPER4 / 'Metrics' / 'revision' / 'aggregate_per_method_14.csv'
OUT_FOLD = PAPER4 / 'Metrics' / 'revision' / 'mae_mape_per_fold_14.csv'

# Display order (matches manuscript Table 1, 14 methods)
ORDER = [
    'Deep_Ensemble_PINN_Knee',
    'Hyper_Deep_Ensemble',
    'Combined_UQ_PINN_Knee',
    'Bootstrap_PINN_Knee',
    'Jackknife_Plus_PINN_Knee',
    'CQR_PINN_Knee',
    'SNGP',
    'Gaussian_Process',
    'NGBoost',
    'Heteroscedastic_MLP_v2',
    'CQR_MLP_v2',
    'Bayesian_LSTM',
    'Ensemble_NN',
    'Last_Layer_Laplace',
]


def main():
    df = pd.read_csv(PER_CELL)
    print(f'Loaded {len(df)} rows from {PER_CELL.name}')

    # Per-fold metrics (for fold-level statistics if needed)
    per_fold = df.groupby(['method', 'n_early', 'fold']).apply(
        lambda g: pd.Series({
            'n_cells': len(g),
            'mae': g.abs_err.mean(),
            'mape_%': g['ape_%'].mean(),
            'rmse': float(np.sqrt(((g.y_true - g.y_pred) ** 2).mean())),
            'median_ae': g.abs_err.median(),
        }), include_groups=False).reset_index()
    per_fold.to_csv(OUT_FOLD, index=False)
    print(f'Wrote {OUT_FOLD} ({len(per_fold)} fold rows)')

    # Aggregate (mean and std over 5 folds)
    agg_rows = []
    for method in ORDER:
        for ne in (50, 100, 150):
            sub = per_fold[(per_fold.method == method) & (per_fold.n_early == ne)]
            if sub.empty:
                continue
            agg_rows.append({
                'method': method,
                'n_early': ne,
                'mae_mean': sub.mae.mean(),
                'mae_std': sub.mae.std(ddof=1),
                'mape_mean': sub['mape_%'].mean(),
                'mape_std': sub['mape_%'].std(ddof=1),
                'rmse_mean': sub.rmse.mean(),
                'rmse_std': sub.rmse.std(ddof=1),
                'n_folds': len(sub),
            })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT_AGG, index=False)
    print(f'Wrote {OUT_AGG} ({len(agg)} method-budget rows)')

    # Pretty print table
    print('\n=== n_early=50 ===')
    sub = agg[agg.n_early == 50][['method', 'mae_mean', 'mae_std', 'mape_mean', 'mape_std', 'rmse_mean']]
    print(sub.to_string(index=False, float_format='%.2f'))

    print('\n=== n_early=100 ===')
    sub = agg[agg.n_early == 100][['method', 'mae_mean', 'mae_std', 'mape_mean', 'mape_std', 'rmse_mean']]
    print(sub.to_string(index=False, float_format='%.2f'))

    print('\n=== n_early=150 ===')
    sub = agg[agg.n_early == 150][['method', 'mae_mean', 'mae_std', 'mape_mean', 'mape_std', 'rmse_mean']]
    print(sub.to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
    main()
