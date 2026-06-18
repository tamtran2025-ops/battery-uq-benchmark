"""Build per-cell prediction CSV for all 12 methods × 3 budgets.

Combines predictions from:
  Paper4_UQ_Comparison/Predictions/  (7 methods)
  Paper_Knee/results/                 (5 additional: bootstrap, jackknife+, sngp, laplace, combined_uq, hetero_v2, cqr_v2)
"""
import numpy as np
import pandas as pd
from pathlib import Path

PAPER4 = Path(__file__).resolve().parent.parent
PAPER_KNEE_RESULTS = PAPER4.parent / 'Paper_Knee' / 'results'

# Map (method-name-in-csv, source-dir) for all 12
METHOD_SOURCES = [
    ('Deep_Ensemble_PINN_Knee',      PAPER4 / 'Predictions' / 'deep_ensemble_preds'),
    ('Ensemble_NN',                  PAPER4 / 'Predictions' / 'ensemble_nn_preds'),
    ('Bayesian_LSTM',                PAPER4 / 'Predictions' / 'bayesian_lstm_preds'),
    ('Gaussian_Process',             PAPER_KNEE_RESULTS / 'gp_preds_v2'),  # V5 Minor 8: n_restarts=10
    ('CQR_PINN_Knee',                PAPER4 / 'Predictions' / 'cqr_pinn_preds'),
    ('Heteroscedastic_MLP_v2',       PAPER_KNEE_RESULTS / 'hetero_preds_v2'),
    ('CQR_MLP_v2',                   PAPER_KNEE_RESULTS / 'cqr_preds_v2'),
    ('Bootstrap_PINN_Knee',          PAPER_KNEE_RESULTS / 'bootstrap_preds'),
    ('Jackknife_Plus_PINN_Knee',     PAPER_KNEE_RESULTS / 'jackknife_plus_preds'),
    ('Combined_UQ_PINN_Knee',        PAPER_KNEE_RESULTS / 'combined_uq_preds'),
    ('SNGP',                         PAPER_KNEE_RESULTS / 'sngp_preds'),
    ('Last_Layer_Laplace',           PAPER_KNEE_RESULTS / 'laplace_preds'),
    ('NGBoost',                      PAPER_KNEE_RESULTS / 'ngboost_preds'),
    ('Hyper_Deep_Ensemble',          PAPER_KNEE_RESULTS / 'hyper_deep_ensemble_preds'),
]

BUDGETS = [50, 100, 150]
FOLDS = [0, 1, 2, 3, 4]


def load_method(method_name, src_dir):
    """Load all (budget, fold) predictions for a method, return DataFrame."""
    rows = []
    for ne in BUDGETS:
        for f in FOLDS:
            p = src_dir / f'preds_ne{ne}_f{f}.npz'
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            y_true = d['y_true']
            cell_names = d['cell_names']
            # Standard ensemble mean prediction
            if 'preds_all' in d.files:
                preds = np.asarray(d['preds_all'])
                if preds.ndim == 2:
                    y_pred = preds.mean(axis=0)
                else:
                    y_pred = preds
            elif 'full_pred' in d.files:
                y_pred = np.asarray(d['full_pred'])
            else:
                # Fallback: skip
                continue
            for cell, yt, yp in zip(cell_names, y_true, y_pred):
                ae = float(abs(yt - yp))
                ape = ae / float(yt) * 100 if yt > 0 else np.nan
                rows.append({
                    'method': method_name,
                    'n_early': ne,
                    'fold': f,
                    'cell': str(cell),
                    'y_true': float(yt),
                    'y_pred': float(yp),
                    'abs_err': ae,
                    'ape_%': ape,
                })
    return pd.DataFrame(rows)


def main():
    all_dfs = []
    for method, src in METHOD_SOURCES:
        df = load_method(method, src)
        n = len(df)
        print(f'{method:35s}  {n:5d} rows  (src: {src.relative_to(src.parents[2]) if len(src.parts)>2 else src})')
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    out = PAPER4 / 'Metrics' / 'revision' / 'all_predictions_per_cell_v3.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f'\nWrote {out} ({len(combined)} rows)')

    # Summary stats
    print('\nPer (method, n_early) cell counts:')
    counts = combined.groupby(['method', 'n_early']).size().unstack(fill_value=0)
    print(counts.to_string())

    # MAPE/MAE per (method, n_early) — for sanity
    print('\nMAE / MAPE per (method, n_early):')
    agg = combined.groupby(['method', 'n_early']).agg(
        mae=('abs_err', 'mean'),
        mape=('ape_%', 'mean'),
        n_cells=('cell', 'count')
    )
    print(agg.round(2).to_string())


if __name__ == '__main__':
    main()
