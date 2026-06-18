"""SHAP failure analysis (V5 Minor 9 fix): feature attribution on top-error cells.

Loads the Deep Ensemble PINN-Knee predictions, identifies the top-K cells with
largest absolute residual at n_early=150, and computes SHAP values via
KernelExplainer (model-agnostic) using the trained ensemble's mean prediction.

Outputs:
  - Metrics/revision/shap_topcell_attributions.csv (cell × feature SHAP values)
  - Metrics/revision/shap_global_importance.csv (feature × mean |SHAP|)
"""
import sys, io, os, pickle, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path
import shap
import torch

PAPER4 = Path(__file__).resolve().parent.parent
PAPER_KNEE = PAPER4.parent / 'Paper_Knee'
sys.path.insert(0, str(PAPER_KNEE / 'scripts'))
sys.path.insert(0, str(PAPER_KNEE / 'scripts' / '_analysis'))
from config import DEVICE, RESULTS_DIR
from features import build_feature_matrix, normalize_features
from models import PINN_Knee
from train import train_pinn_knee

CACHE_PATH = Path(RESULTS_DIR) / '_severson_cache.pkl'
PRED_DIR = Path(RESULTS_DIR) / 'deep_ensemble_preds'
OUT_DIR = PAPER4 / 'Metrics' / 'revision'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NE = 150
N_FOLDS = 5
TOP_K_CELLS = 10  # top-K largest-residual test cells across all folds
N_SHAP_SAMPLES = 100  # KernelExplainer perturbations per cell

FEATURE_NAMES = [f'feat_{i:02d}' for i in range(24)]  # 24 features (verified from build_feature_matrix output dim)


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


def load_residuals():
    """Load DE residuals across all folds at n_early=150 to find top-error cells."""
    rows = []
    for f in range(N_FOLDS):
        d = np.load(PRED_DIR / f'preds_ne{NE}_f{f}.npz', allow_pickle=True)
        y_true = d['y_true']
        preds = d['preds_all'].mean(axis=0)
        for cell, yt, yp in zip(d['cell_names'], y_true, preds):
            rows.append({'cell': str(cell), 'fold': f, 'y_true': float(yt), 'y_pred': float(yp),
                         'abs_err': abs(float(yt - yp))})
    return pd.DataFrame(rows).sort_values('abs_err', ascending=False)


def predict_ensemble_mean(models, X_n):
    """Return mean prediction across ensemble for SHAP."""
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            yhat_log = m(torch.tensor(X_n, dtype=torch.float32, device=DEVICE)).cpu().numpy().squeeze()
            if yhat_log.ndim == 0:
                yhat_log = yhat_log.reshape(1)
            preds.append(np.expm1(yhat_log))
    return np.mean(preds, axis=0)


def main():
    print('SHAP failure analysis — V5 Minor 9')
    with open(CACHE_PATH, 'rb') as f:
        cells = pickle.load(f)
    cells = [c for c in cells if c.get('has_knee_point', False) and c.get('knee_cycle') is not None]
    folds = kfold_split(cells, N_FOLDS)
    cell_index = {c['name']: c for c in cells}

    res = load_residuals()
    print(f'\nTop-{TOP_K_CELLS} highest-residual cells at n_early={NE}:')
    print(res.head(TOP_K_CELLS).to_string(index=False, float_format='%.1f'))

    top_cells_df = res.head(TOP_K_CELLS)

    # Re-train one DE per fold (3 seeds for SHAP background) and run SHAP on top-cells in that fold
    shap_rows = []
    background_per_fold = {}

    for fold_idx in range(N_FOLDS):
        train_cells, cal_cells, test_cells = folds[fold_idx]
        X_tr, y_tr, _, _ = build_feature_matrix(train_cells, NE)
        X_ca, y_ca, _, _ = build_feature_matrix(cal_cells, NE)
        X_te, y_te, _, te_valid = build_feature_matrix(test_cells, NE)
        if len(y_te) == 0: continue
        X_tr_n, X_te_n, X_ca_n, _ = normalize_features(X_tr, X_te, X_ca)
        test_names = [test_cells[i]['name'] for i in te_valid]

        # Top-error cells in this fold
        cells_in_fold = top_cells_df[top_cells_df.fold == fold_idx]
        if cells_in_fold.empty: continue

        print(f'\n=== Fold {fold_idx}: {len(cells_in_fold)} top-error cells, training 3-seed DE for SHAP ===')

        # Train mini ensemble (3 seeds) for SHAP
        models = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed); np.random.seed(seed)
            m = PINN_Knee(n_features=X_tr_n.shape[1]).to(DEVICE)
            m, _ = train_pinn_knee(m, X_tr_n, y_tr, train_cells, NE,
                                   X_val=X_ca_n, y_val=y_ca, use_log_target=True, verbose=False)
            models.append(m)

        # SHAP background = sample of training set
        bg = X_tr_n[np.random.RandomState(42).choice(len(X_tr_n), min(50, len(X_tr_n)), replace=False)]
        explainer = shap.KernelExplainer(lambda X: predict_ensemble_mean(models, X), bg, silent=True)

        for _, row in cells_in_fold.iterrows():
            cell_name = row['cell']
            if cell_name not in test_names:
                continue
            ti = test_names.index(cell_name)
            x = X_te_n[ti:ti+1]
            shap_vals = explainer.shap_values(x, nsamples=N_SHAP_SAMPLES, silent=True)
            shap_vec = np.asarray(shap_vals).flatten()
            entry = {'cell': cell_name, 'fold': fold_idx, 'y_true': row.y_true, 'y_pred': row.y_pred,
                     'abs_err': row.abs_err}
            n_feat_actual = len(shap_vec)
            local_names = FEATURE_NAMES[:n_feat_actual] if n_feat_actual <= len(FEATURE_NAMES) else \
                          [f'feat_{i:02d}' for i in range(n_feat_actual)]
            for fname, sv in zip(local_names, shap_vec):
                entry[f'SHAP_{fname}'] = float(sv)
            shap_rows.append(entry)
            top_idx = np.argsort(-np.abs(shap_vec))[:3]
            print(f'  {cell_name:20s}  AE={row.abs_err:5.1f}  '
                  f'top-3 SHAP: ' + ', '.join(
                      f'{local_names[i]}={shap_vec[i]:+5.1f}' for i in top_idx))

    df = pd.DataFrame(shap_rows)
    df.to_csv(OUT_DIR / 'shap_topcell_attributions.csv', index=False)
    print(f'\nWrote {OUT_DIR / "shap_topcell_attributions.csv"} ({len(df)} cell rows)')

    # Global importance: mean |SHAP| per feature
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    global_imp = df[shap_cols].abs().mean().sort_values(ascending=False)
    global_df = pd.DataFrame({
        'feature': [c.replace('SHAP_', '') for c in global_imp.index],
        'mean_abs_shap': global_imp.values,
    })
    global_df.to_csv(OUT_DIR / 'shap_global_importance.csv', index=False)
    print(f'\nGlobal SHAP importance (top failures):')
    print(global_df.to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
    main()
