"""
Option B: Heteroscedastic PINN-Knee (learn aleatoric uncertainty directly).

Architecture: copy PINN-Knee physics head, add separate log-variance head.
Loss: Gaussian NLL (proper scoring rule) instead of MSE.

Scope: 10 seeds × 3 n_early × 5 folds = 150 runs.
Complements Deep Ensemble (epistemic) with direct aleatoric estimation.
"""
import os, sys, time, pickle, traceback
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

from config import (DEVICE, RESULTS_DIR, EARLY_CYCLE_COUNTS,
                     LEARNING_RATE, WEIGHT_DECAY, N_EPOCHS)
from features import build_feature_matrix, normalize_features

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')
OUT_DIR = os.path.join(RESULTS_DIR, 'hetero_preds')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(10))
N_FOLDS = 5
PATIENCE = 200


class HeteroMLP(nn.Module):
    """MLP with 2 outputs: mean and log-variance.

    Model: μ(x), log σ²(x)
    Loss: Gaussian NLL = 0.5 [(y - μ)² / σ² + log σ²]
    """
    def __init__(self, n_features, hidden=128, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden // 2, 1)
        self.logvar_head = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        z = self.backbone(x)
        mu = self.mean_head(z).squeeze(-1)
        log_var = self.logvar_head(z).squeeze(-1)
        # Clamp log_var to avoid numerical explosions
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var


def gaussian_nll_loss(mu, log_var, y):
    """Per-point Gaussian NLL."""
    var = torch.exp(log_var)
    return 0.5 * torch.mean((y - mu) ** 2 / var + log_var)


def train_hetero(X_tr, y_tr, X_val, y_val, n_features, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HeteroMLP(n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=80)

    y_tr_log = np.log1p(y_tr)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr_log, dtype=torch.float32, device=DEVICE)
    has_val = X_val is not None and len(y_val) > 0
    if has_val:
        y_va_log = np.log1p(y_val)
        X_va_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        y_va_t = torch.tensor(y_va_log, dtype=torch.float32, device=DEVICE)

    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(N_EPOCHS):
        model.train()
        opt.zero_grad()
        mu, log_var = model(X_tr_t)
        loss = gaussian_nll_loss(mu, log_var, y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                vmu, vlv = model(X_va_t)
                vloss = gaussian_nll_loss(vmu, vlv, y_va_t).item()
            sched.step(vloss)
            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait > PATIENCE:
                    break
        else:
            sched.step(loss.item())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_hetero(model, X):
    """Returns (mu, sigma) in RAW cycle units."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        mu_log, log_var = model(X_t)
        sigma_log = torch.sqrt(torch.exp(log_var))
        mu_log = mu_log.cpu().numpy()
        sigma_log = sigma_log.cpu().numpy()
    # Convert log-space Gaussian to raw: approximate via delta method
    mu_raw = np.expm1(mu_log)
    # Raw σ ≈ |d/dx expm1(x)| * σ_log = exp(x) * σ_log
    sigma_raw = np.exp(mu_log) * sigma_log
    return mu_raw, sigma_raw


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
    print('  HETEROSCEDASTIC MLP — Option B for Paper 4 (direct aleatoric)')
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

            preds_all = np.zeros((len(SEEDS), len(y_te)))
            sigmas_all = np.zeros((len(SEEDS), len(y_te)))
            cal_preds_all = np.zeros((len(SEEDS), len(y_ca)))

            for si, seed in enumerate(SEEDS):
                run_i += 1
                elapsed = time.time() - t0
                eta = (elapsed / max(run_i - 1, 1)) * (total - run_i + 1) if run_i > 1 else 0
                print(f'[{run_i:3d}/{total}] ne={ne} f={fold_idx} s={seed}  '
                      f'ETA={int(eta//60):d}m{int(eta%60):02d}s', end=' ', flush=True)
                t1 = time.time()
                try:
                    model = train_hetero(X_tr_n, y_tr, X_ca_n, y_ca,
                                         n_features=X_tr_n.shape[1], seed=seed)
                    mu, sigma = predict_hetero(model, X_te_n)
                    mu_cal, _ = predict_hetero(model, X_ca_n)
                    preds_all[si] = mu
                    sigmas_all[si] = sigma
                    cal_preds_all[si] = mu_cal
                    mae = np.mean(np.abs(mu - y_te))
                    dt = time.time() - t1
                    print(f' MAE={mae:5.0f}  σ_mean={np.mean(sigma):5.0f} ({dt:.0f}s)')
                except Exception as e:
                    print(f' ERROR: {e}')
                    traceback.print_exc()

            np.savez_compressed(
                out_file,
                y_true=y_te,
                preds_all=preds_all,
                sigmas_all=sigmas_all,   # Heteroscedastic-predicted σ
                cell_names=np.array(test_names),
                y_cal=y_ca,
                cal_preds_all=cal_preds_all,
            )

    total_min = (time.time() - t0) / 60
    print('-' * 70)
    print(f'DONE in {total_min:.1f} min')


if __name__ == '__main__':
    main()
