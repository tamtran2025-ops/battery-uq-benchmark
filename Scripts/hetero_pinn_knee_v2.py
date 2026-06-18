"""
Heteroscedastic MLP — REVISED v2 with divergence prevention.

Changes from v1 to fix MAE > 10^13 divergence reported in original benchmark:
  1. Clamp mu_log output to [3, 9] (raw-space [20, 8100] cycles)
  2. Two-phase training: 200-epoch MSE warmup of mean head, then NLL fine-tune
  3. Initialize log_var bias = 0 (σ² = 1 in log space initially)
  4. Aggressive gradient clipping (norm 0.5) during initial 200 epochs
  5. NaN detection with auto-restart at lower LR
  6. log_var clamped to [-8, 8] (raw σ ≈ 0.018 to 55 in log space)

Scope: 10 seeds × 3 n_early × 5 folds = 150 runs.
"""
import os, sys, time, pickle
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
OUT_DIR = os.path.join(RESULTS_DIR, 'hetero_preds_v2')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(10))
N_FOLDS = 5
PATIENCE = 200
MU_LOG_MIN, MU_LOG_MAX = 3.0, 9.0   # raw [20, 8103] cycles — covers Severson knee range
LOG_VAR_MIN, LOG_VAR_MAX = -8.0, 8.0
WARMUP_EPOCHS = 200


class HeteroMLPv2(nn.Module):
    """MLP with 2 outputs: mean and log-variance, both clamped."""
    def __init__(self, n_features, hidden=128, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden // 2, 1)
        self.logvar_head = nn.Linear(hidden // 2, 1)
        # Init log_var bias to 0 (σ² = 1 in log space) for stable start
        nn.init.zeros_(self.logvar_head.bias)
        # Init mean head bias to mid-range
        nn.init.constant_(self.mean_head.bias, 6.0)

    def forward(self, x):
        z = self.backbone(x)
        mu_log = self.mean_head(z).squeeze(-1)
        log_var = self.logvar_head(z).squeeze(-1)
        # Hard clamp both outputs
        mu_log = torch.clamp(mu_log, min=MU_LOG_MIN, max=MU_LOG_MAX)
        log_var = torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
        return mu_log, log_var


def gaussian_nll_loss(mu, log_var, y):
    var = torch.exp(log_var)
    return 0.5 * torch.mean((y - mu) ** 2 / var + log_var)


def mse_log_loss(mu, log_var, y):
    """For warmup: MSE on mean only, ignore variance head."""
    return torch.mean((y - mu) ** 2)


def train_hetero(X_tr, y_tr, X_val, y_val, n_features, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HeteroMLPv2(n_features).to(DEVICE)
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
        # Phase A: MSE warmup; Phase B: full NLL
        if epoch < WARMUP_EPOCHS:
            loss = mse_log_loss(mu, log_var, y_tr_t)
            grad_norm = 0.5
        else:
            loss = gaussian_nll_loss(mu, log_var, y_tr_t)
            grad_norm = 1.0

        # NaN detection
        if torch.isnan(loss) or torch.isinf(loss):
            # Restart with lower LR
            for pg in opt.param_groups:
                pg['lr'] *= 0.1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        opt.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                vmu, vlv = model(X_va_t)
                if epoch < WARMUP_EPOCHS:
                    vloss = mse_log_loss(vmu, vlv, y_va_t).item()
                else:
                    vloss = gaussian_nll_loss(vmu, vlv, y_va_t).item()
            if not (np.isnan(vloss) or np.isinf(vloss)):
                sched.step(vloss)
                if vloss < best_val - 1e-4:
                    best_val = vloss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait > PATIENCE and epoch > WARMUP_EPOCHS:
                        break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_hetero(model, X):
    """Returns (mu, sigma) in RAW cycle units, hard-clamped."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        mu_log, log_var = model(X_t)
        sigma_log = torch.sqrt(torch.exp(log_var))
        mu_log = mu_log.cpu().numpy()
        sigma_log = sigma_log.cpu().numpy()
    # Convert log-space to raw, with safety clamp
    mu_log_safe = np.clip(mu_log, MU_LOG_MIN, MU_LOG_MAX)
    mu_raw = np.expm1(mu_log_safe)
    sigma_raw = np.exp(mu_log_safe) * sigma_log
    sigma_raw = np.clip(sigma_raw, 1e-3, 5000.0)
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
    print('  HETEROSCEDASTIC MLP v2 — divergence-fixed')
    print(f'  Output clamps: mu_log [{MU_LOG_MIN}, {MU_LOG_MAX}], log_var [{LOG_VAR_MIN}, {LOG_VAR_MAX}]')
    print(f'  Warmup: {WARMUP_EPOCHS} epochs MSE, then NLL')
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
                    print(f' MAE={mae:5.0f}  sigma_mean={np.mean(sigma):5.0f} ({dt:.0f}s)')
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f' FAIL: {e}')

            np.savez(out_file,
                     y_true=y_te.astype(np.float32),
                     preds_all=preds_all,
                     sigmas_all=sigmas_all,
                     cell_names=np.array(test_names),
                     y_cal=y_ca.astype(np.float32),
                     cal_preds_all=cal_preds_all)

    print('=' * 70)
    print(f'  COMPLETE in {(time.time()-t0)/60:.1f} min. Output: {OUT_DIR}')
    print('=' * 70)


if __name__ == "__main__":
    main()
