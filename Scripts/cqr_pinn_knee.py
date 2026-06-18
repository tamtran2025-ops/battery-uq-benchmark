"""
Option A (FLAGSHIP): CQR-PINN-Knee — Quantile PINN-Knee + Conformal.

Architecture: PINN-Knee physics head (scalar n_phys) +
              3 parallel NN correction heads (Δ_NN for q_0.025, q_0.5, q_0.975).
              Final predictions: q_i = softplus(n_phys + Δ_NN_i)
              Monotonicity: enforce q_0.025 ≤ q_0.5 ≤ q_0.975 via cumulative softplus.

Loss: Pinball loss at 3 quantiles + physics regularization (composite).
Post-hoc: Conformal adjustment on calibration residuals.

Scope: 10 seeds × 3 n_early × 5 folds = 150 runs.
Paper 4 flagship: first physics-informed CQR for battery UQ.
"""
import os, sys, time, pickle, traceback
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

from config import (DEVICE, RESULTS_DIR, EARLY_CYCLE_COUNTS,
                     LEARNING_RATE, WEIGHT_DECAY, N_EPOCHS, MAX_CYCLE_LIFE)
from features import build_feature_matrix, normalize_features

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')
OUT_DIR = os.path.join(RESULTS_DIR, 'cqr_pinn_preds')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(10))
N_FOLDS = 5
ALPHA = 0.05
QUANTILES = [ALPHA / 2, 0.5, 1 - ALPHA / 2]  # 0.025, 0.5, 0.975
PATIENCE = 200


class QuantilePINN_Knee(nn.Module):
    """Physics-informed quantile regressor for knee-point.

    Physics head: predicts 5 params (a, b, c, d, s) and scalar n_phys (median reference).
    Correction heads: 3 MLP branches, each outputs scalar Δ_i bounded via tanh.
    Monotonicity: q_i = softplus(n_phys + cumulative_delta_i) where
                  cumulative_delta ensures q_0.025 ≤ q_0.5 ≤ q_0.975.
    """
    def __init__(self, n_features, hidden=128, max_cycle=2500.0, dropout=0.1):
        super().__init__()
        self.max_cycle = max_cycle

        # Physics param head (5 bounded params)
        self.phys_param_head = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 5),
        )
        # NN correction for each quantile (3 branches)
        self.delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            ) for _ in range(3)
        ])

    def compute_n_phys(self, params):
        """Log-knee formula: log n_phys = α log b + β log d + γ c + δ"""
        # params: (N, 5) = (a, b, c, d, s) — use b and d (rates)
        a = torch.sigmoid(params[:, 0]) * 0.2 + 0.8   # [0.8, 1.0]
        b = F.softplus(params[:, 1]) * 1e-3 + 1e-5    # small positive rate
        c = torch.sigmoid(params[:, 2]) * 0.5         # [0, 0.5]
        d = F.softplus(params[:, 3]) * 1e-2 + 1e-4    # positive rate
        s = torch.sigmoid(params[:, 4]) * 100 + 10    # [10, 110]

        alpha, beta, gamma, delta = -0.8, -0.3, 1.0, -0.4
        log_n_phys = alpha * torch.log(b) + beta * torch.log(d) + gamma * c + delta
        return log_n_phys, (a, b, c, d, s)

    def forward(self, x):
        # Physics head
        params = self.phys_param_head(x)
        log_n_phys, _ = self.compute_n_phys(params)
        # NN corrections (bounded via tanh * max_cycle/3)
        deltas = [800.0 * torch.tanh(h(x).squeeze(-1)) for h in self.delta_heads]
        # Enforce monotonicity: q_0.025 < q_0.5 < q_0.975
        # δ_0 = raw, δ_1 = δ_0 + softplus(δ_1_raw), δ_2 = δ_1 + softplus(δ_2_raw)
        d0 = deltas[0]
        d1 = d0 + F.softplus(deltas[1] - deltas[0])
        d2 = d1 + F.softplus(deltas[2] - deltas[1])

        # n_phys in raw space
        n_phys = torch.expm1(log_n_phys)
        # Quantiles in raw space (positive via softplus)
        q0 = F.softplus(n_phys + d0)
        q1 = F.softplus(n_phys + d1)
        q2 = F.softplus(n_phys + d2)
        return torch.stack([q0, q1, q2], dim=-1)  # (N, 3)


def pinball_loss(q_pred, target, quantiles):
    """Sum of pinball losses across quantiles.
    q_pred: (N, Q), target: (N,) in RAW space (log-transformed for training).
    """
    target = target.unsqueeze(-1)  # (N, 1)
    diff = target - q_pred
    q = torch.tensor(quantiles, dtype=q_pred.dtype, device=q_pred.device).unsqueeze(0)
    return torch.mean(torch.maximum(q * diff, (q - 1) * diff))


def train_cqr_pinn(X_tr, y_tr, X_val, y_val, n_features, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = QuantilePINN_Knee(n_features, max_cycle=MAX_CYCLE_LIFE).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=80)

    # Log-transform targets
    y_tr_log = np.log1p(y_tr)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr_log, dtype=torch.float32, device=DEVICE)
    has_val = X_val is not None and len(y_val) > 0
    if has_val:
        y_va_log = np.log1p(y_val)
        X_va_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        y_va_t = torch.tensor(y_va_log, dtype=torch.float32, device=DEVICE)

    best_val = float('inf'); best_state = None; wait = 0

    for epoch in range(N_EPOCHS):
        model.train()
        opt.zero_grad()
        q_raw = model(X_tr_t)  # (N, 3) in raw
        # Convert to log for training
        q_log = torch.log1p(q_raw)
        loss = pinball_loss(q_log, y_tr_t, QUANTILES)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                v_raw = model(X_va_t)
                v_log = torch.log1p(v_raw)
                v_loss = pinball_loss(v_log, y_va_t, QUANTILES).item()
            sched.step(v_loss)
            if v_loss < best_val - 1e-4:
                best_val = v_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait > PATIENCE: break
        else:
            sched.step(loss.item())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_quantiles(model, X):
    """Returns (N, 3) quantiles in raw cycle space."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return model(X_t).cpu().numpy()  # already raw


def conformal_calibrate(q_cal, y_cal, alpha=ALPHA):
    q_lo = q_cal[:, 0]; q_hi = q_cal[:, 2]
    E = np.maximum(q_lo - y_cal, y_cal - q_hi)
    n = len(E)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(E)[min(k - 1, n - 1)]


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
    print('  CQR-PINN-Knee (FLAGSHIP) — Option A for Paper 4')
    print('=' * 70)
    print(f'  Quantiles: {QUANTILES}  target coverage {1-ALPHA:.0%}')

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
            if len(y_te) == 0 or len(y_ca) == 0: continue
            X_tr_n, X_te_n, X_ca_n, _ = normalize_features(X_tr, X_te, X_ca)
            test_names = [test_cells[i]['name'] for i in te_valid]

            q_test_all = np.zeros((len(SEEDS), len(y_te), 3))
            q_cal_all = np.zeros((len(SEEDS), len(y_ca), 3))
            c_hats = np.zeros(len(SEEDS))

            for si, seed in enumerate(SEEDS):
                run_i += 1
                elapsed = time.time() - t0
                eta = (elapsed / max(run_i - 1, 1)) * (total - run_i + 1) if run_i > 1 else 0
                print(f'[{run_i:3d}/{total}] ne={ne} f={fold_idx} s={seed}  '
                      f'ETA={int(eta//60):d}m{int(eta%60):02d}s', end=' ', flush=True)
                t1 = time.time()
                try:
                    model = train_cqr_pinn(X_tr_n, y_tr, X_ca_n, y_ca,
                                            n_features=X_tr_n.shape[1], seed=seed)
                    q_test = predict_quantiles(model, X_te_n)
                    q_cal = predict_quantiles(model, X_ca_n)
                    c_hat = conformal_calibrate(q_cal, y_ca)
                    q_test_all[si] = q_test
                    q_cal_all[si] = q_cal
                    c_hats[si] = c_hat
                    mae = np.mean(np.abs(q_test[:, 1] - y_te))
                    picp_raw = np.mean((y_te >= q_test[:, 0]) & (y_te <= q_test[:, 2]))
                    picp_cqr = np.mean((y_te >= q_test[:, 0] - c_hat) &
                                        (y_te <= q_test[:, 2] + c_hat))
                    dt = time.time() - t1
                    print(f' MAE={mae:5.0f} PICP_raw={picp_raw:.2f} PICP_cqr={picp_cqr:.2f} '
                          f'c={c_hat:.0f} ({dt:.0f}s)')
                except Exception as e:
                    print(f' ERROR: {e}')
                    traceback.print_exc()

            np.savez_compressed(
                out_file,
                y_true=y_te,
                q_test_all=q_test_all,
                q_cal_all=q_cal_all,
                c_hats=c_hats,
                y_cal=y_ca,
                cell_names=np.array(test_names),
                preds_all=q_test_all[:, :, 1],
                cal_preds_all=q_cal_all[:, :, 1],
            )

    total_min = (time.time() - t0) / 60
    print('-' * 70)
    print(f'DONE in {total_min:.1f} min')


if __name__ == '__main__':
    main()
