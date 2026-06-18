"""
Conformalized Quantile Regression (CQR) — flagship UQ contribution for Paper 4.

CQR = training-integrated conformal prediction.
  1. Train quantile regressor predicting (q_α/2, q_0.5, q_1-α/2) via pinball loss
  2. Post-hoc conformal adjustment using calibration residuals:
       E_i = max(q_α/2(x_i) - y_i, y_i - q_1-α/2(x_i))
       c_hat = (1-α)(1+1/n)-quantile of {E_i}
  3. Final interval: [q_α/2(x) - c_hat, q_1-α/2(x) + c_hat]

Scope: 10 seeds × 3 n_early × 5 folds = 150 runs, MLP backbone.
ETA: ~15 min on GTX 970M.

Reference: Romano et al. NeurIPS 2019. 'Conformalized Quantile Regression.'
"""
import os, sys, time, pickle, traceback
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

from config import DEVICE, RESULTS_DIR, EARLY_CYCLE_COUNTS, LEARNING_RATE, WEIGHT_DECAY
from features import build_feature_matrix, normalize_features

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')
OUT_DIR = os.path.join(RESULTS_DIR, 'cqr_preds')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(10))
N_FOLDS = 5
ALPHA = 0.05   # 95% target coverage
QUANTILES = [ALPHA / 2, 0.5, 1 - ALPHA / 2]  # [0.025, 0.5, 0.975]
N_EPOCHS = 2000
PATIENCE = 200


class QuantileMLP(nn.Module):
    """MLP outputting 3 quantile predictions."""
    def __init__(self, n_features, hidden=128, n_quantiles=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_quantiles),
        )

    def forward(self, x):
        return self.net(x)


def pinball_loss(pred, target, quantiles):
    """Sum of pinball losses across quantiles.

    pred: (N, Q) predicted quantiles
    target: (N,)
    quantiles: list of Q floats
    """
    target = target.unsqueeze(1)  # (N, 1)
    diff = target - pred  # (N, Q)
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).unsqueeze(0)  # (1, Q)
    loss = torch.maximum(q * diff, (q - 1) * diff).mean()
    return loss


def train_qmlp(X_train, y_train, X_val, y_val, n_features, seed=0, verbose=False):
    """Train quantile MLP with early stopping."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = QuantileMLP(n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=80)

    # log-space targets for stability
    y_tr_t = torch.tensor(np.log1p(y_train), dtype=torch.float32, device=DEVICE)
    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    has_val = X_val is not None and len(y_val) > 0
    if has_val:
        y_va_t = torch.tensor(np.log1p(y_val), dtype=torch.float32, device=DEVICE)
        X_va_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)

    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(N_EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss = pinball_loss(pred, y_tr_t, QUANTILES)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                v_pred = model(X_va_t)
                v_loss = pinball_loss(v_pred, y_va_t, QUANTILES).item()
            sched.step(v_loss)
            if v_loss < best_val - 1e-4:
                best_val = v_loss
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


def predict_quantiles(model, X):
    """Predict all 3 quantiles; return raw-cycle scale."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        q_log = model(X_t).cpu().numpy()
    return np.expm1(q_log)  # (N, 3)


def conformal_calibrate(q_cal, y_cal, alpha=ALPHA):
    """CQR calibration step.

    q_cal: (n_cal, 3) = [q_lo, q_med, q_hi]
    y_cal: (n_cal,)
    Returns c_hat (scalar adjustment).
    """
    q_lo = q_cal[:, 0]
    q_hi = q_cal[:, 2]
    # Non-conformity scores: E_i = max(q_lo - y, y - q_hi)
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
        cal_idx = remaining[:n_cal]
        train_idx = remaining[n_cal:]
        splits.append(([cells[i] for i in train_idx],
                       [cells[i] for i in cal_idx],
                       [cells[i] for i in test_idx]))
    return splits


def main():
    print('=' * 70)
    print('  CQR (Conformalized Quantile Regression) — Paper 4 flagship')
    print('=' * 70)
    print(f'  Quantiles: {QUANTILES} (target {1-ALPHA:.0%} coverage)')

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
            train_cells, cal_cells, test_cells = folds[fold_idx]
            X_tr, y_tr, _, _ = build_feature_matrix(train_cells, ne)
            X_ca, y_ca, _, _ = build_feature_matrix(cal_cells, ne)
            X_te, y_te, _, te_valid = build_feature_matrix(test_cells, ne)
            if len(y_te) == 0 or len(y_ca) == 0:
                continue
            X_tr_n, X_te_n, X_ca_n, _ = normalize_features(X_tr, X_te, X_ca)
            test_names = [test_cells[i]['name'] for i in te_valid]

            # Collect quantile predictions across seeds
            q_test_all = np.zeros((len(SEEDS), len(y_te), 3))  # (seeds, N, 3 quantiles)
            q_cal_all = np.zeros((len(SEEDS), len(y_ca), 3))
            c_hats = np.zeros(len(SEEDS))

            for si, seed in enumerate(SEEDS):
                run_i += 1
                elapsed = time.time() - t0
                avg = elapsed / max(run_i - 1, 1) if run_i > 1 else 0
                eta = avg * (total - run_i + 1)
                print(f'[{run_i:3d}/{total}] ne={ne} f={fold_idx} s={seed}  '
                      f'ETA={int(eta//60):d}m{int(eta%60):02d}s', end=' ', flush=True)
                t1 = time.time()
                try:
                    model = train_qmlp(X_tr_n, y_tr, X_ca_n, y_ca,
                                       n_features=X_tr_n.shape[1], seed=seed)
                    q_test = predict_quantiles(model, X_te_n)
                    q_cal = predict_quantiles(model, X_ca_n)
                    c_hat = conformal_calibrate(q_cal, y_ca, alpha=ALPHA)
                    q_test_all[si] = q_test
                    q_cal_all[si] = q_cal
                    c_hats[si] = c_hat
                    # CQR interval width / MAE
                    mae = np.mean(np.abs(q_test[:, 1] - y_te))  # median pred vs true
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
                os.path.join(OUT_DIR, f'preds_ne{ne}_f{fold_idx}.npz'),
                y_true=y_te,
                q_test_all=q_test_all,    # (seeds, n_test, 3)
                q_cal_all=q_cal_all,      # (seeds, n_cal, 3)
                c_hats=c_hats,            # (seeds,) - conformal adjustments
                y_cal=y_ca,
                cell_names=np.array(test_names),
                # Compatibility: median-only predictions as "preds_all"
                preds_all=q_test_all[:, :, 1],
                cal_preds_all=q_cal_all[:, :, 1],
            )

    total_min = (time.time() - t0) / 60
    print('-' * 70)
    print(f'DONE in {total_min:.1f} min')
    print(f'Predictions: {OUT_DIR}/preds_ne*_f*.npz')


if __name__ == '__main__':
    main()
