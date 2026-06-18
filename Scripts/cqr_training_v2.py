"""
CQR-MLP — REVISED v2 with divergence prevention.

Same fixes as Hetero v2:
  1. Clamp quantile outputs to log-space [3, 9] = raw [20, 8100]
  2. 200-epoch warmup with median-only quantile (q=0.5) MSE on log targets
  3. Aggressive grad clip during warmup
  4. NaN detection
  5. Init bias = 6 for stable start
"""
import os, sys, time, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

from config import DEVICE, RESULTS_DIR, EARLY_CYCLE_COUNTS, LEARNING_RATE, WEIGHT_DECAY
from features import build_feature_matrix, normalize_features

CACHE_PATH = os.path.join(RESULTS_DIR, '_severson_cache.pkl')
OUT_DIR = os.path.join(RESULTS_DIR, 'cqr_preds_v2')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(10))
N_FOLDS = 5
ALPHA = 0.05
QUANTILES = [ALPHA / 2, 0.5, 1 - ALPHA / 2]
N_EPOCHS = 2000
PATIENCE = 200
Q_LOG_MIN, Q_LOG_MAX = 3.0, 9.0
WARMUP_EPOCHS = 200


class QuantileMLPv2(nn.Module):
    def __init__(self, n_features, hidden=128, n_quantiles=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_quantiles),
        )
        # Init final layer bias to mid-range log-space target
        with torch.no_grad():
            self.net[-1].bias.data = torch.tensor([5.5, 6.0, 6.5])  # lo, mid, hi

    def forward(self, x):
        out = self.net(x)
        # Clamp each quantile output
        out = torch.clamp(out, min=Q_LOG_MIN, max=Q_LOG_MAX)
        return out


def pinball_loss(pred, target, quantiles):
    target = target.unsqueeze(1)
    diff = target - pred
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).unsqueeze(0)
    loss = torch.maximum(q * diff, (q - 1) * diff).mean()
    return loss


def median_mse_loss(pred, target):
    """Warmup: only train median (q=0.5) to predict mean via MSE."""
    return torch.mean((pred[:, 1] - target) ** 2)


def train_qmlp(X_train, y_train, X_val, y_val, n_features, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = QuantileMLPv2(n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=80)

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
        if epoch < WARMUP_EPOCHS:
            loss = median_mse_loss(pred, y_tr_t)
            grad_norm = 0.5
        else:
            loss = pinball_loss(pred, y_tr_t, QUANTILES)
            grad_norm = 1.0

        if torch.isnan(loss) or torch.isinf(loss):
            for pg in opt.param_groups:
                pg['lr'] *= 0.1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        opt.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                v_pred = model(X_va_t)
                v_loss = (median_mse_loss(v_pred, y_va_t).item() if epoch < WARMUP_EPOCHS
                          else pinball_loss(v_pred, y_va_t, QUANTILES).item())
            if not (np.isnan(v_loss) or np.isinf(v_loss)):
                sched.step(v_loss)
                if v_loss < best_val - 1e-4:
                    best_val = v_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait > PATIENCE and epoch > WARMUP_EPOCHS:
                        break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_quantiles(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        q_log = model(X_t).cpu().numpy()
    q_log_safe = np.clip(q_log, Q_LOG_MIN, Q_LOG_MAX)
    return np.expm1(q_log_safe)


def conformal_calibrate(q_cal, y_cal, alpha=ALPHA):
    q_lo = q_cal[:, 0]
    q_hi = q_cal[:, 2]
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
    print('  CQR-MLP v2 — divergence-fixed')
    print(f'  Output clamps: q_log [{Q_LOG_MIN}, {Q_LOG_MAX}]')
    print(f'  Warmup: {WARMUP_EPOCHS} epochs median-MSE, then pinball')
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

            q_test_all = np.zeros((len(SEEDS), len(y_te), 3))
            q_cal_all = np.zeros((len(SEEDS), len(y_ca), 3))
            preds_all = np.zeros((len(SEEDS), len(y_te)))
            cal_preds_all = np.zeros((len(SEEDS), len(y_ca)))
            c_hats = np.zeros(len(SEEDS))

            for si, seed in enumerate(SEEDS):
                run_i += 1
                elapsed = time.time() - t0
                eta = (elapsed / max(run_i - 1, 1)) * (total - run_i + 1) if run_i > 1 else 0
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
                    preds_all[si] = q_test[:, 1]  # median as point prediction
                    cal_preds_all[si] = q_cal[:, 1]
                    c_hats[si] = c_hat
                    mae = np.mean(np.abs(q_test[:, 1] - y_te))
                    dt = time.time() - t1
                    print(f' MAE={mae:5.0f}  c_hat={c_hat:5.0f} ({dt:.0f}s)')
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f' FAIL: {e}')

            np.savez(out_file,
                     y_true=y_te.astype(np.float32),
                     q_test_all=q_test_all,
                     q_cal_all=q_cal_all,
                     c_hats=c_hats,
                     y_cal=y_ca.astype(np.float32),
                     cell_names=np.array(test_names),
                     preds_all=preds_all,
                     cal_preds_all=cal_preds_all)

    print('=' * 70)
    print(f'  COMPLETE in {(time.time()-t0)/60:.1f} min. Output: {OUT_DIR}')
    print('=' * 70)


if __name__ == "__main__":
    main()
