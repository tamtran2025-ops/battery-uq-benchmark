"""
Generate UQ figures for Paper 4:
  Fig 1: Reliability diagrams (calibration) per n_early
  Fig 2: Prediction intervals per cell (Gaussian vs Conformal)
  Fig 3: Uncertainty vs error scatter (sharpness-calibration trade-off)
  Fig 4: Expected Calibration Error (ECE) bar chart
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
from config import RESULTS_DIR, FIGURES_DIR, EARLY_CYCLE_COUNTS

PREDS_DIR = os.path.join(RESULTS_DIR, 'deep_ensemble_preds')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_all_preds():
    """Load all per-fold predictions and aggregate."""
    data = {ne: [] for ne in EARLY_CYCLE_COUNTS}
    for ne in EARLY_CYCLE_COUNTS:
        for f in range(5):
            fp = os.path.join(PREDS_DIR, f'preds_ne{ne}_f{f}.npz')
            if not os.path.exists(fp): continue
            d = np.load(fp)
            data[ne].append({
                'y_true': d['y_true'],
                'preds_all': d['preds_all'],
                'y_cal': d['y_cal'] if 'y_cal' in d.files else np.array([]),
                'cal_preds': d['cal_preds_all'] if 'cal_preds_all' in d.files else None,
            })
    return data


def reliability_points(y_true, mu, sigma, n_bins=10):
    """Compute expected vs observed coverage at nominal levels."""
    nominal = np.linspace(0.1, 0.99, n_bins)
    observed = []
    for p in nominal:
        z = norm.ppf(0.5 + p / 2)
        lo, hi = mu - z * sigma, mu + z * sigma
        observed.append(np.mean((y_true >= lo) & (y_true <= hi)))
    return nominal, np.array(observed)


def ece(y_true, mu, sigma, n_bins=10):
    """Expected Calibration Error."""
    nominal, observed = reliability_points(y_true, mu, sigma, n_bins)
    return np.mean(np.abs(nominal - observed))


def fig1_reliability(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for i, ne in enumerate(EARLY_CYCLE_COUNTS):
        ax = axes[i]
        for fold_data in data[ne]:
            y = fold_data['y_true']
            preds = fold_data['preds_all']
            mu = preds.mean(axis=0)
            sigma = np.maximum(preds.std(axis=0, ddof=1), 1)
            nom, obs = reliability_points(y, mu, sigma)
            ax.plot(nom, obs, 'o-', alpha=0.5, color='C0', markersize=4, lw=1)
        # Ideal
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
        ax.set_xlabel('Nominal coverage')
        ax.set_title(f'$n_{{\\mathrm{{early}}}} = {ne}$', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel('Empirical coverage')
            ax.legend(loc='upper left', fontsize=9)
    fig.suptitle('Reliability Diagrams — Deep Ensemble Gaussian Intervals', fontsize=12, y=1.02)
    out = os.path.join(FIGURES_DIR, 'uq_fig1_reliability.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def fig2_intervals(data):
    """Prediction intervals scatter: Gaussian vs Conformal for one fold per n_early."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, ne in enumerate(EARLY_CYCLE_COUNTS):
        ax = axes[i]
        # Use fold 0 for illustration
        if not data[ne]: continue
        d = data[ne][0]
        y = d['y_true']
        preds = d['preds_all']
        mu = preds.mean(axis=0)
        sigma = np.maximum(preds.std(axis=0, ddof=1), 1)

        # Sort by true value
        order = np.argsort(y)
        y = y[order]; mu = mu[order]; sigma = sigma[order]
        x = np.arange(len(y))

        # Gaussian intervals
        ax.fill_between(x, mu - 1.96*sigma, mu + 1.96*sigma,
                         alpha=0.25, color='C0', label='Gaussian 95%')
        ax.scatter(x, y, s=25, c='black', zorder=3, label='True knee')
        ax.plot(x, mu, 'o-', color='C0', ms=4, label='Ensemble mean')

        # Conformal (if available)
        if d['cal_preds'] is not None and len(d['y_cal']) > 0:
            cal_mu = d['cal_preds'].mean(axis=0)
            residuals = np.abs(d['y_cal'] - cal_mu)
            n = len(residuals)
            k = int(np.ceil((n + 1) * 0.95))
            q = np.sort(residuals)[min(k-1, n-1)]
            ax.fill_between(x, mu - q, mu + q, alpha=0.15, color='C3',
                             label=f'Conformal 95% (q={q:.0f})')

        ax.set_xlabel('Test cell (sorted by true knee)')
        ax.set_title(f'$n_{{\\mathrm{{early}}}} = {ne}$, fold 0')
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel('Knee cycle')
            ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Prediction Intervals: Gaussian (naive) vs Conformal (calibrated)', fontsize=12, y=1.02)
    out = os.path.join(FIGURES_DIR, 'uq_fig2_intervals.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def fig3_sharpness(data):
    """Sharpness-calibration scatter: sigma vs |error|"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, ne in enumerate(EARLY_CYCLE_COUNTS):
        ax = axes[i]
        all_sigma, all_err = [], []
        for d in data[ne]:
            preds = d['preds_all']
            mu = preds.mean(axis=0)
            sigma = preds.std(axis=0, ddof=1)
            err = np.abs(d['y_true'] - mu)
            all_sigma.extend(sigma)
            all_err.extend(err)
        ax.scatter(all_sigma, all_err, alpha=0.5, s=15)
        # Identity line
        lim = max(max(all_sigma), max(all_err)) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', lw=1, label='|err| = σ')
        ax.plot([0, lim], [0, 1.96*lim], 'r--', lw=1, label='|err| = 1.96σ')
        ax.set_xlabel('Ensemble σ (epistemic)')
        ax.set_title(f'$n_{{\\mathrm{{early}}}} = {ne}$')
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel('|True - Predicted| (cycles)')
            ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Sharpness vs Accuracy: Ensemble Variance Underestimates Error', fontsize=12, y=1.02)
    out = os.path.join(FIGURES_DIR, 'uq_fig3_sharpness.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def fig4_ece(data):
    """Expected Calibration Error per n_early."""
    ece_vals = []
    for ne in EARLY_CYCLE_COUNTS:
        fold_ece = []
        for d in data[ne]:
            preds = d['preds_all']
            mu = preds.mean(axis=0)
            sigma = np.maximum(preds.std(axis=0, ddof=1), 1)
            fold_ece.append(ece(d['y_true'], mu, sigma))
        ece_vals.append((ne, np.mean(fold_ece), np.std(fold_ece)))

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [str(v[0]) for v in ece_vals]
    means = [v[1] for v in ece_vals]
    stds = [v[2] for v in ece_vals]
    bars = ax.bar(xs, means, yerr=stds, capsize=5, color='C0', alpha=0.7,
                   edgecolor='black')
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{m:.3f}', ha='center', fontsize=9)
    ax.set_xlabel('$n_{\\mathrm{early}}$')
    ax.set_ylabel('Expected Calibration Error')
    ax.set_title('ECE of Deep Ensemble Gaussian Intervals\n(lower = better, 0 = perfect)')
    ax.grid(alpha=0.3, axis='y')
    out = os.path.join(FIGURES_DIR, 'uq_fig4_ece.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
    print(f'\nECE per n_early: {dict((v[0], round(v[1], 3)) for v in ece_vals)}')


def main():
    print('=' * 70)
    print('  UQ FIGURES for Paper 4')
    print('=' * 70)
    data = load_all_preds()
    for ne, folds in data.items():
        print(f'  n_early={ne}: {len(folds)} folds loaded')
    fig1_reliability(data)
    fig2_intervals(data)
    fig3_sharpness(data)
    fig4_ece(data)
    print('All figures saved to', FIGURES_DIR)


if __name__ == '__main__':
    main()
