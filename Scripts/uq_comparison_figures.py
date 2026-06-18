"""
Final comparison figures for Paper 4:
  Fig A: ECE bar chart — PINN-Knee vs Ensemble_NN (vs Bayesian_LSTM)
  Fig B: PICP target line — which methods hit 95%?
  Fig C: Reliability diagrams 3-method overlay
  Fig D: MPIW vs PICP (sharpness-calibration frontier)
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
from config import RESULTS_DIR, FIGURES_DIR, EARLY_CYCLE_COUNTS

METHODS = {
    'Deep Ensemble PINN-Knee': ('deep_ensemble_preds', 'C0', 'o'),
    'CQR-PINN-Knee (flagship)': ('cqr_pinn_preds', 'C1', 'D'),
    'Gaussian Process': ('gp_preds', 'C4', 'v'),
    'Ensemble_NN (5x MLP)': ('ensemble_nn_preds', 'C3', 's'),
    'Bayesian_LSTM': ('bayesian_lstm_preds', 'C2', '^'),
    'Heteroscedastic MLP': ('hetero_preds', 'C5', 'P'),
    'CQR-MLP': ('cqr_preds', 'C6', 'X'),
}


def load_method_data(folder):
    result = {}
    path = os.path.join(RESULTS_DIR, folder)
    if not os.path.exists(path):
        return None
    for ne in EARLY_CYCLE_COUNTS:
        fold_files = sorted(glob.glob(os.path.join(path, f'preds_ne{ne}_f*.npz')))
        fold_files = [f for f in fold_files if '_s' not in os.path.basename(f).replace('preds_', '')]
        folds = []
        for f in fold_files:
            d = np.load(f)
            folds.append(d)
        if folds:
            result[ne] = folds
    return result if result else None


def fig_a_ece(method_data):
    """Expected Calibration Error comparison."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(EARLY_CYCLE_COUNTS))
    width = 0.8 / len(method_data)
    for i, (name, (folder, color, _)) in enumerate(METHODS.items()):
        if name not in method_data: continue
        ece_values = []
        ece_err = []
        for ne in EARLY_CYCLE_COUNTS:
            if ne not in method_data[name]:
                ece_values.append(np.nan); ece_err.append(0); continue
            fold_ece = []
            for d in method_data[name][ne]:
                y = d['y_true']
                preds = d['preds_all']
                mu = preds.mean(axis=0)
                sigma = np.maximum(preds.std(axis=0, ddof=1), 1)
                # ECE
                nominal = np.linspace(0.1, 0.99, 10)
                observed = []
                for p in nominal:
                    z = norm.ppf(0.5 + p / 2)
                    observed.append(np.mean((y >= mu - z*sigma) & (y <= mu + z*sigma)))
                fold_ece.append(np.mean(np.abs(nominal - np.array(observed))))
            ece_values.append(np.mean(fold_ece))
            ece_err.append(np.std(fold_ece))
        offset = (i - len(method_data)/2 + 0.5) * width
        ax.bar(x + offset, ece_values, width, yerr=ece_err, label=name,
                color=color, alpha=0.8, edgecolor='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$n_{{\\mathrm{{early}}}}={ne}$' for ne in EARLY_CYCLE_COUNTS])
    ax.set_ylabel('Expected Calibration Error (lower better)')
    ax.set_title('ECE of Gaussian Intervals (ensemble variance)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(0, color='black', lw=0.5)
    out = os.path.join(FIGURES_DIR, 'paper4_fig_ece_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def fig_b_picp(method_data):
    """PICP vs target 0.95 — which methods hit it?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    kinds = [('Gaussian intervals (ensemble σ)', 'gauss'),
             ('Conformal intervals (calibration)', 'conformal')]
    for ki, (title, kind) in enumerate(kinds):
        ax = axes[ki]
        x = np.arange(len(EARLY_CYCLE_COUNTS))
        width = 0.8 / len(method_data)
        for i, (name, (folder, color, _)) in enumerate(METHODS.items()):
            if name not in method_data: continue
            picps = []
            errs = []
            for ne in EARLY_CYCLE_COUNTS:
                if ne not in method_data[name]:
                    picps.append(np.nan); errs.append(0); continue
                fold_picps = []
                for d in method_data[name][ne]:
                    y = d['y_true']
                    preds = d['preds_all']
                    mu = preds.mean(axis=0)
                    if kind == 'gauss':
                        sigma = np.maximum(preds.std(axis=0, ddof=1), 1)
                        lo = mu - 1.96*sigma; hi = mu + 1.96*sigma
                    else:
                        if 'cal_preds_all' not in d.files or 'y_cal' not in d.files:
                            continue
                        cal_mu = d['cal_preds_all'].mean(axis=0)
                        residuals = np.abs(d['y_cal'] - cal_mu)
                        n = len(residuals)
                        k = int(np.ceil((n+1) * 0.95))
                        q = np.sort(residuals)[min(k-1, n-1)]
                        lo = mu - q; hi = mu + q
                    fold_picps.append(np.mean((y >= lo) & (y <= hi)))
                if fold_picps:
                    picps.append(np.mean(fold_picps))
                    errs.append(np.std(fold_picps))
                else:
                    picps.append(np.nan); errs.append(0)
            offset = (i - len(method_data)/2 + 0.5) * width
            ax.bar(x + offset, picps, width, yerr=errs, label=name,
                    color=color, alpha=0.8, edgecolor='black', capsize=4)
        ax.axhline(0.95, color='red', linestyle='--', lw=1.5, label='Target 95%')
        ax.set_xticks(x)
        ax.set_xticklabels([f'$n_{{\\mathrm{{early}}}}={ne}$' for ne in EARLY_CYCLE_COUNTS])
        ax.set_ylabel('PICP @ 95%')
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis='y')
        if ki == 0: ax.legend(fontsize=8)
    out = os.path.join(FIGURES_DIR, 'paper4_fig_picp_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def fig_c_reliability_overlay(method_data):
    """Reliability diagrams — 3 methods overlaid per n_early."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ci, ne in enumerate(EARLY_CYCLE_COUNTS):
        ax = axes[ci]
        for name, (folder, color, marker) in METHODS.items():
            if name not in method_data or ne not in method_data[name]: continue
            nominal = np.linspace(0.1, 0.99, 10)
            all_observed = []
            for d in method_data[name][ne]:
                y = d['y_true']
                preds = d['preds_all']
                mu = preds.mean(axis=0)
                sigma = np.maximum(preds.std(axis=0, ddof=1), 1)
                obs = []
                for p in nominal:
                    z = norm.ppf(0.5 + p / 2)
                    obs.append(np.mean((y >= mu - z*sigma) & (y <= mu + z*sigma)))
                all_observed.append(obs)
            if all_observed:
                mean_obs = np.mean(all_observed, axis=0)
                ax.plot(nominal, mean_obs, marker=marker, color=color, label=name, lw=1.5, ms=5)
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect')
        ax.set_xlabel('Nominal coverage')
        ax.set_title(f'$n_{{\\mathrm{{early}}}} = {ne}$')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        if ci == 0:
            ax.set_ylabel('Empirical coverage')
            ax.legend(loc='upper left', fontsize=8)
    out = os.path.join(FIGURES_DIR, 'paper4_fig_reliability_overlay.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    print('=' * 70)
    print('  PAPER 4 COMPARISON FIGURES')
    print('=' * 70)
    method_data = {}
    for name, (folder, _, _) in METHODS.items():
        d = load_method_data(folder)
        if d:
            method_data[name] = d
            print(f'  {name}: {sum(len(v) for v in d.values())} fold files loaded')
    if not method_data:
        print('No method data found!')
        return
    fig_a_ece(method_data)
    fig_b_picp(method_data)
    fig_c_reliability_overlay(method_data)


if __name__ == '__main__':
    main()
