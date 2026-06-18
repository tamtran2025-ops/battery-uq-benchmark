"""Holm-Bonferroni correction + BCa-bootstrap CIs for headline-vs-others MAE differences.

Reads cell-level absolute residuals, computes paired delta-MAE (other - DE),
runs paired permutation test (replicates existing), applies Holm correction,
and computes BCa-bootstrap 95 % CIs on each delta.

Output: Metrics/revision/holm_bca_summary.csv with columns
  method, mae_DE, mae_other, delta, p_perm, holm_threshold, holm_significant,
  bca_lower, bca_upper, n_paired
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import bootstrap

PAPER4 = Path(__file__).resolve().parent.parent
PER_CELL = PAPER4 / 'Metrics' / 'revision' / 'all_predictions_per_cell_v3.csv'
PERM = PAPER4 / 'Metrics' / 'revision' / 'permutation_test_cell_level_v3.csv'
OUT = PAPER4 / 'Metrics' / 'revision' / 'holm_bca_summary.csv'

HEADLINE = 'Deep_Ensemble_PINN_Knee'
NE = 150
ALPHA = 0.05
RNG = np.random.default_rng(20260429)


def main():
    per_cell = pd.read_csv(PER_CELL)
    perm = pd.read_csv(PERM)

    # Build paired-cell DE-vs-other absolute residuals at n_early=150
    sub = per_cell[per_cell.n_early == NE]
    pivot = sub.pivot_table(index='cell', columns='method', values='abs_err', aggfunc='first')
    if HEADLINE not in pivot.columns:
        raise SystemExit(f'{HEADLINE} not found in pivot')

    de_abs = pivot[HEADLINE].dropna().values
    de_cells = pivot[HEADLINE].dropna().index

    # Iterate methods at n_early=150 from the perm CSV
    perm_150 = perm[(perm.n_early == NE) &
                    ((perm.method_A == HEADLINE) | (perm.method_B == HEADLINE))].copy()

    rows = []
    for _, r in perm_150.iterrows():
        other = r.method_A if r.method_B == HEADLINE else r.method_B
        if other not in pivot.columns:
            continue
        paired = pivot[[HEADLINE, other]].dropna()
        if len(paired) < 30:
            continue
        de_arr = paired[HEADLINE].values
        ot_arr = paired[other].values
        delta = ot_arr - de_arr  # other - DE; positive means DE is better

        # BCa bootstrap on the mean delta
        try:
            res = bootstrap((delta,), np.mean, confidence_level=0.95,
                            method='BCa', n_resamples=10000, random_state=RNG)
            bca_lo, bca_hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception as e:
            bca_lo, bca_hi = np.nan, np.nan

        rows.append({
            'method': other,
            'n_paired': len(paired),
            'mae_DE': float(de_arr.mean()),
            'mae_other': float(ot_arr.mean()),
            'delta_mean': float(delta.mean()),
            'delta_median': float(np.median(delta)),
            'p_permutation': float(r.p_permutation_two_sided),
            'p_wilcoxon': float(r.wilcoxon_p),
            'bca95_lower': bca_lo,
            'bca95_upper': bca_hi,
        })

    df = pd.DataFrame(rows).sort_values('p_permutation')

    # Holm-Bonferroni step-down: sort by p ascending, threshold at α/(m-i+1)
    m = len(df)
    df = df.reset_index(drop=True)
    df['holm_rank'] = df.index + 1
    df['holm_threshold'] = ALPHA / (m - df['holm_rank'] + 1)
    # Step-down: stop rejecting at first non-rejection
    holm_sig = []
    rejected_so_far = True
    for i, p in enumerate(df['p_permutation']):
        thresh = df.loc[i, 'holm_threshold']
        if rejected_so_far and p < thresh:
            holm_sig.append(True)
        else:
            rejected_so_far = False
            holm_sig.append(False)
    df['holm_significant'] = holm_sig

    # Bonferroni for comparison
    df['bonferroni_threshold'] = ALPHA / m
    df['bonferroni_significant'] = df['p_permutation'] < df['bonferroni_threshold']

    df.to_csv(OUT, index=False)
    print(f'Wrote {OUT} ({len(df)} method pairs)')
    print()
    print('=== Holm vs Bonferroni @ n_early=150 (DE-vs-other family m=13) ===')
    print(f'{"method":30s} {"delta":>8s} {"BCa95_low":>10s} {"BCa95_up":>10s} {"p_perm":>10s} {"holm_thr":>10s} {"holm":>6s} {"bonf":>6s}')
    for _, r in df.iterrows():
        print(f'{r.method:30s} {r.delta_mean:+8.2f} {r.bca95_lower:+10.2f} {r.bca95_upper:+10.2f} '
              f'{r.p_permutation:10.5f} {r.holm_threshold:10.5f} '
              f'{"YES" if r.holm_significant else "no":>6s} '
              f'{"YES" if r.bonferroni_significant else "no":>6s}')

    n_holm = df.holm_significant.sum()
    n_bonf = df.bonferroni_significant.sum()
    print(f'\nHolm rejects: {n_holm}/{len(df)}; Bonferroni rejects: {n_bonf}/{len(df)}')
    if n_holm > n_bonf:
        gained = df[df.holm_significant & ~df.bonferroni_significant].method.tolist()
        print(f'Holm gains: {gained}')


if __name__ == '__main__':
    main()
