"""Permutation test on cell-level paired absolute residuals (V5 Critical 1 fix).

Replaces the V4 Wilcoxon-on-fold-aggregates (n=5, mathematically vacuous after
m=28 Bonferroni) with a paired permutation test on cell-level absolute residuals
(n ≈ 102–110 per budget). For each method-pair we report:
  - n (number of paired cells)
  - mean MAE difference (mae_A − mae_B)
  - paired-permutation p-value (10⁵ sign-flip permutations)
  - Wilcoxon signed-rank on the same paired residuals (sanity check)
  - Bonferroni-corrected significance vs the chosen headline budget (m = 11)

Sign-flip permutation null: residuals exchangeable between methods, so under H0
each cell's signed difference has random ±1 sign.

Reference: Good (2005) "Permutation, Parametric, and Bootstrap Tests of Hypotheses",
Chapter 7. Uses cell-level pairing, which is the appropriate unit since each
cell is held out exactly once across the 5 folds.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon

PAPER4 = Path(__file__).resolve().parent.parent
PER_CELL = PAPER4 / 'Metrics' / 'revision' / 'all_predictions_per_cell_v3.csv'
OUT = PAPER4 / 'Metrics' / 'revision' / 'permutation_test_cell_level_v3.csv'

RNG = np.random.default_rng(20260427)
N_PERMUTATIONS = 100_000
HEADLINE = 'Deep_Ensemble_PINN_Knee'

# 14 methods (display order)
METHODS = [
    'Deep_Ensemble_PINN_Knee',
    'Hyper_Deep_Ensemble',
    'Combined_UQ_PINN_Knee',
    'Bootstrap_PINN_Knee',
    'Jackknife_Plus_PINN_Knee',
    'CQR_PINN_Knee',
    'SNGP',
    'Gaussian_Process',
    'NGBoost',
    'Heteroscedastic_MLP_v2',
    'CQR_MLP_v2',
    'Bayesian_LSTM',
    'Ensemble_NN',
    'Last_Layer_Laplace',
]


def paired_permutation(diff: np.ndarray, n_perm: int = N_PERMUTATIONS) -> float:
    """Two-sided paired permutation p-value (sign-flip test).

    H0: distribution of differences is symmetric about 0.
    """
    n = len(diff)
    obs_mean = float(diff.mean())
    obs_abs = abs(obs_mean)
    # Vectorised sign-flip permutation: ±1 mask for each permutation
    signs = RNG.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_perm, n))
    perm_means = (signs * diff[None, :]).mean(axis=1)
    p = float((np.abs(perm_means) >= obs_abs - 1e-12).sum() / n_perm)
    # Avoid 0 for output: at most 1/n_perm
    return max(p, 1.0 / n_perm)


def run(per_cell: pd.DataFrame, budgets=(50, 100, 150)) -> pd.DataFrame:
    rows = []
    for ne in budgets:
        sub = per_cell[per_cell.n_early == ne]
        # Build cell × method matrix of abs_err
        pivot = sub.pivot_table(index='cell', columns='method', values='abs_err', aggfunc='first')
        pivot = pivot.reindex(columns=METHODS)

        for a, b in combinations(METHODS, 2):
            paired = pivot[[a, b]].dropna()
            if len(paired) < 30:
                continue
            d = paired[a].values - paired[b].values
            mae_a = float(paired[a].mean())
            mae_b = float(paired[b].mean())
            p_perm = paired_permutation(d.astype(np.float64))
            try:
                w_stat, w_p = wilcoxon(paired[a], paired[b], zero_method='wilcox')
            except Exception:
                w_stat, w_p = np.nan, np.nan
            rows.append({
                'n_early': ne,
                'method_A': a, 'method_B': b,
                'n_cells': len(paired),
                'mae_A': mae_a, 'mae_B': mae_b,
                'mean_diff_A_minus_B': float(d.mean()),
                'median_diff_A_minus_B': float(np.median(d)),
                'p_permutation_two_sided': p_perm,
                'wilcoxon_stat': float(w_stat) if not np.isnan(w_stat) else np.nan,
                'wilcoxon_p': float(w_p) if not np.isnan(w_p) else np.nan,
            })
    df = pd.DataFrame(rows)

    # Bonferroni correction within each budget for "headline vs others" family (m=13 for 14 methods)
    df['is_headline_pair'] = (df.method_A == HEADLINE) | (df.method_B == HEADLINE)
    M_HEADLINE = len(METHODS) - 1  # 13 with 14 methods, was 11 with 12
    df['bonferroni_threshold_headline_family'] = 0.05 / M_HEADLINE
    df['significant_uncorrected'] = df.p_permutation_two_sided < 0.05
    df['significant_bonferroni_headline'] = (
        df.is_headline_pair & (df.p_permutation_two_sided < df.bonferroni_threshold_headline_family)
    )
    return df


def main():
    print(f'Loading {PER_CELL}...')
    per_cell = pd.read_csv(PER_CELL)
    print(f'  {len(per_cell)} rows, {per_cell.method.nunique()} methods, budgets {sorted(per_cell.n_early.unique())}')

    print(f'\nRunning paired permutation test ({N_PERMUTATIONS:,} permutations per pair)...')
    df = run(per_cell)
    df.to_csv(OUT, index=False)
    print(f'\nWrote {OUT} ({len(df)} method pairs)')

    # Headline-pair summary at n_early=100 (median budget)
    print('\n=== Headline-pair (DE vs others) at n_early=100 ===')
    hl = df[(df.is_headline_pair) & (df.n_early == 100)].copy()
    hl['other'] = hl.apply(lambda r: r.method_A if r.method_B == HEADLINE else r.method_B, axis=1)
    hl['mae_DE'] = hl.apply(lambda r: r.mae_A if r.method_A == HEADLINE else r.mae_B, axis=1)
    hl['mae_other'] = hl.apply(lambda r: r.mae_B if r.method_A == HEADLINE else r.mae_A, axis=1)
    hl['signed_diff_other_minus_DE'] = hl.mae_other - hl.mae_DE
    cols = ['other', 'n_cells', 'mae_DE', 'mae_other', 'signed_diff_other_minus_DE',
            'p_permutation_two_sided', 'wilcoxon_p', 'significant_bonferroni_headline']
    print(hl[cols].sort_values('p_permutation_two_sided').to_string(index=False, float_format='%.4f'))

    # Same at n_early=150 (final budget)
    print('\n=== Headline-pair (DE vs others) at n_early=150 ===')
    hl = df[(df.is_headline_pair) & (df.n_early == 150)].copy()
    hl['other'] = hl.apply(lambda r: r.method_A if r.method_B == HEADLINE else r.method_B, axis=1)
    hl['mae_DE'] = hl.apply(lambda r: r.mae_A if r.method_A == HEADLINE else r.mae_B, axis=1)
    hl['mae_other'] = hl.apply(lambda r: r.mae_B if r.method_A == HEADLINE else r.mae_A, axis=1)
    hl['signed_diff_other_minus_DE'] = hl.mae_other - hl.mae_DE
    print(hl[cols].sort_values('p_permutation_two_sided').to_string(index=False, float_format='%.4f'))

    # Physics-cluster comparison: DE vs Combined/Bootstrap/Jackknife+
    physics = ['Combined_UQ_PINN_Knee', 'Bootstrap_PINN_Knee', 'Jackknife_Plus_PINN_Knee', 'CQR_PINN_Knee']
    print('\n=== Physics-informed cluster vs DE (claim: indistinguishable plateau) ===')
    for ne in (50, 100, 150):
        print(f'\n  n_early={ne}:')
        sub = df[(df.n_early == ne) & df.is_headline_pair].copy()
        sub['other'] = sub.apply(lambda r: r.method_A if r.method_B == HEADLINE else r.method_B, axis=1)
        sub = sub[sub.other.isin(physics)]
        sub['mae_DE'] = sub.apply(lambda r: r.mae_A if r.method_A == HEADLINE else r.mae_B, axis=1)
        sub['mae_other'] = sub.apply(lambda r: r.mae_B if r.method_A == HEADLINE else r.mae_A, axis=1)
        for _, r in sub.iterrows():
            print(f"    DE vs {r.other:30s}  Δ={r.mae_other - r.mae_DE:+6.2f}  p_perm={r.p_permutation_two_sided:.4f}  "
                  f"sig(α=0.05/11={0.05/11:.4f})={'YES' if r.p_permutation_two_sided < 0.05/11 else 'no'}")


if __name__ == '__main__':
    main()
