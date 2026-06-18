"""Bayesian comparison with region-of-practical-equivalence (ROPE) analysis.

Following Benavoli, Corani, Demšar & Zaffalon (JMLR 18:1–36, 2017,
"Time for a Change"), we compute the posterior probability that
DE outperforms each comparator by more than a practical-equivalence threshold.

ROPE = ±5 cycles (a threshold that should be small relative to the within-fold
±38-cycle noise floor and the smallest practitioner-relevant accuracy gap).

For each method comparison we compute, on the n=102 paired absolute-residual
differences (other - DE):
  - posterior P(delta > +ROPE) — probability "other is significantly worse"
  - posterior P(|delta| <= ROPE) — probability "practically equivalent"
  - posterior P(delta < -ROPE) — probability "other is significantly better"

Using a non-informative t-distribution posterior over the mean difference
(equivalent to Bayesian estimation with a flat prior, n large enough for CLT).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t

PAPER4 = Path(__file__).resolve().parent.parent
PER_CELL = PAPER4 / 'Metrics' / 'revision' / 'all_predictions_per_cell_v3.csv'
OUT = PAPER4 / 'Metrics' / 'revision' / 'bayesian_rope_summary.csv'

HEADLINE = 'Deep_Ensemble_PINN_Knee'
NE = 150
ROPE = 5.0  # cycles


def main():
    per_cell = pd.read_csv(PER_CELL)
    sub = per_cell[per_cell.n_early == NE]
    pivot = sub.pivot_table(index='cell', columns='method', values='abs_err', aggfunc='first')

    rows = []
    for other in pivot.columns:
        if other == HEADLINE:
            continue
        paired = pivot[[HEADLINE, other]].dropna()
        if len(paired) < 30:
            continue
        delta = (paired[other] - paired[HEADLINE]).values
        n = len(delta)
        mean = delta.mean()
        se = delta.std(ddof=1) / np.sqrt(n)
        df = n - 1

        # Posterior is t-distributed with mean and se given n-1 dof
        # P(delta > +ROPE) = P(t > (ROPE - mean) / se)
        p_worse = 1 - t.cdf((ROPE - mean) / se, df=df)
        p_better = t.cdf((-ROPE - mean) / se, df=df)
        p_rope = 1 - p_worse - p_better

        rows.append({
            'other_method': other,
            'n_paired': n,
            'mean_delta': float(mean),
            'se_delta': float(se),
            'P_other_worse_than_DE_by_5cyc': float(p_worse),
            'P_practically_equivalent_within_5cyc': float(p_rope),
            'P_other_better_than_DE_by_5cyc': float(p_better),
        })

    df = pd.DataFrame(rows).sort_values('mean_delta')
    df.to_csv(OUT, index=False)

    print(f'Wrote {OUT} ({len(df)} pairs)')
    print()
    print(f'=== Bayesian comparison with ROPE = ±{ROPE} cycles @ n_early={NE} ===')
    print(f'{"method":30s} {"delta":>8s} {"P(worse)":>10s} {"P(ROPE)":>10s} {"P(better)":>10s}')
    for _, r in df.iterrows():
        print(f'{r.other_method:30s} {r.mean_delta:+8.2f} {r.P_other_worse_than_DE_by_5cyc:>10.4f} '
              f'{r.P_practically_equivalent_within_5cyc:>10.4f} {r.P_other_better_than_DE_by_5cyc:>10.4f}')


if __name__ == '__main__':
    main()
