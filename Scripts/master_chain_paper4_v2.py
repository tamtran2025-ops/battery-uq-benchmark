"""
Master chain v2 — Options A, B, C, F for Paper 4 (+ analyses).

Sequence (runs sequentially, all resume-safe):
  1. Option C: Gaussian Process baseline (~15-20 min)
  2. Option B: Heteroscedastic MLP (~15-20 min GPU)
  3. Option A: CQR-PINN-Knee FLAGSHIP (~30-40 min GPU)
  4. Option F: Ensemble size ablation (offline, 1 min)
  5. Recompute all UQ metrics (1 min)
  6. Regenerate all figures (1 min)
"""
import os, sys, time, subprocess
sys.stdout.reconfigure(encoding='utf-8')

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

STEPS = [
    ('GP baseline (Option C, predictive std)',
     os.path.join(SCRIPTS, '_experiments', 'gp_baseline_preds.py')),
    ('Heteroscedastic MLP (Option B, aleatoric)',
     os.path.join(SCRIPTS, '_experiments', 'hetero_pinn_knee.py')),
    ('CQR-PINN-Knee FLAGSHIP (Option A)',
     os.path.join(SCRIPTS, '_experiments', 'cqr_pinn_knee.py')),
    ('Ensemble size ablation (Option F)',
     os.path.join(SCRIPTS, '_analysis', 'ensemble_size_ablation.py')),
    ('Recompute all UQ metrics',
     os.path.join(SCRIPTS, '_analysis', 'compute_all_uq_metrics.py')),
    ('Regenerate all comparison figures',
     os.path.join(SCRIPTS, '_analysis', 'uq_comparison_figures.py')),
]


def run_step(name, path, i, total):
    print()
    print('=' * 75)
    print(f'[{i}/{total}] {name}')
    print('=' * 75)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, '-u', path],
        capture_output=False,
        text=True,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
    )
    dt = (time.time() - t0) / 60
    if result.returncode != 0:
        print(f'[ERROR] {name} exited {result.returncode} after {dt:.1f} min')
        return False
    print(f'[OK] {name} done in {dt:.1f} min')
    return True


def main():
    print('=' * 75)
    print('  MASTER CHAIN V2 — Paper 4 extensions (A, B, C, F)')
    print('=' * 75)
    t_start = time.time()
    for i, (name, path) in enumerate(STEPS, 1):
        if not os.path.exists(path):
            print(f'[SKIP] {path} not found')
            continue
        ok = run_step(name, path, i, len(STEPS))
        if not ok:
            print(f'CHAIN HALTED at step {i}')
            return 1
    total_min = (time.time() - t_start) / 60
    print()
    print('=' * 75)
    print(f'  ALL CHAIN V2 STEPS COMPLETE in {total_min:.1f} min')
    print('=' * 75)
    return 0


if __name__ == '__main__':
    sys.exit(main())
