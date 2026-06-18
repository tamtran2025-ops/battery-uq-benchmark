"""
Master chain runner for Paper 4 — run ALL remaining experiments + analyses.

Sequence (runs sequentially, resume-safe):
  1. Wait for Ensemble_NN per-cell preds to finish (if still running)
  2. CQR (Conformalized Quantile Regression) — 10 seeds × 3 ne × 5 folds
  3. PINN-Knee MC Dropout — add aleatoric on top of ensemble
  4. Compute ALL UQ metrics (unified)
  5. Generate ALL comparison figures
  6. Final summary report

Launches each as a subprocess; captures output + errors.
"""
import os, sys, time, subprocess
sys.stdout.reconfigure(encoding='utf-8')

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

# Paths
ENSEMBLE_NN_DIR = os.path.join(SCRIPTS, '..', 'results', 'ensemble_nn_preds')
CQR_DIR = os.path.join(SCRIPTS, '..', 'results', 'cqr_preds')

# Scripts to chain (order: fast first, expensive last)
STEPS = [
    ('Ensemble_NN per-cell preds (resume-safe, skips ne=50 done)',
     os.path.join(SCRIPTS, '_experiments', 'ensemble_nn_with_preds.py')),
    ('CQR training (Conformalized Quantile Regression)',
     os.path.join(SCRIPTS, '_experiments', 'cqr_training.py')),
    ('Bayesian_LSTM per-cell predictions (MC Dropout)',
     os.path.join(SCRIPTS, '_experiments', 'bayesian_lstm_with_preds.py')),
    ('Compute all UQ metrics (unified)',
     os.path.join(SCRIPTS, '_analysis', 'compute_all_uq_metrics.py')),
    ('Paper 4 comparison figures (ECE, PICP, reliability)',
     os.path.join(SCRIPTS, '_analysis', 'uq_comparison_figures.py')),
    ('UQ figures v2 (Paper 4 standalone)',
     os.path.join(SCRIPTS, '_analysis', 'uq_figures.py')),
]


def wait_for_ensemble_nn():
    """Deprecated — Ensemble_NN is now part of STEPS (with resume logic)."""
    pass


def run_step(name, script_path, step_num, total):
    print()
    print('=' * 75)
    print(f'[{step_num}/{total}] {name}')
    print('=' * 75)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
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
    print('  MASTER CHAIN FOR PAPER 4 — run everything')
    print('=' * 75)

    t_start = time.time()

    # Chain all steps sequentially
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
    print(f'  ALL CHAIN STEPS COMPLETE in {total_min:.1f} min')
    print('=' * 75)
    return 0


if __name__ == '__main__':
    sys.exit(main())
