"""
Deep Ensemble — train PINN-Knee with 7 additional seeds (3..9).

Uses the existing run_single_knee_experiment from run_experiments.py,
with filtered MODEL_NAMES=['PINN_Knee'] and N_SEEDS=10 to leverage its
resume logic (skips seeds 0,1,2 already done).

Results saved to same RESULTS_CSV — extends main experiment with 7 more seeds.
Foundation for:
  (a) Deep Ensemble UQ paper (next)
  (b) Stronger bootstrap CIs for Paper 7 revision
"""
import os
import sys
import time
import csv
import traceback
import numpy as np

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, '_analysis'))   # uncertainty.py was moved
sys.path.insert(0, os.path.join(SCRIPTS, '_baselines'))  # run_patchtst.py was moved
EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENTS)

# Config
import config
from config import DEVICE, RESULTS_DIR, RESULTS_CSV, EARLY_CYCLE_COUNTS

# Override seed count before importing run_experiments (some module-level constants)
config.N_SEEDS = 10  # was 3; bump to 10 for deep ensemble
import run_experiments as rx
rx.N_SEEDS = 10

from run_experiments import (run_single_knee_experiment, _kfold_split,
                              append_result_to_csv, load_completed_runs)
from data_loader import load_all_cells_with_knees

TARGET_MODEL = 'PINN_Knee'
N_FOLDS = 5
SEEDS = list(range(10))  # 0..9


def main():
    print('=' * 70)
    print('  DEEP ENSEMBLE — PINN-Knee 10 seeds (adds seeds 3..9 to existing 0..2)')
    print('=' * 70)
    print(f'  Device: {DEVICE}')
    print(f'  Target model: {TARGET_MODEL}')
    print(f'  Seeds: {SEEDS}')
    print(f'  n_early: {EARLY_CYCLE_COUNTS}')
    print(f'  Folds: {N_FOLDS}')

    # Load data
    print('\nLoading cells from cache...')
    cells = load_all_cells_with_knees()
    print(f'  {len(cells)} cells with valid knee')

    # Same fold split as main experiment (seed=42)
    folds = _kfold_split(cells, n_folds=N_FOLDS, seed=42)
    print(f'  Fold sizes: {[len(f[2]) for f in folds]}')

    # Resume
    key_cols = ['model', 'n_early', 'seed', 'fold']
    completed = load_completed_runs(RESULTS_CSV, key_cols)

    tasks = []
    for n_early in EARLY_CYCLE_COUNTS:
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                key = f"{TARGET_MODEL}|{n_early}|{seed}|{fold}"
                if key not in completed:
                    tasks.append((n_early, seed, fold))

    total = len(tasks)
    print(f'\n  Completed: {len(completed)} (across all models/seeds)')
    print(f'  {TARGET_MODEL} remaining: {total}')
    print('-' * 70)
    if total == 0:
        print('All seeds already trained.')
        return

    t0 = time.time()
    errors = 0
    for i, (n_early, seed, fold) in enumerate(tasks, 1):
        elapsed = time.time() - t0
        avg = elapsed / max(i-1, 1) if i > 1 else 0
        eta_s = avg * (total - i + 1)
        eta_str = f'{int(eta_s // 3600):d}h{int((eta_s % 3600) // 60):02d}m'
        print(f'[{i:3d}/{total}] ne={n_early} s={seed} f={fold}  ETA={eta_str}', end=' ', flush=True)

        t1 = time.time()
        try:
            train_cells, cal_cells, test_cells = folds[fold]
            result = run_single_knee_experiment(
                TARGET_MODEL, n_early, seed,
                train_cells, cal_cells, test_cells,
            )
            result['fold'] = fold
            append_result_to_csv(RESULTS_CSV, result)
            dt = time.time() - t1
            mae = result.get('MAE', -1)
            status = result.get('status', '?')
            print(f' MAE={mae:6.1f} [{status}] ({dt:.0f}s)')
        except Exception as e:
            errors += 1
            print(f' ERROR: {e}')
            traceback.print_exc()

    total_min = (time.time() - t0) / 60
    print('-' * 70)
    print(f'DONE in {total_min:.1f} min  Errors={errors}/{total}')
    print(f'Results appended to: {RESULTS_CSV}')


if __name__ == '__main__':
    main()
