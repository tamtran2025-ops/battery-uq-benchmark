# battery-uq-benchmark

Code, per-cell predictions, calibration residuals, and computed metrics for the paper:

> **Calibrated Uncertainty Quantification for Lithium-Ion Battery Knee-Point Prediction: A Systematic Benchmark with Mandatory Conformal Post-Processing and Cross-Chemistry Validation**
> Tran Thanh Trang, Tran Nhut Tam — *Journal of Energy Storage* (under review, 2026).

This repository reproduces a head-to-head benchmark of **fourteen uncertainty-quantification (UQ) methods** for early battery knee-point prediction on **110 Severson LFP cells**, with cross-chemistry validation on **19 Tongji-NCM cells**.

## Repository contents

```
Scripts/      Analysis + training code, including the infrastructure modules
              (config.py, features.py, models.py, train.py, knee_detection.py,
              metrics.py) and one script per UQ method / experiment.
Predictions/  Per-method, per-cell prediction .npz files (14 methods).
Metrics/      Computed metrics: MAE / PICP / MPIW / ECE, conformal calibration,
              batch-holdout, stratified coverage, partition-variance, etc.
results/      _severson_cache.pkl - cached Severson early-cycle features (2.3 MB),
              so the pipeline runs without downloading the raw dataset.
```

## Reproduce

Requirements: Python 3, `numpy`, `scipy`, `scikit-learn`, `torch` (CPU is sufficient for the GP and partition-variance paths).

```bash
cd Scripts

# Partition-variance robustness, Gaussian Process baseline (CPU, a few minutes):
python multi_partition_cv.py --partitions 10 --alpha 0.05

# Partition-variance for the headline Deep Ensemble PINN-Knee (GPU):
python multi_partition_cv_DE.py --partitions 5 --seeds 5 --alpha 0.05
```

See `MULTI_PARTITION_HOWTO.md` for details. The raw battery cycling data are publicly
available from the Severson et al. (2019) dataset (https://data.matr.io/1/) - only the
2.3 MB derived feature cache is bundled here.

## Headline results

- **Six-method accuracy plateau** at MAE 111-114 cycles (n_early = 150),
  statistically indistinguishable (cell-level paired permutation + Holm-Bonferroni + ROPE).
- **Conformal post-processing is necessary:** raw PICP 0.00-0.70 -> split-conformal
  restores 0.92-0.98 *marginal* coverage across all 14 methods.
- **CQR-PINN-Knee** restores long-knee-tertile coverage (0.88 vs 0.81) for tail-critical use.

## Citation

If you use this code or data, please cite the paper above (full citation / DOI added upon acceptance).

## License

Released under the [MIT License](LICENSE).
