"""Authoritative stratified knee-tertile split-conformal coverage at n_early=150.

DEF-A: equal-frequency tertiles (high = top third per fold; matches the headline numbers).
DEF-B: fixed cutoff high>710 cycles (top ~14%, n=14) -- shown to expose that the
       "(low<=387, mid 387-710, high>710)" wording is NOT an equal third.
CQR methods use the proper CQR score; Gaussian methods use abs-residual of the mean.
Exact Clopper-Pearson CIs + paired McNemar (CQR-PINN vs Deep Ensemble) on long-knee cells.
Run from 2_CODE/:  python Scripts/recompute_stratified_coverage.py
"""
import numpy as np, glob, os, csv
from scipy.stats import beta, binomtest

ALPHA = 0.05
PRED = "Predictions"
OUT = "Metrics/revision/stratified_coverage_FINAL.csv"
GAUSS = {"deep_ensemble_preds": "Deep Ensemble PINN-Knee", "gp_preds": "Gaussian Process",
         "ensemble_nn_preds": "Ensemble NN", "hetero_preds": "Heteroscedastic MLP",
         "bayesian_lstm_preds": "Bayesian LSTM"}
CQR = {"cqr_pinn_preds": "CQR-PINN-Knee", "cqr_preds": "CQR-MLP"}

def qlvl(n):
    return min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)

def cp(k, n):
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi

def cover_gauss(d):
    pt = np.asarray(d["preds_all"], float).mean(0)
    cpt = np.asarray(d["cal_preds_all"], float).mean(0)
    r = np.abs(np.asarray(d["y_cal"], float) - cpt)
    q = np.quantile(r, qlvl(len(r)))
    y = np.asarray(d["y_true"], float)
    return y, (y >= pt - q) & (y <= pt + q)

def cover_cqr(d):
    qt = np.asarray(d["q_test_all"], float).mean(0)
    qc = np.asarray(d["q_cal_all"], float).mean(0)
    yc = np.asarray(d["y_cal"], float)
    E = np.maximum(qc[:, 0] - yc, yc - qc[:, 2])
    q = np.quantile(E, qlvl(len(E)))
    y = np.asarray(d["y_true"], float)
    return y, (y >= qt[:, 0] - q) & (y <= qt[:, 2] + q)

def collect(md, kind):
    ys, cs = [], []
    for f in sorted(glob.glob(PRED + "/" + md + "/preds_ne150_f*.npz")):
        d = np.load(f, allow_pickle=True)
        y, c = cover_cqr(d) if kind == "cqr" else cover_gauss(d)
        ys.append(y); cs.append(c)
    return ys, cs

def fmt(k, n):
    p = k / n if n else float("nan")
    lo, hi = cp(k, n)
    return "%.3f(%d/%d)[%.2f,%.2f]" % (p, k, n, lo, hi)

rows = []
print("=== DEF-A: equal-frequency tertiles (high = top third per fold) ===")
print("%-24s %17s %17s %17s" % ("Method", "low", "mid", "high"))
for md, name in dict(GAUSS, **CQR).items():
    kind = "cqr" if md in CQR else "gauss"
    ys, cs = collect(md, kind)
    if not ys:
        continue
    kk = {"low": [0, 0], "mid": [0, 0], "high": [0, 0]}
    for y, c in zip(ys, cs):
        t1, t2 = np.percentile(y, [33.333, 66.667])
        for s, m in [("low", y <= t1), ("mid", (y > t1) & (y <= t2)), ("high", y > t2)]:
            kk[s][0] += int(c[m].sum()); kk[s][1] += int(m.sum())
    print("%-24s %17s %17s %17s" % (name, fmt(*kk["low"]), fmt(*kk["mid"]), fmt(*kk["high"])))
    for s in kk:
        k, n = kk[s]; lo, hi = cp(k, n)
        rows.append(["DEF-A_equalthird", name, s, k, n, round(k / n, 4) if n else "", round(lo, 4), round(hi, 4)])

print("\n=== DEF-B: fixed cutoff high>710 (top ~14%, n=14) ===")
for md, name in [("deep_ensemble_preds", "Deep Ensemble PINN-Knee"), ("cqr_pinn_preds", "CQR-PINN-Knee")]:
    kind = "cqr" if md in CQR else "gauss"
    ys, cs = collect(md, kind)
    y = np.concatenate(ys); c = np.concatenate(cs); m = y > 710
    k, n = int(c[m].sum()), int(m.sum()); lo, hi = cp(k, n)
    print("  %-24s high>710: %s" % (name, fmt(k, n)))
    rows.append(["DEF-B_fixed710", name, "high", k, n, round(k / n, 4), round(lo, 4), round(hi, 4)])

ysd, csd = collect("deep_ensemble_preds", "gauss")
ysc, csc = collect("cqr_pinn_preds", "cqr")
b = c = 0
for (yd, cd), (yc, cc) in zip(zip(ysd, csd), zip(ysc, csc)):
    t = np.percentile(yd, 66.667); m = yd > t
    b += int(((~cd[m]) & cc[m]).sum()); c += int((cd[m] & (~cc[m])).sum())
p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
print("\nPaired McNemar (DEF-A high, CQR-PINN vs Deep Ensemble): b=%d c=%d  p=%.3f (%s)" % (b, c, p, "sig" if p < 0.05 else "NOT sig at this n"))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["definition", "method", "tertile", "covered", "n", "picp", "cp_lo", "cp_hi"]); w.writerows(rows)
print("\nSaved -> " + OUT)
