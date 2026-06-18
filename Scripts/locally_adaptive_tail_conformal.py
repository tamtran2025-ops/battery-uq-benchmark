"""Does a locally-adaptive / Mondrian conformal score close the long-knee CONDITIONAL gap?
n_early=150, equal-frequency tertiles, pooled cell-level coverage + Clopper-Pearson CI.
  A) absolute-residual split-conformal (paper default)
  B) normalized score |y-mu|/sigma  (sigma = ensemble seed-std; adaptive width)
  C) Mondrian: separate q_hat per PREDICTED-knee tertile (true tertile unknown at test)
Saves Metrics/revision/locally_adaptive_tail.csv
"""
import numpy as np, glob, os, csv
from scipy.stats import beta

def qlvl(n): return min(1.0, np.ceil((n+1)*0.95)/n)
def cpci(k,n):
    lo=0.0 if k==0 else beta.ppf(.025,k,n-k+1); hi=1.0 if k==n else beta.ppf(.975,k+1,n-k); return lo,hi

def run(md):
    R={k:[[],[],0,0] for k in "ABC"}  # cov_long, w_long, marg_cov, marg_n
    for f in sorted(glob.glob("Predictions/%s/preds_ne150_f*.npz"%md)):
        d=np.load(f,allow_pickle=True)
        y=np.asarray(d["y_true"],float); P=np.asarray(d["preds_all"],float)
        mu=P.mean(0); sd=np.clip(P.std(0),1e-6,None)
        yc=np.asarray(d["y_cal"],float); Pc=np.asarray(d["cal_preds_all"],float)
        muc=Pc.mean(0); sdc=np.clip(Pc.std(0),1e-6,None)
        t2=np.percentile(y,66.667); long=y>t2
        qa=np.quantile(np.abs(yc-muc),qlvl(len(yc)))
        qb=np.quantile(np.abs(yc-muc)/sdc,qlvl(len(yc)))
        pt1,pt2=np.percentile(mu,[33.333,66.667]); g=lambda v:np.where(v<=pt1,0,np.where(v<=pt2,1,2))
        gc,gt=g(muc),g(mu); loC=mu.copy(); hiC=mu.copy()
        for gg in range(3):
            cm=gc==gg; qg=np.quantile(np.abs(yc[cm]-muc[cm]),qlvl(cm.sum())) if cm.sum()>0 else qa
            tm=gt==gg; loC[tm]=mu[tm]-qg; hiC[tm]=mu[tm]+qg
        for k,(lo,hi) in [("A",(mu-qa,mu+qa)),("B",(mu-qb*sd,mu+qb*sd)),("C",(loC,hiC))]:
            cov=(y>=lo)&(y<=hi); w=hi-lo
            R[k][0]+=list(cov[long]); R[k][1]+=list(w[long]); R[k][2]+=int(cov.sum()); R[k][3]+=len(y)
    out=[]
    for k,lab in [("A","abs (default)"),("B","normalized |y-mu|/sd"),("C","Mondrian per-pred-tertile")]:
        cov=np.array(R[k][0]); w=np.array(R[k][1]); kk,n=int(cov.sum()),len(cov); lo,hi=cpci(kk,n)
        print("  %-26s marg=%.3f  long=%.3f [%.2f,%.2f] (%d/%d)  MPIW=%.0f"%(lab,R[k][2]/R[k][3],kk/n,lo,hi,kk,n,w.mean()))
        out.append([md,lab,round(R[k][2]/R[k][3],3),round(kk/n,3),round(lo,3),round(hi,3),round(w.mean())])
    return out

rows=[]
for md,name in [("deep_ensemble_preds","DEEP ENSEMBLE"),("cqr_pinn_preds","CQR-PINN-Knee")]:
    print("\n=== %s (n_early=150) ==="%name); rows+=run(md)
OUT="Metrics/revision/locally_adaptive_tail.csv"
with open(OUT,"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["method","scheme","marginal_PICP","longknee_PICP","cp_lo","cp_hi","longknee_MPIW"]); w.writerows(rows)
print("\nSaved -> "+OUT)
