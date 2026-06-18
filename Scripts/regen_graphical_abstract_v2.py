"""Regenerate Paper 4 graphical abstract — v2 (Fourteen Methods, no 'nan' label).

Issues fixed vs the buggy version in SUBMIT_FINAL_2026-05-18/Paper4_GraphicalAbstract.png:
  1. Title: "10 Methods" -> "Fourteen Methods" (matches manuscript title)
  2. Left panel: only 10 bars + a 'nan' bar -> all 14 methods correctly labeled
  3. Middle calibration panel: kept the same 6 representative methods but with v2 numbers
  4. Right decision tree: minor wording polish; PNG -> PDF + PNG outputs
  5. Aspect ratio: closer to JPS 531x1328 (ratio 2.5) spec — render at 2900x1160 px (2.50)

Data source: Metrics/revision/aggregate_per_method_14.csv (canonical 14-method table).
"""
from __future__ import annotations
import csv
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison")
METRICS = ROOT / "Metrics" / "revision" / "aggregate_per_method_14.csv"
OUT_DIR = ROOT / "SUBMIT_FINAL_2026-05-18"
OUT_PNG = OUT_DIR / "Paper4_GraphicalAbstract.png"
OUT_PDF = OUT_DIR / "Paper4_GraphicalAbstract.pdf"

# ---------- load canonical 14-method MAEs at n_early=150 ----------
records = {}
with open(METRICS, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["n_early"]) == 150:
            records[row["method"]] = float(row["mae_mean"])

DISPLAY = [
    ("Combined_UQ_PINN_Knee",      "Combined UQ PINN-Knee",     "physics"),
    ("Deep_Ensemble_PINN_Knee",    "Deep Ens. PINN-Knee",       "physics"),
    ("Bootstrap_PINN_Knee",        "Bootstrap Ens. PINN-Knee",  "physics"),
    ("Hyper_Deep_Ensemble",        "Hyper-Deep Ensemble",       "physics"),
    ("SNGP",                       "SNGP",                      "non_physics"),
    ("Jackknife_Plus_PINN_Knee",   "Jackknife+ PINN-Knee",      "physics"),
    ("NGBoost",                    "NGBoost",                   "non_physics"),
    ("Gaussian_Process",           "Gaussian Process",          "non_physics"),
    ("Heteroscedastic_MLP_v2",     "Hetero MLP (v2)",           "non_physics"),
    ("CQR_MLP_v2",                 "CQR-MLP (v2)",              "non_physics"),
    ("CQR_PINN_Knee",              "CQR-PINN-Knee",             "physics"),
    ("Bayesian_LSTM",              "Bayesian LSTM",             "non_physics"),
    ("Ensemble_NN",                "Ensemble NN",               "non_physics"),
    ("Last_Layer_Laplace",         "Last-Layer Laplace",        "non_physics"),
]

# sort ascending by MAE for display
methods = sorted(DISPLAY, key=lambda x: records[x[0]])
labels = [m[1] for m in methods]
maes = [records[m[0]] for m in methods]
cats = [m[2] for m in methods]
C_PHYS = "#2E8B57"
C_NON = "#4682B4"
colors = [C_PHYS if c == "physics" else C_NON for c in cats]

# Cap Last-Layer Laplace (MAE 510) for display so other bars stay readable
CAP = 230
display_maes = [min(v, CAP) for v in maes]
clipped_idx = [i for i, v in enumerate(maes) if v > CAP]

# ---------- figure layout ----------
# JPS spec: 1328 x 531 px (W x H), ratio 2.50. We render at 2.5x → ~3320 x 1328 final after tight crop.
# Set figsize so that AFTER bbox_inches="tight" the saved PNG ratio is close to 2.50.
DPI = 200
W_IN = 13.2
H_IN = 6.3
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
# Title
fig.suptitle(
    "Uncertainty Quantification for Battery Knee-Point Prediction: "
    "Fourteen Methods with Physics-Informed Deep Ensembles",
    fontsize=12.5,
    fontweight="bold",
    y=0.965,
)

gs = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.86, bottom=0.15,
                      width_ratios=[1.4, 1.0, 1.1], wspace=0.32)

# ============ PANEL A — MAE bar chart (14 methods, n_early=150) ============
axA = fig.add_subplot(gs[0, 0])
y_pos = list(range(len(labels)))[::-1]  # top to bottom = low to high MAE
bars = axA.barh(y_pos, display_maes, color=colors, edgecolor="black", linewidth=0.6)
axA.set_yticks(y_pos)
axA.set_yticklabels(labels, fontsize=9)
axA.set_xlabel("MAE @ $n_{\\mathrm{early}} = 150$  (cycles)", fontsize=10)
axA.set_xlim(0, CAP + 18)
axA.set_title("Fourteen UQ methods", fontsize=11, fontweight="bold", pad=8)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.grid(axis="x", linestyle=":", color="#CCC", alpha=0.7)

# Annotate clipped bars with real value
for i, (v, dv) in enumerate(zip(maes, display_maes)):
    yp = y_pos[i]
    if v > CAP:
        axA.text(CAP + 2, yp, f"{v:.0f} →", va="center", ha="left",
                 fontsize=8, color="#C44E52", fontweight="bold")
    else:
        axA.text(dv + 2, yp, f"{v:.0f}", va="center", ha="left",
                 fontsize=8, color="#333")

# Legend — put OUTSIDE the chart at bottom so it doesn't overlap the Last-Layer Laplace bar
legend_handles = [
    patches.Patch(color=C_PHYS, label="Physics-informed (PINN-Knee variants)"),
    patches.Patch(color=C_NON,  label="Non-physics baselines"),
]
axA.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
           fontsize=8.5, frameon=False, ncol=2)

# ============ PANEL B — Calibration: raw vs conformal PICP ============
axB = fig.add_subplot(gs[0, 1])
# Short single-line labels (rotated) so they don't overlap each other or the legend
calib_methods = ["Deep Ens.", "Combined UQ", "CQR-PINN", "Ensemble NN", "Bayes LSTM", "GP"]
raw_picp =      [0.38, 0.39, 0.55, 0.08, 0.70, 0.03]
conformal_picp = [0.94, 0.93, 0.92, 0.95, 0.92, 0.92]
x = list(range(len(calib_methods)))
w = 0.36
axB.bar([xx - w/2 for xx in x], raw_picp,
        width=w, color="#C44E52", edgecolor="black", linewidth=0.5, label="Raw")
axB.bar([xx + w/2 for xx in x], conformal_picp,
        width=w, color="#2E8B57", edgecolor="black", linewidth=0.5, label="Conformal")
axB.axhline(0.95, linestyle="--", color="#444", linewidth=1.0)
axB.text(-0.45, 0.965, "target 0.95", fontsize=8, color="#444", ha="left")
axB.set_xticks(x)
axB.set_xticklabels(calib_methods, fontsize=9, rotation=30, ha="right")
axB.set_xlim(-0.7, len(x) - 0.3)
axB.set_ylim(0, 1.08)
axB.set_ylabel("PICP", fontsize=10)
axB.set_title("Calibration:\nsplit-conformal is mandatory", fontsize=11, fontweight="bold", pad=8)
axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
           fontsize=8.5, frameon=False, ncol=2)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.grid(axis="y", linestyle=":", color="#CCC", alpha=0.7)

# ============ PANEL C — Decision tree ============
axC = fig.add_subplot(gs[0, 2])
axC.set_xlim(0, 1)
axC.set_ylim(0, 1)
axC.axis("off")
axC.set_title("Decision tree:\nwhich UQ method?", fontsize=11, fontweight="bold", pad=8)

def add_box(ax, x, y, w, h, text, facecolor="#FFFCC9", edgecolor="#444", fontsize=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#222")

def add_arrow(ax, x1, y1, x2, y2, color="#444", lw=1.4, head=0.18):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle=f"-|>,head_width={head},head_length={head*1.7}",
        lw=lw, color=color, mutation_scale=16))

# Root: physics prior?
add_box(axC, 0.18, 0.84, 0.64, 0.11, "Have physics prior?", "#FFEFA0", weight="bold", fontsize=10)

# Branch YES (left, green) and NO (right, blue)
axC.text(0.21, 0.795, "YES", fontsize=9, color="#2E8B57", fontweight="bold", ha="center")
axC.text(0.79, 0.795, "NO",  fontsize=9, color="#4682B4", fontweight="bold", ha="center")

# Arrows from root to row-1 headline boxes
add_arrow(axC, 0.34, 0.84, 0.25, 0.76, color="#2E8B57", lw=1.6)
add_arrow(axC, 0.66, 0.84, 0.75, 0.76, color="#4682B4", lw=1.6)

# Row 1 — HEADLINE method per branch
add_box(axC, 0.02, 0.59, 0.46, 0.17,
        "HEADLINE\nDeep Ensemble\nPINN-Knee (K=5)\nMAE 111.6", "#D8EFD8", weight="bold", fontsize=8.5)
add_box(axC, 0.52, 0.59, 0.46, 0.17,
        "HEADLINE\nGaussian Process\nor SNGP\nMAE 113.6–125.3", "#D8E4F0", weight="bold", fontsize=8.5)

# Sub-branch arrows: row-1 → row-2 alternatives (within same colour family)
# Endpoint MUST sit on the ALTERNATIVE box top edge (y=0.50) so the arrowhead touches it
add_arrow(axC, 0.25, 0.59, 0.25, 0.50, color="#2E8B57", lw=1.2, head=0.14)
add_arrow(axC, 0.75, 0.59, 0.75, 0.50, color="#4682B4", lw=1.2, head=0.14)

# Sub-labels OUTSIDE the arrows (left of left arrow, right of right arrow)
# to avoid overlap in centre of panel
axC.text(0.18, 0.555, "if tail-PICP\nmatters", fontsize=7.5, color="#2E8B57",
         ha="right", va="center", fontstyle="italic")
axC.text(0.82, 0.555, "if sequential\nfeatures", fontsize=7.5, color="#4682B4",
         ha="left", va="center", fontstyle="italic")

# Row 2 — Alternative method per branch
add_box(axC, 0.02, 0.36, 0.46, 0.14,
        "ALTERNATIVE\nCQR-PINN-Knee\n(tail-aware, interval)", "#D8EFD8", fontsize=8)
add_box(axC, 0.52, 0.36, 0.46, 0.14,
        "ALTERNATIVE\nBayesian LSTM /\nEnsemble NN", "#D8E4F0", fontsize=8)

# Not recommended (warning) — separate box, 3 short lines so text fits inside
add_box(axC, 0.02, 0.10, 0.96, 0.17,
        "⚠ NOT recommended (naïve):\nHeteroscedastic MLP / CQR-MLP\n(overflow without v2 stabilisation)",
        "#FADBD8", weight="bold", fontsize=7.8)

# "+ split-conformal mandatory" tag at very bottom — applies to ALL branches
axC.text(0.5, 0.04,
         "+ split-conformal wrapper at every branch (PICP 0.92–0.98)",
         ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#444",
         fontweight="bold")

# ---------- save ----------
OUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Verify dims
from PIL import Image
im = Image.open(OUT_PNG)
print(f"Saved: {OUT_PNG.name}")
print(f"  Dimensions: {im.size[0]} x {im.size[1]} px  (W x H)")
print(f"  JPS spec:   1328 x 531 px (W x H) or proportionally more")
print(f"  Ratio W:H = {im.size[0]/im.size[1]:.2f}   (JPS spec ratio: {1328/531:.2f})")
print(f"Saved: {OUT_PDF.name}  (vector)")

# Verify methods count
print(f"\nMethods displayed: {len(labels)}  (target: 14)")
print(f"Methods list:")
for i, (lbl, v) in enumerate(zip(labels, maes), 1):
    print(f"  {i:>2}. {lbl:<26}  MAE {v:6.1f}")
