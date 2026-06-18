"""Regenerate Figure 8 v3: 4-leaf right side, no arrow crossings."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COLOR_Q = "#FFF3B0"
COLOR_M_PHYS = "#B6E5B6"
COLOR_M_GP = "#B5D9F0"
COLOR_M_NN = "#E0BBE4"
EDGE = "#2C2C2C"

FS_Q = 11
FS_M = 10
FS_M_SUB = 9
FS_LABEL = 10
FS_TITLE = 14
FS_NOTE = 9

fig, ax = plt.subplots(figsize=(15, 10))

def qbox(x, y, w, h, text, color=COLOR_Q):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor=EDGE, linewidth=1.4))
    ax.text(x, y, text, ha='center', va='center',
            fontsize=FS_Q, weight='bold', color=EDGE)

def mbox(x, y, w, h, main, sub=None, color=COLOR_M_PHYS):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor=EDGE, linewidth=1.2))
    if sub:
        ax.text(x, y+h*0.18, main, ha='center', va='center',
                fontsize=FS_M, weight='bold', color=EDGE)
        ax.text(x, y-h*0.20, sub, ha='center', va='center',
                fontsize=FS_M_SUB, style='italic', color=EDGE)
    else:
        ax.text(x, y, main, ha='center', va='center',
                fontsize=FS_M, weight='bold', color=EDGE)

def arrow(x1, y1, x2, y2, label="", offset=(0.005, 0.005)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=EDGE,
                                lw=1.5, mutation_scale=14))
    if label:
        ax.text((x1+x2)/2 + offset[0], (y1+y2)/2 + offset[1], label,
                ha='center', fontsize=FS_LABEL, weight='bold',
                color="#8B0000",
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.92, pad=2))

# Y levels
Y_ROOT = 0.93
Y_Q1 = 0.76
Y_Q2 = 0.58
Y_M = 0.36
Y_BOTTOM = 0.12

# Box sizes
QW, QH = 0.24, 0.06
QW_SM, QH_SM = 0.16, 0.06
MW, MH = 0.16, 0.10

# ===== ROOT =====
qbox(0.5, Y_ROOT, QW, QH, "Physics prior available?")

# ===== LEVEL 2 =====
qbox(0.20, Y_Q1, QW_SM+0.02, QH_SM, "Tail accuracy critical?")
qbox(0.70, Y_Q1, QW_SM-0.02, QH_SM, "Dataset size?")

arrow(0.42, Y_ROOT-QH/2, 0.25, Y_Q1+QH_SM/2, "Yes")
arrow(0.58, Y_ROOT-QH/2, 0.66, Y_Q1+QH_SM/2, "No")

# ===== LEVEL 3 LEFT: Physics methods =====
mbox(0.09, Y_Q2-0.05, MW-0.02, MH,
     "CQR-PINN-Knee",
     "+12 pp tail PICP\nsharper intervals",
     color=COLOR_M_PHYS)
mbox(0.30, Y_Q2-0.05, MW+0.04, MH,
     "Deep Ensemble PINN-Knee",
     "K=5, [Combined UQ if epistemic /\naleatoric needed]",
     color=COLOR_M_PHYS)

arrow(0.16, Y_Q1-QH_SM/2, 0.11, Y_Q2+MH/2-0.05, "Yes")
arrow(0.24, Y_Q1-QH_SM/2, 0.28, Y_Q2+MH/2-0.05, "No")

# ===== LEVEL 3 RIGHT: Two sub-questions =====
qbox(0.58, Y_Q2, QW_SM-0.02, QH_SM, "$N \\geq 500$ cells?")
qbox(0.82, Y_Q2, QW_SM-0.02, QH_SM, "Sequential features?")

arrow(0.66, Y_Q1-QH_SM/2, 0.60, Y_Q2+QH_SM/2, "")
arrow(0.74, Y_Q1-QH_SM/2, 0.80, Y_Q2+QH_SM/2, "")

# ===== LEVEL 4: 4 leaf methods on right side (all same Y) =====
mbox(0.51, Y_M, MW-0.02, MH,
     "Gaussian Process",
     "+ noise-floor prior",
     color=COLOR_M_GP)
mbox(0.67, Y_M, MW-0.02, MH,
     "SNGP [46]",
     "distance-aware DL",
     color=COLOR_M_GP)
mbox(0.83, Y_M, MW-0.02, MH,
     "Bayesian LSTM",
     "MC Dropout",
     color=COLOR_M_NN)
mbox(0.99, Y_M, MW-0.02, MH,
     "Ensemble NN",
     "+ split-conformal",
     color=COLOR_M_NN)

arrow(0.55, Y_Q2-QH_SM/2, 0.52, Y_M+MH/2, "Small")
arrow(0.61, Y_Q2-QH_SM/2, 0.66, Y_M+MH/2, "Large")
arrow(0.79, Y_Q2-QH_SM/2, 0.83, Y_M+MH/2, "Yes")
arrow(0.85, Y_Q2-QH_SM/2, 0.97, Y_M+MH/2, "No")

# ===== BOTTOM: split-conformal emphasis =====
ax.text(0.5, Y_BOTTOM,
        "ALL methods wrapped in split-conformal calibration\n"
        "(reduces ECE 7–10×; PICP from 0.00–0.70 to 0.92–0.98)",
        ha='center', va='center',
        fontsize=11, weight='bold', color="#7A0000",
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='#FFE4E1', edgecolor='#8B0000', linewidth=1.8))

# ===== BOTTOM CAUTION =====
ax.text(0.5, 0.02,
        "Not recommended on this dataset size: Heteroscedastic MLP / CQR-MLP "
        "(naïve) diverge to MAE 10¹³ without output clamping;  "
        "Last-Layer Laplace suffers Hessian ill-conditioning at N≈70 (MAE up to 1,610).",
        ha='center', va='center',
        fontsize=FS_NOTE, style='italic', color="#5C5C5C")

# ===== TITLE =====
ax.set_title("Practitioner's decision tree for UQ method selection (fourteen-method benchmark)",
             fontsize=FS_TITLE, weight='bold', pad=14, color=EDGE)

ax.set_xlim(-0.02, 1.08)
ax.set_ylim(-0.02, 1.0)
ax.axis('off')

plt.tight_layout()
import os as _os
out_png = r"D:\Project Python\PythonProject9\00_Battery_Paper1_UQ-Benchmark_(RESS)\REVISION_Package_RESS\RESS_figures\Figure_8.png"
_os.makedirs(_os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
