"""Regenerate Figure 8: Practitioner's decision tree for UQ method selection."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 9))

# Define box style helpers
def question_box(ax, x, y, w, h, text, color="#fff8dc"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04",
                          facecolor=color, edgecolor='black', linewidth=1.2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=10, weight='bold')

def method_box(ax, x, y, w, h, text, color="#90ee90"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04",
                          facecolor=color, edgecolor='black', linewidth=1.0)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9.5)

def arrow(ax, x1, y1, x2, y2, label="", style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle=style, color='black', lw=1.2))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.02, label, ha='center',
                fontsize=9, style='italic',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))


# === Tree layout ===
# Root: physics prior?
question_box(ax, 0.5, 0.95, 0.30, 0.06, "Physics prior available?", color="#fff8dc")

# Yes branch: tail accuracy or headline?
question_box(ax, 0.22, 0.78, 0.22, 0.05, "Tail accuracy critical?", color="#fffacd")
arrow(ax, 0.42, 0.92, 0.27, 0.81, "Yes")

# No branch: dataset size?
question_box(ax, 0.78, 0.78, 0.22, 0.05, "Dataset size?", color="#fffacd")
arrow(ax, 0.58, 0.92, 0.73, 0.81, "No")

# YES-PHYSICS / Tail-critical: CQR-PINN-Knee
method_box(ax, 0.10, 0.60, 0.22, 0.07,
           "CQR-PINN-Knee\n(+12 pp tail PICP,\nsharper intervals)",
           color="#98fb98")
arrow(ax, 0.18, 0.75, 0.13, 0.65, "Yes")

# YES-PHYSICS / Headline: Deep Ensemble + (Combined UQ for triage)
method_box(ax, 0.34, 0.60, 0.24, 0.07,
           "Deep Ensemble PINN-Knee (K=5)\n[Combined UQ if epistemic/aleatoric needed]",
           color="#90ee90")
arrow(ax, 0.27, 0.75, 0.34, 0.65, "No")

# NO-PHYSICS branches
question_box(ax, 0.65, 0.60, 0.20, 0.05, "$N \\geq 500$ cells?", color="#fffacd")
arrow(ax, 0.73, 0.75, 0.68, 0.64, "")

# Branch left of bottom-question
question_box(ax, 0.92, 0.60, 0.16, 0.05, "Sequential\nfeatures?", color="#fffacd")
arrow(ax, 0.84, 0.75, 0.91, 0.64, "")

# Small N: GP
method_box(ax, 0.55, 0.42, 0.16, 0.06, "Gaussian Process\n+ noise-floor prior",
           color="#add8e6")
arrow(ax, 0.62, 0.57, 0.57, 0.46, "Small")

# Large N: SNGP
method_box(ax, 0.75, 0.42, 0.16, 0.06, "SNGP [46]\n(distance-aware DL)",
           color="#add8e6")
arrow(ax, 0.69, 0.57, 0.74, 0.46, "Large")

# Sequential: Bayesian LSTM
method_box(ax, 0.95, 0.42, 0.14, 0.06, "Bayesian LSTM\n(MC Dropout)",
           color="#dda0dd")
arrow(ax, 0.93, 0.57, 0.95, 0.46, "Yes")

# No sequential: Ensemble NN
method_box(ax, 0.78, 0.27, 0.16, 0.06, "Ensemble NN\n+ split-conformal",
           color="#dda0dd")
arrow(ax, 0.92, 0.57, 0.81, 0.30, "No")

# Bottom note: split-conformal
ax.text(0.5, 0.13, "ALL methods wrapped in split-conformal calibration\n"
                    "(reduces ECE 7-10×; PICP from 0.00–0.70 → 0.92–0.98)",
        ha='center', va='center',
        fontsize=11, style='italic', weight='bold',
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='#ffe4e1', edgecolor='#8b0000', linewidth=1.5))

# NOT recommended box
ax.text(0.5, 0.04, "⚠ NOT recommended on this dataset size:\n"
                    "Heteroscedastic MLP / CQR-MLP (naïve) → diverge to MAE 10¹³ without output clamping;\n"
                    "Last-Layer Laplace → Hessian ill-conditioning at N≈70 (MAE up to 1,610)",
        ha='center', va='center',
        fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='#fff0f0', edgecolor='#a52a2a', linewidth=1))

ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.0)
ax.axis('off')
ax.set_title("Practitioner's decision tree for UQ method selection (12-method benchmark)",
             fontsize=13, weight='bold', pad=10)

plt.tight_layout()
out = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\JPS_Submission_Files\figures\Figure_8.png"
plt.savefig(out, dpi=600, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {out}")
