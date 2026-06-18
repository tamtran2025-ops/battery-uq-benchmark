"""Wrapper: re-run Fig 3-7 regen scripts and save BOTH .png and .pdf vector to SUBMIT_FINAL folder.

Monkey-patches matplotlib.pyplot.savefig to detect .png output and additionally save a .pdf companion.
Also redirects each script's OUT constant to point at SUBMIT_FINAL_2026-05-18/figures/.
Existing PNGs in SUBMIT_FINAL are overwritten with the freshly-rendered ones (same data).
"""
import os, sys, io, runpy
_ORIG_STDOUT = sys.stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_NEW = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison\SUBMIT_FINAL_2026-05-18\figures"
os.makedirs(OUT_NEW, exist_ok=True)

# Monkey-patch savefig: when a .png is written, also write a .pdf companion
_orig_savefig = plt.savefig
def _patched_savefig(fname, *args, **kwargs):
    fname = str(fname)
    # Force OUT redirection: replace any path containing JPS_Submission_Files\figures with SUBMIT_FINAL\figures
    if "JPS_Submission_Files" in fname and "figures" in fname:
        fname = os.path.join(OUT_NEW, os.path.basename(fname))
    out = _orig_savefig(fname, *args, **kwargs)
    if fname.lower().endswith(".png"):
        pdf_name = fname[:-4] + ".pdf"
        # Render PDF with same kwargs minus dpi (vector doesn't need it)
        pdf_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
        _orig_savefig(pdf_name, *args, **pdf_kwargs)
        print(f"  + PDF: {os.path.basename(pdf_name)}")
    return out

plt.savefig = _patched_savefig

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGETS = [
    os.path.join(SCRIPTS_DIR, "regen_figures_3_5_6_7.py"),
    os.path.join(SCRIPTS_DIR, "regen_figure_4_real.py"),
]

def _strip_stdout_wrap(src: str) -> str:
    out_lines = []
    for ln in src.splitlines():
        if "TextIOWrapper(sys.stdout.buffer" in ln:
            out_lines.append("# stripped by export_figures_pdf wrapper: " + ln.strip())
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)

for script in TARGETS:
    print(f"\n=== Running {os.path.basename(script)} ===", flush=True)
    try:
        with open(script, "r", encoding="utf-8") as f:
            src = _strip_stdout_wrap(f.read())
        ns = {"__name__": "__main__", "__file__": script}
        exec(compile(src, script, "exec"), ns)
    except SystemExit:
        pass
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}", flush=True)

print("\n=== Done. Outputs in: ===")
print(f"  {OUT_NEW}")
