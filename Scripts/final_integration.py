"""Final integration after all GPU runs complete:
1. Plot lambda ablation Supp Fig S3
2. Plot physics params Supp Fig S4
3. Re-export DOCX with all updates
4. Run final cluster check
"""
import os, subprocess, sys

ROOT = r"D:\Project Python\PythonProject9\Paper 7\Paper4_UQ_Comparison"
PYTHON = sys.executable

def run(name, cmd):
    print(f"\n=== {name} ===")
    r = subprocess.run(cmd, cwd=ROOT, shell=True)
    return r.returncode == 0

# 1. Generate the 2 new supplementary figures
run("Plot lambda ablation", [PYTHON, "Scripts/plot_lambda_ablation.py"])
run("Plot physics params", [PYTHON, "Scripts/plot_physics_params.py"])

# 2. Re-export manuscript DOCX
run("Convert manuscript", [PYTHON, "convert_to_docx_v3.py"])
import shutil
shutil.copy(os.path.join(ROOT, "Paper4_Manuscript.docx"),
            os.path.join(ROOT, "JPS_Submission_Files", "Paper4_Manuscript.docx"))
run("Embed figures", [PYTHON, "embed_figures.py"])
shutil.copy(os.path.join(ROOT, "JPS_Submission_Files", "Paper4_Manuscript.docx"),
            os.path.join(ROOT, "Paper4_Manuscript.docx"))

# 3. Re-export supplementary
run("Convert supplementary", [PYTHON, "convert_supp_to_docx.py"])
shutil.copy(os.path.join(ROOT, "Paper4_Supplementary.docx"),
            os.path.join(ROOT, "JPS_Submission_Files", "Paper4_Supplementary.docx"))

# 4. Re-export cover letter
run("Convert cover letter", [PYTHON, "convert_cover_to_docx.py"])
shutil.copy(os.path.join(ROOT, "Paper4_CoverLetter.docx"),
            os.path.join(ROOT, "JPS_Submission_Files", "Paper4_CoverLetter.docx"))

# 5. Cluster check
run("Cluster check", [PYTHON, "smart_cluster_check.py"])

print("\n=== FINAL FILES ===")
for f in ["Paper4_Manuscript.docx", "Paper4_Supplementary.docx",
          "Paper4_CoverLetter.docx", "Paper4_Highlights.docx"]:
    p = os.path.join(ROOT, "JPS_Submission_Files", f)
    if os.path.exists(p):
        print(f"  {f}: {os.path.getsize(p)/1024:.0f} KB")
