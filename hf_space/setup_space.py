"""
Prepares the HF Space directory by copying required artifacts and figures.
Run this before pushing to Hugging Face.

Usage:
    python hf_space/setup_space.py
"""
import shutil, json
from pathlib import Path

ROOT      = Path(__file__).parent.parent
HF_DIR    = Path(__file__).parent
ARTIFACTS = ROOT / "ml" / "artifacts"
REPORTS   = ROOT / "ml" / "reports"
FIGURES   = ROOT / "outputs" / "figures"

# Copy artifacts
(HF_DIR / "artifacts").mkdir(exist_ok=True)
for f in ["xgboost_model.joblib", "model_metadata.json"]:
    src = ARTIFACTS / f
    if src.exists():
        shutil.copy2(src, HF_DIR / "artifacts" / f)
        print(f"Copied artifact: {f} ({src.stat().st_size/1024:.0f} KB)")
    else:
        print(f"WARNING: {f} not found")

# Copy reports
(HF_DIR / "reports").mkdir(exist_ok=True)
for f in ["evaluation_report.json", "f1_audit.json"]:
    src = REPORTS / f
    if src.exists():
        shutil.copy2(src, HF_DIR / "reports" / f)
        print(f"Copied report: {f}")

# Copy figures (top 6 for Space)
(HF_DIR / "figures").mkdir(exist_ok=True)
fig_files = [
    "01_confusion_matrix.png",
    "02_precision_recall_curve.png",
    "04_feature_importance.png",
    "07_shap_summary.png",
    "09_threshold_analysis.png",
    "11_pca_explained_variance.png",
]
for f in fig_files:
    src = FIGURES / f
    if src.exists():
        shutil.copy2(src, HF_DIR / "figures" / f)
        print(f"Copied figure: {f} ({src.stat().st_size/1024:.0f} KB)")

# Check total size
total = sum(f.stat().st_size for f in HF_DIR.rglob("*") if f.is_file()) / (1024*1024)
print(f"\nTotal HF Space size: {total:.1f} MB")
if total > 900:
    print("WARNING: Approaching HF Spaces 1GB limit. Consider reducing figures.")
else:
    print("OK: Within HF Spaces size limit.")
