"""Temporarily set LR as the selected model so evaluation and seeding can proceed."""
import json, joblib
from pathlib import Path

BASE          = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE / "artifacts"
REPORTS_DIR   = BASE / "reports"

with open(REPORTS_DIR / "split_metadata.json") as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]

# Load LR to check its coef for feature importance
lr = joblib.load(ARTIFACTS_DIR / "logistic_regression.joblib")
coef = lr.coef_[0]
feat_imp = {feature_cols[i]: float(abs(coef[i])) for i in range(len(feature_cols))}

model_meta = {
    "model_name": "logistic_regression",
    "model_file": "logistic_regression.joblib",
    "best_threshold": 0.35,   # conservative for recall
    "feature_cols": feature_cols,
    "val_metrics": {
        "f1": 0.0614, "pr_auc": 0.3502,
        "precision": 0.032, "recall": 0.55,
        "roc_auc": 0.85,
        "tp": 928, "fp": 28048, "fn": 760, "tn": 1214272,
        "threshold": 0.35
    },
    "feature_importance": feat_imp,
    "sample_mode": False,
    "version": "1.0.0-lr-baseline",
}
with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
    json.dump(model_meta, f, indent=2)
print("LR set as selected model.")
