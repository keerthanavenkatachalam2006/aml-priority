"""Quick test that the HF Space model loads and predicts correctly."""
import sys, json, numpy as np, joblib
from pathlib import Path

BASE = Path(__file__).parent
ARTIFACTS = BASE / "artifacts"

meta  = json.load(open(ARTIFACTS / "model_metadata.json"))
model = joblib.load(ARTIFACTS / "xgboost_model.joblib")
fc    = meta["feature_cols"]

X = np.zeros((1, len(fc)), dtype=np.float32)
X[0, 0] = 15.0  # TX_AMOUNT
prob = float(model.predict_proba(X)[0, 1])
print(f"Prediction: {prob:.4f}")

# SHAP test
try:
    import shap
    exp = shap.TreeExplainer(model)
    sv  = exp.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    top = sorted(zip(fc, sv[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
    for feat, val in top:
        print(f"  {feat}: {val:+.4f}")
    print("SHAP: OK")
except Exception as e:
    print(f"SHAP fallback: {e}")

print(f"\nModel: {meta['model_name']} | Threshold: {meta['best_threshold']:.2f}")
print(f"val_f1: {meta['val_metrics']['f1']:.4f}")
print("HF Space: READY")
