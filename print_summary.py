import json
from pathlib import Path

base = Path("aml-priority/ml")
ev = json.load(open(base / "reports" / "evaluation_report.json"))
tr = json.load(open(base / "reports" / "training_results.json"))
tm = ev["test_metrics"]
cm = ev["confusion_matrix"]

print("=== FINAL EXPERIMENT SUMMARY ===")
print()
print("Dataset: AMLSim 100K accounts, 12,476,012 transactions (0.137% fraud)")
print("Split:   Train 0-139 (8.74M rows) | Val 140-159 (1.24M) | Test 160-199 (2.49M)")
print()
print("MODEL COMPARISON (Validation Set):")
lr = tr["logistic_regression"]["val_metrics"]
xg = tr["xgboost"]["val_metrics"]
print(f"  LogisticRegression  F1={lr['f1']:.4f}  PR-AUC={lr['pr_auc']:.4f}  Recall={lr['recall']:.4f}  Train={tr['logistic_regression']['train_time_s']}s")
print(f"  XGBoost (selected)  F1={xg['f1']:.4f}  PR-AUC={xg['pr_auc']:.4f}  Recall={xg['recall']:.4f}  Tune={tr['xgboost']['tune_time_s']}s")
print()
print("SELECTED MODEL TEST RESULTS (XGBoost, threshold=85%):")
print(f"  Precision: {tm['precision']*100:.2f}%")
print(f"  Recall:    {tm['recall']*100:.2f}%")
print(f"  F1-Score:  {tm['f1']*100:.2f}%")
print(f"  PR-AUC:    {tm['pr_auc']*100:.2f}%")
print(f"  ROC-AUC:   {tm['roc_auc']*100:.4f}%")
print(f"  TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}")
print(f"  False Positive Rate: {tm['false_positive_rate']*100:.4f}%")
print()
print("Alerts in DB: 3,641 (scored test transactions above medium threshold)")
print()
print("NOTE: Near-perfect metrics reflect regularity of AMLSim simulated patterns.")
print("      Real-world AML fraud detection F1 is typically 0.1-0.6.")
