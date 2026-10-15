# Figure Index — AML Priority Visualizations
All figures use actual evaluation data. No synthetic results.
Generated from: `python ml/src/analysis/generate_visualizations.py`

| # | File | Description | Data Source |
|---|------|-------------|-------------|
| 1 | `01_confusion_matrix.png` | Confusion Matrix | XGBoost test predictions vs true labels |
| 2 | `02_precision_recall_curve.png` | Precision-Recall Curve | Test set probabilities + PR-AUC |
| 3 | `03_roc_curve.png` | ROC Curve | Test set probabilities + ROC-AUC |
| 4 | `04_feature_importance.png` | Feature Importance | XGBoost gain scores |
| 5 | `05_risk_score_distribution.png` | Risk Score Distribution | Test probabilities by class |
| 6 | `06_pca_projection.png` | PCA 2D Projection | Training sample 200K, PCA fitted on 200K |
| 7 | `07_shap_summary.png` | SHAP Summary / Fallback | TreeExplainer on 2000 test samples |
| 8 | `08_shap_dependence.png` | SHAP Dependence Plot | Top feature, 2000 test samples |
| 9 | `09_threshold_analysis.png` | Threshold Sensitivity | Test set precision/recall/F1 vs threshold |
| 10 | `10_error_analysis.png` | Error Analysis | FP/FN by amount, step, risk score |
| 11 | `11_pca_explained_variance.png` | PCA Explained Variance | 16-component PCA on training sample |
| 12 | `12_umap_projection.png` | UMAP 2D Projection | Training sample 8000 rows |
| 13 | `13_tsne_projection.png` | t-SNE 2D Projection | Training sample 3000 rows |
