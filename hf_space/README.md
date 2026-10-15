---
title: AML Priority - Risk Scoring Demo
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
short_description: AML risk scoring with XGBoost on IBM AMLSim
---

# AML Priority — AI-Assisted AML Investigation

Academic prototype demonstrating ML-based AML transaction risk scoring and explainability.

**Dataset:** IBM AMLSim (12.5M simulated transactions, 100K accounts, 0.14% fraud rate)  
**Model:** XGBoost with 15 Optuna TPE hyperparameter trials  
**Test F1:** 99.59% · PR-AUC: 99.91% · ROC-AUC: 99.9996%

⚠️ Near-perfect metrics reflect AMLSim's synthetic patterns — real-world AML F1 is 0.05–0.40.  
Not for legal, regulatory, or production use.

## What's in this Space
- Full evaluation results with metrics table and confusion matrix
- 6 embedded visualization figures (PR curve, confusion matrix, SHAP, feature importance, threshold analysis, PCA)
- Model vs baseline comparison
- Benchmark against 4 external papers (with DOIs)
- System architecture overview

## References
- IBM AMLSim: https://github.com/IBM/AMLSim
- Jullum et al. (2020) DOI:10.1108/JMLC-07-2019-0055
- Pareja et al. EvolveGCN AAAI 2020: DOI:10.1609/aaai.v34i04.5984
