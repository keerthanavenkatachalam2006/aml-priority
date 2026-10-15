# AML Priority — Rubric Completion Checklist

## Rubric 1: Data Engineering & Visualization (8 marks)

| Requirement | Status | File / Evidence |
|---|---|---|
| Complex tabular dataset (12.5M rows, 100K accounts) | IMPLEMENTED & VERIFIED | `100Kvertices-10Medges/` — AMLSim IBM |
| Advanced feature engineering (16 leakage-safe features) | IMPLEMENTED & VERIFIED | `ml/src/preprocess.py` |
| Chronological split with temporal boundaries | IMPLEMENTED & VERIFIED | `ml/reports/split_metadata.json` |
| Chunked processing (no full RAM load) | IMPLEMENTED & VERIFIED | `ml/src/preprocess.py` — 1M row chunks |
| Rolling account features (past-only) | IMPLEMENTED & VERIFIED | `ml/src/preprocess.py` — audit confirms no leakage |
| Dimensionality reduction — PCA (all 16 components) | IMPLEMENTED & VERIFIED | `ml/src/analysis/dimensionality_reduction.py`, `ml/artifacts/pca_full.joblib`, `ml/reports/pca_variance.json` — 10 PCs for 90% variance |
| Dimensionality reduction — UMAP | IMPLEMENTED & VERIFIED | `outputs/dim_reduction/umap_projection.parquet` |
| Dimensionality reduction — t-SNE | IMPLEMENTED & VERIFIED | `outputs/dim_reduction/tsne_projection.parquet` |
| 5+ visualizations | IMPLEMENTED & VERIFIED (13 figures) | `outputs/figures/` — all non-empty PNGs |
| Confusion matrix heatmap | VERIFIED | `outputs/figures/01_confusion_matrix.png` (52 KB) |
| Precision-Recall curve | VERIFIED | `outputs/figures/02_precision_recall_curve.png` (39 KB) |
| ROC curve | VERIFIED | `outputs/figures/03_roc_curve.png` (52 KB) |
| Feature importance | VERIFIED | `outputs/figures/04_feature_importance.png` (87 KB) |
| Risk score distribution | VERIFIED | `outputs/figures/05_risk_score_distribution.png` (43 KB) |
| PCA 2D projection | VERIFIED | `outputs/figures/06_pca_projection.png` (164 KB) |
| SHAP summary plot | VERIFIED | `outputs/figures/07_shap_summary.png` (169 KB) |
| SHAP dependence plot | VERIFIED | `outputs/figures/08_shap_dependence.png` (58 KB) |
| Threshold analysis | VERIFIED | `outputs/figures/09_threshold_analysis.png` (90 KB) |
| Error analysis plot | VERIFIED | `outputs/figures/10_error_analysis.png` (116 KB) |
| PCA explained variance | VERIFIED | `outputs/figures/11_pca_explained_variance.png` (87 KB) |
| UMAP projection | VERIFIED | `outputs/figures/12_umap_projection.png` (134 KB) |
| t-SNE projection | VERIFIED | `outputs/figures/13_tsne_projection.png` (198 KB) |
| Figure index | VERIFIED | `outputs/figures/README.md` |

**Regenerate command:** `python ml/src/analysis/generate_visualizations.py`

---

## Rubric 2: Advanced Modeling & Pipeline Optimization (10 marks)

| Requirement | Status | File / Evidence |
|---|---|---|
| XGBoost advanced model | IMPLEMENTED & VERIFIED | `ml/artifacts/xgboost_model.joblib` |
| Logistic Regression baseline | IMPLEMENTED & VERIFIED | `ml/artifacts/logistic_regression.joblib` |
| Optuna TPE hyperparameter tuning (15 trials) | IMPLEMENTED & VERIFIED | `ml/src/train.py`, `ml/reports/optuna_trials.json`, `ml/artifacts/xgboost_best_params.json` |
| Modular pipeline (preprocess→train→evaluate→predict) | VERIFIED | 5 separate scripts in `ml/src/` |
| MLflow experiment tracking | IMPLEMENTED & VERIFIED | `ml/src/mlflow_train.py`, `outputs/mlruns/mlflow.db` — run_id=d77bddd90ac5 |
| MLflow logs params, metrics, artifacts | VERIFIED | `ml/reports/mlflow_run_summary.json` — val_f1=0.9985 logged |
| Scaler fitted on train only | VERIFIED | `ml/artifacts/scaler.joblib` — fit in preprocess.py on train split |
| Feature-order consistency check | VERIFIED | `tests/ml/test_analysis.py::test_feature_order_matches_model` |
| model_config.yaml | VERIFIED | `ml/artifacts/model_config.yaml` |
| SHAP explainability | IMPLEMENTED & VERIFIED | `ml/src/analysis/generate_visualizations.py`, `backend/app/api/routes/explainability.py` |
| `/api/explain/transaction` endpoint | VERIFIED | `tests/ml/test_analysis.py::test_explain_endpoint_available` PASS |

**MLflow UI command:** `mlflow ui --backend-store-uri sqlite:///aml-priority/outputs/mlruns/mlflow.db`

---

## Rubric 3: Benchmarking & Comparative Analysis (10 marks)

| Requirement | Status | File / Evidence |
|---|---|---|
| LR vs XGBoost internal comparison | VERIFIED | `ml/reports/training_results.json` |
| External literature comparison | IMPLEMENTED | `docs/benchmark_references.md` — 4 papers with DOIs |
| F1 score audit (verify no leakage) | VERIFIED | `ml/src/analysis/audit_f1.py`, `ml/reports/f1_audit.json` — 11/11 checks pass, 0 leakage |
| Error analysis by segment | VERIFIED | `ml/src/analysis/error_analysis.py`, `ml/reports/error_analysis.json` |
| PR-AUC, ROC-AUC, F1, confusion matrix on test | VERIFIED | `ml/reports/evaluation_report.json` |
| Threshold sensitivity analysis | VERIFIED | `ml/reports/evaluation_report.json::threshold_table` + figure 09 |
| Real-world AML comparison & limitations | DOCUMENTED | `docs/benchmark_references.md` — real-world F1 0.05-0.40 context |

**Test metrics (reproduced 3x):** Precision=99.66%, Recall=99.78%, F1=99.72%, PR-AUC=99.91%

---

## Rubric 4: Deployment & Containerization (8 marks)

### Local FastAPI App (4/4)

| Requirement | Status | File / Evidence |
|---|---|---|
| FastAPI app with 20 endpoints | VERIFIED | `backend/app/main.py` + route files |
| Pydantic input validation | VERIFIED | `backend/app/schemas/schemas.py` |
| Real-time inference with `/api/model/predictions/score` | VERIFIED | test_score_prediction PASS |
| SHAP explainability endpoint `/api/explain/transaction` | VERIFIED | 19/19 integration checks |
| 28/28 API checks pass | VERIFIED | `backend/verify_api.py` output |
| 64/64 automated tests pass | VERIFIED | `pytest tests/` output |
| Frontend running on :5173 | VERIFIED | HTTP 200 confirmed |

**Start commands:**
```powershell
# Backend
Push-Location aml-priority\backend
$env:PYTHONPATH="$(Get-Location)"
python -m uvicorn app.main:app --port 8000
# API docs: http://localhost:8000/api/docs

# Frontend
Push-Location aml-priority\frontend
npm run dev
# UI: http://localhost:5173
```

### Docker & Cloud (partial — 2/4)

| Requirement | Status | File / Evidence |
|---|---|---|
| Dockerfile (backend) — correct, PYTHONPATH fixed | IMPLEMENTED | `backend/Dockerfile` — `PYTHONPATH=/app/backend:/app/ml/src` |
| Dockerfile (frontend) — multi-stage nginx | IMPLEMENTED | `frontend/Dockerfile` — built successfully (94.6 MB) |
| docker-compose.yml — volumes, healthcheck, ports | IMPLEMENTED | `docker-compose.yml` |
| .dockerignore | IMPLEMENTED | `.dockerignore` — excludes dataset, processed data, node_modules |
| Backend image built (799 MB) | VERIFIED | `docker build` completed — image `aml-priority-backend:latest` created 14-Oct-2026 |
| Frontend image built (94.6 MB) | VERIFIED | `docker build` completed — image `aml-priority-frontend:latest` created |
| `docker-compose up` end-to-end test | PARTIALLY VERIFIED | Backend container started but crashed on first build (missing PYTHONPATH); fixed in current Dockerfile; Docker Desktop unavailable for re-test |
| HF Space Gradio app | IMPLEMENTED | `hf_space/app.py`, `hf_space/requirements.txt` — artifacts copied, tested locally |
| Public cloud URL | PENDING — requires your HF account | See `docs/cloud_deployment_guide.md` |

**Docker commands (when Docker Desktop is running):**
```powershell
# Rebuild with fix and start
docker build -t aml-priority-backend -f aml-priority/backend/Dockerfile aml-priority/
docker-compose -f aml-priority/docker-compose.yml up -d

# Test health
Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing
Invoke-WebRequest http://localhost:3000 -UseBasicParsing
```

---

## Rubric 5: Documentation & Report Quality (4 marks)

| Requirement | Status | File / Evidence |
|---|---|---|
| README with all commands, architecture, metrics | IMPLEMENTED | `README.md` — 300+ lines |
| Data dictionary | IMPLEMENTED | `ml/reports/data_dictionary.json` |
| Model config YAML | IMPLEMENTED | `ml/artifacts/model_config.yaml` |
| Benchmark/literature reference | IMPLEMENTED | `docs/benchmark_references.md` |
| Cloud deployment guide | IMPLEMENTED | `docs/cloud_deployment_guide.md` |
| Figure index | IMPLEMENTED | `outputs/figures/README.md` |
| Code modularity (separate scripts per concern) | VERIFIED | 13 Python modules across ml/src/ and backend/ |
| IEEE-format PDF report | PENDING — not yet written | Produce after deployment URL obtained |

---

## Summary

| Component | Max | Achieved | Key Evidence |
|---|---|---|---|
| Data Engineering & Visualization | 8 | **7-8** | 13 figures, PCA+UMAP+t-SNE, 16 features, chunked processing |
| Advanced Modeling & Pipeline | 10 | **9-10** | XGBoost + Optuna 15 trials + MLflow + SHAP API |
| Benchmarking & Comparative Analysis | 10 | **8-9** | 4 DOI citations, F1 audit, error analysis, threshold table |
| Local Deployment | 4 | **4** | 64 tests pass, 28 API checks pass, frontend :5173 live |
| Docker & Cloud | 4 | **2-3** | Both images built; docker-compose needs Docker Desktop restart; HF Space ready |
| Documentation | 4 | **3** | README, benchmark doc, model config, deployment guide |
| **Total** | **40** | **33-37** | |

## Remaining Manual Actions

1. **Docker test**: Start Docker Desktop → `docker-compose -f aml-priority/docker-compose.yml up --build` → verify `http://localhost:3000`
2. **HF Spaces**: Create account → run `python aml-priority/hf_space/setup_space.py` → push → get public URL
3. **IEEE Report**: Write 10-page PDF using results from `ml/reports/evaluation_report.json`, figures from `outputs/figures/`, benchmark from `docs/benchmark_references.md`
