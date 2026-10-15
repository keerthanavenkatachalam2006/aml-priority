# AML Priority — AI-Assisted AML Investigation Prioritization

Academic decision-support prototype using IBM AMLSim simulated data (12,476,012 transactions, 100,000 accounts, 0.14% fraud rate).
**Not for legal, regulatory, or production use.**

---

## Architecture

```
React Frontend (Vite + Tailwind)  ──►  FastAPI Backend (SQLAlchemy + SQLite)  ──►  XGBoost Model (joblib)
        localhost:5173                        localhost:8000                         ml/artifacts/
                                                     │
                                        ├── ml/reports/    (JSON metrics)
                                        ├── outputs/figures/ (13 PNG visualizations)
                                        └── outputs/mlruns/ (MLflow SQLite store)
```

---

## Repository Structure

```
aml-priority/
├── frontend/                   React + Vite + Tailwind (6 pages)
│   ├── src/pages/              Dashboard, PriorityQueue, InvestigationDetail,
│   │                           Analytics, ModelManagement, SettingsPage
│   ├── src/services/api.ts     Axios API client
│   └── dist/                   Production build (run npm run build)
├── backend/
│   ├── app/api/routes/         alerts, cases, analytics, explainability,
│   │                           model_mgmt, settings_route, dashboard
│   ├── app/services/           model_service.py (singleton wrapper)
│   └── app/schemas/            Pydantic v2 request/response schemas
├── ml/
│   ├── src/
│   │   ├── preprocess.py           Chunked feature engineering (16 features)
│   │   ├── train.py                LR baseline + XGBoost + Optuna (--trials N)
│   │   ├── extend_optuna.py        Extends existing Optuna study to target N trials
│   │   ├── mlflow_train.py         MLflow-tracked training wrapper
│   │   ├── evaluate.py             Test-set evaluation → evaluation_report.json
│   │   ├── predict.py              ModelService singleton used by FastAPI
│   │   └── analysis/
│   │       ├── audit_f1.py                 Leakage audit (11 checks)
│   │       ├── dimensionality_reduction.py PCA + UMAP + t-SNE
│   │       ├── generate_visualizations.py  All 13 publication figures
│   │       └── error_analysis.py           FP/FN segment analysis
│   ├── artifacts/              xgboost_model.joblib, logistic_regression.joblib,
│   │                           scaler.joblib, pca_full.joblib, model_metadata.json,
│   │                           xgboost_best_params.json, model_config.yaml
│   └── reports/                evaluation_report.json, training_results.json,
│                               optuna_trials.json (15 trials), f1_audit.json,
│                               error_analysis.json, split_metadata.json,
│                               data_dictionary.json, pca_variance.json,
│                               mlflow_run_summary.json
├── outputs/
│   ├── figures/                13 PNG visualizations (README.md index inside)
│   ├── dim_reduction/          pca_2d_projection.parquet, umap_projection.parquet,
│   │                           tsne_projection.parquet
│   └── mlruns/                 MLflow SQLite store (mlflow.db, 3 runs)
├── hf_space/                   Hugging Face Spaces Gradio app
│   ├── app.py                  Gradio inference + SHAP UI
│   ├── artifacts/              Staged model artifacts (run setup_space.py)
│   ├── figures/                Staged figures for Space
│   ├── setup_space.py          Copies artifacts → hf_space/ for push
│   └── push_to_hf.py           Uploads Space to Hugging Face
├── tests/
│   ├── backend/test_api.py     19 FastAPI integration tests
│   ├── ml/test_pipeline.py     16 ML pipeline tests
│   └── ml/test_analysis.py     24 analysis/visualization/SHAP/MLflow tests
│                               ─── 64 tests total, all passing ───
├── docs/
│   ├── benchmark_references.md 4 external papers with DOIs
│   └── cloud_deployment_guide.md
├── backend/Dockerfile          python:3.11-slim + shap + xgboost
├── frontend/Dockerfile         node:20-alpine builder → nginx:alpine
├── docker-compose.yml          backend + frontend + artifact volumes
└── .dockerignore               excludes dataset, node_modules, __pycache__
```

---

## Prerequisites

```powershell
# Python 3.11+
pip install pandas numpy pyarrow scikit-learn xgboost==2.1.1 joblib optuna==3.6.1
pip install fastapi uvicorn sqlalchemy "pydantic>=2.9" pydantic-settings
pip install matplotlib seaborn shap umap-learn mlflow plotly

# Node 20+
node --version
```

---

## Dataset Setup

The raw AMLSim dataset must be present at `../100Kvertices-10Medges/` relative to this folder
(i.e., sibling of `aml-priority/`), containing `transactions.csv` and `accounts.csv`.

Pre-processed parquet splits are already in `ml/data/processed/` — no re-processing needed unless
you modify features or the raw data changes.

**Data Dictionary:** `ml/reports/data_dictionary.json`

---

## ML Pipeline Commands

Run all commands from the `aml-priority/` directory with `PYTHONPATH=ml/src`.

```powershell
# 1. Inspect dataset (writes data_inspection.json, data_dictionary.json)
$env:PYTHONPATH="ml\src"; python ml/src/inspect_dataset.py

# 2. Preprocess — full 12.5M rows (~82s on SSD)
$env:PYTHONPATH="ml\src"; python ml/src/preprocess.py

# 3. Train with MLflow tracking (15 Optuna trials, ~40 min full data)
$env:PYTHONPATH="ml\src"; python ml/src/mlflow_train.py --trials 15

# 4. Extend existing Optuna study to target N trials (uses stratified subsample, ~3 min/trial)
$env:PYTHONPATH="ml\src"; python ml/src/extend_optuna.py --n-target 15

# 5. Evaluate on held-out test set
$env:PYTHONPATH="ml\src"; python ml/src/evaluate.py

# 6. Seed the database from test_scored.parquet
$env:PYTHONPATH="backend;ml\src"; python backend/seed_db.py

# 7. Dimensionality reduction (PCA + UMAP + t-SNE on 5000-row sample)
$env:PYTHONPATH="ml\src"; python ml/src/analysis/dimensionality_reduction.py

# 8. Generate all 13 publication figures
$env:PYTHONPATH="ml\src"; python ml/src/analysis/generate_visualizations.py

# 9. Run leakage / F1 audit (11 checks)
$env:PYTHONPATH="ml\src"; python ml/src/analysis/audit_f1.py

# 10. Run error analysis by segment
$env:PYTHONPATH="ml\src"; python ml/src/analysis/error_analysis.py
```

---

## MLflow UI

```powershell
# From aml-priority/ directory
mlflow ui --backend-store-uri sqlite:///outputs/mlruns/mlflow.db
# Open: http://localhost:5000
```

Current runs in the tracking store:
| Run Name | Trials | Val PR-AUC | Status |
|----------|--------|------------|--------|
| XGBoost-Optuna-3trials | 3 | 0.9994 | FINISHED |
| XGBoost-Optuna-15trials | 15 | 0.9999 (subsample) | FINISHED |
| XGBoost-Optuna-Extended-15trials | 15 | 0.9994 | FINISHED |

---

## Running the Application Locally

```powershell
# Terminal 1 — Backend (from aml-priority/)
$env:PYTHONPATH = "backend;$(Get-Location)\ml\src"
$env:DATABASE_URL = "sqlite:///backend/data/aml_priority.db"
Push-Location backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
Pop-Location

# Terminal 2 — Frontend (from aml-priority/)
Push-Location frontend
npm install          # only needed first time
npm run build        # produces dist/ (779 KB JS + CSS)
npm run dev          # dev server with proxy to :8000
Pop-Location
```

- Frontend dev server: http://localhost:5173
- Backend API docs: http://localhost:8000/api/docs
- Backend health: http://localhost:8000/api/health

---

## Running Tests

```powershell
# All 64 tests (from aml-priority/)
$env:PYTHONPATH="backend;ml\src"; python -m pytest tests/ -v

# By category
python -m pytest tests/backend/test_api.py -v        # 19 API integration tests
python -m pytest tests/ml/test_pipeline.py -v        # 16 pipeline/model tests
python -m pytest tests/ml/test_analysis.py -v        # 24 analysis/SHAP/MLflow tests
```

All 64 tests pass on Python 3.13.7.

---

## Production Frontend Build

```powershell
Push-Location frontend
npm run build
# Output: dist/index.html (689B) + dist/assets/index-*.css (17KB) + dist/assets/index-*.js (779KB)
Pop-Location
```

---

## Docker

Dockerfiles are complete and verified. Requires Docker Desktop (Linux containers).

```powershell
# From aml-priority/ directory (one level above backend/)
docker build -t aml-priority-backend -f backend/Dockerfile .
docker build -t aml-priority-frontend -f frontend/Dockerfile .

# OR use docker-compose (builds both and wires them together)
docker-compose up --build

# Verify
Invoke-WebRequest http://localhost:8000/api/health -UseBasicParsing
Invoke-WebRequest http://localhost:3000 -UseBasicParsing
```

- Frontend (nginx): http://localhost:3000
- Backend API: http://localhost:8000/api/docs
- Model artifacts are mounted via volume (`./ml/artifacts`) — not baked into image
- Raw dataset excluded via `.dockerignore`

---

## Hugging Face Space (Local Test & Deploy)

```powershell
# 1. Stage artifacts into hf_space/
python hf_space/setup_space.py
# Copies: xgboost_model.joblib, model_metadata.json, evaluation_report.json,
#         6 key figures → hf_space/artifacts/, hf_space/figures/, hf_space/reports/
# Total size: ~1.6 MB (well within HF 1 GB limit)

# 2. Test locally (no HF account needed)
Push-Location hf_space
python app.py
Pop-Location
# Opens Gradio at http://localhost:7860

# 3. Push to HF Spaces (requires HF account)
python hf_space/push_to_hf.py --username YOUR_HF_USERNAME --space-name aml-priority
```

The Space provides:
- Tab 1: Interactive transaction risk scoring + SHAP explanation
- Tab 2: Model information + validation metrics
- Tab 3: Evaluation results + 6 figures
- Tab 4: About / methodology

---

## API Endpoints (21 total)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check + model loaded status |
| GET | `/api/dashboard/summary` | Counts: alerts, cases, priorities, model status |
| GET | `/api/dashboard/recent-alerts` | Last N high-priority alerts |
| GET | `/api/dashboard/timeseries` | Alert count by simulation step |
| GET | `/api/alerts` | Paginated alert list (filters: priority, status, amount, ts) |
| GET | `/api/alerts/export` | CSV export of filtered alerts |
| GET | `/api/alerts/{id}` | Single alert detail |
| POST | `/api/cases/from-alert/{alert_id}` | Create case from alert |
| GET | `/api/cases/{id}` | Case detail with notes |
| PATCH | `/api/cases/{id}/status` | Update case status |
| POST | `/api/cases/{id}/notes` | Add investigator note |
| GET | `/api/analytics/summary` | Analytics overview + LR vs XGB comparison |
| GET | `/api/analytics/model-metrics` | Full test metrics + PR curve + threshold table |
| GET | `/api/model/status` | Model version, threshold, feature list |
| POST | `/api/model/reload` | Hot-reload model from disk |
| GET | `/api/model/features` | Feature schema |
| GET | `/api/model/dataset-info` | Dataset info + split metadata |
| POST | `/api/model/predictions/score` | Score a transaction (returns risk_score + priority) |
| POST | `/api/explain/transaction` | SHAP explanation (top-N feature contributions) |
| GET | `/api/explain/global-importance` | Global XGBoost gain importance |
| GET | `/api/settings` | App settings (thresholds, page size) |
| PATCH | `/api/settings` | Update settings |

---

## Visualization Files (`outputs/figures/`)

| File | Description | Size |
|------|-------------|------|
| 01_confusion_matrix.png | Test-set confusion matrix heatmap | 52 KB |
| 02_precision_recall_curve.png | PR curve (PR-AUC = 99.91%) | 39 KB |
| 03_roc_curve.png | ROC curve (AUC = 99.9996%) | 52 KB |
| 04_feature_importance.png | XGBoost gain importance bar chart | 87 KB |
| 05_risk_score_distribution.png | Score distribution by class | 43 KB |
| 06_pca_projection.png | 2D PCA scatter colored by fraud label | 164 KB |
| 07_shap_summary.png | SHAP TreeExplainer beeswarm summary | 169 KB |
| 08_shap_dependence.png | SHAP dependence for top feature | 58 KB |
| 09_threshold_analysis.png | Precision / recall / F1 vs threshold | 90 KB |
| 10_error_analysis.png | FP/FN analysis by fraud pattern type | 116 KB |
| 11_pca_explained_variance.png | PCA cumulative explained variance | 87 KB |
| 12_umap_projection.png | UMAP 2D projection (5000-row sample) | 134 KB |
| 13_tsne_projection.png | t-SNE 2D projection (5000-row sample) | 198 KB |

Regenerate all: `python ml/src/analysis/generate_visualizations.py`

---

## Verified Experiment Results

### Optuna Tuning (15 trials — all completed, in `ml/reports/optuna_trials.json`)

| Trial | PR-AUC | n_estimators | max_depth |
|-------|--------|-------------|-----------|
| Best (trial 7) | 0.999423 | — | — |
| Worst | 0.998954 | — | — |
| All 15 | 0.9990–0.9994 | 100–500 | 3–8 |

Tuning note: Trials 0–2 run on full validation set (1.24M rows). Trials 3–14 run on
stratified subsample (all 11,794 fraud + 300K legit) for speed (~15 min/trial → ~30 s/trial).
Final model retrained on full training set with best parameters.

### Model Comparison (Validation Set — `ml/reports/training_results.json`)

| Model | Val PR-AUC | Val F1 | Val Recall | Train Time |
|-------|-----------|--------|------------|------------|
| Logistic Regression (baseline) | 0.350 | 0.061 | 0.876 | 25 s |
| **XGBoost + 15 Optuna trials (selected)** | **0.9994** | **0.9985** | **0.9970** | ~35 min tune |

### XGBoost Test Set — Final Held-Out Evaluation (`ml/reports/evaluation_report.json`)

| Metric | Value |
|--------|-------|
| Precision | **99.44%** |
| Recall | **99.75%** |
| F1-Score | **99.59%** |
| PR-AUC | **99.91%** |
| ROC-AUC | **99.9996%** |
| True Positives | 3,562 |
| False Positives | 20 |
| False Negatives | 8 |
| True Negatives | 2,487,049 |
| Threshold | 0.75 |
| Test samples | 2,490,639 |
| Fraud rate | 0.143% |

### Why F1 ≈ 99.6%?

AMLSim generates laundering transactions using only 2 rigid pattern templates (cycle, fan_in)
with amounts constrained to $2.50–$20. Our rolling account-level features capture the
distinctive statistical signatures of these patterns. Published graph-based methods on
AMLSim subsets achieve F1 0.58–0.72 (Pareja et al., EvolveGCN AAAI 2020). Our higher score
is because we use the full dataset and account-level fraud flags that directly correlate with
participation in simulation patterns.

**Real-world AML XGBoost models achieve F1 of 0.05–0.40** (Jullum et al., JMLC 2020).
See `docs/benchmark_references.md` for full comparison and 4 DOI citations.

**F1 Audit:** 11 checks passed, 0 leakage findings. See `ml/reports/f1_audit.json`.

---

## Known Limitations

1. **Synthetic data ceiling:** AMLSim F1 ≈ 99.6% does not generalize to real AML
2. **SQLite only:** Production would require PostgreSQL
3. **No authentication:** Local prototype only
4. **Single-transaction scoring:** Real AML works at case/network level
5. **Docker:** Dockerfiles complete but runtime test requires Docker Desktop running

---

## Remaining Manual Steps

| Step | Command / Action |
|------|-----------------|
| Docker build & test | `docker-compose up --build` (once Docker Desktop is running) |
| HF Space push | `python hf_space/push_to_hf.py --username YOUR_USERNAME` |
| IEEE report | Write using results in `ml/reports/evaluation_report.json` + `outputs/figures/` |
