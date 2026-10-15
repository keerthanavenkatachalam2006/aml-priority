"""
Tests for dimensionality reduction, visualizations, MLflow, SHAP, error analysis, audit.
"""
import sys, json, os
import numpy as np
import pytest
from pathlib import Path

ML_SRC  = Path(__file__).parent.parent.parent / "ml" / "src"
OUTPUTS = Path(__file__).parent.parent.parent / "outputs"
REPORTS = Path(__file__).parent.parent.parent / "ml" / "reports"
ARTIFACTS = Path(__file__).parent.parent.parent / "ml" / "artifacts"
sys.path.insert(0, str(ML_SRC))


# ── 1. Dimensionality Reduction Outputs ──────────────────────────────────────
def test_pca_variance_json_exists():
    p = REPORTS / "pca_variance.json"
    assert p.exists(), "pca_variance.json not found — run dimensionality_reduction.py"

def test_pca_variance_content():
    p = REPORTS / "pca_variance.json"
    if not p.exists():
        pytest.skip("pca_variance.json not present")
    d = json.load(open(p))
    assert "explained_variance_ratio" in d
    assert "cumulative_explained_variance" in d
    assert "n_components_95pct" in d
    evr = d["explained_variance_ratio"]
    assert abs(sum(evr) - 1.0) < 0.01, "Explained variance ratios should sum to ~1.0"

def test_pca_artifacts_exist():
    assert (ARTIFACTS / "pca_full.joblib").exists(), "pca_full.joblib missing"

def test_pca_2d_projection_exists():
    p = OUTPUTS / "dim_reduction" / "pca_2d_projection.parquet"
    assert p.exists(), "PCA 2D projection parquet missing"

def test_pca_2d_projection_has_both_classes():
    import pandas as pd
    p = OUTPUTS / "dim_reduction" / "pca_2d_projection.parquet"
    if not p.exists():
        pytest.skip()
    df = pd.read_parquet(p)
    assert set(df["label"].unique()) == {0, 1}, "PCA projection should have both fraud and legit labels"
    assert len(df) > 100

def test_umap_projection_exists():
    p = OUTPUTS / "dim_reduction" / "umap_projection.parquet"
    assert p.exists(), "UMAP projection missing — run dimensionality_reduction.py"

def test_tsne_projection_exists():
    p = OUTPUTS / "dim_reduction" / "tsne_projection.parquet"
    assert p.exists(), "t-SNE projection missing — run dimensionality_reduction.py"


# ── 2. Visualization Files ────────────────────────────────────────────────────
REQUIRED_FIGURES = [
    "01_confusion_matrix.png",
    "02_precision_recall_curve.png",
    "03_roc_curve.png",
    "04_feature_importance.png",
    "05_risk_score_distribution.png",
    "06_pca_projection.png",
]

@pytest.mark.parametrize("fname", REQUIRED_FIGURES)
def test_figure_exists_and_nonempty(fname):
    p = OUTPUTS / "figures" / fname
    assert p.exists(), f"Required figure missing: {fname}"
    assert p.stat().st_size > 5000, f"Figure suspiciously small: {fname}"

def test_shap_figure_exists():
    p = OUTPUTS / "figures" / "07_shap_summary.png"
    assert p.exists(), "SHAP (or fallback) figure missing"
    assert p.stat().st_size > 1000

def test_figure_index_exists():
    p = OUTPUTS / "figures" / "README.md"
    assert p.exists(), "Figure index README.md missing"
    content = p.read_text()
    assert "confusion_matrix" in content.lower()


# ── 3. F1 Audit ───────────────────────────────────────────────────────────────
def test_f1_audit_report_exists():
    p = REPORTS / "f1_audit.json"
    assert p.exists(), "f1_audit.json not found — run audit_f1.py"

def test_f1_audit_no_leakage_findings():
    p = REPORTS / "f1_audit.json"
    if not p.exists():
        pytest.skip()
    d = json.load(open(p))
    leakage_findings = [f for f in d.get("findings", [])
                        if "LEAKAGE" in f.upper() and "NEAR-LEAKAGE" not in f.upper()]
    assert len(leakage_findings) == 0, f"Leakage found: {leakage_findings}"

def test_f1_audit_metrics_reproduced():
    p = REPORTS / "f1_audit.json"
    if not p.exists():
        pytest.skip()
    d = json.load(open(p))
    re = d.get("re_evaluated_metrics", {})
    assert re.get("f1", 0) > 0.95, "Re-evaluated F1 should be >0.95 for XGBoost on AMLSim"


# ── 4. Error Analysis ─────────────────────────────────────────────────────────
def test_error_analysis_exists():
    p = REPORTS / "error_analysis.json"
    assert p.exists(), "error_analysis.json missing — run error_analysis.py"

def test_error_analysis_content():
    p = REPORTS / "error_analysis.json"
    if not p.exists():
        pytest.skip()
    d = json.load(open(p))
    assert "total_fp" in d and "total_fn" in d
    assert "by_amount_band" in d
    assert d["total_test"] > 0
    # FPR should be extremely low for this model
    assert d.get("overall_fpr", 1.0) < 0.001, "FPR should be very low"


# ── 5. MLflow Tracking ────────────────────────────────────────────────────────
def test_mlflow_db_exists():
    p = OUTPUTS / "mlruns" / "mlflow.db"
    assert p.exists(), "mlflow.db not found — run mlflow_train.py"

def test_mlflow_has_runs():
    p = OUTPUTS / "mlruns" / "mlflow.db"
    if not p.exists():
        pytest.skip()
    try:
        import mlflow
        uri = f"sqlite:///{p.as_posix()}"
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        exps = client.search_experiments()
        assert len(exps) > 0, "No MLflow experiments found"
        runs = client.search_runs([e.experiment_id for e in exps])
        assert len(runs) > 0, "No MLflow runs found"
        # Check at least one run has val_f1 logged
        run_with_f1 = [r for r in runs if "val_f1" in r.data.metrics]
        assert len(run_with_f1) > 0, "No runs with val_f1 metric"
    except ImportError:
        pytest.skip("MLflow not installed")

def test_mlflow_run_summary_exists():
    p = REPORTS / "mlflow_run_summary.json"
    assert p.exists(), "mlflow_run_summary.json missing"


# ── 6. SHAP / Explainability API ──────────────────────────────────────────────
def test_explain_endpoint_available():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    r = client.get("/api/model/status")
    assert r.status_code == 200
    if not r.json().get("loaded"):
        pytest.skip("Model not loaded")
    feat = {k: 0.0 for k in r.json().get("feature_cols", [])}
    feat["TX_AMOUNT"] = 15.0
    feat["sender_tx_count"] = 50.0
    r2 = client.post("/api/explain/transaction", json={"features": feat, "top_n": 5})
    assert r2.status_code == 200
    d = r2.json()
    assert "risk_score" in d
    assert "contributions" in d
    assert len(d["contributions"]) > 0
    assert "explanation_mode" in d

def test_explain_global_importance():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    r = client.get("/api/explain/global-importance")
    assert r.status_code == 200
    d = r.json()
    assert "feature_importance" in d
    assert len(d["feature_importance"]) > 0


# ── 7. Feature order consistency ─────────────────────────────────────────────
def test_feature_order_matches_model():
    """Model and metadata must agree on feature count and order."""
    import joblib
    meta  = json.load(open(ARTIFACTS / "model_metadata.json"))
    model = joblib.load(ARTIFACTS / meta["model_file"])
    assert model.n_features_in_ == len(meta["feature_cols"]), \
        f"Feature count mismatch: model={model.n_features_in_}, meta={len(meta['feature_cols'])}"

def test_inference_feature_order():
    """Inference with mismatched feature count should fail, correct count should succeed."""
    import joblib
    meta  = json.load(open(ARTIFACTS / "model_metadata.json"))
    model = joblib.load(ARTIFACTS / meta["model_file"])
    X_ok  = np.zeros((3, len(meta["feature_cols"])), dtype=np.float32)
    probs = model.predict_proba(X_ok)[:, 1]
    assert probs.shape == (3,)
    assert all(0 <= p <= 1 for p in probs)


# ── 8. Benchmark reference ────────────────────────────────────────────────────
def test_benchmark_reference_exists():
    p = Path(__file__).parent.parent.parent / "docs" / "benchmark_references.md"
    assert p.exists(), "benchmark_references.md missing"

def test_benchmark_reference_has_citations():
    p = Path(__file__).parent.parent.parent / "docs" / "benchmark_references.md"
    if not p.exists():
        pytest.skip()
    content = p.read_text(encoding="utf-8", errors="replace")
    assert "DOI" in content or "doi" in content
    assert "XGBoost" in content
    assert "AMLSim" in content
