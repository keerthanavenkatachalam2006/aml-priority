import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import ModelMetricsOut, ConfusionMatrix
from app.core.config import settings

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _load_eval_report() -> dict:
    base = Path(settings.ml_reports_dir)
    for name in ["evaluation_report.json", "evaluation_report_sample.json"]:
        p = base / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


def _load_training_results() -> dict:
    base = Path(settings.ml_reports_dir)
    for name in ["training_results.json", "training_results_sample.json"]:
        p = base / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


@router.get("/summary")
def analytics_summary():
    report   = _load_eval_report()
    training = _load_training_results()
    if not report:
        raise HTTPException(503, "Evaluation report not available. Run evaluate.py first.")
    tm = report.get("test_metrics", {})
    cm = report.get("confusion_matrix", {})
    return {
        "test_metrics": tm,
        "val_metrics":  report.get("val_metrics", {}),
        "confusion_matrix": cm,
        "pr_auc": tm.get("pr_auc"),
        "f1":     tm.get("f1"),
        "model_comparison": _build_comparison(training),
        "sample_mode": tm.get("sample_mode", False),
    }


@router.get("/model-metrics", response_model=ModelMetricsOut)
def model_metrics():
    report = _load_eval_report()
    if not report:
        raise HTTPException(503, "Evaluation report not available.")
    tm = report.get("test_metrics", {})
    cm = report.get("confusion_matrix", {})
    return ModelMetricsOut(
        model_name=tm.get("model_name", "unknown"),
        threshold=tm.get("threshold", 0.5),
        test_precision=tm.get("precision", 0),
        test_recall=tm.get("recall", 0),
        test_f1=tm.get("f1", 0),
        test_pr_auc=tm.get("pr_auc", 0),
        test_roc_auc=tm.get("roc_auc", 0),
        confusion_matrix=ConfusionMatrix(**cm) if cm else ConfusionMatrix(tn=0,fp=0,fn=0,tp=0),
        false_positive_rate=tm.get("false_positive_rate", 0),
        pr_curve=report.get("pr_curve"),
        threshold_table=report.get("threshold_table"),
        feature_importance=report.get("feature_importance"),
        score_distribution=report.get("score_distribution"),
        val_metrics=report.get("val_metrics"),
        sample_mode=tm.get("sample_mode", False),
    )


def _build_comparison(training: dict) -> list:
    result = []
    for model_key in ["logistic_regression", "xgboost"]:
        if model_key in training:
            m   = training[model_key]
            vm  = m.get("val_metrics", {})
            result.append({
                "model":      m.get("model_name", model_key),
                "val_f1":     vm.get("f1", 0),
                "val_pr_auc": vm.get("pr_auc", 0),
                "val_recall": vm.get("recall", 0),
                "train_time_s": m.get("train_time_s", 0),
                "selected":   training.get("selected_model") == model_key,
            })
    return result
