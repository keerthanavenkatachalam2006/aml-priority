import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.schemas import ModelStatusOut, PredictionRequest, PredictionResponse
from app.services.model_service import get_model_svc
from app.core.config import settings

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/status", response_model=ModelStatusOut)
def model_status():
    svc = get_model_svc()
    s   = svc.status()
    return ModelStatusOut(**s)


@router.post("/reload")
def reload_model():
    """Force reload the model artifact from disk."""
    from app.services.model_service import reload_model_svc
    svc = reload_model_svc()
    return svc.status()


@router.get("/features")
def get_feature_schema():
    svc = get_model_svc()
    if not svc.is_loaded:
        raise HTTPException(503, "Model not loaded")
    return {"feature_cols": svc.feature_cols, "count": len(svc.feature_cols)}


@router.get("/dataset-info")
def dataset_info():
    base = Path(settings.ml_reports_dir)
    result = {}
    for name in ["data_inspection.json"]:
        p = base / name
        if p.exists():
            with open(p) as f:
                result = json.load(f)
    meta_p = base / "split_metadata.json"
    if not meta_p.exists():
        meta_p = base / "split_metadata_sample.json"
    if meta_p.exists():
        with open(meta_p) as f:
            result["split_metadata"] = json.load(f)
    return result


@router.post("/predictions/score", response_model=PredictionResponse)
def score_transaction(body: PredictionRequest):
    svc = get_model_svc()
    if not svc.is_loaded:
        return PredictionResponse(risk_score=None, error=svc.load_error)
    result = svc.score(body.features)
    if "error" in result:
        return PredictionResponse(risk_score=None, error=result["error"])
    score = result["risk_score"]
    return PredictionResponse(
        risk_score=score,
        priority=svc.priority_label(score),
    )
