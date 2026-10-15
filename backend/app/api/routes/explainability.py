"""
SHAP explainability endpoint.
Returns local explanation for a specific alert's risk score.
"""
import sys, logging
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "ml" / "src"))

log = logging.getLogger(__name__)
router = APIRouter(prefix="/explain", tags=["explainability"])

_explainer = None
_mode = "unavailable"


def _get_explainer():
    """Lazily initialize SHAP TreeExplainer or fallback."""
    global _explainer, _mode
    if _explainer is not None:
        return _explainer, _mode
    try:
        import shap
        import joblib
        from app.services.model_service import get_model_svc
        svc = get_model_svc()
        if not svc.is_loaded:
            _mode = "unavailable"
            return None, _mode
        _explainer = shap.TreeExplainer(svc.model)
        _mode = "shap"
        log.info("SHAP TreeExplainer initialized")
    except Exception as e:
        log.warning(f"SHAP unavailable: {e}. Using XGBoost gain fallback.")
        _mode = "gain_fallback"
    return _explainer, _mode


class ExplainRequest(BaseModel):
    features: Dict[str, float]
    top_n: int = 8


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: Optional[float] = None
    direction: str          # "increases_risk" | "decreases_risk" | "neutral"
    rank: int


class ExplainResponse(BaseModel):
    risk_score: float
    priority: str
    explanation_mode: str   # "shap" | "gain_fallback"
    contributions: List[FeatureContribution]
    disclaimer: str = (
        "These explanations describe model behaviour on AMLSim simulated data. "
        "They do not prove financial crime and should not be used as the sole basis "
        "for any legal or regulatory action."
    )


@router.post("/transaction", response_model=ExplainResponse)
def explain_transaction(body: ExplainRequest):
    """
    Return SHAP-based (or gain-fallback) feature contributions for a single transaction.
    """
    from app.services.model_service import get_model_svc
    import numpy as np

    svc = get_model_svc()
    if not svc.is_loaded:
        raise HTTPException(503, "Model not loaded")

    feature_cols = svc.feature_cols
    X = np.array([[body.features.get(f, 0.0) for f in feature_cols]], dtype=np.float32)
    risk_score = float(svc.model.predict_proba(X)[0, 1])
    priority   = svc.priority_label(risk_score)

    explainer, mode = _get_explainer()

    if mode == "shap" and explainer is not None:
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]
        contributions = []
        for i, (feat, val, sv_i) in enumerate(
                sorted(zip(feature_cols,
                           [body.features.get(f, 0.0) for f in feature_cols],
                           sv),
                       key=lambda x: abs(x[2]), reverse=True)[:body.top_n]):
            contributions.append(FeatureContribution(
                feature=feat, value=float(val), shap_value=float(sv_i),
                direction="increases_risk" if sv_i > 0.01 else
                          "decreases_risk" if sv_i < -0.01 else "neutral",
                rank=i + 1,
            ))
    else:
        # Gain fallback
        scores = svc.model.feature_importances_
        pairs  = sorted(zip(feature_cols, scores,
                            [body.features.get(f, 0.0) for f in feature_cols]),
                        key=lambda x: x[1], reverse=True)[:body.top_n]
        contributions = []
        for i, (feat, gain, val) in enumerate(pairs):
            contributions.append(FeatureContribution(
                feature=feat, value=float(val), shap_value=None,
                direction="increases_risk" if gain > 0.05 else "neutral",
                rank=i + 1,
            ))

    return ExplainResponse(
        risk_score=risk_score,
        priority=priority,
        explanation_mode=mode,
        contributions=contributions,
    )


@router.get("/global-importance")
def global_feature_importance():
    """Return global XGBoost feature importance (gain)."""
    from app.services.model_service import get_model_svc
    svc = get_model_svc()
    if not svc.is_loaded:
        raise HTTPException(503, "Model not loaded")
    scores = svc.model.feature_importances_
    result = sorted(
        [{"feature": svc.feature_cols[i], "importance": float(scores[i])}
         for i in range(len(svc.feature_cols))],
        key=lambda x: x["importance"], reverse=True
    )
    return {
        "mode": "xgboost_gain",
        "disclaimer": "XGBoost gain importance. For SHAP-based explanations use /explain/transaction.",
        "feature_importance": result,
    }
