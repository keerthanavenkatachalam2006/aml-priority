"""Pydantic request/response schemas."""
from datetime import datetime
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field, field_validator


# ── Alert schemas ─────────────────────────────────────────────────────────────
class AlertBase(BaseModel):
    tx_id:               int
    sender_account_id:   int
    receiver_account_id: int
    tx_amount:           float
    timestamp:           int
    risk_score:          float
    priority:            str
    is_fraud_label:      bool
    alert_type:          Optional[str] = None


class AlertOut(AlertBase):
    id:         int
    created_at: datetime
    case_id:    Optional[int] = None
    case_status: Optional[str] = None

    model_config = {"from_attributes": True}


class AlertListResponse(BaseModel):
    total:   int
    page:    int
    size:    int
    items:   List[AlertOut]


# ── Case schemas ──────────────────────────────────────────────────────────────
class CaseNoteOut(BaseModel):
    id:         int
    author:     str
    content:    str
    created_at: datetime
    model_config = {"from_attributes": True}


class CaseOut(BaseModel):
    id:          int
    alert_id:    int
    status:      str
    assigned_to: Optional[str]
    created_at:  datetime
    updated_at:  datetime
    resolved_at: Optional[datetime]
    notes:       List[CaseNoteOut] = []
    alert:       Optional[AlertOut] = None
    model_config = {"from_attributes": True}


class CaseStatusUpdate(BaseModel):
    status:      str = Field(..., pattern="^(OPEN|UNDER_REVIEW|ESCALATED|RESOLVED)$")
    assigned_to: Optional[str] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        allowed = {"OPEN", "UNDER_REVIEW", "ESCALATED", "RESOLVED"}
        if v not in allowed:
            raise ValueError(f"Status must be one of {allowed}")
        return v


class CaseNoteCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    author:  str = Field(default="Analyst", max_length=100)


# ── Dashboard schemas ─────────────────────────────────────────────────────────
class DashboardSummary(BaseModel):
    total_transactions:   int
    total_accounts:       int
    total_alerts:         int
    high_priority_count:  int
    medium_priority_count: int
    low_priority_count:   int
    open_cases:           int
    resolved_cases:       int
    fraud_label_count:    int
    model_status:         str
    model_name:           Optional[str] = None
    model_version:        Optional[str] = None


class TimeSeriesPoint(BaseModel):
    timestamp: int
    count:     int
    fraud_count: int = 0


# ── Analytics schemas ─────────────────────────────────────────────────────────
class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int


class ModelMetricsOut(BaseModel):
    model_name:        str
    threshold:         float
    test_precision:    float
    test_recall:       float
    test_f1:           float
    test_pr_auc:       float
    test_roc_auc:      float
    confusion_matrix:  ConfusionMatrix
    false_positive_rate: float
    pr_curve:          Optional[Dict[str, Any]] = None
    threshold_table:   Optional[List[Dict]]     = None
    feature_importance: Optional[Dict[str, float]] = None
    score_distribution: Optional[Dict[str, Any]] = None
    val_metrics:       Optional[Dict[str, Any]] = None
    sample_mode:       bool = False


# ── Model status schemas ──────────────────────────────────────────────────────
class ModelStatusOut(BaseModel):
    loaded:          bool
    model_name:      Optional[str]  = None
    version:         Optional[str]  = None
    threshold:       Optional[float] = None
    feature_count:   Optional[int]  = None
    feature_cols:    Optional[List[str]] = None
    val_metrics:     Optional[Dict[str, Any]] = None
    sample_mode:     Optional[bool] = None
    prediction_unit: Optional[str]  = None
    error:           Optional[str]  = None


# ── Settings schemas ──────────────────────────────────────────────────────────
class SettingsOut(BaseModel):
    id:                   int
    high_threshold:       float
    medium_threshold:     float
    default_page_size:    int
    app_theme:            str
    model_config = {"from_attributes": True}


class SettingsUpdate(BaseModel):
    high_threshold:    Optional[float] = Field(None, ge=0.0, le=1.0)
    medium_threshold:  Optional[float] = Field(None, ge=0.0, le=1.0)
    default_page_size: Optional[int]   = Field(None, ge=5, le=200)
    app_theme:         Optional[str]   = Field(None, pattern="^(dark|light)$")


# ── Prediction schemas ────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    risk_score: Optional[float]
    priority:   Optional[str] = None
    error:      Optional[str] = None
