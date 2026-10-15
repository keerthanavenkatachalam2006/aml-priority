import io
import csv
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.repositories.alert_repo import get_alerts, get_alert_by_id
from app.schemas.schemas import AlertOut, AlertListResponse
from app.models.orm_models import Alert

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _alert_out(a) -> AlertOut:
    return AlertOut(
        id=a.id, tx_id=a.tx_id,
        sender_account_id=a.sender_account_id,
        receiver_account_id=a.receiver_account_id,
        tx_amount=a.tx_amount, timestamp=a.timestamp,
        risk_score=a.risk_score, priority=a.priority,
        is_fraud_label=a.is_fraud_label, alert_type=a.alert_type,
        created_at=a.created_at,
        case_id=a.case.id if a.case else None,
        case_status=a.case.status if a.case else None,
    )


@router.get("", response_model=AlertListResponse)
def list_alerts(
    page:       int            = Query(1, ge=1),
    size:       int            = Query(25, ge=1, le=200),
    priority:   Optional[str] = Query(None),
    status:     Optional[str] = Query(None),
    search:     Optional[str] = Query(None),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
    min_ts:     Optional[int]  = Query(None),
    max_ts:     Optional[int]  = Query(None),
    sort_by:    str            = Query("risk_score"),
    sort_dir:   str            = Query("desc"),
    db:         Session        = Depends(get_db),
):
    total, items = get_alerts(
        db, page=page, size=size, priority=priority, status=status,
        search=search, min_amount=min_amount, max_amount=max_amount,
        min_ts=min_ts, max_ts=max_ts, sort_by=sort_by, sort_dir=sort_dir,
    )
    return AlertListResponse(
        total=total, page=page, size=size,
        items=[_alert_out(a) for a in items],
    )


@router.get("/export")
def export_alerts_csv(
    priority:   Optional[str] = Query(None),
    status:     Optional[str] = Query(None),
    search:     Optional[str] = Query(None),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
    db:         Session        = Depends(get_db),
):
    total, items = get_alerts(
        db, page=1, size=5000, priority=priority, status=status,
        search=search, min_amount=min_amount, max_amount=max_amount,
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["alert_id","tx_id","sender","receiver","amount",
                     "timestamp","risk_score","priority","is_fraud","alert_type","case_status"])
    for a in items:
        writer.writerow([a.id, a.tx_id, a.sender_account_id, a.receiver_account_id,
                         a.tx_amount, a.timestamp, round(a.risk_score, 4),
                         a.priority, a.is_fraud_label, a.alert_type or "",
                         a.case.status if a.case else ""])
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=alerts.csv"})


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    a = get_alert_by_id(db, alert_id)
    if not a:
        raise HTTPException(404, f"Alert {alert_id} not found")
    return _alert_out(a)


# ── Submit a new transaction for scoring ─────────────────────────────────────

class SubmitTransactionRequest(BaseModel):
    tx_id:               int   = Field(..., description="Unique transaction ID")
    sender_account_id:   int   = Field(..., description="Sender account ID")
    receiver_account_id: int   = Field(..., description="Receiver account ID")
    tx_amount:           float = Field(..., gt=0, description="Transaction amount in $")
    timestamp:           int   = Field(..., ge=0, le=999, description="Simulation step or day number")
    # Rolling account-history features
    sender_tx_count:           float = Field(0, ge=0)
    sender_out_volume:         float = Field(0, ge=0)
    sender_avg_amount:         float = Field(0, ge=0)
    sender_n_counterparties:   float = Field(0, ge=0)
    sender_fraud_account:      float = Field(0, ge=0, le=1)
    sender_tx_behavior:        float = Field(1, ge=1, le=5)
    sender_init_balance:       float = Field(0, ge=0)
    receiver_tx_count:         float = Field(0, ge=0)
    receiver_in_volume:        float = Field(0, ge=0)
    receiver_avg_amount:       float = Field(0, ge=0)
    receiver_n_counterparties: float = Field(0, ge=0)
    receiver_fraud_account:    float = Field(0, ge=0, le=1)
    receiver_tx_behavior:      float = Field(1, ge=1, le=5)
    amount_to_sender_avg_ratio: float = Field(1, ge=0)


class SubmitTransactionResponse(BaseModel):
    alert_id:   int
    tx_id:      int
    risk_score: float
    priority:   str
    message:    str


@router.post("/submit", response_model=SubmitTransactionResponse, status_code=201)
def submit_transaction(body: SubmitTransactionRequest, db: Session = Depends(get_db)):
    """
    Score a new transaction with the XGBoost model and add it to the alert priority queue.
    Returns the assigned risk score, priority label, and the new alert ID.
    """
    from app.services.model_service import get_model_svc
    svc = get_model_svc()
    if not svc.is_loaded:
        raise HTTPException(503, "Model not loaded — cannot score transaction")

    # Check for duplicate tx_id
    existing = db.query(Alert).filter(Alert.tx_id == body.tx_id).first()
    if existing:
        raise HTTPException(409, f"Transaction {body.tx_id} already exists as alert {existing.id}")

    # Build feature dict matching the model's expected feature_cols order
    feature_dict = {
        "TX_AMOUNT":                  body.tx_amount,
        "TIMESTAMP":                  float(body.timestamp),
        "sender_tx_count":            body.sender_tx_count,
        "receiver_tx_count":          body.receiver_tx_count,
        "sender_out_volume":          body.sender_out_volume,
        "receiver_in_volume":         body.receiver_in_volume,
        "sender_avg_amount":          body.sender_avg_amount,
        "receiver_avg_amount":        body.receiver_avg_amount,
        "sender_n_counterparties":    body.sender_n_counterparties,
        "receiver_n_counterparties":  body.receiver_n_counterparties,
        "sender_fraud_account":       body.sender_fraud_account,
        "receiver_fraud_account":     body.receiver_fraud_account,
        "sender_tx_behavior":         body.sender_tx_behavior,
        "receiver_tx_behavior":       body.receiver_tx_behavior,
        "sender_init_balance":        body.sender_init_balance,
        "amount_to_sender_avg_ratio": body.amount_to_sender_avg_ratio,
    }

    result     = svc.score(feature_dict)
    risk_score = result["risk_score"]
    priority   = svc.priority_label(risk_score)

    # Save to database
    alert = Alert(
        tx_id               = body.tx_id,
        sender_account_id   = body.sender_account_id,
        receiver_account_id = body.receiver_account_id,
        tx_amount           = body.tx_amount,
        timestamp           = body.timestamp,
        risk_score          = risk_score,
        priority            = priority,
        is_fraud_label      = False,   # unknown for new submissions
        alert_type          = "manual_submission",
        created_at          = datetime.now(timezone.utc).replace(tzinfo=None),
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    return SubmitTransactionResponse(
        alert_id   = alert.id,
        tx_id      = body.tx_id,
        risk_score = risk_score,
        priority   = priority,
        message    = f"Transaction scored and added to priority queue as alert #{alert.id}",
    )
