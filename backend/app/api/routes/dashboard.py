from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.repositories.alert_repo import (
    get_dashboard_counts, get_recent_alerts, get_alert_timeseries
)
from app.schemas.schemas import DashboardSummary, AlertOut
from app.services.model_service import get_model_svc
from typing import List

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

TOTAL_TRANSACTIONS = 0  # loaded from metadata at startup
TOTAL_ACCOUNTS     = 100_000


@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db)):
    counts  = get_dashboard_counts(db)
    svc     = get_model_svc()
    status  = svc.status()
    tx_meta = _load_tx_meta()

    return DashboardSummary(
        total_transactions=tx_meta.get("total_transactions", 0),
        total_accounts=TOTAL_ACCOUNTS,
        **counts,
        model_status="loaded" if status["loaded"] else "not_loaded",
        model_name=status.get("model_name"),
        model_version=status.get("version"),
    )


@router.get("/recent-alerts", response_model=List[AlertOut])
def recent_high_alerts(limit: int = 10, db: Session = Depends(get_db)):
    alerts = get_recent_alerts(db, limit=limit)
    return [_alert_out(a) for a in alerts]


@router.get("/timeseries")
def alert_timeseries(db: Session = Depends(get_db)):
    return get_alert_timeseries(db)


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


def _load_tx_meta() -> dict:
    import json
    from pathlib import Path
    from app.core.config import settings
    p = Path(settings.ml_reports_dir) / "split_metadata.json"
    ps = Path(settings.ml_reports_dir) / "split_metadata_sample.json"
    for path in [p, ps]:
        if path.exists():
            with open(path) as f:
                m = json.load(f)
            total = m.get("train_rows", 0) + m.get("val_rows", 0) + m.get("test_rows", 0)
            return {"total_transactions": total}
    return {"total_transactions": 12_476_012}
