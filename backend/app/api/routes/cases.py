from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.repositories.alert_repo import (
    get_cases, get_case_by_id, get_or_create_case,
    update_case_status, add_case_note, get_alert_by_id
)
from app.schemas.schemas import (
    CaseOut, CaseStatusUpdate, CaseNoteCreate, CaseNoteOut, AlertOut
)

router = APIRouter(prefix="/cases", tags=["cases"])


def _alert_out(a) -> AlertOut:
    return AlertOut(
        id=a.id, tx_id=a.tx_id,
        sender_account_id=a.sender_account_id,
        receiver_account_id=a.receiver_account_id,
        tx_amount=a.tx_amount, timestamp=a.timestamp,
        risk_score=a.risk_score, priority=a.priority,
        is_fraud_label=a.is_fraud_label, alert_type=a.alert_type,
        created_at=a.created_at,
        case_id=None, case_status=None,
    )


def _case_out(c) -> CaseOut:
    return CaseOut(
        id=c.id, alert_id=c.alert_id, status=c.status,
        assigned_to=c.assigned_to,
        created_at=c.created_at, updated_at=c.updated_at,
        resolved_at=c.resolved_at,
        notes=[CaseNoteOut(id=n.id, author=n.author, content=n.content,
                           created_at=n.created_at) for n in (c.notes or [])],
        alert=_alert_out(c.alert) if c.alert else None,
    )


@router.get("", response_model=dict)
def list_cases(
    page:   int           = Query(1, ge=1),
    size:   int           = Query(25, ge=1, le=200),
    status: Optional[str] = Query(None),
    db:     Session       = Depends(get_db),
):
    total, items = get_cases(db, page=page, size=size, status=status)
    return {"total": total, "page": page, "size": size,
            "items": [_case_out(c) for c in items]}


@router.post("/from-alert/{alert_id}", response_model=CaseOut)
def create_case_from_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = get_alert_by_id(db, alert_id)
    if not alert:
        raise HTTPException(404, f"Alert {alert_id} not found")
    case = get_or_create_case(db, alert_id)
    return _case_out(get_case_by_id(db, case.id))


@router.get("/{case_id}", response_model=CaseOut)
def get_case(case_id: int, db: Session = Depends(get_db)):
    c = get_case_by_id(db, case_id)
    if not c:
        raise HTTPException(404, f"Case {case_id} not found")
    return _case_out(c)


@router.patch("/{case_id}/status", response_model=CaseOut)
def patch_case_status(case_id: int, body: CaseStatusUpdate, db: Session = Depends(get_db)):
    c = update_case_status(db, case_id, body.status, body.assigned_to)
    if not c:
        raise HTTPException(404, f"Case {case_id} not found")
    return _case_out(get_case_by_id(db, case_id))


@router.post("/{case_id}/notes", response_model=CaseNoteOut)
def post_case_note(case_id: int, body: CaseNoteCreate, db: Session = Depends(get_db)):
    c = get_case_by_id(db, case_id)
    if not c:
        raise HTTPException(404, f"Case {case_id} not found")
    note = add_case_note(db, case_id, body.content, body.author)
    return CaseNoteOut(id=note.id, author=note.author,
                       content=note.content, created_at=note.created_at)
