"""Alert and Case repository: all DB queries live here."""
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc, asc, or_
from app.models.orm_models import Alert, Case, CaseNote, AppSettings


# ── Alert queries ─────────────────────────────────────────────────────────────
def get_alerts(
    db: Session,
    page: int = 1,
    size: int = 25,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    min_ts: Optional[int] = None,
    max_ts: Optional[int] = None,
    sort_by: str = "risk_score",
    sort_dir: str = "desc",
) -> Tuple[int, List[Alert]]:
    q = db.query(Alert).outerjoin(Case, Alert.id == Case.alert_id)

    if priority:
        q = q.filter(Alert.priority == priority.upper())
    if status:
        q = q.filter(Case.status == status.upper())
    if search:
        term = f"%{search}%"
        q = q.filter(or_(
            Alert.tx_id.cast(db.bind.dialect.name == "sqlite" and str or None).like(term)
            if False else Alert.tx_id == (int(search) if search.isdigit() else -1),
        ))
    if min_amount is not None:
        q = q.filter(Alert.tx_amount >= min_amount)
    if max_amount is not None:
        q = q.filter(Alert.tx_amount <= max_amount)
    if min_ts is not None:
        q = q.filter(Alert.timestamp >= min_ts)
    if max_ts is not None:
        q = q.filter(Alert.timestamp <= max_ts)

    total = q.count()

    order_col = {
        "risk_score": Alert.risk_score,
        "timestamp":  Alert.timestamp,
        "tx_amount":  Alert.tx_amount,
        "created_at": Alert.created_at,
    }.get(sort_by, Alert.risk_score)

    q = q.order_by(desc(order_col) if sort_dir == "desc" else asc(order_col))
    items = q.offset((page - 1) * size).limit(size).options(
        joinedload(Alert.case)
    ).all()
    return total, items


def get_alert_by_id(db: Session, alert_id: int) -> Optional[Alert]:
    return db.query(Alert).options(joinedload(Alert.case)).filter(Alert.id == alert_id).first()


def bulk_insert_alerts(db: Session, records: list):
    db.bulk_insert_mappings(Alert, records)
    db.commit()


# ── Case queries ──────────────────────────────────────────────────────────────
def get_or_create_case(db: Session, alert_id: int) -> Case:
    case = db.query(Case).filter(Case.alert_id == alert_id).first()
    if not case:
        case = Case(alert_id=alert_id, status="OPEN")
        db.add(case)
        db.commit()
        db.refresh(case)
    return case


def get_case_by_id(db: Session, case_id: int) -> Optional[Case]:
    return db.query(Case).options(
        joinedload(Case.notes),
        joinedload(Case.alert),
    ).filter(Case.id == case_id).first()


def get_cases(
    db: Session,
    page: int = 1,
    size: int = 25,
    status: Optional[str] = None,
) -> Tuple[int, List[Case]]:
    q = db.query(Case).options(joinedload(Case.alert))
    if status:
        q = q.filter(Case.status == status.upper())
    total = q.count()
    items = q.order_by(desc(Case.created_at)).offset((page - 1) * size).limit(size).all()
    return total, items


def update_case_status(db: Session, case_id: int, status: str, assigned_to: Optional[str] = None) -> Optional[Case]:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        return None
    case.status     = status
    case.updated_at = _now()
    if assigned_to is not None:
        case.assigned_to = assigned_to
    if status == "RESOLVED":
        case.resolved_at = _now()
    db.commit()
    db.refresh(case)
    return case


def add_case_note(db: Session, case_id: int, content: str, author: str = "Analyst") -> CaseNote:
    note = CaseNote(case_id=case_id, content=content, author=author)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


# ── Dashboard counts ──────────────────────────────────────────────────────────
def get_dashboard_counts(db: Session) -> dict:
    total_alerts = db.query(func.count(Alert.id)).scalar() or 0
    high         = db.query(func.count(Alert.id)).filter(Alert.priority == "HIGH").scalar() or 0
    medium       = db.query(func.count(Alert.id)).filter(Alert.priority == "MEDIUM").scalar() or 0
    low          = db.query(func.count(Alert.id)).filter(Alert.priority == "LOW").scalar() or 0
    fraud_labels = db.query(func.count(Alert.id)).filter(Alert.is_fraud_label == True).scalar() or 0
    open_cases   = db.query(func.count(Case.id)).filter(Case.status.in_(["OPEN", "UNDER_REVIEW", "ESCALATED"])).scalar() or 0
    resolved     = db.query(func.count(Case.id)).filter(Case.status == "RESOLVED").scalar() or 0
    return {
        "total_alerts":         total_alerts,
        "high_priority_count":  high,
        "medium_priority_count": medium,
        "low_priority_count":   low,
        "fraud_label_count":    fraud_labels,
        "open_cases":           open_cases,
        "resolved_cases":       resolved,
    }


def get_recent_alerts(db: Session, limit: int = 10) -> List[Alert]:
    return (db.query(Alert)
            .filter(Alert.priority == "HIGH")
            .order_by(desc(Alert.risk_score))
            .limit(limit).all())


def get_alert_timeseries(db: Session) -> List[dict]:
    from sqlalchemy import Integer, case as sa_case
    rows = (db.query(Alert.timestamp,
                     func.count(Alert.id).label("count"),
                     func.sum(sa_case((Alert.is_fraud_label == True, 1), else_=0)).label("fraud_count"))
            .group_by(Alert.timestamp)
            .order_by(Alert.timestamp)
            .all())
    return [{"timestamp": r.timestamp, "count": r.count,
             "fraud_count": int(r.fraud_count or 0)} for r in rows]


# ── Settings ──────────────────────────────────────────────────────────────────
def get_settings(db: Session) -> AppSettings:
    s = db.query(AppSettings).first()
    if not s:
        s = AppSettings()
        db.add(s)
        db.commit()
        db.refresh(s)
    return s


def update_settings(db: Session, **kwargs) -> AppSettings:
    s = get_settings(db)
    for k, v in kwargs.items():
        if v is not None and hasattr(s, k):
            setattr(s, k, v)
    s.updated_at = _now()
    db.commit()
    db.refresh(s)
    return s
