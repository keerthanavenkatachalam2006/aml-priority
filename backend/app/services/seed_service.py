"""
Seed the SQLite DB from the evaluated/scored parquet and alerts.csv.
Run once after training & evaluation.
"""
import json, logging
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from app.models.orm_models import Alert, AppSettings
from app.db.database import engine, Base

log = logging.getLogger(__name__)

ML_BASE       = Path(__file__).parent.parent.parent.parent / "ml"
PROCESSED_DIR = ML_BASE / "data" / "processed"
# Dataset lives two levels above aml-priority/
DATA_DIR      = Path(__file__).parent.parent.parent.parent.parent / "100Kvertices-10Medges"
ARTIFACTS_DIR = ML_BASE / "artifacts"


def _priority(score: float, high_t: float = 0.7, medium_t: float = 0.4) -> str:
    if score >= high_t:
        return "HIGH"
    if score >= medium_t:
        return "MEDIUM"
    return "LOW"


def seed_alerts(db: Session, high_t: float = 0.7, medium_t: float = 0.4,
                sample_mode: bool = False) -> int:
    """
    Load the scored test parquet and alerts.csv, join them, and insert
    into the alerts table. Returns number of records inserted.
    Only inserts alerts with risk_score >= medium_t to keep DB manageable.
    """
    suffix    = "_sample" if sample_mode else ""
    scored_p  = PROCESSED_DIR / f"test_scored{suffix}.parquet"

    if not scored_p.exists():
        log.warning(f"Scored parquet not found: {scored_p}. Run evaluate.py first.")
        return 0

    log.info("Loading scored test parquet...")
    df = pd.read_parquet(scored_p)
    log.info(f"Scored rows: {len(df):,}")

    # Load alerts.csv for alert_type
    alerts_csv = DATA_DIR / "alerts.csv"
    alert_types = {}
    if alerts_csv.exists():
        al = pd.read_csv(alerts_csv, usecols=["TX_ID", "ALERT_TYPE"])
        alert_types = dict(zip(al["TX_ID"].values, al["ALERT_TYPE"].values))

    # Filter to model-flagged suspicious OR true positives (keep TP+FP alerts)
    mask = (df["risk_score"] >= medium_t) | (df["IS_FRAUD"] == True)
    df_alerts = df[mask].copy()
    log.info(f"Rows to insert as alerts: {len(df_alerts):,}")

    if len(df_alerts) == 0:
        log.warning("No alerts to seed.")
        return 0

    # Clear existing alerts
    db.query(Alert).delete()
    db.commit()

    records = []
    for row in df_alerts.itertuples(index=False):
        tx_id = int(row.TX_ID)
        records.append({
            "tx_id":               tx_id,
            "sender_account_id":   int(row.SENDER_ACCOUNT_ID),
            "receiver_account_id": int(row.RECEIVER_ACCOUNT_ID),
            "tx_amount":           float(row.TX_AMOUNT),
            "timestamp":           int(row.TIMESTAMP),
            "risk_score":          float(row.risk_score),
            "priority":            _priority(float(row.risk_score), high_t, medium_t),
            "is_fraud_label":      bool(row.IS_FRAUD),
            "alert_type":          alert_types.get(tx_id, None),
        })

    CHUNK = 5000
    for i in range(0, len(records), CHUNK):
        db.bulk_insert_mappings(Alert, records[i:i+CHUNK])
        db.commit()
        log.info(f"  Inserted {min(i+CHUNK, len(records)):,} / {len(records):,}")

    log.info(f"Seeded {len(records)} alerts.")
    return len(records)


def ensure_tables():
    Base.metadata.create_all(bind=engine)
    log.info("Tables created/verified.")


def ensure_default_settings(db: Session):
    if not db.query(AppSettings).first():
        db.add(AppSettings())
        db.commit()
        log.info("Default settings inserted.")
