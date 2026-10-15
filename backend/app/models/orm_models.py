"""SQLAlchemy ORM models."""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, Text, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from app.db.database import Base

def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Alert(Base):
    __tablename__ = "alerts"
    id                   = Column(Integer, primary_key=True, index=True)
    tx_id                = Column(Integer, index=True, nullable=False)
    sender_account_id    = Column(Integer, index=True, nullable=False)
    receiver_account_id  = Column(Integer, index=True, nullable=False)
    tx_amount            = Column(Float, nullable=False)
    timestamp            = Column(Integer, nullable=False, index=True)
    risk_score           = Column(Float, nullable=False)
    priority             = Column(String(10), nullable=False, index=True)
    is_fraud_label       = Column(Boolean, nullable=False)
    alert_type           = Column(String(50), nullable=True)
    created_at           = Column(DateTime, default=_now)

    case = relationship("Case", back_populates="alert", uselist=False)

    __table_args__ = (
        Index("ix_alerts_risk_score", "risk_score"),
        Index("ix_alerts_priority_created", "priority", "created_at"),
    )


class Case(Base):
    __tablename__ = "cases"
    id           = Column(Integer, primary_key=True, index=True)
    alert_id     = Column(Integer, ForeignKey("alerts.id"), unique=True, nullable=False)
    status       = Column(String(30), default="OPEN", index=True)
    assigned_to  = Column(String(100), nullable=True)
    created_at   = Column(DateTime, default=_now)
    updated_at   = Column(DateTime, default=_now, onupdate=_now)
    resolved_at  = Column(DateTime, nullable=True)

    alert = relationship("Alert", back_populates="case")
    notes = relationship("CaseNote", back_populates="case", order_by="CaseNote.created_at")


class CaseNote(Base):
    __tablename__ = "case_notes"
    id         = Column(Integer, primary_key=True, index=True)
    case_id    = Column(Integer, ForeignKey("cases.id"), nullable=False, index=True)
    author     = Column(String(100), default="Analyst")
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=_now)

    case = relationship("Case", back_populates="notes")


class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    id           = Column(Integer, primary_key=True, index=True)
    model_name   = Column(String(100), nullable=False)
    version      = Column(String(50), nullable=False)
    loaded_at    = Column(DateTime, default=_now)
    threshold    = Column(Float, nullable=False)
    sample_mode  = Column(Boolean, default=False)
    metrics_json = Column(Text, nullable=True)


class AppSettings(Base):
    __tablename__ = "app_settings"
    id                   = Column(Integer, primary_key=True)
    high_threshold       = Column(Float, default=0.7)
    medium_threshold     = Column(Float, default=0.4)
    default_page_size    = Column(Integer, default=25)
    app_theme            = Column(String(20), default="dark")
    updated_at           = Column(DateTime, default=_now, onupdate=_now)
