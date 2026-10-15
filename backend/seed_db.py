"""Run this once to create tables and seed the alert database."""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from app.db.database import SessionLocal, engine
from app.models.orm_models import Base
from app.services.seed_service import seed_alerts, ensure_tables, ensure_default_settings

ensure_tables()
db = SessionLocal()
try:
    ensure_default_settings(db)
    n = seed_alerts(db, high_t=0.7, medium_t=0.4)
    print(f"Seeded {n} alerts into the database.")
finally:
    db.close()
