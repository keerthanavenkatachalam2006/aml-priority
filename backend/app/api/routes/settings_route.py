from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.repositories.alert_repo import get_settings, update_settings
from app.schemas.schemas import SettingsOut, SettingsUpdate

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsOut)
def read_settings(db: Session = Depends(get_db)):
    return get_settings(db)


@router.patch("", response_model=SettingsOut)
def patch_settings(body: SettingsUpdate, db: Session = Depends(get_db)):
    updated = update_settings(db, **body.model_dump(exclude_none=True))
    return updated
