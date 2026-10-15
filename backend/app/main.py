"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import engine, SessionLocal
from app.models.orm_models import Base
from app.api.routes import dashboard, alerts, cases, analytics, model_mgmt, settings_route
from app.api.routes.explainability import router as explain_router
from app.repositories.alert_repo import get_settings as db_get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        db_get_settings(db)  # ensure default settings row
    finally:
        db.close()

    log.info("Loading ML model...")
    from app.services.model_service import get_model_svc
    svc = get_model_svc()
    log.info(f"Model status: {svc.status()}")

    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers under /api
app.include_router(dashboard.router,      prefix="/api")
app.include_router(alerts.router,         prefix="/api")
app.include_router(cases.router,          prefix="/api")
app.include_router(analytics.router,      prefix="/api")
app.include_router(model_mgmt.router,     prefix="/api")
app.include_router(settings_route.router, prefix="/api")
app.include_router(explain_router,        prefix="/api")


@app.get("/api/health")
def health():
    from app.services.model_service import get_model_svc
    svc = get_model_svc()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "model_loaded": svc.is_loaded,
    }
