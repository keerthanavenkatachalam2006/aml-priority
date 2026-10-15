from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    app_name:    str  = "AML Priority"
    app_version: str  = "1.0.0"
    debug:       bool = False
    database_url: str = ""

    ml_artifacts_dir: str = str(
        Path(__file__).parent.parent.parent.parent / "ml" / "artifacts"
    )
    ml_reports_dir: str = str(
        Path(__file__).parent.parent.parent.parent / "ml" / "reports"
    )

    high_threshold:   float = 0.7
    medium_threshold: float = 0.4

    cors_origins: list = ["http://localhost:5173", "http://localhost:3000", "http://frontend:5173"]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
