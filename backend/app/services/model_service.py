"""Thin adapter that loads the ML predict module."""
import sys
from pathlib import Path

# Add ml/src to path so we can import predict.py directly
ML_SRC = Path(__file__).parent.parent.parent.parent.parent / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from predict import ModelService, get_model_service  # noqa: E402

_svc = None


def get_model_svc() -> ModelService:
    global _svc
    if _svc is None:
        _svc = get_model_service()
    return _svc


def reload_model_svc() -> ModelService:
    global _svc
    _svc = ModelService()
    return _svc
