"""
Model service: loads artifacts once and exposes a consistent prediction interface.
Used by the FastAPI backend.
"""
import json, logging
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

BASE          = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE / "artifacts"
REPORTS_DIR   = BASE / "reports"

# Priority thresholds (overridden by settings from DB at runtime)
DEFAULT_HIGH_THRESHOLD   = 0.7
DEFAULT_MEDIUM_THRESHOLD = 0.4


class ModelService:
    def __init__(self):
        self.model          = None
        self.scaler         = None
        self.meta           = None
        self.feature_cols   = []
        self.threshold      = 0.5
        self.is_loaded      = False
        self.load_error     = None
        self._load()

    def _load(self):
        try:
            meta_path = ARTIFACTS_DIR / "model_metadata.json"
            if not meta_path.exists():
                self.load_error = "model_metadata.json not found – run train.py first"
                log.warning(self.load_error)
                return

            with open(meta_path) as f:
                self.meta = json.load(f)

            model_path = ARTIFACTS_DIR / self.meta["model_file"]
            if not model_path.exists():
                self.load_error = f"Model file not found: {model_path}"
                log.warning(self.load_error)
                return

            self.model        = joblib.load(model_path)
            self.feature_cols = self.meta["feature_cols"]
            self.threshold    = self.meta.get("best_threshold", 0.5)

            # Load scaler for LR
            if self.meta["model_name"] == "logistic_regression":
                scaler_path = ARTIFACTS_DIR / "scaler.joblib"
                if not scaler_path.exists():
                    scaler_path = ARTIFACTS_DIR / "scaler_sample.joblib"
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)

            self.is_loaded  = True
            self.load_error = None
            log.info(f"Model loaded: {self.meta['model_name']} "
                     f"(threshold={self.threshold:.2f}, features={len(self.feature_cols)})")

        except Exception as e:
            self.load_error = str(e)
            log.error(f"Failed to load model: {e}")

    def status(self) -> dict:
        if self.is_loaded:
            return {
                "loaded": True,
                "model_name": self.meta["model_name"],
                "version": self.meta.get("version", "1.0.0"),
                "threshold": self.threshold,
                "feature_count": len(self.feature_cols),
                "feature_cols": self.feature_cols,
                "val_metrics": self.meta.get("val_metrics", {}),
                "sample_mode": self.meta.get("sample_mode", False),
                "prediction_unit": "transaction",
            }
        return {"loaded": False, "error": self.load_error}

    def score(self, feature_dict: dict) -> dict:
        """
        Score a single transaction given a dict of feature values.
        Returns risk_score, priority, and any available explanation.
        """
        if not self.is_loaded:
            return {"error": self.load_error or "Model not loaded", "risk_score": None}

        # Build feature vector
        X = np.array([[feature_dict.get(f, 0.0) for f in self.feature_cols]],
                     dtype=np.float32)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        prob = float(self.model.predict_proba(X)[0, 1])
        return {"risk_score": prob}

    def score_batch(self,
                    feature_matrix: np.ndarray,
                    high_t: float = DEFAULT_HIGH_THRESHOLD,
                    medium_t: float = DEFAULT_MEDIUM_THRESHOLD) -> np.ndarray:
        """Score a numpy feature matrix, return probabilities."""
        if not self.is_loaded:
            raise RuntimeError(self.load_error or "Model not loaded")
        X = feature_matrix.astype(np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    @staticmethod
    def priority_label(score: float,
                       high_t: float = DEFAULT_HIGH_THRESHOLD,
                       medium_t: float = DEFAULT_MEDIUM_THRESHOLD) -> str:
        if score >= high_t:
            return "HIGH"
        if score >= medium_t:
            return "MEDIUM"
        return "LOW"


# Singleton instance used by the FastAPI app
_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service
