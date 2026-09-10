"""
CycloneAI Backend Detection Service
===================================
Production integration for satellite-based cyclone pattern classification and intensity estimation.
Powered by the 5-Fold Soft-Voting Ensemble (Approach A - Aggressive CE weights).
Directly computes predictions on real satellite data; zero hardcoded mock fallbacks.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.satellite.inference import (
    EnsembleSatellitePredictor,
    DEFAULT_ENSEMBLE_CHECKPOINTS,
    CLASS_NAMES,
)

logger = logging.getLogger("cycloneai.detection_service")


class DetectionService:
    """
    Backend service wrapping the production 5-fold satellite CNN ensemble.
    Accepts .npy arrays or standard image files (.png, .jpg), runs soft-voting,
    and returns IMD classification, wind speed, Dvorak T-number, and estimated pressure.
    """

    def __init__(self, checkpoint_paths: Optional[list] = None):
        self.checkpoint_paths = checkpoint_paths or DEFAULT_ENSEMBLE_CHECKPOINTS
        self.predictor = None
        self._initialize_predictor()

    def _initialize_predictor(self):
        """Validates checkpoint files and loads the 5-fold ensemble into memory."""
        missing = [p for p in self.checkpoint_paths if not os.path.exists(p)]
        if missing:
            err_msg = (
                f"DetectionService initialization error: The following ensemble checkpoint(s) "
                f"were not found: {missing}. Please ensure model weights are tracked and available."
            )
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        try:
            self.predictor = EnsembleSatellitePredictor(checkpoint_paths=self.checkpoint_paths)
            print(f"DetectionService: Production 5-fold ensemble loaded successfully ({len(self.predictor.models)} models).")
        except Exception as e:
            logger.error(f"DetectionService: Failed to instantiate EnsembleSatellitePredictor: {e}")
            raise RuntimeError(f"Failed to load satellite ensemble predictor: {e}") from e

    def run_inference(self, image_path: str) -> Dict[str, Any]:
        """
        Loads satellite image, runs 5-fold soft-voting ensemble CNN,
        and returns IMD classification + estimated intensity parameters.
        Preserves backward compatibility with frontend contract while eliminating hardcoded stubs.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Satellite image not found: {image_path}")

        if self.predictor is None:
            self._initialize_predictor()

        try:
            res = self.predictor.predict_file(image_path)
            pattern_type = res["predicted_class"]
            probs = res.get("probabilities", {})
            confidence = float(probs.get(pattern_type, 0.0))
            wind_speed = float(res["predicted_wind_speed_knots"])

            # Empirical Dvorak T-number derivation based on continuous wind speed (IMD standards)
            if "Low Pressure" in pattern_type or wind_speed < 17.0:
                t_number = 1.0
            elif "Depression" in pattern_type and "Deep" not in pattern_type:
                t_number = max(1.5, min(2.0, round((wind_speed - 17.0) / 10.0 * 0.5 + 1.5, 1)))
            else:
                t_number = max(2.0, min(3.0, round((wind_speed - 28.0) / 6.0 * 0.5 + 2.5, 1)))

            # Empirical central pressure estimation (Atkinson & Holliday formula approximation)
            estimated_pressure_hpa = round(1010.0 - (wind_speed * 0.65), 1)

            return {
                "pattern_type": pattern_type,
                "confidence": round(confidence, 3),
                "wind_speed_knots": round(wind_speed, 1),
                "dvorak_t_number": round(t_number, 1),
                "estimated_pressure_hpa": estimated_pressure_hpa,
                "probabilities": probs,
                "ensemble_size": res.get("ensemble_size", 5),
                "inference_mode": res.get("inference_mode", "ensemble_soft_vote"),
                "latency_ms": res.get("latency_ms", 0.0),
            }
        except Exception as e:
            logger.error(f"DetectionService inference error on {image_path}: {e}")
            # Do NOT return hardcoded mock data: re-raise so callers receive true error status
            raise RuntimeError(f"DetectionService inference failed on {image_path}: {e}") from e


# Instantiate singleton
detection_service = DetectionService()
