"""
Production Inference Module for CycloneAI Satellite Models.
Default Production Path: 5-Model Soft-Voting Ensemble (Approach A - Aggressive CE weights).
Legacy Path: Single-checkpoint prediction (ResidualSatelliteCNN or ResidualSatelliteRegressor).
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, Any, Union, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.classification.model import ResidualSatelliteCNN
try:
    from models.satellite.training.model import ResidualSatelliteRegressor
except ImportError:
    ResidualSatelliteRegressor = None

CLASS_NAMES = [
    'Low Pressure Area (< 17 kts)',
    'Depression (17-27 kts)',
    'Deep Depression (28-33 kts)'
]

DEFAULT_ENSEMBLE_CHECKPOINTS = [
    os.path.join(BASE_DIR, "models", "consolidated_cv", f"A_fold_{i}.pth")
    for i in range(1, 6)
]
DEFAULT_SINGLE_CHECKPOINT = os.path.join(BASE_DIR, "models", "consolidated_cv", "final_model.pth")


def map_intensity_to_imd_category(wind_kts: float) -> str:
    """Categorizes continuous wind speed into standard IMD cyclone development stages."""
    if wind_kts < 17.0:
        return 'Low Pressure Area (< 17 kts)'
    elif wind_kts <= 27.0:
        return 'Depression (17-27 kts)'
    else:
        return 'Deep Depression (28-33 kts)'


def load_satellite_image(file_path: str) -> np.ndarray:
    """
    Loads satellite image from .npy array or standard image format (.png, .jpg, .jpeg).
    Returns 128x128 float32 array normalized to [0.0, 1.0].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Satellite image file not found: {file_path}")
    
    if file_path.endswith(".npy"):
        arr = np.load(file_path).astype(np.float32)
    else:
        from PIL import Image
        img = Image.open(file_path).convert("L")
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
    return arr


class EnsembleSatellitePredictor:
    """
    Production 5-Fold Ensemble Predictor using Soft-Voting.
    Combines predictions from all 5 cross-validation fold models.
    """
    def __init__(self, checkpoint_paths: Optional[List[str]] = None, device: Optional[str] = None):
        if checkpoint_paths is None:
            checkpoint_paths = DEFAULT_ENSEMBLE_CHECKPOINTS
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        self.checkpoint_paths = []

        for p in checkpoint_paths:
            abs_p = p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
            if not os.path.exists(abs_p):
                raise FileNotFoundError(f"Ensemble checkpoint not found at: {abs_p}")
            
            ckpt = torch.load(abs_p, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            
            model = ResidualSatelliteCNN(in_channels=1, num_classes=3)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            self.checkpoint_paths.append(abs_p)

    def _prepare_tensor(self, arr: np.ndarray) -> torch.Tensor:
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            tensor = torch.from_numpy(arr).unsqueeze(0)
        elif arr.ndim == 4:
            tensor = torch.from_numpy(arr)
        else:
            raise ValueError(f"Expected array of shape (128, 128) or (1, 128, 128), got {arr.shape}")
        return tensor.to(self.device)

    def predict_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Runs soft-voting ensemble inference on a single 128x128 float32 numpy array."""
        tensor = self._prepare_tensor(arr)
        
        t0 = time.perf_counter()
        prob_list = []
        wind_list = []

        with torch.no_grad():
            for model in self.models:
                logits, reg = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                wind = float(reg.squeeze().item())
                prob_list.append(probs)
                wind_list.append(wind)

        # Average softmax probabilities across all 5 models (soft voting)
        avg_probs = np.mean(prob_list, axis=0)
        pred_idx = int(np.argmax(avg_probs))
        pred_wind = float(np.mean(wind_list))
        pred_class = CLASS_NAMES[pred_idx]
        latency = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_class": pred_class,
            "class_index": pred_idx,
            "probabilities": {
                CLASS_NAMES[i]: round(float(avg_probs[i]), 4) for i in range(3)
            },
            "predicted_wind_speed_knots": round(pred_wind, 2),
            "ensemble_size": len(self.models),
            "inference_mode": "ensemble_soft_vote",
            "latency_ms": round(latency, 2),
        }

    def predict_file(self, file_path: str) -> Dict[str, Any]:
        arr = load_satellite_image(file_path)
        res = self.predict_array(arr)
        res["file_path"] = file_path
        res["filename"] = os.path.basename(file_path)
        return res

    def predict_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        results = []
        valid_exts = (".npy", ".png", ".jpg", ".jpeg")
        for root, _, files in os.walk(dir_path):
            for f in sorted(files):
                if f.lower().endswith(valid_exts):
                    p = os.path.join(root, f)
                    try:
                        res = self.predict_file(p)
                        results.append(res)
                    except Exception as e:
                        print(f"Error predicting {p}: {e}", file=sys.stderr)
        return results


class LegacySingleSatellitePredictor:
    """
    Legacy Inference wrapper for a single CycloneAI Satellite CNN checkpoint.
    Non-default path retained for backward compatibility.
    """
    def __init__(self, checkpoint_path: str = None, device: str = None):
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_SINGLE_CHECKPOINT
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(BASE_DIR, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint_path

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

        has_class_head = any("fc_class" in k for k in state_dict.keys())
        if has_class_head:
            self.model_type = "classification"
            self.model = ResidualSatelliteCNN(in_channels=1, num_classes=3)
            self.model.load_state_dict(state_dict)
        else:
            self.model_type = "regression"
            if ResidualSatelliteRegressor is None:
                from models.satellite.training.model import ResidualSatelliteRegressor as RSR
                self.model = RSR(in_channels=1, dropout=0.3)
            else:
                self.model = ResidualSatelliteRegressor(in_channels=1, dropout=0.3)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict_array(self, arr: np.ndarray) -> Dict[str, Any]:
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            tensor = torch.from_numpy(arr).unsqueeze(0)
        elif arr.ndim == 4:
            tensor = torch.from_numpy(arr)
        else:
            raise ValueError(f"Expected array of shape (128, 128) or (1, 128, 128), got {arr.shape}")

        tensor = tensor.to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            if self.model_type == "classification":
                logits, reg = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                pred_idx = int(torch.argmax(logits, dim=1).item())
                pred_wind = float(reg.squeeze().item())
                pred_class = CLASS_NAMES[pred_idx]
                latency = (time.perf_counter() - t0) * 1000.0

                return {
                    "predicted_class": pred_class,
                    "class_index": pred_idx,
                    "probabilities": {
                        CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(3)
                    },
                    "predicted_wind_speed_knots": round(pred_wind, 2),
                    "inference_mode": "legacy_single_model",
                    "latency_ms": round(latency, 2),
                }
            else:
                pred = self.model(tensor)
                pred_val = float(pred.squeeze().item())
                cat = map_intensity_to_imd_category(pred_val)
                latency = (time.perf_counter() - t0) * 1000.0

                return {
                    "predicted_intensity_knots": round(pred_val, 2),
                    "imd_category": cat,
                    "inference_mode": "legacy_single_model",
                    "latency_ms": round(latency, 2),
                }

    def predict_file(self, file_path: str) -> Dict[str, Any]:
        arr = load_satellite_image(file_path)
        res = self.predict_array(arr)
        res["file_path"] = file_path
        res["filename"] = os.path.basename(file_path)
        return res

    def predict_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        results = []
        valid_exts = (".npy", ".png", ".jpg", ".jpeg")
        for root, _, files in os.walk(dir_path):
            for f in sorted(files):
                if f.lower().endswith(valid_exts):
                    p = os.path.join(root, f)
                    try:
                        res = self.predict_file(p)
                        results.append(res)
                    except Exception as e:
                        print(f"Error predicting {p}: {e}", file=sys.stderr)
        return results


# Default predictor class points to the 5-fold ensemble
SatellitePredictor = EnsembleSatellitePredictor
SatelliteEnsemblePredictor = EnsembleSatellitePredictor
SatelliteIntensityPredictor = EnsembleSatellitePredictor


def predict_satellite_image(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    use_ensemble: bool = True
) -> Dict[str, Any]:
    """
    Convenience API for single image prediction.
    By default (use_ensemble=True, checkpoint_path=None), runs the 5-fold soft-voting ensemble.
    If checkpoint_path is explicitly provided or use_ensemble=False, uses legacy single-model predictor.
    """
    if checkpoint_path is not None or not use_ensemble:
        predictor = LegacySingleSatellitePredictor(checkpoint_path=checkpoint_path)
    else:
        predictor = EnsembleSatellitePredictor()
    return predictor.predict_file(image_path)


def main():
    parser = argparse.ArgumentParser(description="CycloneAI Satellite CNN Production Inference (5-Fold Ensemble Default)")
    parser.add_argument("--image", type=str, help="Path to a single .npy satellite array or image (.png, .jpg)")
    parser.add_argument("--dir", type=str, help="Path to directory containing satellite files")
    parser.add_argument("--single", action="store_true", help="Run in legacy single-model mode instead of 5-fold ensemble")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (triggers legacy single-model mode)")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON or CSV file")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    if args.single or args.checkpoint is not None:
        ckpt = args.checkpoint if args.checkpoint is not None else DEFAULT_SINGLE_CHECKPOINT
        predictor = LegacySingleSatellitePredictor(checkpoint_path=ckpt)
    else:
        predictor = EnsembleSatellitePredictor()

    if args.image:
        result = predictor.predict_file(args.image)
        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
    elif args.dir:
        results = predictor.predict_directory(args.dir)
        print(f"Processed {len(results)} images.")
        if args.output:
            if args.output.endswith(".csv"):
                import pandas as pd
                pd.DataFrame(results).to_csv(args.output, index=False)
            else:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            for r in results[:10]:
                print(f"{r['filename']}: {r['predicted_class']} ({r['predicted_wind_speed_knots']} kt)")
            if len(results) > 10:
                print(f"... and {len(results) - 10} more.")


if __name__ == "__main__":
    main()
