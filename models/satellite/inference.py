"""
Standalone Inference Module for CycloneAI Satellite Models.
Supports both:
1. Multi-task ResidualSatelliteCNN checkpoints (e.g. models/consolidated_cv/final_model.pth)
2. Pure regression ResidualSatelliteRegressor checkpoints (e.g. models/satellite/checkpoints/best_satellite_regressor.pth)
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, Any, Union, List

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.classification.model import ResidualSatelliteCNN
from models.satellite.training.model import ResidualSatelliteRegressor

CLASS_NAMES = [
    'Low Pressure Area (< 17 kts)',
    'Depression (17-27 kts)',
    'Deep Depression (28-33 kts)'
]

DEFAULT_CHECKPOINT = os.path.join(BASE_DIR, "models", "consolidated_cv", "final_model.pth")


def map_intensity_to_imd_category(wind_kts: float) -> str:
    """Categorizes continuous wind speed into standard IMD cyclone development stages."""
    if wind_kts < 17.0:
        return 'Low Pressure Area (< 17 kts)'
    elif wind_kts <= 27.0:
        return 'Depression (17-27 kts)'
    else:
        return 'Deep Depression (28-33 kts)'


class SatellitePredictor:
    """
    Inference wrapper for trained CycloneAI Satellite CNN models.
    """
    def __init__(self, checkpoint_path: str = None, device: str = None):
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(BASE_DIR, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint_path

        # Load checkpoint weights
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

        # Check if classification head is present in state dict
        has_class_head = any("fc_class" in k for k in state_dict.keys())

        if has_class_head:
            self.model_type = "classification"
            self.model = ResidualSatelliteCNN(in_channels=1, num_classes=3)
            self.model.load_state_dict(state_dict)
        else:
            self.model_type = "regression"
            if ResidualSatelliteRegressor is None:
                from training.model import ResidualSatelliteRegressor as RSR
                self.model = RSR(in_channels=1, dropout=0.3)
            else:
                self.model = ResidualSatelliteRegressor(in_channels=1, dropout=0.3)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Runs inference on a single 128x128 float32 numpy array."""
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 1, 128, 128)
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
                    "latency_ms": round(latency, 2),
                }

    def predict_file(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Array file not found: {file_path}")
        arr = np.load(file_path)
        res = self.predict_array(arr)
        res["file_path"] = file_path
        res["filename"] = os.path.basename(file_path)
        return res

    def predict_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        results = []
        for root, _, files in os.walk(dir_path):
            for f in sorted(files):
                if f.endswith(".npy"):
                    p = os.path.join(root, f)
                    try:
                        res = self.predict_file(p)
                        results.append(res)
                    except Exception as e:
                        print(f"Error predicting {p}: {e}", file=sys.stderr)
        return results


def predict_satellite_image(image_path: str, checkpoint_path: str = None) -> Dict[str, Any]:
    """Convenience functional API for single image prediction."""
    predictor = SatellitePredictor(checkpoint_path=checkpoint_path)
    return predictor.predict_file(image_path)


# Retain backwards compatibility class name
SatelliteIntensityPredictor = SatellitePredictor


def main():
    parser = argparse.ArgumentParser(description="CycloneAI Satellite CNN Inference")
    parser.add_argument("--image", type=str, help="Path to a single .npy satellite array")
    parser.add_argument("--dir", type=str, help="Path to directory containing .npy satellite arrays")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON or CSV file")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    predictor = SatellitePredictor(checkpoint_path=args.checkpoint)

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
                if "predicted_class" in r:
                    print(f"{r['filename']}: {r['predicted_class']} ({r['predicted_wind_speed_knots']} kt)")
                else:
                    print(f"{r['filename']}: {r['imd_category']} ({r['predicted_intensity_knots']} kt)")
            if len(results) > 10:
                print(f"... and {len(results) - 10} more.")


if __name__ == "__main__":
    main()
