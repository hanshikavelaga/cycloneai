"""
CycloneAI Standalone Satellite Image Inference Module (Phase 16)
================================================================
Provides a production-grade inference interface for the trained ResidualSatelliteCNN.
Accepts raw numpy arrays or file paths, runs zero-leakage normalization, and produces
deterministic classifications and continuous wind speed intensity forecasts.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.classification.model import ResidualSatelliteCNN

DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "models", "classification", "satellite_cnn_best.pth")
DEFAULT_LABELS_PATH = os.path.join(REPO_ROOT, "models", "classification", "label_mapping.json")

_MODEL_CACHE = None
_LABELS_CACHE = None


def load_model(checkpoint_path: str = DEFAULT_MODEL_PATH, labels_path: str = DEFAULT_LABELS_PATH):
    """Loads model and labels with memory caching."""
    global _MODEL_CACHE, _LABELS_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _LABELS_CACHE

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    with open(labels_path, "r") as f:
        labels_meta = json.load(f)

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = len(labels_meta["idx_to_class"])
    model = ResidualSatelliteCNN(in_channels=1, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _MODEL_CACHE = model
    _LABELS_CACHE = labels_meta
    return _MODEL_CACHE, _LABELS_CACHE


def preprocess_image(image_input) -> torch.Tensor:
    """
    Preprocesses various input formats into a torch Tensor of shape (1, 1, 128, 128).
    Handles:
      - File path (.npy, .png, .jpg)
      - NumPy array: 2D (128, 128), 3D (1, 128, 128), (128, 128, 1), (128, 128, 3)
      - Dynamic resizing to (128, 128)
      - Dynamic normalization to [0.0, 1.0]
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        if image_input.endswith(".npy"):
            arr = np.load(image_input)
        else:
            from PIL import Image
            img = Image.open(image_input).convert("L")
            arr = np.array(img, dtype=np.float32)
    elif isinstance(image_input, np.ndarray):
        arr = image_input.copy()
    elif isinstance(image_input, torch.Tensor):
        arr = image_input.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)

    # Convert RGB/Multi-channel to single channel
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            if arr.shape[0] == 3:
                arr = 0.2989 * arr[0] + 0.5870 * arr[1] + 0.1140 * arr[2]
            else:
                arr = arr[0]
        elif arr.shape[-1] in (1, 3):
            if arr.shape[-1] == 3:
                arr = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
            else:
                arr = arr[:, :, 0]

    val_max = float(np.max(arr))
    if val_max > 1.0 and val_max <= 255.0:
        arr = arr / 255.0

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    if tensor.shape[-2:] != (128, 128):
        tensor = F.interpolate(tensor, size=(128, 128), mode="bilinear", align_corners=False)

    return tensor


def predict_satellite_image(image_input, checkpoint_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    End-to-end inference function for satellite image intensity and genesis stage.
    """
    model, labels_meta = load_model(checkpoint_path)
    tensor = preprocess_image(image_input)

    with torch.no_grad():
        logits, reg_pred = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_wind = float(reg_pred.squeeze().item())
        pred_wind = max(0.0, round(pred_wind, 2))

    idx_to_class = labels_meta["idx_to_class"]
    prob_dict = {idx_to_class[str(i)]: round(float(probs[i]), 4) for i in range(len(probs))}

    return {
        "status": "success",
        "prediction": idx_to_class[str(pred_idx)],
        "predicted_class_id": pred_idx,
        "confidence": round(float(probs[pred_idx]), 4),
        "class_probabilities": prob_dict,
        "predicted_wind_speed_knots": pred_wind,
        "model_version": "ResidualSatelliteCNN_v1.0",
        "framework": f"PyTorch {torch.__version__}"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycloneAI Satellite CNN Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to satellite tensor or image file")
    args = parser.parse_args()

    if args.image:
        result = predict_satellite_image(args.image)
        print(json.dumps(result, indent=2))
    else:
        test_sample = os.path.join(
            REPO_ROOT, "data", "processed", "satellite", "hursat_genesis_2015157N13069_ASHOBAA_0.npy"
        )
        if os.path.exists(test_sample):
            print(f"Testing on held-out test sample: {os.path.basename(test_sample)}")
            result = predict_satellite_image(test_sample)
            print(json.dumps(result, indent=2))
        else:
            print("Please provide an image path using --image <path>")

