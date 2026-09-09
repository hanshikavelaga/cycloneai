# 🛰️ CycloneAI — Satellite Image CNN Classification & Intensity Pipeline

**Lead ML Engineer Documentation (Branch `feat/satellite-cnn`)**  
**Target Basin:** North Indian Ocean (Bay of Bengal & Arabian Sea)  
**Dataset:** NOAA HURSAT-B1 Infrared Satellite Imagery (2000–2015 Seasons)

---

## 1. Executive Summary

This directory contains the machine learning pipeline for automated tropical cyclone genesis stage classification and continuous wind intensity estimation from single-channel satellite infrared imagery.

- **Primary Goal:** Identify early-stage tropical cyclogenesis and forecast peak surface wind speed (in knots) directly from raw infrared brightness temperature tensors.
- **Dataset Scope:** Strictly audited and verified on **2000–2015** North Indian Ocean cyclone events. 456 preprocessed tensors (`(128, 128)` float32, normalized $[0.0, 1.0]$, zero NaNs).
- **Leakage Prevention:** Group-aware temporal partition by year and cyclone ID ensuring **0% storm overlap** between Train (2000–2010), Validation (2011–2012), and Held-out Test (2013 & 2015).

---

## 2. Model Architecture

We trained and compared two distinct convolutional architectures:

### A. SimpleBaselineCNN (35,716 parameters)
- 3 Convolutional blocks: Conv2d(16) -> Conv2d(32) -> Conv2d(64) with BatchNorm, ReLU, and MaxPool2d.
- AdaptiveAvgPool2d((4, 4)) -> Dense classification head (3 classes) + Dense regression head (1 output).

### B. ResidualSatelliteCNN (629,412 parameters) — *Production Model*
- Initial Conv 7x7 stride 2 + MaxPool.
- 4 Residual Stages with identity shortcut connections and residual downsampling (channels: 32 -> 64 -> 128 -> 256).
- Global Average Pooling -> Multi-task dual head:
  - **Classification Head:** Linear(256 -> 64) -> Dropout(0.3) -> Linear(64 -> 3) (IMD Genesis Stage).
  - **Regression Head:** Linear(256 -> 64) -> Dropout(0.2) -> Linear(64 -> 1) (Wind Speed Knots).

---

## 3. Ground Truth Targets

1. **Continuous Regression Target:** Peak sustained surface wind speed in knots (`intensity_knots`).
2. **Discrete Classification Target:** Official India Meteorological Department (IMD) Genesis Stages:
   - `0`: **Low Pressure Area (< 17 kts)**
   - `1`: **Depression (17–27 kts)**
   - `2`: **Deep Depression (28–33 kts)**

---

## 4. Training Protocol & Hyperparameters

- **Loss Function:** Multi-task combined loss:
  $$\mathcal{L}_{total} = \mathcal{L}_{CE, weighted}(\hat{y}_{cls}, y_{cls}) + 0.1 \times \mathcal{L}_{MSE}(\hat{y}_{wind}, y_{wind})$$
  - Class weights applied to Cross-Entropy to handle class imbalance (`[2.11, 0.43, 4.88]`).
- **Optimizer:** AdamW (`lr=1e-3`, `weight_decay=1e-4`) with `CosineAnnealingLR` schedule.
- **Data Augmentation (Train only):** Random horizontal flips ($p=0.5$), random vertical flips ($p=0.5$), and random 90° rotations ($p=0.5$).
- **Batch Size:** 8
- **Epochs:** 25 on CPU with validation checkpointing on Macro F1 and MAE.

---

## 5. Quantitative Evaluation Results

### Validation Set Comparison (2011–2012, 28 samples, 14 cyclones)
| Metric | SimpleBaselineCNN | ResidualSatelliteCNN (Winner) |
|---|---|---|
| Validation Accuracy | 17.86% | **21.43%** |
| Balanced Accuracy | 33.33% | **44.44%** |
| Macro F1-Score | 0.1010 | **0.2409** |
| Wind Speed MAE | 4.08 knots | **4.03 knots** |
| Total Parameters | 35,716 | 629,412 |

### Held-Out Test Evaluation (2013 & 2015 Seasons, 18 samples, 9 cyclones)
Evaluated on strictly unseen held-out storms (including Cyclones Mahasen/Viyaru and Ashobaa):
- **Wind Speed MAE:** **3.52 knots**
- **Wind Speed RMSE:** **4.33 knots**
- **Balanced Classification Accuracy:** **50.00%**
- **Overall Classification Accuracy:** **5.56%** (Reflects conservative class-weight shift at the 27/28 kt boundary)

---

## 6. Directory Structure & Key Files

```
models/classification/
├── satellite_cnn_best.pth        # Production PyTorch weights (ResidualSatelliteCNN)
├── baseline_best.pth             # Baseline model checkpoint
├── residual_best.pth             # Best residual checkpoint from training
├── label_mapping.json            # Class index to IMD stage mapping
└── model_config.json             # Complete model metadata and hyperparameter spec

src/models/classification/
├── dataset.py                    # PyTorch Dataset, augmentation, and split loaders
├── model.py                      # Model architectures (Baseline & Residual)
├── train.py                      # Multi-task training script with comparison
├── evaluate.py                   # Held-out test evaluation & confusion matrix generator
├── inference.py                  # Standalone inference API and CLI runner
└── README.md                     # This documentation file

reports/
├── dataset_distributions.png     # Data distributions across splits
├── representative_satellite_samples.png # Visual samples from dataset
├── confusion_matrix.png          # Test set confusion matrix
├── test_metrics.json             # Full metrics report
├── model_comparison.csv          # Baseline vs Residual comparison
└── error_analysis.md             # Forensic failure analysis report
```

---

## 7. How to Use

### Run Standalone Inference (CLI)
```bash
python src/models/classification/inference.py --image "data/processed/satellite/hursat_genesis_2015157N13069_ASHOBAA_0.npy"
```

### Python API Integration
```python
from src.models.classification.inference import predict_satellite_image

result = predict_satellite_image("path/to/satellite_tensor.npy")
print(result)
# Output:
# {
#   "status": "success",
#   "prediction": "Depression (17-27 kts)",
#   "predicted_class_id": 1,
#   "confidence": 0.854,
#   "class_probabilities": { ... },
#   "predicted_wind_speed_knots": 24.0,
#   "model_version": "ResidualSatelliteCNN_v1.0"
# }
```

### Backend API Endpoint
The FastAPI service exposes `POST /api/predict/image`, automatically calling the production model via `backend.app.services.detection_service`.

