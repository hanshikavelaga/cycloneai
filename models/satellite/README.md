# CycloneAI: Satellite CNN Production Branch (2000–2015)

Production-grade PyTorch implementation of the **5-Fold Residual Satellite CNN Ensemble (`ResidualSatelliteCNN`)** for contemporaneous tropical cyclone intensity estimation and 3-stage IMD development classification from single-channel normalized NOAA HURSAT-B1 infrared satellite imagery ($128 \times 128$).

---

## 1. Production Model: 5-Fold Soft-Voting Ensemble

The shipped production model is a **5-fold ensemble** of `ResidualSatelliteCNN` networks trained with **Approach A (Aggressive Inverse-Frequency Cross-Entropy Weights)** under cyclone-grouped 5-fold cross-validation on the 2000–2010 development pool ($N = 188$).

- **Ensemble Combination Method:** Soft-voting (averaging softmax probability distributions across all 5 fold models; taking argmax for discrete classification; averaging auxiliary regression heads for wind speed in knots).
- **Checkpoints (5 Folds):**
  - `models/consolidated_cv/A_fold_1.pth`
  - `models/consolidated_cv/A_fold_2.pth`
  - `models/consolidated_cv/A_fold_3.pth`
  - `models/consolidated_cv/A_fold_4.pth`
  - `models/consolidated_cv/A_fold_5.pth`
- **Total Parameters:** $5 \times 629,412 = 3,147,060$ parameters.
- **Legacy Single Checkpoint:** `models/consolidated_cv/final_model.pth` (retained for backward compatibility).

---

## 2. Final Verified Evaluation Results (Frozen Test Set 2011–2015)

Evaluated on the frozen, held-out test split ($N = 46$ verified observations across 23 unique cyclones from 2011–2015):

| Metric | 5-Fold Ensemble (Production) | Single Model (`final_model.pth`) | Delta Improvement |
|---|:---:|:---:|:---:|
| **Test Accuracy** | **60.87%** (28 / 46) | 41.30% (19 / 46) | **+19.57%** |
| **Test Balanced Accuracy** | **45.65%** | 31.98% | **+13.67%** |
| **Test Macro F1** | **0.3810** | 0.2764 | **+0.1046 (+37.8% rel)** |
| **Test Wind MAE** | **6.69 knots** | 5.36 knots | +1.33 kt |
| **Test Wind RMSE** | **7.66 knots** | 9.08 knots | **-1.42 kt** |

### Per-Class Performance Breakdown (Ensemble)

| IMD Class | Precision | Recall | F1-Score | Support | Status |
|---|:---:|:---:|:---:|:---:|---|
| **Low Pressure Area (< 17 kts)** | 0.00% | **0.00%** (0 / 6) | 0.0000 | 6 | Weak (unreliable detection) |
| **Depression (17–27 kts)** | 78.79% | **70.27%** (26 / 37) | **0.7429** | 37 | **Strong** (high reliability) |
| **Deep Depression (28–33 kts)** | 28.57% | **66.67%** (2 / 3) | **0.4000** | 3 | Moderate (improved minority recall) |

---

## 3. Known Limitations (Stated Plainly, No Softening)

1. **Low Pressure Area Detection Remains Unreliable (0% Recall):**
   - The model currently cannot distinguish nascent Low Pressure Areas (<17 kt) from Depressions (17–27 kt). All 6 test samples of this class were predicted as Depression.
2. **Small Dataset Training Pool:**
   - The model was trained on only 234 verified satellite observations total (188 in the 2000–2010 development pool, 46 in the 2011–2015 test pool).
3. **Multi-Pass Latency Overhead:**
   - The ensemble requires running 5 forward passes per prediction (~25–220 ms total on CPU based on per-model latency), which is fast and well within operational limits for 3-hourly satellite feeds.
4. **Temporal Scope:**
   - Evaluated on historical 2011–2015 data only.

---

## 4. Quickstart & Production Inference

### Default CLI Inference (5-Fold Ensemble Soft-Voting)
```powershell
python models/satellite/inference.py --image "data/processed/satellite/hursat_genesis_2000088N08090_MISSING_0.npy"
```

**Example Output:**
```json
{
  "predicted_class": "Depression (17-27 kts)",
  "class_index": 1,
  "probabilities": {
    "Low Pressure Area (< 17 kts)": 0.0215,
    "Depression (17-27 kts)": 0.8932,
    "Deep Depression (28-33 kts)": 0.0853
  },
  "predicted_wind_speed_knots": 23.45,
  "ensemble_size": 5,
  "inference_mode": "ensemble_soft_vote",
  "latency_ms": 28.5
}
```

### Legacy Single-Model CLI Inference
```powershell
python models/satellite/inference.py --image "data/processed/satellite/hursat_genesis_2000088N08090_MISSING_0.npy" --single
```

### Python API Call
```python
from models.satellite.inference import predict_satellite_image

# Runs 5-model ensemble by default
result = predict_satellite_image("data/processed/satellite/hursat_genesis_2000088N08090_MISSING_0.npy")
print(result)
```

---

## 5. Directory Structure

```
models/
├── consolidated_cv/
│   ├── A_fold_1.pth ... A_fold_5.pth     # Production 5-fold ensemble checkpoints
│   ├── final_model.pth                   # Legacy single-model checkpoint
│   └── ensemble/
│       ├── ensemble_test_metrics.json    # Verified test performance of ensemble
│       ├── ensemble_confusion_matrix.csv # Test confusion matrix
│       └── ensemble_predictions.csv       # Per-sample test predictions
├── satellite/
│   ├── INTEGRATION.md                    # Canonical integration contract
│   ├── README.md                         # Production branch documentation (this file)
│   ├── inference.py                      # Production inference CLI & API
│   └── dataset_audit/
│       └── verified_satellite_manifest.csv # 234 verified observations manifest
```
