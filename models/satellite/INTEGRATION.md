# Satellite CNN Integration Contract (2000–2015 Production Branch)

This document establishes the canonical integration specification for the **production-shipped satellite infrared convolutional neural network model** developed on the 2000–2015 HURSAT dataset.

---

## 1. Production Model Artifacts & Architecture

- **Shipped Production Model:** **5-Fold Ensemble (Approach A — Aggressive Inverse-Frequency Cross-Entropy Weights)**
- **Combination Method:** Soft-voting (averaging softmax class probabilities across all 5 fold models; taking argmax for discrete class assignment; averaging auxiliary regression outputs for continuous intensity).
- **Architecture Class:** `ResidualSatelliteCNN` (defined in `src/models/classification/model.py`)
- **Total Parameters:** 5 x 629,412 = 3,147,060 parameters across the 5 ensemble members.
- **Model Checkpoint Files:**
  1. `models/consolidated_cv/A_fold_1.pth` (2,547,485 bytes)
  2. `models/consolidated_cv/A_fold_2.pth` (2,547,485 bytes)
  3. `models/consolidated_cv/A_fold_3.pth` (2,547,485 bytes)
  4. `models/consolidated_cv/A_fold_4.pth` (2,547,485 bytes)
  5. `models/consolidated_cv/A_fold_5.pth` (2,547,485 bytes)
- **Legacy Single Checkpoint (Non-Default):** `models/consolidated_cv/final_model.pth` (retained for backward compatibility).

---

## 2. Input Specification & Preprocessing

- **Input Modality:** NOAA HURSAT-B1 Infrared Brightness Temperature.
- **Accepted File Format:** `.npy` file containing a NumPy array.
- **Expected Data Type:** `float32`.
- **Expected Input Shape:** `(128, 128)` (2D array) or `(1, 128, 128)` (3D tensor with channel dimension).
- **Preprocessing:** MinMax normalization to [0.0, 1.0].
  - Normalization formula: `(T_ir - T_min) / (T_max - T_min)` where physical arrays in `data/processed/satellite/` are pre-normalized float32 arrays in [0.0, 1.0].
- **Data Augmentation at Inference:** **None.** Input images must be passed without rotation, translation, or flipping.

---

## 3. Output Specification & Classes

The ensemble aggregates predictions from all 5 fold models:
1. **Classification Head:** Soft-voted averaged class probabilities over 3 discrete IMD genesis categories.
2. **Regression Head:** Averaged auxiliary continuous maximum sustained wind speed in knots.

### Output Classes (Exact Strings & Canonical Order)

| Class Index | Class Name | Operational Wind Speed Range |
|:---:|---|---|
| **0** | `Low Pressure Area (< 17 kts)` | < 17.0 knots (< 31 km/h) |
| **1** | `Depression (17-27 kts)` | 17.0 - 27.0 knots (31 - 51 km/h) |
| **2** | `Deep Depression (28-33 kts)` | 28.0 - 33.0 knots (52 - 61 km/h) |

### Soft-Voting Inference Procedure
For an input tensor x, each fold model m in {1, ..., 5} outputs classification logits z^(m) and regression scalar w^(m):
1. Compute softmax probability vector for each fold:
   p_c^(m) = exp(z_c^(m)) / sum_j exp(z_j^(m)),  c in {0, 1, 2}
2. Average softmax probability vectors across all 5 models (soft voting):
   bar_p_c = (1/5) * sum_{m=1}^5 p_c^(m)
3. Assign final predicted discrete class via argmax:
   hat_c = argmax_{c in {0, 1, 2}} bar_p_c
4. Compute ensemble predicted continuous wind speed:
   hat_w = (1/5) * sum_{m=1}^5 hat_w^(m)

---

## 4. Final Verified Evaluation Results (Frozen Test Set 2011–2015)

Evaluated on the frozen, held-out test partition (N = 46 verified observations from 2011–2015):

| Metric | 5-Fold Ensemble (Production) | Single Model (`final_model.pth`) | Delta Improvement |
|---|:---:|:---:|:---:|
| **Test Accuracy** | **60.87%** (28 / 46) | 41.30% (19 / 46) | **+19.57%** |
| **Test Balanced Accuracy** | **45.65%** | 31.98% | **+13.67%** |
| **Test Macro F1** | **0.3810** | 0.2764 | **+0.1046 (+37.8% rel)** |
| **Test Wind MAE** | **6.69 knots** | 5.36 knots | +1.33 kt |
| **Test Wind RMSE** | **7.66 knots** | 9.08 knots | **-1.42 kt** |

### Per-Class Performance Breakdown (Ensemble)
- **Low Pressure Area (< 17 kts):** Recall = **0.00%** (0 / 6), Precision = 0.00%, F1 = 0.0000 (Support = 6)
- **Depression (17–27 kts):** Recall = **70.27%** (26 / 37), Precision = 78.79%, F1 = **0.7429** (Support = 37)
- **Deep Depression (28–33 kts):** Recall = **66.67%** (2 / 3), Precision = 28.57%, F1 = **0.4000** (Support = 3)

---

## 5. Inference Usage & API

### Standalone CLI Execution (Ensemble Default)
```powershell
python models/satellite/inference.py --image "data/processed/satellite/hursat_genesis_2000088N08090_MISSING_0.npy"
```

### Python API Call
```python
from models.satellite.inference import predict_satellite_image

# Runs 5-model soft-voting ensemble by default
result = predict_satellite_image(
    image_path="data/processed/satellite/hursat_genesis_2000088N08090_MISSING_0.npy"
)

print(result)
```

**Expected Return Format:**
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
  "latency_ms": 32.4
}
```

---

## 6. Known Limitations (Stated Plainly, No Softening)

1. **Low Pressure Area Detection Remains Unreliable (0% Recall):**
   - The model currently cannot distinguish nascent Low Pressure Areas (<17 kt) from Depressions (17–27 kt). All 6 test samples of this class were predicted as Depression.
2. **Small Dataset Training Pool:**
   - The model was developed on a total pool of only 234 verified satellite observations (188 in the 2000–2010 training/validation pool, 46 in the 2011–2015 test pool).
3. **Multi-Pass Latency Overhead:**
   - The ensemble requires running 5 forward passes per prediction. On standard multi-core CPU, latency is approximately 25–220 ms per image, which is fully acceptable for operational 3-hourly forecasting.
4. **Temporal Scope Constraint:**
   - The test set covers the historical period 2011–2015 only.
