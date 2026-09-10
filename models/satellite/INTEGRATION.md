# Satellite CNN Integration Contract (2000–2015 Branch)

This document establishes the canonical integration specification for the satellite infrared convolutional neural network model developed on the 2000–2015 HURSAT dataset.

---

## 1. Model Artifact & Architecture

- **Winning Model Location:** `models/consolidated_cv/final_model.pth`
- **Architecture Class:** `ResidualSatelliteCNN` (defined in `src/models/classification/model.py`)
- **Total Parameters:** 629,412
- **Training Formulation:** Multi-task residual CNN trained with **Approach A (Aggressive Inverse-Frequency Cross-Entropy Weights)** selected via cyclone-grouped 5-fold cross-validation on 2000–2010 pooled data.

---

## 2. Input Specification & Preprocessing

- **Input Modality:** NOAA HURSAT-B1 Infrared Brightness Temperature.
- **Accepted File Format:** `.npy` file containing a NumPy array.
- **Expected Data Type:** `float32`.
- **Expected Input Shape:** `(128, 128)` (2D array) or `(1, 128, 128)` (3D tensor with channel dimension).
- **Preprocessing:** MinMax normalization to $[0.0, 1.0]$.
  - Normalization formula: `(T_ir - T_min) / (T_max - T_min)` where physical arrays in `data/processed/satellite/` are pre-normalized float32 arrays in $[0.0, 1.0]$.
- **Data Augmentation at Inference:** **None.** Input images must be passed without rotation, translation, or flipping.

---

## 3. Output Specification & Classes

The network produces two heads:
1. **Classification Head:** Logits over 3 discrete IMD genesis categories.
2. **Regression Head:** Auxiliary continuous maximum sustained wind speed in knots.

### Output Classes (Exact Strings & Canonical Order)

| Class Index | Class Name | Operational Wind Speed Range |
|:---:|---|---|
| **0** | `Low Pressure Area (< 17 kts)` | $< 17.0\text{ knots}$ ($< 31\text{ km/h}$) |
| **1** | `Depression (17-27 kts)` | $17.0 - 27.0\text{ knots}$ ($31 - 51\text{ km/h}$) |
| **2** | `Deep Depression (28-33 kts)` | $28.0 - 33.0\text{ knots}$ ($52 - 61\text{ km/h}$) |

### Confidence Output
Confidence scores are calculated via Softmax over the classification logits:
$$\text{confidence}_c = \frac{\exp(z_c)}{\sum_{j=0}^2 \exp(z_j)}$$
The sum of all three class probabilities strictly equals $1.0$.

---

## 4. Inference Function Signature & Usage

### Standalone CLI Execution
```powershell
python models/satellite/inference.py --image "data/processed/satellite/hursat_genesis_2000088N08090_MISSING_0.npy" --checkpoint "models/consolidated_cv/final_model.pth"
```

### Python API Call
Located in [`models/satellite/inference.py`](file:///C:/Users/hasin/Desktop/cycloneai/models/satellite/inference.py):

```python
from models.satellite.inference import predict_satellite_image

result = predict_satellite_image(
    image_path="data/processed/satellite/hursat_genesis_2000088N08090_MISSING_0.npy",
    checkpoint_path="models/consolidated_cv/final_model.pth"
)

print(result)
```

**Expected Return Format:**
```json
{
  "predicted_class": "Depression (17-27 kts)",
  "class_index": 1,
  "probabilities": {
    "Low Pressure Area (< 17 kts)": 0.0089,
    "Depression (17-27 kts)": 0.9456,
    "Deep Depression (28-33 kts)": 0.0455
  },
  "predicted_wind_speed_knots": 22.21,
  "latency_ms": 5.9
}
```

---

## 5. Known Limitations & Operational Constraints

1. **Frozen Test-Set Generalization (2011–2015):**
   - **Test Accuracy:** **41.30%** (19 / 46 correct).
   - **Balanced Accuracy:** **31.98%**.
   - **Macro F1:** **0.2764** (27.64%).
   - **Wind Speed MAE:** **5.36 knots** (RMSE: 9.08 knots).
2. **Severe Class Imbalance in Training Pool:**
   - Trained on the 188-observation pooled 2000–2010 development set containing only 13 Deep Depression observations ($6.9\%$) and 31 Low Pressure Area observations ($16.5\%$).
   - Minorities carry higher predictive uncertainty. In Approach A, aggressive weights preserve minority sensitivity (16.7% Low recall, 33.3% Deep Depression recall on test) but cause false positive classifications on Depression cases (12 Depression samples misclassified as Low, 8 as Deep Depression).
3. **Temporal Coverage Constraint:**
   - This model is strictly trained and evaluated on HURSAT observations from **2000–2015**.
   - Performance on **2016+** modern INSAT-3D / Kalpana / Himawari imagery is unverified by this model and falls under the purview of the teammate 2016–2025 branch (`feat/satellite-cnn`).
4. **Physical Information Ceiling of Single-Channel IR:**
   - Cloud-top brightness temperature alone cannot unambiguously distinguish boundary-layer wind speed for disorganized, nascent convective systems (15–35 knots). Multimodal fusion with ERA5 reanalysis fields (wind shear, moisture, SST) is recommended for operational deployment.
