# Satellite-Only Tropical Cyclone Intensity Estimation (2000–2015)

Production-grade PyTorch implementation of a residual convolutional neural network (**ResidualSatelliteRegressor**) for estimating contemporaneous tropical cyclone intensity (maximum sustained wind speed in knots) from normalized NOAA HURSAT-B1 infrared satellite imagery ($128 \times 128$).

---

## 1. Overview & Dataset Protocol

- **Input Modality:** Single-channel normalized infrared brightness temperature ($1 \times 128 \times 128$, float32 $\in [0.0, 1.0]$).
- **Target:** Ground-truth contemporaneous maximum sustained wind speed in knots (`intensity_knots`).
- **Data Period:** Fixed 2000–2015 window ($N = 234$ verified physical observations across 113 unique cyclones).
- **Physical Exclusions:** 222 unindexed/unverified images quarantined and completely excluded.
- **Scientific Task:** Contemporaneous intensity/development estimation (knots). **Does not** claim or fabricate future cyclogenesis prediction.

### Strict Chronological Partitioning (Zero Cyclone Leakage)

| Split | Years | Observations | Unique Cyclones | Target Mean (kt) | Role |
|---|---|---|---|---|---|
| **Train** | 2000–2008 | 158 | 75 | 22.85 | Backpropagation, Augmentations |
| **Validation** | 2009–2010 | 30 | 15 | 24.33 | Early Stopping & Model Selection |
| **Held-Out Test** | 2011–2015 | 46 | 23 | 24.89 | Single-Pass Frozen Evaluation |

> **Firewall Guarantees:**
> - `TEST_DATA_ACCESSED_DURING_TRAINING = 0`: Test samples were never accessed during training, feature scaling, hyperparameter tuning, or threshold selection.
> - `FINAL_TEST_EVALUATION = 1`: The test split was evaluated exactly once after the best checkpoint was frozen.
> - **Zero Cyclone Overlap:** Intersection of cyclone identifiers between Train, Validation, and Test is strictly $\emptyset$.

---

## 2. Model Architecture (`ResidualSatelliteRegressor`)

The model contains **620,961 parameters** and is structured as:
1. **Stem:** `Conv2d(1, 32, k=3, s=2, p=1)` $\to$ `BatchNorm2d(32)` $\to$ `ReLU` (Output: $32 \times 64 \times 64$).
2. **Stage 1:** `ResidualBlock(32, 64, stride=2)` (Output: $64 \times 32 \times 32$).
3. **Stage 2:** `ResidualBlock(64, 128, stride=2)` (Output: $128 \times 16 \times 16$).
4. **Stage 3:** `ResidualBlock(128, 128, stride=2)` (Output: $128 \times 8 \times 8$).
5. **Global Pooling:** `AdaptiveAvgPool2d((1, 1))` (Output: $128$).
6. **Regression Head:** `Linear(128, 128)` $\to$ `ReLU` $\to$ `Dropout(0.3)` $\to$ `Linear(128, 32)` $\to$ `ReLU` $\to$ `Linear(32, 1)`.

---

## 3. Directory Structure

```
models/satellite/
├── checkpoints/
│   └── best_satellite_regressor.pth        # Best model weights (Epoch 12, Val MAE: 3.34 kt)
├── dataset_audit/
│   ├── verified_satellite_manifest.csv     # 234 verified observations manifest
│   ├── satellite_2000_2015_audit.json      # Machine-readable audit summary
│   ├── satellite_2000_2015_audit.md        # Detailed audit report
│   ├── split_plan.json                     # Split specification
│   └── year_distribution.csv              # Per-year counts and statistics
├── evaluation/
│   ├── validation_metrics.json             # Validation split performance metrics
│   ├── test_metrics.json                   # Final test split performance metrics
│   ├── test_predictions.csv                # Sample-level predictions and residuals
│   └── confusion_matrix.csv                # Derived 3-class IMD confusion matrix
├── plots/
│   ├── training_curves.png                 # Loss and MAE learning trajectories
│   ├── prediction_vs_actual.png            # Scatter plot with ±5 kt error band
│   └── residual_distribution.png           # Error histogram and normal density fit
├── training/
│   ├── dataset.py                          # SatelliteDataset and DataLoader
│   ├── model.py                            # ResidualSatelliteRegressor definition
│   ├── train_satellite_regressor.py        # End-to-end training pipeline
│   ├── training_config.json                # Complete hyperparameter record
│   └── training_history.json               # Epoch-by-epoch loss and MAE logs
├── experiment_report.md                    # Comprehensive scientific markdown report
├── inference.py                            # Standalone CLI inference tool
└── README.md                               # System documentation (this file)
```

---

## 4. Quickstart & Inference

### Predict on a Single Satellite Image (.npy)
```powershell
python models/satellite/inference.py --image "data/processed/satellite/hursat_genesis_2000088N08090_MISSING_0.npy"
```
**Example Output:**
```json
{
  "predicted_intensity_knots": 21.21,
  "imd_category": "Depression (17-27 kts)",
  "file_path": "data/processed/satellite/hursat_genesis_2000088N08090_MISSING_0.npy",
  "filename": "hursat_genesis_2000088N08090_MISSING_0.npy"
}
```

### Batch Predict on a Directory of Images
```powershell
python models/satellite/inference.py --dir "data/processed/satellite" --output "models/satellite/batch_predictions.csv"
```

---

## 5. Pipeline Reproduction

To reproduce training and evaluation from scratch:

```powershell
# 1. Run training (respects random_seed=42, early stopping on Val MAE)
python models/satellite/training/train_satellite_regressor.py

# 2. Run single-pass test evaluation
python models/satellite/evaluation/evaluate.py

# 3. Regenerate diagnostic plots
python models/satellite/plots/generate_plots.py
```

---

## 6. Key Scientific Findings

1. **Training Convergence:** The model converges cleanly with early stopping triggering at epoch 20 (best model at **epoch 12** with validation MAE of **3.34 kt**).
2. **Error Margins:** On the held-out 2011–2015 test set, **65.2%** of predictions fall within $\pm 5\text{ kt}$ and **95.7%** fall within $\pm 10\text{ kt}$.
3. **Derived Classification:** Overall classification accuracy into IMD stages is **76.1%**, heavily driven by Depression stage prevalence (the modal stage in development tracks).
4. **Satellite-Only Upper Bound:** Residual satellite CNN regression achieves reasonable operational bounds for weak developmental stages, but compresses predictions toward the central mean ($\sigma_{\text{pred}}/\sigma_{\text{true}} = 0.630$). This empirically demonstrates the necessity of multimodal fusion with environmental reanalysis (e.g., ERA5 vertical wind shear, moisture, and SST) for resolving intensity variations beyond cloud-top IR patterns.
