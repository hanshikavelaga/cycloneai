# CycloneAI -- Regression-Derived Categorization vs. Learned Classification Head
**Evaluation Split:** 2009-2010 Validation Set (30 samples across 15 cyclones)
**Evaluation Status:** Strict Validation-Only Evaluation (Frozen 2011-2015 Test Set NOT Evaluated)

## 1. Objective
Evaluate whether converting continuous wind speed predictions (v_hat, in knots) into IMD-style genesis categories via fixed physical meteorological thresholds provides a more reliable, balanced classification than the learned 3-class classification head (`fc_class`).

## 2. Fixed Thresholds
The thresholds applied to predicted wind speed (v_hat) are fixed by official IMD genesis definitions and are strictly un-tuned:
* **Low Pressure Area (< 17 kts):** v_hat < 17.0 kts
* **Depression (17-27 kts):** 17.0 <= v_hat <= 27.0 kts
* **Deep Depression (28-33 kts):** v_hat >= 28.0 kts

## 3. Validation Dataset Description
* **Split Seasons:** 2009 & 2010
* **Sample Count:** 30 satellite observations
* **Unique Cyclones:** 15
* **True Class Distribution:**
  * Low Pressure Area: 6 samples (20.0%)
  * Depression: 23 samples (76.7%)
  * Deep Depression: 1 sample (3.3%)
* **Wind Speed Range:** 15.0 to 28.0 knots (Mean: 22.0 kts, Std: 4.32 kts)

## 4. Regression Quality Metrics on Validation Set
* **Mean Absolute Error (MAE):** 3.75 knots
* **Root Mean Squared Error (RMSE):** 4.67 knots
* **Coefficient of Determination ($R^2$):** -0.167
* **Mean True Wind Speed:** 22.0 knots (Std: 4.32 kts)
* **Mean Predicted Wind Speed:** 23.9 knots (Std: 2.53 kts)
* **Predicted Wind Range:** [20.1, 31.46] knots

> **Key Meteorological Observation on Regression Variance:**
> The regression head exhibits significant conditional-mean shrinkage (variance compression). While ground-truth wind speeds range from 15.0 to 28.0 knots ($\sigma = 4.32$ kts), model predictions have a standard deviation of only $\sigma = 2.53$ kts. Crucially, the minimum predicted wind speed across all 30 validation samples is **20.1 knots**, which is strictly above the 17.0 knot threshold for Low Pressure Area.

## 5. Direct Comparison Table
| Metric | Approach A: Learned Classification Head | Approach B: Regression-Derived Category | Delta (B vs. A) |
|---|---|---|---|
| **Overall Accuracy** | 46.67% (14/30) | 66.67% (20/30) | **+20.00%** |
| **Balanced Accuracy** | **72.71%** | 28.99% | **-43.72% (Severely Degraded)** |
| **Macro Precision** | 0.4940 | 0.2469 | -0.2471 |
| **Macro Recall** | **0.7271** | 0.2899 | **-0.4372** |
| **Macro F1-Score** | **0.4128** | 0.2667 | **-0.1461** |
| **Weighted F1-Score** | 0.5031 | 0.6133 | +0.1102 |

## 6. Confusion Matrices
### Approach A: Classification Head
```
               Predicted
               Low   Dep   Deep
True Low         5     0      1
True Dep         9     8      6
True Deep        0     0      1
```

### Approach B: Regression-Derived Category
```
               Predicted
               Low   Dep   Deep
True Low         0     6      0
True Dep         0    20      3
True Deep        0     1      0
```

## 7. Per-Class Performance Comparison
| Category | Support | Head Recall | Reg Recall | Head Precision | Reg Precision | Head F1 | Reg F1 |
|---|---|---|---|---|---|---|---|
| **Low Pressure Area (< 17 kts)** | 6 | **83.3%** | 0.0% | 35.7% | 0.0% | 0.5000 | 0.0000 |
| **Depression (17-27 kts)** | 23 | **34.8%** | 87.0% | 100.0% | 74.1% | 0.5161 | 0.8000 |
| **Deep Depression (28-33 kts)** | 1 | **100.0%** | 0.0% | 12.5% | 0.0% | 0.2222 | 0.0000 |

## 8. Predicted Class Distributions
* **Ground Truth:** 6 Low (20.0%), 23 Depression (76.7%), 1 Deep Depression (3.3%)
* **Approach A (Classification Head):** 14 Low (46.7%), 8 Depression (26.7%), 8 Deep Depression (26.7%) -- *Actively predicts all three classes*
* **Approach B (Regression-Derived):** 0 Low (0.0%), 27 Depression (90.0%), 3 Deep Depression (10.0%) -- *Severe collapse into Depression*

## 9. Boundary Analysis
### Region 1: Low / Depression Boundary (15-19 knots)
* **Sample Count:** 10 samples (6 Low Pressure at 15.0 kts, 4 Depression at 18.0 kts)
* **Classification Head Accuracy:** **70.0%** (7/10 correct)
* **Regression-Derived Accuracy:** 40.0% (4/10 correct)
* **Mean Absolute Wind Error:** 6.47 kts
* **Why Approach B fails at the 15-19 kt boundary:** All 6 Low Pressure samples (true wind = 15.0 kts) have predicted wind speeds between 20.8 and 25.1 kts (average prediction: 22.8 kts). Because every prediction is >= 17.0 kts, Approach B misclassifies 100% of Low Pressure samples into Depression.

### Region 2: Depression / Deep Depression Boundary (26-30 knots)
* **Sample Count:** 1 sample (1 Deep Depression at 28.0 kts)
* **Classification Head Accuracy:** **100.0%** (Correctly identified as Deep Depression)
* **Regression-Derived Accuracy:** 0.0% (Predicted 26.71 kts -> Misclassified as Depression)
* **Mean Absolute Wind Error:** 1.29 kts

## 10. Regression Error Analysis & Verification of '26 of 33' Claim
### A. Verification on 2009-2010 Validation Data (16 Classification Head Errors)
* Total misclassifications by classification head: 16 / 30
* Absolute wind error <= 1.0 knot: **4** (25.0%)
* Absolute wind error <= 2.0 knots: **6** (37.5%)
* Absolute wind error <= 3.0 knots: **9** (56.2%)
* Absolute wind error <= 5.0 knots: **13** (81.2%)

### B. Audit of Previously Recorded Frozen Test Errors (33 Errors in `error_analysis.md`)
* From the existing static records (no re-evaluation conducted):
  * Error <= 1.0 kt: **8 / 33** (24.2%)
  * Error <= 2.0 kt: **13 / 33** (39.4%)
  * Error <= 3.0 kt: **23 / 33** (69.7%) *(Note: The informal '26 of 33' claim corresponded to errors <= 3.6 kt; the exact count within 3.0 kt is 23/33)*
  * Error <= 5.0 kt: **27 / 33** (81.8%)

## 11. Complete Validation Sample Audit Table
| Sample | Filename | Cyclone ID | Year | True Wind | True Class | Pred Wind | Head Class | Reg Class | Head OK | Reg OK |
|---|---|---|---|---|---|---|---|---|---|---|
| 00 | `hursat_genesis_2009104N13088_BIJLI_0.npy` | `2009104N13088` | 2009 | 25.0 | Depression (17-27 kts) | 25.1 | Deep Depression (28-33 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 01 | `hursat_genesis_2009104N13088_BIJLI_3.npy` | `2009104N13088` | 2009 | 28.0 | Deep Depression (28-33 kts) | 26.7 | Deep Depression (28-33 kts) | Depression (17-27 kts) | CORRECT | WRONG |
| 02 | `hursat_genesis_2009143N17089_AILA_0.npy` | `2009143N17089` | 2009 | 25.0 | Depression (17-27 kts) | 21.9 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 03 | `hursat_genesis_2009143N17089_AILA_3.npy` | `2009143N17089` | 2009 | 25.0 | Depression (17-27 kts) | 20.8 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 04 | `hursat_genesis_2009174N18072_MISSING_0.npy` | `2009174N18072` | 2009 | 25.0 | Depression (17-27 kts) | 24.8 | Deep Depression (28-33 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 05 | `hursat_genesis_2009174N18072_MISSING_3.npy` | `2009174N18072` | 2009 | 25.0 | Depression (17-27 kts) | 24.6 | Depression (17-27 kts) | Depression (17-27 kts) | CORRECT | CORRECT |
| 06 | `hursat_genesis_2009177N23069_MISSING_0.npy` | `2009176N23069` | 2009 | 25.0 | Depression (17-27 kts) | 22.4 | Depression (17-27 kts) | Depression (17-27 kts) | CORRECT | CORRECT |
| 07 | `hursat_genesis_2009177N23069_MISSING_3.npy` | `2009176N23069` | 2009 | 25.0 | Depression (17-27 kts) | 22.1 | Depression (17-27 kts) | Depression (17-27 kts) | CORRECT | CORRECT |
| 08 | `hursat_genesis_2009201N21089_MISSING_0.npy` | `2009201N21089` | 2009 | 25.0 | Depression (17-27 kts) | 23.2 | Depression (17-27 kts) | Depression (17-27 kts) | CORRECT | CORRECT |
| 09 | `hursat_genesis_2009201N21089_MISSING_3.npy` | `2009201N21089` | 2009 | 25.0 | Depression (17-27 kts) | 22.0 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 10 | `hursat_genesis_2009247N19089_MISSING_0.npy` | `2009247N19089` | 2009 | 15.0 | Low Pressure Area (< 17 kts) | 20.4 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | CORRECT | WRONG |
| 11 | `hursat_genesis_2009247N19089_MISSING_3.npy` | `2009247N19089` | 2009 | 18.0 | Depression (17-27 kts) | 20.1 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 12 | `hursat_genesis_2009313N12071_PHYAN_0.npy` | `2009313N11072` | 2009 | 20.0 | Depression (17-27 kts) | 25.8 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 13 | `hursat_genesis_2009313N12071_PHYAN_3.npy` | `2009313N11072` | 2009 | 25.0 | Depression (17-27 kts) | 26.4 | Deep Depression (28-33 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 14 | `hursat_genesis_2009345N07085_WARD_0.npy` | `2009344N06085` | 2009 | 25.0 | Depression (17-27 kts) | 23.2 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 15 | `hursat_genesis_2009345N07085_WARD_2.npy` | `2009344N06085` | 2009 | 25.0 | Depression (17-27 kts) | 24.4 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 16 | `hursat_genesis_2010136N09057_BANDU_0.npy` | `2010136N09057` | 2010 | 15.0 | Low Pressure Area (< 17 kts) | 23.9 | Deep Depression (28-33 kts) | Depression (17-27 kts) | WRONG | WRONG |
| 17 | `hursat_genesis_2010136N09057_BANDU_3.npy` | `2010136N09057` | 2010 | 15.0 | Low Pressure Area (< 17 kts) | 20.6 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | CORRECT | WRONG |
| 18 | `hursat_genesis_2010137N10090_LAILA_0.npy` | `2010137N10090` | 2010 | 15.0 | Low Pressure Area (< 17 kts) | 22.9 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | CORRECT | WRONG |
| 19 | `hursat_genesis_2010137N10090_LAILA_3.npy` | `2010137N10090` | 2010 | 18.0 | Depression (17-27 kts) | 21.4 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | WRONG | CORRECT |
| 20 | `hursat_genesis_2010151N14065_PHET_0.npy` | `2010151N14065` | 2010 | 15.0 | Low Pressure Area (< 17 kts) | 25.1 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | CORRECT | WRONG |
| 21 | `hursat_genesis_2010151N14065_PHET_3.npy` | `2010151N14065` | 2010 | 18.0 | Depression (17-27 kts) | 24.4 | Depression (17-27 kts) | Depression (17-27 kts) | CORRECT | CORRECT |
| 22 | `hursat_genesis_2010280N18085_MISSING_0.npy` | `2010280N17085` | 2010 | 25.0 | Depression (17-27 kts) | 31.5 | Deep Depression (28-33 kts) | Deep Depression (28-33 kts) | WRONG | WRONG |
| 23 | `hursat_genesis_2010280N18085_MISSING_3.npy` | `2010280N17085` | 2010 | 25.0 | Depression (17-27 kts) | 28.5 | Deep Depression (28-33 kts) | Deep Depression (28-33 kts) | WRONG | WRONG |
| 24 | `hursat_genesis_2010286N18090_MISSING_0.npy` | `2010286N18090` | 2010 | 25.0 | Depression (17-27 kts) | 21.7 | Depression (17-27 kts) | Depression (17-27 kts) | CORRECT | CORRECT |
| 25 | `hursat_genesis_2010286N18090_MISSING_3.npy` | `2010286N18090` | 2010 | 25.0 | Depression (17-27 kts) | 22.5 | Depression (17-27 kts) | Depression (17-27 kts) | CORRECT | CORRECT |
| 26 | `hursat_genesis_2010293N17093_GIRI_0.npy` | `2010293N17093` | 2010 | 15.0 | Low Pressure Area (< 17 kts) | 24.4 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | CORRECT | WRONG |
| 27 | `hursat_genesis_2010293N17093_GIRI_2.npy` | `2010293N17093` | 2010 | 18.0 | Depression (17-27 kts) | 23.5 | Depression (17-27 kts) | Depression (17-27 kts) | CORRECT | CORRECT |
| 28 | `hursat_genesis_2010341N15082_MISSING_0.npy` | `2010341N14082` | 2010 | 25.0 | Depression (17-27 kts) | 27.5 | Deep Depression (28-33 kts) | Deep Depression (28-33 kts) | WRONG | WRONG |
| 29 | `hursat_genesis_2010341N15082_MISSING_2.npy` | `2010341N14082` | 2010 | 25.0 | Depression (17-27 kts) | 25.4 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | WRONG | CORRECT |

## 12. Final Validation-Based Decision
> **OFFICIAL CONCLUSION:**
> **Regression-derived categorization is NOT supported by the current validation evidence.**

### Detailed Rationale:
1. **The Accuracy Paradox (Raw vs. Balanced):** While Approach B achieves a higher raw accuracy (66.67% vs. 46.67%), this is entirely an artifact of majority-class bias. Because 76.7% of the validation set is Depression and the regression head compresses predictions into the 20-27 kt range, Approach B classifies 27 out of 30 samples as Depression.
2. **Catastrophic Minority Class Collapse in Approach B:**
   * **Low Pressure Recall:** **0.00% (0 / 6)** under Approach B, compared to **83.33% (5 / 6)** under Approach A.
   * **Deep Depression Recall:** **0.00% (0 / 1)** under Approach B, compared to **100.00% (1 / 1)** under Approach A.
   * **Balanced Accuracy:** Collapses from **72.71%** (Approach A) down to **28.99%** (Approach B).
   * **Macro F1:** Drops from **0.4128** down to **0.2667**.
3. **Physical Explanation (Conditional Mean Shrinkage):** A regression model trained with MSE minimizes squared error by predicting the conditional expectation E[Y|X]. In a severely skewed intensity dataset with mean ~22.8 kts, MSE pulls predictions towards the center. The minimum predicted wind speed on validation is **20.10 kts**. Thus, applying a rigid cutoff at 17.0 kts without calibration ensures that zero samples will ever be classified as Low Pressure Area.
4. **Strict Stop Condition Followed:** The 2011-2015 frozen test set was **NOT** evaluated, touched, or inspected. No thresholds were tuned.
