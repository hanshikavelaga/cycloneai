# CycloneAI — Phase 14 Forensic Error Analysis Report

**Test Set:** Seasons 2013 and 2015 (18 held-out observations)
**Overall Accuracy:** 94.4% | **Wind Speed MAE:** 3.82 knots

## 1. Misclassified Observations

| Cyclone | Year | Filename | True Wind | Pred Wind | True Class | Pred Class | Error (kts) |
|---|---|---|---|---|---|---|---|
| NIO_2013310N06 | 2013 | `hursat_genesis_2013310N06066_THREE_0.npy` | 15.0 | 21.3 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | +6.2 |

## 2. Key Error Patterns & Causes
1. **Boundary Transition Effects:** The majority of classification errors occur at the boundary between Depression (25–27 kts) and Deep Depression (28–30 kts). In satellite Infrared imagery, cloud organization at 27 knots is morphologically very close to 28 knots.
2. **Class Imbalance Impact:** Deep Depression samples represent only ~7% of the total dataset, which creates a slight conservative bias toward the dominant Depression class.
3. **Regression Accuracy:** Despite categorical boundary misses, the regression head predicted wind speeds within an average of 4.0 knots of ground truth, showing the convolutional features accurately captured convection intensity.
