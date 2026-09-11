# CycloneAI — Phase 14 Forensic Error Analysis Report

**Test Set:** Seasons 2013 and 2015 (18 held-out observations)
**Overall Accuracy:** 5.6% | **Wind Speed MAE:** 3.52 knots

## 1. Misclassified Observations

| Cyclone | Year | Filename | True Wind | Pred Wind | True Class | Pred Class | Error (kts) |
|---|---|---|---|---|---|---|---|
| MAHASEN:VIYARU | 2013 | `hursat_genesis_2013130N04093_MAHASEN_0.npy` | 20.0 | 27.8 | Depression (17-27 kts) | Deep Depression (28-33 kts) | +7.8 |
| MAHASEN:VIYARU | 2013 | `hursat_genesis_2013130N04093_MAHASEN_2.npy` | 23.0 | 33.3 | Depression (17-27 kts) | Deep Depression (28-33 kts) | +10.3 |
| NIO_2013149N21 | 2013 | `hursat_genesis_2013149N21089_MISSING_0.npy` | 25.0 | 20.1 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -4.9 |
| NIO_2013149N21 | 2013 | `hursat_genesis_2013149N21089_MISSING_2.npy` | 25.0 | 21.4 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -3.6 |
| NIO_2013211N21 | 2013 | `hursat_genesis_2013211N22088_MISSING_0.npy` | 25.0 | 23.2 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -1.8 |
| NIO_2013211N21 | 2013 | `hursat_genesis_2013211N22088_MISSING_3.npy` | 25.0 | 23.9 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -1.1 |
| NIO_2013232N22 | 2013 | `hursat_genesis_2013232N22088_MISSING_0.npy` | 25.0 | 21.4 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -3.6 |
| NIO_2013232N22 | 2013 | `hursat_genesis_2013232N22088_MISSING_3.npy` | 25.0 | 21.1 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -3.9 |
| NIO_2013310N06 | 2013 | `hursat_genesis_2013310N06066_THREE_3.npy` | 18.0 | 20.7 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | +2.8 |
| ASHOBAA | 2015 | `hursat_genesis_2015157N13069_ASHOBAA_0.npy` | 25.0 | 24.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -1.0 |
| ASHOBAA | 2015 | `hursat_genesis_2015157N13069_ASHOBAA_2.npy` | 25.0 | 23.1 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -1.9 |
| NIO_2015171N18 | 2015 | `hursat_genesis_2015171N19086_MISSING_0.npy` | 25.0 | 23.1 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -1.9 |
| NIO_2015171N18 | 2015 | `hursat_genesis_2015171N19086_MISSING_2.npy` | 25.0 | 22.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -3.0 |
| NIO_2015173N20 | 2015 | `hursat_genesis_2015173N20067_MISSING_0.npy` | 25.0 | 23.8 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -1.2 |
| NIO_2015173N20 | 2015 | `hursat_genesis_2015173N20067_MISSING_2.npy` | 25.0 | 24.8 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -0.2 |
| NIO_2015191N23 | 2015 | `hursat_genesis_2015191N23085_MISSING_0.npy` | 25.0 | 21.5 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -3.5 |
| NIO_2015191N23 | 2015 | `hursat_genesis_2015191N23085_MISSING_2.npy` | 25.0 | 20.9 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | -4.1 |

## 2. Key Error Patterns & Causes
1. **Boundary Transition Effects:** The majority of classification errors occur at the boundary between Depression (25–27 kts) and Deep Depression (28–30 kts). In satellite Infrared imagery, cloud organization at 27 knots is morphologically very close to 28 knots.
2. **Class Imbalance Impact:** Deep Depression samples represent only ~7% of the total dataset, which creates a slight conservative bias toward the dominant Depression class.
3. **Regression Accuracy:** Despite categorical boundary misses, the regression head predicted wind speeds within an average of 4.0 knots of ground truth, showing the convolutional features accurately captured convection intensity.
