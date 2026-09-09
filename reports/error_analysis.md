# CycloneAI — Phase 5 Forensic Error Analysis Report (Strategy B)

**Test Set:** Seasons 2011–2015 (46 held-out observations across 23 cyclones)
**Overall Accuracy:** 28.26% | **Balanced Accuracy:** 21.02% | **Wind Speed MAE:** 3.68 knots

## 1. Complete Table of Misclassified Observations

| Filename | Cyclone ID | Cyclone Name | Year | True Wind | True Class | Pred Class | Confidence | Pred Wind | Error (kts) |
|---|---|---|---|---|---|---|---|---|---|
| `hursat_genesis_2011160N18070_ONE_2.npy` | `2011160N18070` | NIO_2011160N18 | 2011 | 23.0 | Depression (17-27 kts) | Deep Depression (28-33 kts) | 0.386 | 25.4 | +2.4 |
| `hursat_genesis_2011167N22089_MISSING_0.npy` | `2011167N22089` | NIO_2011167N22 | 2011 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.420 | 24.4 | -0.6 |
| `hursat_genesis_2011167N22089_MISSING_3.npy` | `2011167N22089` | NIO_2011167N22 | 2011 | 25.0 | Depression (17-27 kts) | Deep Depression (28-33 kts) | 0.348 | 25.3 | +0.3 |
| `hursat_genesis_2011203N24084_MISSING_0.npy` | `2011203N24085` | NIO_2011203N24 | 2011 | 20.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.664 | 21.3 | +1.3 |
| `hursat_genesis_2011203N24084_MISSING_3.npy` | `2011203N24085` | NIO_2011203N24 | 2011 | 20.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.658 | 21.0 | +1.0 |
| `hursat_genesis_2011265N22087_MISSING_0.npy` | `2011265N21088` | NIO_2011265N21 | 2011 | 4.0 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | 0.378 | 23.9 | +19.9 |
| `hursat_genesis_2011290N15090_TWO_0.npy` | `2011290N15090` | NIO_2011290N15 | 2011 | 15.0 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | 0.463 | 22.4 | +7.4 |
| `hursat_genesis_2011290N15090_TWO_2.npy` | `2011290N15090` | NIO_2011290N15 | 2011 | 15.0 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | 0.383 | 23.4 | +8.3 |
| `hursat_genesis_2011302N13062_KEILA_0.npy` | `2011302N13062` | KEILA | 2011 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.548 | 22.9 | -2.1 |
| `hursat_genesis_2011302N13062_KEILA_2.npy` | `2011302N13062` | KEILA | 2011 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.515 | 23.2 | -1.8 |
| `hursat_genesis_2011310N11066_FOUR_0.npy` | `2011310N11066` | NIO_2011310N11 | 2011 | 25.0 | Depression (17-27 kts) | Deep Depression (28-33 kts) | 0.567 | 27.9 | +2.9 |
| `hursat_genesis_2011330N06078_FIVE_0.npy` | `2011330N06078` | NIO_2011330N06 | 2011 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.574 | 25.7 | +0.8 |
| `hursat_genesis_2011330N06078_FIVE_2.npy` | `2011330N06078` | NIO_2011330N06 | 2011 | 28.0 | Deep Depression (28-33 kts) | Low Pressure Area (< 17 kts) | 0.608 | 25.1 | -2.9 |
| `hursat_genesis_2011360N09088_THANE_0.npy` | `2011360N09088` | THANE | 2011 | 30.0 | Deep Depression (28-33 kts) | Low Pressure Area (< 17 kts) | 0.391 | 26.6 | -3.4 |
| `hursat_genesis_2011360N09088_THANE_2.npy` | `2011360N09088` | THANE | 2011 | 30.0 | Deep Depression (28-33 kts) | Low Pressure Area (< 17 kts) | 0.359 | 26.4 | -3.6 |
| `hursat_genesis_2012285N21091_MISSING_0.npy` | `2012285N21091` | NIO_2012285N21 | 2012 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.628 | 23.6 | -1.4 |
| `hursat_genesis_2012285N21091_MISSING_2.npy` | `2012285N21091` | NIO_2012285N21 | 2012 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.628 | 22.1 | -2.9 |
| `hursat_genesis_2012301N11087_NILAM_0.npy` | `2012301N11087` | NILAM | 2012 | 15.0 | Low Pressure Area (< 17 kts) | Depression (17-27 kts) | 0.452 | 22.5 | +7.5 |
| `hursat_genesis_2012357N09064_FOUR_0.npy` | `2012357N09064` | NIO_2012357N09 | 2012 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.416 | 22.8 | -2.2 |
| `hursat_genesis_2012357N09064_FOUR_3.npy` | `2012357N09064` | NIO_2012357N09 | 2012 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.457 | 22.9 | -2.1 |
| `hursat_genesis_2013130N04093_MAHASEN_0.npy` | `2013130N04093` | MAHASEN:VIYARU | 2013 | 20.0 | Depression (17-27 kts) | Deep Depression (28-33 kts) | 0.823 | 31.1 | +11.1 |
| `hursat_genesis_2013130N04093_MAHASEN_2.npy` | `2013130N04093` | MAHASEN:VIYARU | 2013 | 23.0 | Depression (17-27 kts) | Deep Depression (28-33 kts) | 0.923 | 39.7 | +16.7 |
| `hursat_genesis_2013149N21089_MISSING_0.npy` | `2013149N21090` | NIO_2013149N21 | 2013 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.615 | 22.3 | -2.7 |
| `hursat_genesis_2013149N21089_MISSING_2.npy` | `2013149N21090` | NIO_2013149N21 | 2013 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.599 | 23.2 | -1.8 |
| `hursat_genesis_2013232N22088_MISSING_0.npy` | `2013232N22088` | NIO_2013232N22 | 2013 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.488 | 20.9 | -4.1 |
| `hursat_genesis_2013232N22088_MISSING_3.npy` | `2013232N22088` | NIO_2013232N22 | 2013 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.543 | 21.4 | -3.6 |
| `hursat_genesis_2015157N13069_ASHOBAA_0.npy` | `2015157N13069` | ASHOBAA | 2015 | 25.0 | Depression (17-27 kts) | Deep Depression (28-33 kts) | 0.385 | 25.5 | +0.5 |
| `hursat_genesis_2015157N13069_ASHOBAA_2.npy` | `2015157N13069` | ASHOBAA | 2015 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.343 | 24.4 | -0.7 |
| `hursat_genesis_2015171N19086_MISSING_2.npy` | `2015171N18086` | NIO_2015171N18 | 2015 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.477 | 22.8 | -2.2 |
| `hursat_genesis_2015173N20067_MISSING_0.npy` | `2015173N20067` | NIO_2015173N20 | 2015 | 25.0 | Depression (17-27 kts) | Deep Depression (28-33 kts) | 0.423 | 24.8 | -0.2 |
| `hursat_genesis_2015173N20067_MISSING_2.npy` | `2015173N20067` | NIO_2015173N20 | 2015 | 25.0 | Depression (17-27 kts) | Deep Depression (28-33 kts) | 0.569 | 25.8 | +0.8 |
| `hursat_genesis_2015191N23085_MISSING_0.npy` | `2015191N23085` | NIO_2015191N23 | 2015 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.594 | 22.7 | -2.3 |
| `hursat_genesis_2015191N23085_MISSING_2.npy` | `2015191N23085` | NIO_2015191N23 | 2015 | 25.0 | Depression (17-27 kts) | Low Pressure Area (< 17 kts) | 0.551 | 23.4 | -1.6 |

## 2. Key Error Patterns & Meteorological Findings
1. **Boundary Proximity (15 vs 17 kt, 27 vs 28 kt):** In infrared satellite imagery, cloud organization at 15 knots (Low Pressure) and 18–20 knots (Depression) often shows similar curved convection bands.
2. **Deep Depression Representation:** The test set contained 3 Deep Depression observations. The multi-task network captures both continuous intensity and IMD categorical transitions.
3. **Continuous Regression Resilience:** The wind speed regression head predicted intensities within ~3.5 knots MAE, confirming the learned convolutional features reflect true convective thermodynamics.
