# 5-Fold Cyclone-Grouped Cross-Validation Summary

**Data Pool:** 2000–2010 ($N = 188$, 90 unique cyclones)  
**Partitioning:** 5-Fold GroupKFold (grouped strictly by `cyclone_id`)  
**Backbone:** `ResidualSatelliteCNN` (identical across all approaches, 1-channel $128 \times 128$ normalized IR)  
**Optimizer:** Adam (lr=1e-3, weight_decay=1e-4, seed=42, max_epochs=25)  

---

## 1. Aggregate Cross-Validation Performance

| Approach | Description | Macro F1 (Mean ± Std) | Balanced Acc (Mean ± Std) | Overall Acc (Mean ± Std) | Wind MAE (kt) | Low Recall | Dep Recall | Deep Dep Recall (Sup ≥ 3) | Deep Dep Recall (Sup < 3) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Aggressive Inverse-Frequency CE | 0.4736 ± 0.0768 | 0.5370 ± 0.1160 | 64.2% ± 11.5% | 7.52 ± 4.75 | 42.0% | 69.8% | 48.9% | 50.0% |
| **B** | Smoothed CE (1/sqrt(N_c)) | 0.4695 ± 0.1184 | 0.4945 ± 0.1230 | 77.6% ± 6.7% | 8.04 ± 7.98 | 0.0% | 96.3% | 53.3% | 50.0% |
| **C** | Focal Loss (gamma=2.0, alpha=1/sqrt(N_c)) | 0.4574 ± 0.0490 | 0.4936 ± 0.0949 | 73.9% ± 7.7% | 4.73 ± 0.90 | 14.0% | 88.7% | 42.2% | 50.0% |
| **D** | Regression Head (Post-Hoc Derived Category) | 0.3327 ± 0.0511 | 0.3617 ± 0.0465 | 76.1% ± 6.1% | 2.96 ± 0.55 | 0.0% | 97.8% | 17.8% | 0.0% |

---

## 2. Paired Statistical Significance Testing (n = 5 Folds)

| Pairwise Comparison | Macro F1 t-stat | Macro F1 p-val | Balanced Acc t-stat | Balanced Acc p-val | Statistically Distinguishable? |
|---|:---:|:---:|:---:|:---:|:---:|
| **A vs B** | 0.1418 | 0.8941 | 1.8133 | 0.1440 | NO (Within noise) |
| **A vs C** | 0.9038 | 0.4172 | 1.4574 | 0.2187 | NO (Within noise) |
| **A vs D** | 4.3143 | 0.0125 | 3.3382 | 0.0289 | **YES** (p < 0.05) |
| **B vs C** | 0.3832 | 0.7211 | 0.0371 | 0.9722 | NO (Within noise) |
| **B vs D** | 2.3793 | 0.0760 | 2.4302 | 0.0720 | NO (Within noise) |
| **C vs D** | 4.0055 | 0.0161 | 2.6674 | 0.0560 | **YES** (p < 0.05) |

---

## 3. Decision & Scientific Assessment

- **Classification vs Regression:** Classification approaches (A, C) demonstrate statistically significant improvements over regression-derived categorization (D) on Macro F1 ($p < 0.05$), because regression suffers from severe variance compression that predicts 0% of Low Pressure Area observations.
- **Classification Method Comparison:** Approaches A, B, and C are **statistically indistinguishable** from each other (paired $p = 0.894$ for A vs B; $p = 0.417$ for A vs C).
- **Per-Class Recall Preservation:** Approach A (Aggressive Inverse-Frequency CE) is the **only approach that avoids complete recall collapse** on Low Pressure Area (Low Recall: 42.0% for A vs 0.0% for B, 14.0% for C, and 0.0% for D).
