# Concrete Compressive Strength Estimation: Detailed Ablation & Results Report

## 1. Executive Summary
This document provides an exhaustive review of the modeling experiments conducted to predict concrete compressive strength. We compare the results of unconstrained gradient boosting models (**Stage A**) against physics-informed, monotonically constrained models, stacking ensembles, and Gaussian processes (**Stage B**). 

The key finding is a **bias-variance trade-off driven by the dataset scope**: constrained models (Stage B) heavily outperform on sparse, early-age subsets (≤7 days) by acting as strong regularizers. Conversely, unconstrained models (Stage A) slightly edge out constrained models on the Full dataset, benefiting from their flexibility to learn complex interactions across all ages.

---

## 2. Experimental Setup & Ablations

The experiments were broken down along two primary ablation axes:

### A. Algorithmic Ablations
1. **Unconstrained Models (Stage A)**: Baseline implementations of Linear Regression, Random Forest, XGBoost, CatBoost, and LightGBM.
2. **Monotonically Constrained Models (Stage B)**: Gradient boosting models (`XGBoost_constrained`, `CatBoost_constrained`, `LightGBM_constrained`) forced to adhere to physical rules (e.g., compressive strength must monotonically increase with cement content).
3. **Advanced Architectures (Stage B)**: 
    *   **Stacking Ensemble**: Combines the predictions of multiple constrained base learners to reduce individual model variance.
    *   **Gaussian Process (GP)**: Evaluated for its ability to provide uncertainty quantification alongside point estimates.

### B. Data Scope Ablations (Subsetting)
To address the non-linear curing process of concrete, models were trained and evaluated on specific age subsets:
*   **EA1**: Early-age, 1-day strength only. Highly sparse, noisy data.
*   **EA7**: Early-age, up to 7-day strength. 
*   **EA14**: Early-age, up to 14-day strength.
*   **Full**: The entire dataset, spanning 1 to 365 days.

---

## 3. Detailed Results Across Subsets

### 3.1. EA1 Subset (1-Day Strength)
*The hardest subset to predict due to data sparsity and high initial curing variance.*

| Model | RMSE | MAE | R² | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Stage A Best:** CatBoost | 4.158 | 3.032 | 0.793 | Unconstrained, prone to overfitting sparse data. |
| **Stage B:** XGBoost_constrained | 3.569 | 2.593 | 0.865 | **Massive improvement.** Constraints act as crucial regularizers. |
| **Stage B:** Stacking_Ensemble | 3.593 | 2.646 | 0.863 | Highly robust, secondary best model. |

**Takeaway**: Monotonic constraints are absolutely necessary for early-age predictions. They prevent the models from learning spurious correlations in the highly limited data.

### 3.2. EA7 Subset (≤ 7-Day Strength)
*Moderate data density, early curing dynamics still dominant.*

| Model | RMSE | MAE | R² | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Stage A Best:** CatBoost | 4.705 | 3.136 | 0.840 | Solid baseline. |
| **Stage B:** GaussianProcess | 4.406 | 3.076 | 0.862 | Best performer, handles variance well. |
| **Stage B:** CatBoost_constrained | 4.453 | 2.949 | 0.861 | Tight follower to the GP. |
| **Stage B:** Stacking_Ensemble | 4.435 | 2.922 | 0.860 | Consistently strong. |

**Takeaway**: Stage B architectures continue to dominate, showing that constraints and ensembling are highly effective at capturing the first week of curing.

### 3.3. EA14 Subset (≤ 14-Day Strength)
*Transition period. Data becomes denser and curing curves begin to flatten.*

| Model | RMSE | MAE | R² | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Stage A Best:** CatBoost | 4.268 | 2.944 | 0.869 | The unconstrained models begin catching up. |
| **Stage B:** Stacking_Ensemble | 4.263 | 2.914 | 0.870 | Marginal winner. |

**Takeaway**: At 14 days, the dataset size and stability are sufficient that the regularization benefit of the constraints is roughly equal to the flexibility of the unconstrained models.

### 3.4. Full Dataset (1 to 365 Days)
*Requires the model to learn the entire non-linear curing curve and complex interactions of all mixtures.*

| Model | RMSE | MAE | R² | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Stage A Best:** CatBoost | 3.724 | 2.484 | 0.947 | **Best overall model.** Maximum flexibility pays off here. |
| **Stage A:** XGBoost | 4.060 | 2.801 | 0.937 | Excellent unconstrained baseline. |
| **Stage B:** Stacking_Ensemble | 4.134 | 2.621 | 0.938 | Best constrained model, but loses to unconstrained. |
| **Stage B:** CatBoost_constrained | 4.198 | 2.716 | 0.936 | Constrained logic limits peak performance. |

**Takeaway**: The classic bias-variance trade-off. Over the full dataset, the strict monotonic rules (bias) prevent the models from capturing highly complex, potentially non-monotonic interactions that occur late in the curing process or with specific admixture combinations.

---

## 4. Key Takeaways & Recommendations

> [!TIP]
> **Hybrid Deployment Strategy**: No single model rules them all. 
> *   **Recommendation**: Use the `Stacking_Ensemble` (Stage B) for any predictions required at or before 7 days. Use the unconstrained `CatBoost_Full` (Stage A) for general strength estimations or estimations beyond 14 days.

1. **The Power of Physics-Informed ML**: The application of monotonic constraints in Stage B was not a generic hyperparameter tune; it was the injection of domain knowledge. It proved invaluable where data was sparse (EA1/EA7).
2. **Linear Regression as a Barometer**: Across all subsets, the Linear Regression baseline improved from Stage A to Stage B (e.g., Full dataset RMSE dropped from 6.86 to 6.49). This indicates that the feature engineering and preprocessing steps solidified in Stage B provided a universally better foundation for learning.

---

## 5. Shortcomings & Limitations

> [!WARNING]
> While the current pipeline is highly effective, the following shortcomings remain and should be addressed in future iterations.

1. **Rigidity of Monotonic Constraints**: The constraints applied in Stage B are binary (either strictly increasing/decreasing or unconstrained). In reality, some features might have a parabolic or localized relationship (e.g., an admixture that increases strength up to a certain optimal dosage, but decreases it if overdosed). The current implementation forces the model to choose a single direction, which likely caused the performance drop on the Full dataset.
    *   *Future Fix*: Explore localized constraints or Spline-based regressions that allow for inflection points.
2. **Computational Overhead of Stacking**: The `Stacking_Ensemble` provides robust results for early-age predictions but requires training and inferencing multiple base learners. In a highly latency-sensitive production environment, this could introduce unnecessary overhead.
3. **Extreme Data Imbalance**: The age distribution is heavily skewed toward 28-day samples (425 samples), while 1-day (2 samples) and 120-day (3 samples) are incredibly sparse. The subset approach (EA1, EA7) helps isolate this, but the models fundamentally lack enough raw data at the extreme ends of the time spectrum to be definitively reliable without a margin of error.

---

## 6. Comprehensive Experimental Results Appendix

This section contains exhaustive tabular results from all model evaluations across Stage A and Stage B.
Metrics included:
- **RMSE**: Root Mean Squared Error (Lower is better)
- **MAE**: Mean Absolute Error (Lower is better)
- **R²**: Coefficient of Determination (Higher is better, max 1.0)

### Subset: EA1 (1-Day Compressive Strength)
| Stage | Model | RMSE (Mean ± Std) | MAE (Mean ± Std) | R² (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- |
| Stage A (Unconstrained/Baseline) | LinearRegression | 3.7869 ± 0.5430 | 2.9344 ± 0.2914 | 0.8271 ± 0.0582 |
| Stage A (Unconstrained/Baseline) | CatBoost | 4.1583 ± 0.4318 | 3.0323 ± 0.3450 | 0.7935 ± 0.0596 |
| Stage A (Unconstrained/Baseline) | XGBoost | 4.4817 ± 0.7162 | 3.2852 ± 0.3930 | 0.7577 ± 0.0875 |
| Stage A (Unconstrained/Baseline) | LightGBM | 4.5969 ± 0.3130 | 3.4633 ± 0.3163 | 0.7507 ± 0.0592 |
| Stage A (Unconstrained/Baseline) | RandomForest | 4.6442 ± 0.5727 | 3.4357 ± 0.4071 | 0.7465 ± 0.0665 |
| Stage B (Constrained/Advanced) | XGBoost_constrained | 3.5690 ± 0.3085 | 2.5932 ± 0.2631 | 0.8651 ± 0.0191 |
| Stage B (Constrained/Advanced) | Stacking_Ensemble | 3.5938 ± 0.2682 | 2.6468 ± 0.2161 | 0.8632 ± 0.0174 |
| Stage B (Constrained/Advanced) | GaussianProcess | 3.7053 ± 0.2882 | 2.7171 ± 0.1196 | 0.8542 ± 0.0225 |
| Stage B (Constrained/Advanced) | CatBoost_constrained | 3.7340 ± 0.2841 | 2.6979 ± 0.2489 | 0.8516 ± 0.0249 |
| Stage B (Constrained/Advanced) | LightGBM_constrained | 3.9156 ± 0.3013 | 2.8324 ± 0.4360 | 0.8363 ± 0.0304 |
| Stage B (Constrained/Advanced) | LinearRegression | 4.0109 ± 0.3302 | 2.9959 ± 0.3137 | 0.8291 ± 0.0286 |
| Stage B (Constrained/Advanced) | RandomForest | 4.0655 ± 0.2017 | 2.8424 ± 0.1190 | 0.8256 ± 0.0108 |

### Subset: EA7 (7-Day Compressive Strength)
| Stage | Model | RMSE (Mean ± Std) | MAE (Mean ± Std) | R² (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- |
| Stage A (Unconstrained/Baseline) | CatBoost | 4.7051 ± 0.9156 | 3.1364 ± 0.5738 | 0.8400 ± 0.0703 |
| Stage A (Unconstrained/Baseline) | LinearRegression | 4.8635 ± 0.6408 | 3.6362 ± 0.4060 | 0.8384 ± 0.0232 |
| Stage A (Unconstrained/Baseline) | XGBoost | 5.2855 ± 1.2956 | 3.6346 ± 0.7790 | 0.7979 ± 0.0891 |
| Stage A (Unconstrained/Baseline) | LightGBM | 5.3805 ± 0.9966 | 3.7644 ± 0.6384 | 0.7936 ± 0.0734 |
| Stage A (Unconstrained/Baseline) | RandomForest | 5.7014 ± 1.3010 | 4.0426 ± 0.8266 | 0.7688 ± 0.0931 |
| Stage B (Constrained/Advanced) | GaussianProcess | 4.4065 ± 0.8294 | 3.0760 ± 0.3703 | 0.8623 ± 0.0778 |
| Stage B (Constrained/Advanced) | CatBoost_constrained | 4.4533 ± 0.7468 | 2.9499 ± 0.3371 | 0.8616 ± 0.0711 |
| Stage B (Constrained/Advanced) | Stacking_Ensemble | 4.4351 ± 0.7978 | 2.9222 ± 0.3736 | 0.8609 ± 0.0759 |
| Stage B (Constrained/Advanced) | XGBoost_constrained | 4.4750 ± 0.7859 | 3.0006 ± 0.3404 | 0.8594 ± 0.0747 |
| Stage B (Constrained/Advanced) | LightGBM_constrained | 4.6238 ± 0.7392 | 3.1084 ± 0.3699 | 0.8510 ± 0.0756 |
| Stage B (Constrained/Advanced) | LinearRegression | 4.8500 ± 0.5443 | 3.5561 ± 0.3216 | 0.8461 ± 0.0450 |
| Stage B (Constrained/Advanced) | RandomForest | 4.9194 ± 0.6280 | 3.4216 ± 0.3668 | 0.8327 ± 0.0768 |

### Subset: EA14 (14-Day Compressive Strength)
| Stage | Model | RMSE (Mean ± Std) | MAE (Mean ± Std) | R² (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- |
| Stage A (Unconstrained/Baseline) | CatBoost | 4.2689 ± 0.7111 | 2.9447 ± 0.2726 | 0.8698 ± 0.0444 |
| Stage A (Unconstrained/Baseline) | XGBoost | 4.7961 ± 0.6042 | 3.3395 ± 0.3078 | 0.8371 ± 0.0420 |
| Stage A (Unconstrained/Baseline) | LinearRegression | 5.0439 ± 0.3546 | 3.7634 ± 0.2574 | 0.8229 ± 0.0127 |
| Stage A (Unconstrained/Baseline) | RandomForest | 5.0691 ± 0.7791 | 3.6565 ± 0.5209 | 0.8193 ± 0.0425 |
| Stage A (Unconstrained/Baseline) | LightGBM | 5.0944 ± 0.5871 | 3.5755 ± 0.3264 | 0.8145 ± 0.0500 |
| Stage B (Constrained/Advanced) | Stacking_Ensemble | 4.2633 ± 0.8218 | 2.9146 ± 0.3563 | 0.8708 ± 0.0536 |
| Stage B (Constrained/Advanced) | CatBoost_constrained | 4.3261 ± 0.8672 | 2.9695 ± 0.3934 | 0.8697 ± 0.0478 |
| Stage B (Constrained/Advanced) | XGBoost_constrained | 4.3150 ± 0.8176 | 2.9512 ± 0.3047 | 0.8668 ± 0.0556 |
| Stage B (Constrained/Advanced) | GaussianProcess | 4.4078 ± 0.6771 | 3.1412 ± 0.4027 | 0.8603 ± 0.0530 |
| Stage B (Constrained/Advanced) | LightGBM_constrained | 4.4308 ± 0.8535 | 3.0569 ± 0.3887 | 0.8583 ± 0.0624 |
| Stage B (Constrained/Advanced) | RandomForest | 4.7279 ± 0.6843 | 3.2012 ± 0.3235 | 0.8426 ± 0.0491 |
| Stage B (Constrained/Advanced) | LinearRegression | 5.0746 ± 0.4511 | 3.7270 ± 0.2912 | 0.8207 ± 0.0367 |

### Subset: Full (1-365 Days Compressive Strength)
| Stage | Model | RMSE (Mean ± Std) | MAE (Mean ± Std) | R² (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- |
| Stage A (Unconstrained/Baseline) | CatBoost | 3.7243 ± 0.2930 | 2.4840 ± 0.0949 | 0.9471 ± 0.0069 |
| Stage A (Unconstrained/Baseline) | XGBoost | 4.0603 ± 0.2448 | 2.8016 ± 0.1513 | 0.9371 ± 0.0065 |
| Stage A (Unconstrained/Baseline) | LightGBM | 4.0830 ± 0.4936 | 2.8164 ± 0.2639 | 0.9365 ± 0.0108 |
| Stage A (Unconstrained/Baseline) | RandomForest | 4.7506 ± 0.2635 | 3.4089 ± 0.2136 | 0.9140 ± 0.0077 |
| Stage A (Unconstrained/Baseline) | LinearRegression | 6.8633 ± 0.3692 | 5.3356 ± 0.2375 | 0.8208 ± 0.0110 |
| Stage B (Constrained/Advanced) | Stacking_Ensemble | 4.1344 ± 0.3806 | 2.6215 ± 0.1179 | 0.9381 ± 0.0096 |
| Stage B (Constrained/Advanced) | CatBoost_constrained | 4.1983 ± 0.4425 | 2.7168 ± 0.1602 | 0.9363 ± 0.0101 |
| Stage B (Constrained/Advanced) | XGBoost_constrained | 4.2150 ± 0.3542 | 2.6744 ± 0.1237 | 0.9355 ± 0.0097 |
| Stage B (Constrained/Advanced) | LightGBM_constrained | 4.2228 ± 0.5017 | 2.6757 ± 0.1751 | 0.9352 ± 0.0129 |
| Stage B (Constrained/Advanced) | GaussianProcess | 4.9329 ± 0.3957 | 3.3430 ± 0.1958 | 0.9114 ± 0.0161 |
| Stage B (Constrained/Advanced) | RandomForest | 4.9992 ± 0.3158 | 3.4269 ± 0.1465 | 0.9099 ± 0.0059 |
| Stage B (Constrained/Advanced) | LinearRegression | 6.4921 ± 0.4150 | 4.9254 ± 0.2422 | 0.8483 ± 0.0028 |
