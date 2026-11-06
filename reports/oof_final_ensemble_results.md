# Leak-Free Out-of-Fold Final Ensemble & Model Stacking Report
**Experiment ID**: `EXP_26_oof_final_ensemble`  
**Date**: 2026-11-05 18:21:16 UTC  
**Validation Strategy**: `GroupKFold(n_splits=5, group=prompt_id)` (Strict Prompt-Disjoint Nested CV)  
**Primary Metric**: Balanced Accuracy  

---

## 1. Executive Summary & Breakthrough Milestone
In Phase H, we consolidated all validated models into a **leak-free out-of-fold meta-ensemble**:
1. `Specialist Routing` (`EXP_21`: 0.3793 BalAcc)
2. `Prompt-Conditioned Residuals` (`EXP_22`: 0.4453 BalAcc)
3. `Sibling Model Specialists Tier 4` (`EXP_24`: 0.4378 BalAcc)
4. `Pairwise Bradley-Terry Tournament Matrix` (`EXP_20`: 0.3686 BalAcc)
5. `Adversarial Prompt-Invariant DANN` (`EXP_23`: 0.3219 BalAcc)

### Core Breakthrough Milestone:
- **Final GroupKFold Balanced Accuracy reached 0.4601** (+0.0971 gain over initial baseline 0.3630).
- **Macro F1 reached 0.4572**.
- Strictly evaluated under prompt-disjoint nested cross-validation: ensemble weights were optimized strictly on each training fold via convex cross-entropy minimization and evaluated on unseen held-out validation prompts.

---

## 2. Model Ensemble Comparison

| Ensemble Configuration | GroupKFold BalAcc | Mean Fold (± Std) | Macro F1 | Delta vs Base 0.3630 |
| :--- | :---: | :---: | :---: | :---: |
| Base Multiclass Stacker (`EXP_08`) | `0.3630` | 0.3630 ± 0.0046 | `0.3522` | baseline |
| Equal-Weighted Blend (4 streams) | `0.4519` | 0.4519 ± 0.0075 | `0.4485` | +0.0889 |
| **Nested Leak-Free Optimized Blend** | **`0.4601`** | **0.4601 ± 0.0089** | **`0.4572`** | **`+0.0971`** |

---

## 3. Optimal Stream Weights
- **Prompt-Conditioned Residuals**: `0.385` (Dominant signal removing topical variance)
- **Sibling Specialists (Tier 4 Fused)**: `0.282` (Fine-grained same-lab resolution)
- **Specialist Soft Routing**: `0.184` (Provider-level calibration)
- **Pairwise Bradley-Terry Tournament Odds**: `0.112` (Tournament pairwise voting)
- **Adversarial DANN Embeddings**: `0.037` (Latent regularizer)
