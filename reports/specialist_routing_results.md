# Soft Specialist Routing & Calibrated Meta-Classifier Report
**Experiment ID**: `EXP_21_specialist_routing`  
**Date**: 2026-11-05 17:58:30 UTC  
**Validation Strategy**: `GroupKFold(n_splits=5, group=prompt_id)` (Strict Prompt-Disjoint)  
**Primary Metric**: Balanced Accuracy  

---

## 1. Executive Summary & Progression
In Phase D, we replaced hard hierarchical routing with **Soft Specialist Routing**—a calibrated multi-level architecture that simultaneously inputs:
1. **Global Base Multi-Class Probabilities** (18d, from stacked stylometry + syntax + likelihood + fusion)
2. **Provider 6-way Routing Probabilities** (6d, 63.32% provider accuracy)
3. **Provider-Conditioned Joint Likelihoods** (18d, $p_m \cdot P(\text{provider}(m))$)
4. **Pairwise Attribution Matrix Bradley-Terry Odds** (18d, from Phase C)
5. **Character 4-Gram Likelihood Profiles** (18d)
6. **Pure Stylometric Fingerprints** (180d)

The combined feature representation is calibrated via strongly regularized multinomial logistic regression ($C=0.01$).

### Key Empirical Result:
- **Balanced Accuracy reached 0.3793** (+0.0163 gain over base 0.3630, +0.0107 gain over Pairwise Matrix 0.3686).
- **Macro F1 reached 0.3712**.
- Soft routing completely avoids the cascading error failure mode of hard hierarchy (-0.0074 drop), while capturing the decisive benefits of provider clustering and pairwise sibling resolution.

---

## 2. Benchmark Comparison

| Model Architecture | GroupKFold BalAcc | Mean Fold (± Std) | Macro F1 | Delta vs Base |
| :--- | :---: | :---: | :---: | :---: |
| Base Multiclass Stacker (`EXP_08`) | `0.3630` | 0.3630 ± 0.0046 | `0.3522` | baseline |
| Hard Hierarchical Routing (`EXP_14`) | `0.3556` | 0.3556 ± 0.0082 | `0.3440` | -0.0074 |
| Pairwise Attribution Matrix (`EXP_20`) | `0.3686` | 0.3686 ± 0.0062 | `0.3623` | +0.0056 |
| **Soft Specialist Routing (`EXP_21`)** | **`0.3793`** | **0.3793 ± 0.0084** | **`0.3712`** | **`+0.0163`** |

---

## 3. Fold-by-Fold Breakdown
- Fold 1: `0.3852`
- Fold 2: `0.3704`
- Fold 3: `0.3722`
- Fold 4: `0.3889`
- Fold 5: `0.3798`

Consistent positive gains across all 5 folds under strict prompt-disjoint evaluation.
