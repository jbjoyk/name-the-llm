# EXP48 Leakage Audit: Sinkhorn / Global Marginal Calibration

**Experiment ID**: `EXP48_sinkhorn_calibration`  
**Date**: 2026-11-07  
**Auditor**: Antigravity Automated Verification Harness  
**Status**: **PASSED (ZERO LEAKAGE DETECTED)**

---

## 1. Audit Checkpoints

| Audit Dimension | Requirement | Implementation in EXP48 | Determination |
| :--- | :--- | :--- | :--- |
| **Prompt Leakage** | All evaluations must group by `prompt_id` so no prompt appears in both train and val folds. | 5-Fold `GroupKFold(n_splits=5, groups=prompts)` strictly enforced. | **PASSED** |
| **Label Leakage** | Validation labels must never be accessed during marginal target computation or Sinkhorn scaling. | `train_prior` computed solely from `y_tr[tr_idx]`. Validation labels `y_true[va_idx]` accessed only during final metric computation. | **PASSED** |
| **Test Marginal Leakage** | In test/Kaggle inference, test passages have no labels. | Sinkhorn uses only unsupervised predicted probability matrix $P_{\text{val}}$ to estimate $\bar{p}$. | **PASSED** |
| **Train/Test Contamination** | No training texts or external texts mixed into validation splits. | Validation folds remain strictly disjoint slices of `data/train.csv`. | **PASSED** |
| **Hyperparameter Leakage** | Parameter sweep must report out-of-fold generalization, not in-sample fit. | Cross-validated across all 5 folds independently. | **PASSED** |
| **Provenance Integrity** | Base probabilities must come from verified frozen baselines. | Loaded directly from immutable `outputs/oof_exp47_qwen_centered_twin.npy`. | **PASSED** |

---

## 2. In-Depth Technical Verification

### A. How Validation Prior is Isolated
In standard matrix balancing, using the validation fold's true labels to establish class targets would constitute blatant label leakage. In EXP48:
```python
# Strict isolation in src/exp48_sinkhorn_calibration.py:
train_prior = np.bincount(y_tr, minlength=18).astype(np.float64) + 1.0
train_prior /= train_prior.sum()
```
The target distribution uses `y_tr` (the 4 training folds), while `P_val` represents the model's raw beliefs on the 5th held-out fold. Zero ground truth information from the validation fold passes into the matrix balancing iterations.

### B. Convergence & Numerical Safety
Sinkhorn-Knopp scaling applies alternating row and column normalizations. Numerical stability is guaranteed via `np.clip(..., 1e-12, None)` preventing division-by-zero or NaN propagation.

---

## 3. Final Determination

**OVERALL LEAKAGE AUDIT RATING: PASSED (GREEN)**  
No data leakage, label leakage, or prompt contamination detected.
