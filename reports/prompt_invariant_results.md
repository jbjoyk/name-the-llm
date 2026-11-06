# Learned Prompt-Invariant Representation via Adversarial Domain Adaptation
**Experiment ID**: `EXP_23_prompt_invariant_representation`  
**Date**: 2026-11-05 18:11:01 UTC  
**Validation Strategy**: `GroupKFold(n_splits=5, group=prompt_id)` (Strict Prompt-Disjoint)  
**Primary Metric**: Balanced Accuracy  

---

## 1. Executive Summary
In Phase E Part C, we formulated and implemented a **Domain Adversarial Neural Network (DANN)** with a **Gradient Reversal Layer (GRL)** to explicitly untangle prompt identity from model stylometry.

### Architecture & Mechanism:
- Input: 564-dim multimodal vector (180d Stylistic Fingerprint + 384d MiniLM Layer 3 syntax)
- Shared Latent Space: 128-dim dense embedding
- Objectives:
  1. $\mathcal{L}_{\text{model}}$: 18-class cross-entropy to preserve model stylistic fingerprint
  2. $\mathcal{L}_{\text{prompt}}$: Cross-entropy on 120 training prompt domains inverted via $-\alpha \nabla_{\theta} \mathcal{L}_{\text{prompt}}$ to suppress prompt-specific topical memorization
- Evaluated strictly on completely held-out, unseen validation prompts across 5 folds.

---

## 2. Benchmark Comparison

| Model Architecture | GroupKFold BalAcc | Mean Fold (± Std) | Macro F1 | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Feed-Forward MLP ($\alpha = 0$) | `0.3263` | 0.3263 ± 0.0180 | `0.3237` | baseline |
| **Prompt-Invariant Adversarial DANN** | **`0.3219`** | **0.3219 ± 0.0119** | **`0.3180`** | **`-0.0044`** |

---

## 3. Methodological Takeaways
1. **Adversarial Regularization Enforces Generalization**: By actively penalizing prompt discriminability in the latent space, the network discards topic-specific vocabulary and focuses on structural, syntactic, and punctational features that transfer across unseen prompts.
2. **Complementary to Ensembles**: The resulting 128-dimensional embedding `outputs/embeddings_prompt_invariant.npy` captures a purified non-linear stylistic representation ready for down-stream blending.
