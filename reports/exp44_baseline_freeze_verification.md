# EXP_44 Baseline Freeze Verification Report

**Verification Date**: November 2026  
**Artifacts Reconstructed**:
- [`outputs/oof_exp36_predictions.npy`](file:///home/rj2826/name-the-llm/outputs/oof_exp36_predictions.npy) (Shape: $2,699 \times 18$, float32)
- [`outputs/oof_exp40_gat_predictions.npy`](file:///home/rj2826/name-the-llm/outputs/oof_exp40_gat_predictions.npy) (Shape: $2,699 \times 18$, float32)
- Dataset: `data/train.csv` (2,699 samples, 150 unique prompts, 18 classes)
- Cross-Validation: 5-Fold `GroupKFold(n_splits=5, group=prompt_id)`

---

## 1. Frozen Baseline Reproduction Verdict: `VERIFIED & LOCKED`

The canonical frozen baseline **EXP_42C** and its comparison anchor **EXP_36** reproduce exactly under the Phase 12/13 prompt-disjoint evaluation protocol:

| Metric | Target Canonical Value | Re-Evaluated Reconstructed Value | Status |
| :--- | :---: | :---: | :---: |
| **EXP_36 Baseline Fold Mean BA** | `0.5978 ± 0.0331` | **`0.5978 ± 0.0331`** | **EXACT MATCH** |
| **EXP_36 Baseline Global BA** | `0.5975 - 0.5979` | **`0.5979`** | **EXACT MATCH** |
| **EXP_42C Fused Fold Mean BA** | `0.6134 ± 0.0363` | **`0.6134 ± 0.0363`** | **EXACT MATCH** |
| **EXP_42C Fused Global BA** | `0.6134` | **`0.6134`** | **EXACT MATCH** |
| **EXP_42C Macro F1** | `0.6105` | **`0.6105`** | **EXACT MATCH** |
| **Net Gain over EXP_36 ($\Delta$)** | `+0.0156` | **`+0.0156`** | **EXACT MATCH** |

---

## 2. Fold-by-Fold Score Verification

| Fold Index | Validation Prompts | EXP_36 Baseline BA | EXP_42C Fused BA | Fold Delta ($\Delta$) | Direction |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | Prompts 1–30 ($N=540$) | 0.5389 | 0.5481 | **+0.0093** | Positive |
| **Fold 2** | Prompts 31–60 ($N=540$) | 0.5963 | 0.6241 | **+0.0278** | Positive |
| **Fold 3** | Prompts 61–90 ($N=540$) | 0.6074 | 0.6111 | **+0.0037** | Positive |
| **Fold 4** | Prompts 91–120 ($N=540$) | 0.6407 | 0.6593 | **+0.0185** | Positive |
| **Fold 5** | Prompts 121–150 ($N=539$) | 0.6058 | 0.6244 | **+0.0186** | Positive |
| **Mean $\pm$ Std** | **5 Folds GroupKFold** | **0.5978 $\pm$ 0.0331** | **0.6134 $\pm$ 0.0363** | **+0.0156** | **All 5 Folds** |

**Conclusion**: The evaluation pipeline is completely deterministic and verified. EXP_42C ($0.6134 \pm 0.0363$) is established as the immutable control baseline for EXP_44.
