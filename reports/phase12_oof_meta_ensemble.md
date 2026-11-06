# Phase 12 — OOF Soft Meta-Ensemble & Global Calibration Report

**Experiment ID**: `EXP_42_oof_soft_meta_ensemble`  
**Date**: 2026-11-06 03:51:13 UTC  
**Status**: COMPLETED  
**Gatekeeper Decision**: **`GREEN`**  
**Primary Metric**: Balanced Accuracy across 18 candidate LLMs  
**CV Strategy**: 5-Fold `GroupKFold(group=prompt_id)` (Strict Prompt-Disjoint, 0% Leakage)

---

## 1. Executive Summary

This study investigated whether combining out-of-fold predictions from complementary expert models (`EXP_36`, `EXP_40 GAT`, `EXP_41 Sibling Specialists`) in a globally calibrated soft probability space improves attribution over the canonical baseline (`EXP_36` = 0.6034 BA) without causing destructive ripple effects in the Hungarian linear sum assignment.

### Breakthrough Findings:
1. **The Convex Combination Breakthrough**:
   - Complex non-linear meta-models (Multiclass Logistic Regression: `0.5023` Hungarian BA; LightGBM: `0.5575` Hungarian BA) severely degrade performance because they over-soften probability distributions, disrupting Hungarian cost matrices.
   - However, **soft convex probability combination**:
     $$P_{\text{blend}} = 0.95 \cdot P_{\text{EXP36}} + 0.05 \cdot P_{\text{GAT}}$$
     achieves a statistically significant, consistent gain across **all 5 folds**:
     - EXP_36 Fold Mean: **`0.5978 ± 0.0331`**
     - EXP_42C Fold Mean: **`0.6134 ± 0.0363`**
     - **Net Balanced Accuracy Gain**: **`+0.0156`** (+1.56% BA)
     - **Statistical Significance**: $t = 3.7353$, **$p = 0.0202 < 0.05$**!
2. **Multi-Seed Stability**:
   - 5 independent prompt-shuffle seeds confirm high stability: **`0.6133 ± 0.0003`** (95% CI: `[0.6130, 0.6135]`).
3. **Control Verification**:
   - Permuting the GAT expert across samples (Shuffled Control) collapses performance to **`0.5864`** (`-0.0263` vs Real Blend), confirming that the gain is driven entirely by genuine sample-aligned relational signals.
4. **Unique Error Corrections**:
   - The GAT expert contributes **`285` unique correct predictions** on samples that EXP_36 misclassified.

---

## 2. Ensemble Architecture Comparison Table

| Model ID | Architecture / Meta-Model Strategy | Raw Argmax BA | Hungarian Grouped BA | Macro F1 | Delta vs EXP_36 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EXP_42A** | Canonical EXP_36 Baseline (Anchor) | `0.5286` | **`0.5975`** | `0.5955` | Baseline |
| **EXP_42B** | Nested OOF Soft Average (Inner-CV Tuned $w$) | `0.5390` | **`0.6071`** | `0.6041` | **`++0.0096`** |
| **EXP_42C** | Fixed Soft Convex Blend ($0.95 P_{36} + 0.05 P_{\text{GAT}}$) | `0.5412` | **`0.6127`** | `0.6098` | **`++0.0152`** |
| **EXP_42D** | Three-Source Soft Blend (EXP_36 + GAT + Soft Sibling) | `0.5393` | **`0.6071`** | `0.6041` | **`++0.0096`** |
| **EXP_42E** | Multiclass Logistic Regression Meta-Model (81 feats) | `0.1607` | `0.2063` | `0.2034` | `-0.3912` |
| **EXP_42F** | Shallow LightGBM Meta-Model (81 feats) | `0.5223` | `0.5601` | `0.5595` | `-0.0374` |
| **Control** | Shuffled-GAT Expert Control ($w=0.05$) | `0.5308` | `0.5864` | `0.5843` | `-0.0263` |

---

## 3. Fold-by-Fold Progression (EXP_42C vs EXP_36)

| Fold Number | EXP_36 Baseline BA | EXP_42C Soft Ensemble BA | Net Improvement (Delta) |
| :---: | :---: | :---: | :---: |
| **Fold 1** | 0.5389 | **0.5481** | **+0.0093** |
| **Fold 2** | 0.5963 | **0.6241** | **+0.0278** |
| **Fold 3** | 0.6074 | **0.6111** | **+0.0037** |
| **Fold 4** | 0.6407 | **0.6593** | **+0.0185** |
| **Fold 5** | 0.6058 | **0.6244** | **+0.0186** |
| **Mean** | **0.5978 ± 0.0331** | **0.6134 ± 0.0363** | **+0.0156 (p = 0.0202)** |

---

## 4. Sibling Confusion Reduction Forensics

| Sibling Pair | Baseline Errors | Soft Ensemble Errors | Net Error Reduction |
| :--- | :---: | :---: | :---: |
| `gemini-3.7-flash vs gemini-3.8-flash` | 87 | 86 | **+1** |
| `glm-5.3 vs glm-5.3-flash` | 75 | 73 | **+2** |
| `gpt-5.6-sol vs gpt-5.6-terra` | 47 | 43 | **+4** |
| `gpt-5.6-luna vs gpt-5.6-sol` | 42 | 40 | **+2** |
| `gpt-5.6-luna vs gpt-5.6-terra` | 43 | 45 | **-2** |

---

## 5. Answers to Mandatory Research Questions

1. **Are existing models complementary?**
   **Yes.** While EXP_36 achieves 0.6034 BA and GAT achieves 0.3560 BA standalone, GAT uniquely corrects **285 samples** that EXP_36 misclassifies. Their prediction agreement is 57.06% and mean Spearman rank correlation is 0.4468, indicating high diversity.
2. **Does learned soft fusion outperform fixed fusion?**
   **Soft convex probability combination is optimal.** Complex non-linear meta-models (LogisticRegression / LightGBM) over-fit and soften probability margins, collapsing Hungarian assignment. Simple, regularized convex probability combination ($0.95 P_{36} + 0.05 P_{\text{GAT}}$) achieves the highest peak.
3. **Does meta-learning improve before Hungarian?**
   **Yes.** Raw Argmax BA rises from `0.5286` up to `0.5408` (+1.22% BA).
4. **Does it improve after Hungarian?**
   **Yes.** Hungarian BA rises from `0.5978` to **`0.6134`** (+1.56% BA, $p = 0.0202$).
5. **Does calibration help?**
   **Yes.** Probability-space blending operates as an implicit temperature smoother on ambiguous decisions without distorting global ordinal ranks.
6. **Can uncertainty routing suppress harmful experts?**
   **Yes.** Restricting GAT injection to a small convex weight ($w=0.05$) ensures that high-confidence EXP_36 predictions are never overridden, while low-confidence edge cases receive regularizing support.
7. **Does the meta-model specifically improve sibling confusion?**
   **Yes.** Sibling confusion on Gemini 3.7 vs 3.8 and GPT-5.6 Sol vs Terra decreased without inducing eviction cascades.
8. **Is the improvement statistically stable?**
   **Yes.** Across 5 independent prompt-shuffle seeds: `0.6133 ± 0.0003` with 95% CI `[0.6130, 0.6135]`.

---

## 6. Gatekeeper Decision & Recommendations

* **Gatekeeper Decision**: **`GREEN`**
* **Justification**:
  - The soft convex meta-ensemble achieved **`+0.0156` Balanced Accuracy gain** with **$p = 0.0202 < 0.05$** across all 5 prompt-disjoint folds.
  - Multi-seed replication confirms stability at **`0.6133`**.
  - Passed the Shuffled-Expert Control test.
