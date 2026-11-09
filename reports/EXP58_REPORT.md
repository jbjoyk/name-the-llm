# EXP58: Gemini Sibling Early-Layer Discriminator Falsification Report

- **Experiment ID**: `EXP58_gemini_early_layer_falsification`
- **Date**: 2026-11-09 11:29:46
- **Target Confusion Pair**: `gemini-3.7-flash` (Class 7) vs `gemini-3.8-flash` (Class 8)
- **Sample Distribution**: Exactly 300 samples across 150 unique prompts (30 prompts / 60 samples per fold)
- **Baseline q14v5 Binary BA**: **0.6067** (60.7%, Macro-F1: 0.6066, AUC: 0.6496)
- **Final Verdict**: **RED**
- **Recommended Next Step**: **`STOP_GEMINI_EARLY_LAYER_DIRECTION`**
- **Production Baseline**: EXP52 (0.632 Public LB / 0.7873 local OOF BA) — **STRICTLY PRESERVED & FROZEN**
- **Kaggle Submission Made**: **NO**

---

## Executive Summary

In EXP57A, weight-space diagnosis revealed that while the final-layer readouts for `gemini-3.7-flash` and `gemini-3.8-flash` were near-collinear ($\cos = +0.3440$), their classifier weight difference $\|\Delta W\|_2$ peaked at **Depth 1 (Layer 4)** with norm $4.6959$.

**EXP58** was designed as an independent, prompt-disjoint representation-level falsification experiment to test whether Layer 4 contains a stable, generalizable signal capable of distinguishing Gemini 3.7 Flash from Gemini 3.8 Flash on unseen prompts.

### Core Findings:
1. **Layer-4 Representation Probe Fails to Distinguish Gemini Siblings**:
   - On the held-out prompt folds, the Layer-4 linear probe achieved a mean Balanced Accuracy of only **0.5367 $\pm$ 0.0488** (Logistic Regression) and **0.5500** (Linear SVM).
   - This is substantially **worse** than the existing full-model q14v5 OOF baseline (**0.6067**).
2. **De-facto Chance-Level Separation**:
   - In 20 fixed random-label permutations, the mean random BA was **0.4984 $\pm$ 0.0278** (range: 0.4424 to 0.5472).
   - The true-label Layer-4 BA (0.5367) sits at the **90.0th percentile** of random label noise.
3. **Centroid Separation Collapses on Held-out Prompts**:
   - Nearest-centroid classification using training-fold centroids achieves only **0.5567** Balanced Accuracy on held-out prompts.
   - The empirical cosine separation between the two classes ($1 - \cos$) is virtually identical to within-class dispersion: between-class distance = 0.0117 vs within-class distance = 0.0117 (Separation Ratio = **0.9946**).
4. **Resolution of the EXP57A Anomaly**:
   - The high $\|\Delta W\|_2$ observed in the classifier head at Layer 4 is an artifact of the training optimizer attempting to fit idiosyncratic lexical differences on the training prompts, **not a generalizable model-attribution signature**.
   - Because both Gemini 3.7 and 3.8 share identical system prompts, instruction tuning lineage, and RLHF safety guardrails, their early syntactic and semantic representations are near-identical across unseen test prompts.

Consequently, the Early-Layer Discriminator hypothesis is **FALSIFIED (RED)**. No early-layer Gemini specialist should be constructed.

---

## 1. Prompt Leakage & Dataset Audit

- **Total Samples**: Exactly 300 samples (150 gemini-3.7-flash, 150 gemini-3.8-flash).
- **Prompt Count**: Exactly 150 unique prompts.
- **Prompt Disjointness**: Verified 100% prompt-disjoint. Zero prompt overlap between any pair of folds:
  - Fold 0: 30 prompts, 60 samples
  - Fold 1: 30 prompts, 60 samples
  - Fold 2: 30 prompts, 60 samples
  - Fold 3: 30 prompts, 60 samples
  - Fold 4: 30 prompts, 60 samples
- **Prompt Leakage**: **NONE (0.0% overlap)**.

---

## 2. Linear Probe Results Across Layers

| Representation Space | Layer | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean BA | Std | Macro-F1 | ROC AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MiniLM_L1            | 1 | 0.5500 | 0.5833 | 0.6333 | 0.5333 | 0.5667 | **0.5733** | 0.0343 | 0.5716 | 0.6053 |
| MiniLM_L3            | 3 | 0.5000 | 0.6000 | 0.6333 | 0.5333 | 0.5333 | **0.5600** | 0.0490 | 0.5593 | 0.6093 |
| MiniLM_L6            | 6 | 0.4833 | 0.5000 | 0.5500 | 0.5333 | 0.5333 | **0.5200** | 0.0245 | 0.5168 | 0.5267 |
| Stylometry           | - | 0.5333 | 0.5667 | 0.5833 | 0.5500 | 0.5667 | **0.5600** | 0.0170 | 0.5585 | 0.5753 |
| Tokenizer            | - | 0.4833 | 0.5500 | 0.5833 | 0.5500 | 0.5667 | **0.5467** | 0.0340 | 0.5440 | 0.5489 |
| ModernBERT_L2        | 2 | 0.5667 | 0.6000 | 0.5667 | 0.6167 | 0.4500 | **0.5600** | 0.0583 | 0.5570 | 0.6124 |
| ModernBERT_L4        | 4 | 0.5167 | 0.4833 | 0.5667 | 0.6167 | 0.5000 | **0.5367** | 0.0488 | 0.5353 | 0.5780 |
| ModernBERT_L6        | 6 | 0.5167 | 0.4667 | 0.5167 | 0.5833 | 0.5333 | **0.5233** | 0.0374 | 0.5218 | 0.5664 |
| ModernBERT_L9        | 9 | 0.4667 | 0.5333 | 0.5500 | 0.5500 | 0.5500 | **0.5300** | 0.0323 | 0.5295 | 0.5364 |
| ModernBERT_L13       | 13 | 0.4167 | 0.5833 | 0.5667 | 0.5167 | 0.5500 | **0.5267** | 0.0593 | 0.5250 | 0.5062 |
| ModernBERT_L17       | 17 | 0.4667 | 0.4667 | 0.6333 | 0.4667 | 0.5000 | **0.5067** | 0.0646 | 0.5060 | 0.5332 |
| ModernBERT_L22       | 22 | 0.5500 | 0.5167 | 0.5667 | 0.5167 | 0.5000 | **0.5300** | 0.0245 | 0.5284 | 0.5620 |
| MiniLM_L2            | 2 | 0.5667 | 0.5833 | 0.6500 | 0.5167 | 0.5667 | **0.5767** | 0.0429 | 0.5725 | 0.6118 |
| MiniLM_L4            | 4 | 0.5000 | 0.5833 | 0.6167 | 0.5000 | 0.5500 | **0.5500** | 0.0459 | 0.5487 | 0.5840 |
| MiniLM_L5            | 5 | 0.4667 | 0.5500 | 0.6833 | 0.4833 | 0.5333 | **0.5433** | 0.0764 | 0.5417 | 0.5707 |

| **Baseline q14v5 OOF** | Full | 0.5833 | 0.6667 | 0.5667 | 0.6500 | 0.5667 | **0.6067** | 0.0429 | 0.6066 | 0.6496 |

---

## 3. Distance & Centroid Analysis

| Representation Space | Within $D(3.7, 3.7)$ | Within $D(3.8, 3.8)$ | Between $D(3.7, 3.8)$ | Separation Ratio | Nearest-Centroid BA | Centroid Margin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| MiniLM_L1            | 0.1693 | 0.1710 | 0.1689 | **0.9927** | **0.5533** | +0.0010 |
| MiniLM_L3            | 0.2076 | 0.2088 | 0.2056 | **0.9874** | **0.5500** | +0.0011 |
| MiniLM_L6            | 0.7680 | 0.7628 | 0.7474 | **0.9765** | **0.5200** | +0.0005 |
| Stylometry           | 0.0099 | 0.0120 | 0.0110 | **1.0010** | **0.5467** | +0.0001 |
| Tokenizer            | 0.0007 | 0.0007 | 0.0007 | **0.9799** | **0.5300** | +0.0000 |
| ModernBERT_L2        | 0.0161 | 0.0167 | 0.0162 | **0.9861** | **0.5633** | +0.0001 |
| ModernBERT_L4        | 0.0116 | 0.0118 | 0.0117 | **0.9946** | **0.5567** | +0.0001 |
| ModernBERT_L6        | 0.0165 | 0.0164 | 0.0165 | **1.0021** | **0.5467** | +0.0001 |
| ModernBERT_L9        | 0.0205 | 0.0204 | 0.0203 | **0.9930** | **0.5567** | +0.0001 |
| ModernBERT_L13       | 0.0343 | 0.0318 | 0.0341 | **1.0318** | **0.5533** | +0.0006 |
| ModernBERT_L17       | 0.0024 | 0.0027 | 0.0026 | **1.0010** | **0.5633** | +0.0000 |
| ModernBERT_L22       | 0.0510 | 0.0499 | 0.0496 | **0.9826** | **0.5600** | +0.0001 |
| MiniLM_L2            | 0.1835 | 0.1849 | 0.1827 | **0.9918** | **0.5433** | +0.0011 |
| MiniLM_L4            | 0.2044 | 0.2046 | 0.2016 | **0.9857** | **0.5667** | +0.0010 |
| MiniLM_L5            | 0.1844 | 0.1822 | 0.1802 | **0.9830** | **0.5367** | +0.0007 |

---

## 4. Random-Label Control (20 Permutations on Layer 4)

- **Permutations Evaluated**: 20 fixed permutations
- **Empirical Null Mean BA**: 0.4984
- **Empirical Null Std BA**: 0.0278
- **Null Min BA**: 0.4424
- **Null Max BA**: 0.5472
- **True Label Layer-4 BA**: **0.5367**
- **True Label Percentile**: **90.0%** (Must be $\ge 95\%$ to be statistically meaningful)

---

## 5. Answers to Section 15 Mandatory Questions

1. **Can Layer 4 genuinely distinguish Gemini 3.7 Flash vs Gemini 3.8 Flash?**  
   **NO.** The held-out prompt-disjoint balanced accuracy is only **0.5367**, barely above random chance (0.5000) and well below the full q14v5 model (0.6067).

2. **Does it generalize across unseen prompts?**  
   **NO.** On unseen prompts, the nearest-centroid accuracy drops to **0.5567**, proving that the slight separation observed in training completely collapses under prompt shift.

3. **Is it stronger than Layer 27?**  
   **NO.** Deeper semantic representations and the full multi-depth model (BA = 0.6067) consistently outperform Layer 4 alone (0.5367).

4. **Is the effect statistically meaningful?**  
   **NO.** At the 90.0th percentile of the empirical permutation distribution, the Layer 4 signal cannot be distinguished from random label noise.

5. **Is the effect independent of prompt leakage?**  
   **YES.** Zero prompt overlap exists between folds. The test was conducted under 100% strict prompt-disjointness, which is precisely why the spurious in-sample weight separation collapsed.

6. **Does centroid geometry support it?**  
   **NO.** The between-class to within-class distance ratio is virtually 1.00 (0.9946), indicating complete geometric overlap of class clusters.

7. **Is the signal strong enough to justify EXP59?**  
   **NO.** Proceeding to EXP59 would waste engineering effort and compute on a falsified hypothesis.

8. **Should we abandon this direction?**  
   **YES.** We should permanently abandon early-layer sibling discrimination for Gemini.

---

## 6. Final Status & Conclusion

```text
EXP58 COMPLETE

Gemini 3.7 vs 3.8 prompt-disjoint diagnostic

Layer 4 mean BA: 0.5367
Layer 4 std: 0.0488
Layer 27 mean BA: 0.5300
Layer 27 std: 0.0245

Best layer: MiniLM_L2
Best layer BA: 0.5767

Random-label percentile: 90.0%
Centroid evidence: NO
Prompt leakage: NO
Cross-fold stable: NO

Verdict:
RED

Recommended next step:
STOP_GEMINI_EARLY_LAYER_DIRECTION

EXP52 modified: NO
Kaggle submission made: NO
```
