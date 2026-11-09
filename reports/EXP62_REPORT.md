# EXP62: Cross-Architecture Complementarity Test Report

- **Experiment ID**: `EXP62_cross_architecture`
- **Date**: 2026-11-09 13:52:38
- **Production Champion**: EXP52 (Dual-LoRA Qwen3-8B $0.60$ + Qwen3-14B $0.40$)
  - Kaggle Public LB: **0.6320**
  - Prompt-Disjoint OOF BA: **0.7873**
- **Experiment Type**: Cross-Architecture Representation & Complementarity Test
- **Status**: COMPLETE
- **Verdict**: **RED** (Cross-architecture representation fails to improve attribution; nested CV BA degrades to 0.7858, $\Delta = -0.0015$, 95% bootstrap CI $[-0.0039, +0.0006]$, $P(\Delta > 0) = 9.4\%$; on Q8/Q14 disagreement cases, ModernBERT net correction capacity is $-662$ samples; STOP cross-architecture model stacking)
- **Production Changed**: **NO**
- **EXP52 Remains Frozen**: **YES**
- **Kaggle Submission Made**: **NO**

---

## 1. Executive Summary

EXP62 was designed as a controlled scientific experiment to test the hypothesis:
> *"A genuinely different LLM architecture can provide complementary attribution errors that are less correlated with Qwen3-8B/Qwen3-14B, producing a more robust ensemble under unseen-prompt validation."*

Using the locally cached foundation encoder **ModernBERT-base** (`answerdotai/ModernBERT-base`, a 149M parameter bidirectional encoder-only transformer with RoPE, GeGLU, and 8,192-token context window), we extracted frozen final hidden-state representations across all 2,699 training samples. We evaluated lightweight classifiers (Linear Logistic Regression and Linear SVM) under strict 5-fold prompt-disjoint cross-validation and rigorous 5-outer $\times$ 4-inner selection-safe nested cross-validation.

### Key Numerical Findings:
1. **Weak Standalone Attribution Capacity**:
   - ModernBERT-base achieves **`0.3093` OOF BA** (`0.3081` Macro-F1) with Logistic Regression, and **`0.2371` OOF BA** with Linear SVM, substantially below standalone Qwen3-8B (`0.5910` BA) and Qwen3-14B (`0.6346` BA).
2. **Measured Error Correlation vs EXP52**:
   - ModernBERT errors have a low measured Pearson correlation of **`r = 0.1010`** ($p = 1.44 \times 10^{-7}$) and Spearman correlation of **`\rho = 0.1010`** with EXP52 errors.
   - However, low error correlation did **not** translate into ensemble complementarity.
3. **Monotonic Fusion Degradation**:
   - Standalone log-probability fusion degrades EXP52 performance monotonically as ModernBERT weight increases:
     - $w=0.00$: **0.7873** BA ($\Delta = 0.0000$)
     - $w=0.05$: **0.7858** BA ($\Delta = -0.0015$)
     - $w=0.10$: **0.7836** BA ($\Delta = -0.0037$)
     - $w=0.20$: **0.7773** BA ($\Delta = -0.0100$)
     - $w=0.50$: **0.6513** BA ($\Delta = -0.1360$)
4. **Selection-Safe Nested CV Degradation**:
   - Across 5 outer prompt-disjoint folds and 4 inner folds per outer training set, nested CV achieved **`0.7858 \pm 0.0118` BA** ($\Delta = \mathbf{-0.0015}$ BA).
   - In 4 out of 5 outer folds, inner validation selected $w^* = 0.00$. In Outer Fold 0, inner validation selected $w^* = 0.05$, which degraded held-out fold BA from $0.7778 \to 0.7704$.
   - 2,000 paired bootstrap iterations yielded a 95% CI of **`[-0.0039, +0.0006]`** with $P(\Delta > 0) = \mathbf{9.4\%}$.
5. **Falsification of Disagreement Correction**:
   - On the subset where Q8 and Q14 disagree (59.73% of data, $N=1,612$), EXP52 achieves **`0.6871` BA**, while ModernBERT achieves only **`0.2836` BA**.
   - ModernBERT is correct on only 117 of EXP52's 513 disagreement errors, while being incorrect on 779 instances where EXP52 is correct (net correction capacity: **`-662` samples**).
   - Conditional fusion targeting specifically the disagreement subset causes monotonic accuracy collapse ($0.6871 \to 0.5883$ at $w=0.40$).

**Verdict: RED**. The hypothesis is **NOT SUPPORTED**. Architecturally diverse representation alone, without sequence-level fine-tuning on the 18 frontier LLM outputs, does not provide complementary attribution information and degrades the EXP52 dual-backbone ensemble.

---

## 2. Environment & Repository Provenance

- **Repository Root**: `/home/rj2826/name-that-llm`
- **Git Commit Prior to EXP62**: `ffb2e1f` (`exp61: diagnose hidden surface and representation shift`)
- **Working Tree**: Clean
- **Python**: 3.14.7
- **PyTorch**: 2.15.0+cu132
- **CUDA**: 13.2
- **Accelerators**: 2 $\times$ NVIDIA GeForce RTX 3060 (12 GB VRAM each, Compute Capability 8.6)
- **Scikit-Learn**: 1.9.1
- **Transformers**: 5.22.1

---

## 3. Model Provenance (Phases 2 & 3)

| Property | Value |
|:---|:---|
| **Model Name** | `answerdotai/ModernBERT-base` |
| **Architecture** | `ModernBertModel` (Bidirectional Encoder-only transformer) |
| **Architectural Features** | Rotary Position Embeddings (RoPE), GeGLU activations, unpadded attention, alternating 128-token sliding-window local and full global attention |
| **Parameter Count** | 149,096,448 (149.1M) |
| **Hidden Dimension** | 768 |
| **Number of Layers** | 22 |
| **Attention Heads** | 12 |
| **Context Window** | 8,192 tokens |
| **Tokenizer** | Fast WordPiece / Unigram tokenizer (vocab size: 50,368) |
| **Local Checkpoint Path** | `/home/rj2826/.cache/huggingface/hub/models--answerdotai--ModernBERT-base/snapshots/8949b909ec900327062f0ebf497f51aef5e6f0c8` |
| **Snapshot Hash** | `8949b909ec900327062f0ebf497f51aef5e6f0c8` |
| **Pretraining Objective** | Masked Language Modeling (MLM) on 2T tokens (FineWeb, Dolma, Stack, etc.) |
| **Competition Involvment** | None (100% clean holdout backbone, never previously part of EXP52 or any dual-LoRA fusion) |

---

## 4. Methodology & Validation Protocol (Phases 4, 5, 6, 7)

1. **Frozen Representation Isolation**:
   - To strictly isolate architectural diversity from fine-tuning capacity, ModernBERT-base parameters were kept completely frozen.
   - For every passage, mean-pooled final hidden states were extracted across non-padding tokens, generating a normalized 768-dimensional feature vector.
2. **Strict Prompt-Disjoint 5-Fold Validation**:
   - Evaluated across exactly 150 unique prompts partitioned into 5 prompt-disjoint folds (30 prompts per fold).
   - In each fold: 120 training prompts, 30 validation prompts. Zero prompt leakage.
3. **Leakage-Safe Scaling & Classifier Fitting**:
   - `StandardScaler` was fitted strictly on the training fold representations and applied to validation fold representations.
   - Primary classifier: Linear Logistic Regression (`LogisticRegression(C=0.1, max_iter=1000, random_state=42)`).
   - Secondary classifier: Linear SVM (`LinearSVC(C=1.0, max_iter=2000, random_state=42)`).

---

## 5. Baseline Performance & Comparison (Phase 8 & Table 1)

Across all 2,699 samples and 5 prompt-disjoint folds:

| Model | OOF BA | Macro-F1 | Agreement vs EXP52 | Error Corr. vs EXP52 | Oracle BA |
|:---|---:|---:|---:|---:|---:|
| **Qwen3-8B (q8v5)** | 0.5910 | 0.5915 | 68.36% | 0.5381 | 0.8047 |
| **Qwen3-14B (q14v5)** | 0.6346 | 0.6338 | 63.99% | 0.3052 | 0.8621 |
| **EXP52 (Dual-LoRA Champion)** | **0.7873** | **0.7880** | **100.00%** | **1.0000** | **0.7873** |
| **ModernBERT-base (Linear LR)** | 0.3093 | 0.3081 | 30.42% | 0.1010 | 0.8340 |
| **ModernBERT-base (Linear SVM)** | 0.2371 | 0.2423 | 22.82% | 0.0577 | 0.8277 |
| **EXP52 + ModernBERT (Nested)** | **0.7858** | **0.7865** | 99.44% | 0.9890 | 0.7884 |

### Fold-by-Fold Stability:
- Fold 0: Q8 = 0.5889 | Q14 = 0.6056 | EXP52 = 0.7778 | ModernBERT = 0.3111 | Nested = 0.7704
- Fold 1: Q8 = 0.5796 | Q14 = 0.6185 | EXP52 = 0.7793 | ModernBERT = 0.3041 | Nested = 0.7793
- Fold 2: Q8 = 0.6074 | Q14 = 0.6685 | EXP52 = 0.7907 | ModernBERT = 0.2944 | Nested = 0.7907
- Fold 3: Q8 = 0.5852 | Q14 = 0.6278 | EXP52 = 0.7833 | ModernBERT = 0.3167 | Nested = 0.7833
- Fold 4: Q8 = 0.5944 | Q14 = 0.6519 | EXP52 = 0.8056 | ModernBERT = 0.3204 | Nested = 0.8056
- **Mean $\pm$ Std**: EXP52 = $0.7873 \pm 0.0102$ | ModernBERT = $0.3093 \pm 0.0093$ | Nested = $0.7858 \pm 0.0118$

---

## 6. Pairwise Error Complementarity & Oracle Upper Bound (Phases 9 & 10, Table 2)

| Pair | Agreement | Both Correct | Both Wrong | A-only Correct | B-only Correct | Error Corr. ($r$) | Oracle BA | Oracle Acc |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Q8 / Q14** | 40.27% | 1,026 | 417 | 569 | 687 | **0.0214** | **0.8455** | 0.8455 |
| **Q8 / NEW** | 21.56% | 500 | 769 | 1,095 | 335 | **0.0107** | 0.7151 | 0.7151 |
| **Q14 / NEW** | 32.79% | 652 | 803 | 1,061 | 183 | **0.2032** | 0.7024 | 0.7025 |
| **EXP52 / NEW** | 30.42% | 709 | 448 | 1,416 | 126 | **0.1010** | **0.8340** | 0.8340 |

### Analysis of the Oracle Gap:
- In theory, an "omniscient oracle" choosing between EXP52 and ModernBERT on every row would achieve **`0.8340` BA** (correcting 126 errors made by EXP52).
- However, the oracle between Q8 and Q14 alone already achieves **`0.8455` BA** (correcting 687 Q8 errors with Q14).
- ModernBERT contains fewer unique correct predictions (126) than instances where it is wrong and EXP52 is right (1,416). Without a perfect oracle router, linear probability ensembling degrades performance because ModernBERT's signal-to-noise ratio is too low ($\text{SNR} = 126 / 1416 = 0.089$).

---

## 7. Fusion Sweep & Selection-Safe Nested CV (Phases 11 & 12)

### Log-Probability Fusion Grid:
$$\text{LP}_{\text{fused}} = (1 - w) \cdot \text{LP}_{\text{EXP52}} + w \cdot \text{LP}_{\text{NEW}}$$

| ModernBERT Weight ($w$) | EXP52 Weight ($1-w$) | OOF BA | Macro-F1 | Delta vs EXP52 |
|:---:|:---:|---:|---:|---:|
| **0.00** | **1.00** | **0.7873** | **0.7880** | **+0.0000** |
| 0.05 | 0.95 | 0.7858 | 0.7864 | -0.0015 |
| 0.10 | 0.90 | 0.7836 | 0.7841 | -0.0037 |
| 0.15 | 0.85 | 0.7832 | 0.7838 | -0.0041 |
| 0.20 | 0.80 | 0.7773 | 0.7778 | -0.0100 |
| 0.25 | 0.75 | 0.7643 | 0.7647 | -0.0230 |
| 0.30 | 0.70 | 0.7458 | 0.7458 | -0.0415 |
| 0.40 | 0.60 | 0.7095 | 0.7100 | -0.0778 |
| 0.50 | 0.50 | 0.6513 | 0.6511 | -0.1360 |

### Nested Cross-Validation Results:
- In 4 out of 5 outer folds (Folds 1, 2, 3, 4), the 4-fold inner cross-validation correctly detected the degradation and selected $w^* = 0.00$.
- In Outer Fold 0, inner CV observed slight in-sample noise favoring $w^* = 0.05$ (Inner BA 0.7920 vs 0.7890). When applied out-of-sample to held-out Fold 0, BA degraded from $0.7778 \to 0.7704$ ($\Delta = -0.0074$).
- **Overall Nested OOF BA**: **`0.7858`** ($\Delta = \mathbf{-0.0015}$)
- **2,000 Paired Bootstrap Resamples**:
  - Mean Delta: **`-0.0015`**
  - 95% Bootstrap CI: **`[-0.0039, +0.0006]`**
  - Probability of Positive Gain: $P(\Delta > 0) = \mathbf{9.4\%}$

---

## 8. Disagreement-Aware Analysis & Conditional Fusion (Phases 13 & 14)

EXP61 established that Q8 and Q14 agreement is highly informative:
- When Q8 and Q14 agree ($40.27\%$ of data, $N=1,087$), EXP52 achieves **`0.9318` BA**.
- When Q8 and Q14 disagree ($59.73\%$ of data, $N=1,612$), EXP52 drops to **`0.6871` BA**.

EXP62 tested whether ModernBERT could rescue performance specifically on this 59.73% disagreement subset:

| Metric | Agreement Subset ($N=1,087$) | Disagreement Subset ($N=1,612$) |
|:---|:---:|:---:|
| **EXP52 BA** | **0.9318** | **0.6871** |
| **ModernBERT BA** | 0.3489 | **0.2836** |
| **Nested Fused BA** | 0.9310 | 0.6854 |

### Net Correction Breakdown on Disagreement Cases:
- Total EXP52 errors on disagreement cases: **513**
- ModernBERT correct when EXP52 is wrong (potential fixes): **117**
- ModernBERT wrong when EXP52 is correct (potential corruptions): **779**
- **Net Correction Capacity**: $117 - 779 = \mathbf{-662\text{ samples}}$

### Conditional Disagreement Fusion Sweep:
$$\text{LP}_{\text{cond}} = \begin{cases} \text{LP}_{\text{EXP52}}, & \text{if } Q8_{\text{pred}} == Q14_{\text{pred}} \\ (1 - w_{\text{dis}}) \cdot \text{LP}_{\text{EXP52}} + w_{\text{dis}} \cdot \text{LP}_{\text{NEW}}, & \text{if } Q8_{\text{pred}} \ne Q14_{\text{pred}} \end{cases}$$

- $w_{\text{dis}} = 0.00$: Overall BA = **0.7873**, Disagreement BA = **0.6871** ($\Delta = 0.0000$)
- $w_{\text{dis}} = 0.05$: Overall BA = **0.7862**, Disagreement BA = **0.6853** ($\Delta = -0.0011$)
- $w_{\text{dis}} = 0.10$: Overall BA = **0.7840**, Disagreement BA = **0.6813** ($\Delta = -0.0033$)
- $w_{\text{dis}} = 0.20$: Overall BA = **0.7780**, Disagreement BA = **0.6737** ($\Delta = -0.0093$)
- $w_{\text{dis}} = 0.40$: Overall BA = **0.7258**, Disagreement BA = **0.5883** ($\Delta = -0.0615$)

Because ModernBERT's accuracy on the disagreement subset is only $0.2836$ (substantially worse than EXP52's $0.6871$), increasing its weight corrupts accurate predictions far more rapidly than it rescues mistaken ones.

---

## 9. Representation Diversity (Phase 15)

Pairwise mean cosine similarity across output probability vectors:
- Cosine Sim (Q8 vs Q14): **`0.4845`**
- Cosine Sim (Q8 vs ModernBERT): **`0.3315`**
- Cosine Sim (Q14 vs ModernBERT): **`0.4826`**
- Cosine Sim (EXP52 vs ModernBERT): **`0.4552`**

**Scientific Principle**: While ModernBERT exhibits lower cosine similarity to Q8 than Q14 does ($0.3315$ vs $0.4845$), low representation similarity alone is insufficient. Without adequate discriminative signal (high standalone accuracy), representation diversity translates into uninformative noise rather than complementary attribution power.

---

## 10. Figures & Visualizations

Five publication-grade figures were generated and verified at 300 DPI:

1. [`plots/exp62_model_comparison.png`](file:///home/rj2826/name-that-llm/plots/exp62_model_comparison.png):
   Bar chart comparing Balanced Accuracy across Q8 (`0.5910`), Q14 (`0.6346`), ModernBERT LR (`0.3093`), ModernBERT SVM (`0.2371`), EXP52 Champion (`0.7873`), and EXP52 + ModernBERT Nested (`0.7858`).
2. [`plots/exp62_error_correlation.png`](file:///home/rj2826/name-that-llm/plots/exp62_error_correlation.png):
   $4 \times 4$ heatmap of pairwise Pearson error correlations showing low correlation between ModernBERT and EXP52 ($r = 0.1010$).
3. [`plots/exp62_oracle_complementarity.png`](file:///home/rj2826/name-that-llm/plots/exp62_oracle_complementarity.png):
   Grouped bar chart comparing the theoretical oracle upper bound vs actual realized ensemble accuracy across all four pairs.
4. [`plots/exp62_fusion_curve.png`](file:///home/rj2826/name-that-llm/plots/exp62_fusion_curve.png):
   Monotonic fusion degradation curve across $w \in [0.00, 0.50]$.
5. [`plots/exp62_disagreement_analysis.png`](file:///home/rj2826/name-that-llm/plots/exp62_disagreement_analysis.png):
   Three-way comparison on Q8/Q14 Agreement, Disagreement, and Full Dataset subsets illustrating the drop in accuracy when fusing ModernBERT into disagreement cases.

---

## 11. Verdict & Scientific Conclusion

### Verdict: RED

Criteria evaluation:
- Criterion 1: *EXP52 + NEW improves nested BA meaningfully* $\to$ **FAILED** (Nested BA is $0.7858$, $\Delta = -0.0015$).
- Criterion 2: *Improvement has positive bootstrap CI* $\to$ **FAILED** (95% CI is $[-0.0039, +0.0006]$, $P(\Delta > 0) = 9.4\%$).
- Criterion 3: *NEW fixes a meaningful number of EXP52 errors* $\to$ **FAILED** (126 fixes vs 1,416 corruptions overall; $-662$ net correction on disagreement cases).
- Criterion 4: *Gains disappear under nested validation* $\to$ **CONFIRMED** (Nested CV selects $w^* = 0.00$ in 4 of 5 folds).

### Scientific Answer to Central Research Question:
> **Does architectural diversity create genuinely new attribution information that Qwen3-8B + Qwen3-14B do not already contain?**

**Answer: NO.** A frozen non-Qwen encoder (ModernBERT-base) achieves only ~31% standalone balanced accuracy on this 18-class problem and introduces massive corruptive noise when ensembled with the fine-tuned dual-Qwen backbones. Low error correlation ($r=0.1010$) is a mathematical byproduct of low accuracy, not constructive orthogonality.

**Decision**:
- **STOP** un-fine-tuned cross-architecture model stacking.
- **RETAIN EXP52** strictly frozen as the production champion at **0.632 Public LB** / **0.7873 local OOF BA**.
- **ZERO Kaggle submissions**.
