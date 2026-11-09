# EXP57B: Class-Head Norm Equalization Validation Report

**Experiment ID**: EXP57B  
**Date**: November 9, 2026  
**Status**: COMPLETE  
**Verdict**: **RED**  
**Recommended Next Step**: `STOP_WEIGHT_SPACE_EDITING`  
**Kaggle Submission Made**: NO (Strict Rule Enforced)  
**EXP52 Baseline Preserved**: YES (Frozen at 0.632 Public LB / 0.7873 local OOF BA)

---

## Executive Summary

In EXP57A, weight-space diagnosis revealed a striking observational correlation between class-head classifier weight L2 norm ($\|W_c\|_2$) and per-class OOF recall ($r = 0.9528, p = 2.1 \times 10^{-9}$). 

**EXP57B** was designed as a strict, inference-time, prompt-disjoint causal intervention to test whether equalizing classifier head norms via $W'_c = W_c / N_c^\gamma$ improves LLM attribution accuracy or whether the association was merely an artifact of class frequency/calibration.

The experimental results definitively **reject** the causal utility of weight-space norm equalization:
1. **Marginal, Non-Significant OOF Gain on Standalone Plain Head**: At $\gamma = 1.00$, Variant A gained a nominal $+0.0011$ Balanced Accuracy (BA $0.6346 \to 0.6357$), but Macro-F1 dropped ($0.6338 \to 0.6335$), the 95% paired bootstrap CI spans zero ($[-0.0021, +0.0044]$), and $P(\Delta > 0) = 0.7585$.
2. **One-Fold Driven Artifact**: The $+0.0011$ gain was entirely driven by a single fold (Fold 3: $+0.0111$), while Fold 1 dropped ($-0.0037$), Fold 4 dropped ($-0.0019$), and Folds 0 & 2 showed zero change.
3. **Mechanistic Hypothesis Falsified**: The core hypothesis predicted that suppressed-norm classes (`glm-5.3`, `glm-5.3-flash`, `gemini-3.7-flash`, `kimi-k3`) would experience recall gains. In reality, their mean recall **degraded** by $-0.0117$ (`glm-5.3` fell $-0.0267$, `gemini-3.7-flash` fell $-0.0200$). Meanwhile, dominant-norm classes (`gpt-6-astra`, `gemini-3.1-pro-preview`, `claude-opus-5`) actually gained $+0.0044$ recall.
4. **Permutation Control Fails**: Randomly shuffling the norm assignments across classes yielded an average $\Delta \text{BA} = +0.0006 \pm 0.0020$. The original assignment was at the **65.0th percentile** of random permutations, proving that the $+0.0011$ delta is well within random logit perturbation noise.
5. **Ensemble Degradation**: In the production-grade twin-head architecture (Variant D) and the frozen EXP52 dual-LoRA ensemble ($0.60 \text{ Q8} + 0.40 \text{ Q14}$), norm equalization degraded performance across all $\gamma > 0$ (EXP52 BA dropped from $0.7873$ to $0.7847$, $\Delta = -0.0026$).

Consequently, weight-space norm editing is classified as **RED** and weight-space editing should be stopped.

---

## 1. Baseline Reproduction Verification

Before testing any norm intervention, the baseline prompt-disjoint 5-fold OOF metrics were reproduced exactly against the verified reference standards:

| Model Component | Reproduced BA | Reference BA | Reproduced Macro-F1 | Status |
|:---|:---:|:---:|:---:|:---:|
| **q14v5 Plain Head (Variant A)** | **0.6346** | 0.6346 | 0.6338 | Verified Match ($\Delta = 0.0000$) |
| **q14v5 Centered Head (Variant B)** | **0.4842** | 0.4842 | 0.4534 | Verified Match ($\Delta = 0.0000$) |
| **q14v5 Twin Combined @ 0.35 (Variant C)** | **0.6491** | 0.6491 | 0.6409 | Verified Match ($\Delta = 0.0000$) |
| **EXP52 Ensemble ($0.60 \text{ Q8} + 0.40 \text{ Q14}$)** | **0.7873** | 0.7873 | 0.7880 | Verified Match ($\Delta = 0.0000$) |

All 2,699 samples across all 5 prompt-disjoint folds matched down to four decimal places.

---

## 2. Pre-Specified Gamma Grid Sweep

Norm equalization was evaluated across the pre-specified exponent grid $\gamma \in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]$ for all four model variants.

### Gamma Sweep Summary Table

| Variant | $\gamma$ | Balanced Acc | Macro-F1 | $\Delta$ BA | Bootstrap 95% CI | $P(\Delta > 0)$ | Mean Entropy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Variant A (Plain Head)** | 0.00 | 0.6346 | 0.6338 | +0.0000 | [0.0000, 0.0000] | 0.0000 | 0.6542 |
| | 0.10 | 0.6342 | 0.6330 | -0.0004 | [-0.0017, +0.0008] | 0.2550 | 0.7830 |
| | 0.20 | 0.6335 | 0.6322 | -0.0011 | [-0.0028, +0.0004] | 0.0780 | 0.9380 |
| | 0.30 | 0.6335 | 0.6321 | -0.0011 | [-0.0028, +0.0004] | 0.0780 | 1.1070 |
| | 0.40 | 0.6353 | 0.6338 | +0.0007 | [-0.0015, +0.0032] | 0.7390 | 1.2849 |
| | 0.50 | 0.6350 | 0.6333 | +0.0004 | [-0.0022, +0.0031] | 0.6165 | 1.4688 |
| | 0.75 | 0.6346 | 0.6326 | +0.0000 | [-0.0029, +0.0029] | 0.5050 | 1.9392 |
| | **1.00** | **0.6357** | **0.6335** | **+0.0011** | **[-0.0021, +0.0044]** | **0.7585** | 2.3789 |
| **Variant B (Centered Head)** | 0.00 | 0.4842 | 0.4534 | +0.0000 | [0.0000, 0.0000] | 0.0000 | 0.7026 |
| | 0.10 | 0.4831 | 0.4527 | -0.0011 | [-0.0039, +0.0014] | 0.1895 | 0.8622 |
| | 0.20 | 0.4823 | 0.4521 | -0.0019 | [-0.0051, +0.0013] | 0.1125 | 1.0564 |
| | 0.50 | 0.4827 | 0.4535 | -0.0015 | [-0.0070, +0.0040] | 0.2885 | 1.7644 |
| | 1.00 | 0.4809 | 0.4535 | -0.0034 | [-0.0110, +0.0037] | 0.1855 | 2.7667 |
| **Variant C (Twin Baseline)** | any | 0.6491 | 0.6409 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.4994 |
| **Variant D (Twin Scale-Preserving)** | 0.00 | 0.6491 | 0.6409 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.4994 |
| | 0.10 | 0.6483 | 0.6399 | -0.0007 | [-0.0019, 0.0000] | 0.0000 | 0.4989 |
| | 0.20 | 0.6483 | 0.6399 | -0.0007 | [-0.0019, 0.0000] | 0.0000 | 0.4984 |
| | 0.30 | 0.6487 | 0.6401 | -0.0004 | [-0.0017, +0.0008] | 0.2660 | 0.4982 |
| | 0.40 | 0.6487 | 0.6401 | -0.0004 | [-0.0020, +0.0012] | 0.3330 | 0.4981 |
| | 0.50 | 0.6483 | 0.6395 | -0.0007 | [-0.0025, +0.0011] | 0.2200 | 0.4983 |
| | 1.00 | 0.6483 | 0.6388 | -0.0007 | [-0.0034, +0.0020] | 0.3075 | 0.5008 |

---

## 3. Fold-by-Fold Stability Analysis

A core requirement of EXP57B is cross-fold stability. If an apparent improvement is concentrated in one fold while other folds degrade, the candidate must be rejected.

### Fold-by-Fold Results for Variant A (Plain Head)

| $\gamma$ | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold Mean | Fold Std |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 (Base) | 0.6278 | 0.6418 | 0.6389 | 0.6407 | 0.6241 | 0.6347 | 0.0073 |
| 0.10 | 0.6259 | 0.6400 | 0.6389 | 0.6426 | 0.6241 | 0.6343 | 0.0077 |
| 0.20 | 0.6259 | 0.6381 | 0.6389 | 0.6426 | 0.6222 | 0.6336 | 0.0080 |
| 0.30 | 0.6259 | 0.6381 | 0.6389 | 0.6426 | 0.6222 | 0.6336 | 0.0080 |
| 0.40 | 0.6278 | 0.6400 | 0.6407 | 0.6463 | 0.6222 | 0.6354 | 0.0089 |
| 0.50 | 0.6278 | 0.6381 | 0.6407 | 0.6463 | 0.6222 | 0.6350 | 0.0088 |
| 0.75 | 0.6259 | 0.6381 | 0.6389 | 0.6481 | 0.6222 | 0.6347 | 0.0094 |
| **1.00** | **0.6278** | **0.6381** | **0.6389** | **0.6519** | **0.6222** | **0.6358** | **0.0102** |
| **$\Delta (\gamma=1.0 - 0.0)$** | **+0.0000** | **-0.0037** | **+0.0000** | **+0.0111** | **-0.0019** | **+0.0011** | — |

**Fold Stability Diagnosis**:
- **Fold 3 is the ONLY fold that improved** ($+0.0111$).
- **Fold 1 degraded** by $-0.0037$.
- **Fold 4 degraded** by $-0.0019$.
- **Folds 0 and 2 were unchanged** ($0.0000$).
The apparent $+0.0011$ pooled OOF gain is an artifact of Fold 3 over-compensating for losses in Folds 1 and 4. The effect is **NOT cross-fold stable**.

---

## 4. Per-Class Mechanistic Analysis

The theoretical hypothesis asserted that smaller classifier weight norms suppress class recall by under-scaling logits, predicting that equalizing norms would preferentially elevate recall on low-norm classes without disproportionately harming high-norm classes.

### Per-Class Recall and Precision Dynamics ($\gamma = 1.00$)

| Model Class | Mean $\|W_c\|_2$ | Recall Base | Recall Equalized | $\Delta$ Recall | Precision Base | Precision Equalized | $\Delta$ Precision | Target Category |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `glm-5.3` | **7.55** | 0.3200 | 0.2933 | **-0.0267** | 0.3221 | 0.3165 | -0.0056 | **Low-Norm Target** |
| `glm-5.3-flash` | **7.58** | 0.3667 | 0.3667 | **+0.0000** | 0.3846 | 0.3901 | +0.0055 | **Low-Norm Target** |
| `gemini-3.7-flash` | **7.67** | 0.4733 | 0.4533 | **-0.0200** | 0.5145 | 0.5312 | +0.0168 | **Low-Norm Target** |
| `kimi-k3` | **7.97** | 0.5133 | 0.5133 | **+0.0000** | 0.6063 | 0.6210 | +0.0147 | **Low-Norm Target** |
| `gemini-3.8-flash` | 8.01 | 0.5667 | 0.5867 | +0.0200 | 0.5449 | 0.5535 | +0.0086 | Mid-Norm |
| `claude-fable-5` | 8.06 | 0.4430 | 0.4497 | +0.0067 | 0.4000 | 0.4012 | +0.0012 | Mid-Norm |
| `gpt-5.6-sol` | 8.23 | 0.5267 | 0.5267 | +0.0000 | 0.5683 | 0.5809 | +0.0125 | Mid-Norm |
| `claude-opus-4-8` | 8.36 | 0.6933 | 0.6867 | -0.0067 | 0.7172 | 0.7153 | -0.0020 | Mid-Norm |
| `gpt-5.6-terra` | 8.50 | 0.6200 | 0.6200 | +0.0000 | 0.5962 | 0.5962 | +0.0000 | Mid-Norm |
| `gpt-5.6-luna` | 8.57 | 0.6600 | 0.6600 | +0.0000 | 0.6188 | 0.6111 | -0.0076 | Mid-Norm |
| `gemini-3.6-flash` | 8.59 | 0.7333 | 0.7533 | +0.0200 | 0.7097 | 0.7019 | -0.0078 | Mid-Norm |
| `qwen3.8-max` | 8.63 | 0.7467 | 0.7533 | +0.0067 | 0.8000 | 0.7902 | -0.0098 | Mid-Norm |
| `claude-sonnet-5` | 8.76 | 0.7267 | 0.7267 | +0.0000 | 0.6943 | 0.6812 | -0.0130 | Mid-Norm |
| `gpt-5.5` | 8.84 | 0.6667 | 0.6733 | +0.0067 | 0.6329 | 0.6352 | +0.0023 | Mid-Norm |
| `claude-opus-4-6` | 8.89 | 0.8133 | 0.8133 | +0.0000 | 0.8026 | 0.7821 | -0.0206 | Mid-Norm |
| `gemini-3.1-pro-preview` | **9.16** | 0.8667 | 0.8667 | **+0.0000** | 0.8553 | 0.8497 | -0.0056 | **High-Norm Target** |
| `claude-opus-5` | **9.16** | 0.8133 | 0.8267 | **+0.0133** | 0.7922 | 0.7848 | -0.0074 | **High-Norm Target** |
| `gpt-6-astra` | **9.43** | 0.8733 | 0.8733 | **+0.0000** | 0.8562 | 0.8562 | +0.0000 | **High-Norm Target** |

**Summary of Target Groups**:
- **Low-Norm Targets**: Mean $\Delta$ Recall = **$-0.0117$** (DEGRADED). Specifically, `glm-5.3` dropped from $0.3200 \to 0.2933$ and `gemini-3.7-flash` dropped from $0.4733 \to 0.4533$.
- **High-Norm Targets**: Mean $\Delta$ Recall = **$+0.0044$** (IMPROVED).
The empirical behavior is the exact **opposite** of the hypothesis prediction.

---

## 5. Control Experiments

### Control 1: Random Norm Permutation Test (20 Iterations)
To test whether the class-specific assignment of norms matters, the class norms were randomly permuted across the 18 classes 20 times:
- **Original Class-Specific $\Delta$ BA**: $+0.0011$
- **Permutation Mean $\Delta$ BA**: $+0.0006 \pm 0.0020$
- **Permutation Range**: $[-0.0026, +0.0063]$
- **Empirical Percentile of Original Result**: **65.0%**

Because 35% of purely random shuffles achieved an equal or higher Balanced Accuracy than the true weight norms (with some permutations reaching $+0.0063$), the true weight norms possess **no privileged class-specific alignment**.

### Control 2: Logit Bias & Temperature Calibration Equivalence
For linear classifier $z_c = W_c \cdot x + b_c$, equalizing weight norm $W'_c = W_c / N_c^\gamma$ scales logits as $z'_c = z_c / N_c^\gamma$.
Expanding $N_c = \bar{N} + \delta N_c$ to first order around mean norm $\bar{N}$:
$$\frac{1}{N_c^\gamma} \approx \bar{N}^{-\gamma} \left( 1 - \gamma \frac{\delta N_c}{\bar{N}} \right)$$
For a representative feature $x$ with mean logit level $\bar{z}$:
$$z'_c \approx \frac{z_c}{\bar{N}^\gamma} - \gamma \bar{z} \bar{N}^{-\gamma-1} (N_c - \bar{N})$$
This demonstrates that norm equalization is mathematically equivalent to:
1. A global temperature rescaling by $T = \bar{N}^\gamma$.
2. An additive class logit bias $\beta_c \propto -(N_c - \bar{N})$.

Because global scalar temperature scaling on a single head does not alter the argmax ranking ($\Delta \text{BA} = 0.0000$ identically for all $T \in [0.70, 2.00]$), any minor shifts in argmax are solely due to tiny, noisy class-specific logit bias shifts. This confirms that norm equalization is merely an uncalibrated post-hoc logit offset, not a novel weight-space editing phenomenon.

---

## 6. Frozen EXP52 Integration

The ultimate test for any candidate improvement is whether it survives when integrated into the best validated ensemble: **EXP52** ($0.60 \text{ Q8} + 0.40 \text{ Q14}$).

| System Configuration | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean Fold BA | OOF BA | $\Delta$ OOF BA | 95% Bootstrap CI | $P(\Delta > 0)$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP52 Baseline ($0.60\text{Q8} + 0.40\text{Q14}$)** | 0.7778 | 0.7793 | 0.7907 | 0.7833 | 0.8056 | 0.7873 | **0.7873** | 0.0000 | [0.0000, 0.0000] | 0.0000 |
| **Candidate EXP52 ($0.60\text{Q8} + 0.40\text{Q14}_{\text{norm\_eq}}$)** | 0.7741 | 0.7756 | 0.7852 | 0.7815 | 0.8074 | 0.7848 | **0.7847** | **-0.0026** | **[-0.0059, +0.0008]** | **0.0615** |

When injected into EXP52:
- OOF BA **drops by $-0.0026$** ($0.7873 \to 0.7847$).
- 4 out of 5 folds experience performance drops (Fold 0: $-0.0037$, Fold 1: $-0.0037$, Fold 2: $-0.0055$, Fold 3: $-0.0018$).
- Probability of positive delta is a negligible **6.15%**.

---

## 7. Answers to Mandatory EXP57B Questions (Section 19)

1. **Does class-head norm actually causally affect attribution performance under inference-time intervention?**  
   **No.** The observational correlation between class-head norm and recall in EXP57A does not reflect a causal mechanism that can be leveraged at inference time. Equalizing norms does not improve balanced accuracy.
2. **What gamma performs best?**  
   On Variant A (Plain head alone), $\gamma = 1.00$ showed the highest nominal BA ($0.6357$, $+0.0011$), but this was not statistically significant and degraded Macro-F1. For all other variants (B, D, EXP52), $\gamma = 0.00$ (the unedited baseline) was strictly optimal.
3. **Is the improvement stable across all five prompt-disjoint folds?**  
   **No.** At $\gamma = 1.00$, 2 folds degraded, 2 were unchanged, and only 1 fold improved.
4. **Do low-norm classes improve?**  
   **No.** Low-norm target classes degraded on average by $-0.0117$ in recall (`glm-5.3` lost $-0.0267$, `gemini-3.7-flash` lost $-0.0200$).
5. **Do high-norm classes deteriorate?**  
   **No.** High-norm target classes actually gained $+0.0044$ in recall.
6. **Does global temperature scaling reproduce the effect?**  
   Global temperature scaling does not change single-head argmax predictions ($\Delta \text{BA} = 0.0000$), demonstrating that the observed shifts are due to minor class offset distortions rather than sharpness adjustments.
7. **Does the permutation control reject the class-specific hypothesis?**  
   **Yes.** True class norm assignment was at the 65th percentile of random permutations, which is indistinguishable from random noise.
8. **Does the effect survive inside frozen EXP52?**  
   **No.** When integrated into EXP52, OOF BA falls from $0.7873$ to $0.7847$ ($\Delta = -0.0026$, with $P(\Delta > 0) = 0.0615$).
9. **Is this genuinely promising for weight-space programming, or merely classifier calibration?**  
   It is merely an unprincipled classifier logit shift. It does not represent genuine weight-space programming.
10. **Should EXP57C be attempted?**  
    **No.** Weight-space editing on classifier heads has been thoroughly refuted.

---

## 8. Artifact Inventory

All generated artifacts are archived in `experiments/EXP57B_norm_equalization/`:
- `gamma_sweep.csv`: Full metric log across 8 gamma values and 4 model variants.
- `temperature_sweep.csv`: Calibration sweep across 8 temperature values.
- `fold_results.csv`: Complete 5-fold cross-validation metrics across all conditions.
- `per_class_effects.csv`: Detailed recall, precision, and F1 deltas for all 18 LLM classes.
- `permutation_control.csv`: Results of 20 random norm permutation iterations.
- `exp52_integration.csv`: Direct comparison of baseline EXP52 vs candidate EXP52 across all folds.
- `confusion_matrix_baseline.png`: 18x18 normalized confusion matrix of baseline q14v5.
- `confusion_matrix_best_gamma.png`: 18x18 normalized confusion matrix of equalized head at $\gamma = 1.00$.
- `class_recall_delta.png`: Bar plot of per-class recall deltas sorted by weight norm.
- `gamma_vs_ba.png`: Comparative trajectory of BA vs $\gamma$ across the four variants.
- `temperature_vs_ba.png`: Calibration and entropy curves vs temperature.
