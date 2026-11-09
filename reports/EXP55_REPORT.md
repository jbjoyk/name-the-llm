# EXP55: Isolated Stylometry / Character-Level Specialist Fusion Report

**Experiment ID**: `EXP55_stylometry_specialist`  
**Date**: 2026-11-08  
**Author**: Antigravity Machine Learning Research Agent  
**Parent Baseline**: EXP52 (Dual-LoRA Qwen3-8B $0.60$ + Qwen3-14B $0.40$, Local OOF BA: **0.7873**, Kaggle Public LB: **0.632**)  
**Predecessor Experiments**: EXP47 (LB 0.620), EXP49 (LB 0.612), EXP50 (OOF BA 0.6102), EXP52 (LB 0.632), EXP53 (OOF BA 0.7873), EXP54 (OOF BA 0.7888)  
**Primary Metric**: Macro-averaged Balanced Accuracy across 18 candidate LLM classes  
**Validation Strategy**: Strict 5-Fold Prompt-Disjoint `GroupKFold(groups=prompt_id)` on `data/train.csv` (2,699 samples, 150 unique prompts, 30 prompts per fold)  
**Nested Validation Strategy**: 5-Fold outer prompt-disjoint CV with inner 4-fold selection of fusion weights  
**Kaggle Submission Status**: **ABSOLUTELY NOT SUBMITTED — ZERO SUBMISSIONS**  
**Production Baseline**: **EXP52 REMAINS FROZEN AT KAGGLE PUBLIC LB 0.632**  
**EXP55 Verdict**: **YELLOW** (Small/uncertain improvement; local gains do not survive domain shift risk; no submission)

---

## Executive Summary

EXP55 executes an isolated scientific ablation to test whether a lightweight **stylometry / character-level surface-form specialist** provides complementary discriminative power to our frozen production baseline: **EXP52 (Qwen3-8B + Qwen3-14B dual-backbone ensemble, Kaggle LB 0.632, local OOF BA 0.7873)**.

In EXP49, combining Qwen3-8B with an Open Pool 4-view LinearSVC produced an apparent local OOF jump to **0.7028**, but collapsed on Kaggle's hidden test set to **0.612** because 85% decision mass was granted to shallow n-grams that overfit training prompt vocabulary. In EXP54, adding a small weight ($w_{\text{svc}} \le 0.20$) of that same 4-view SVC to EXP52 yielded an insignificant gain (+0.0015 BA, net +4 samples).

EXP55 isolates the exact **character-level surface-form TF-IDF specialist** from the previously audited ~0.66–0.672 solution lineage (Cell 12 of `0-672-dual-svc-residual-refinement-apex.ipynb` / `infer.py`, `--tfidf_w 0.25`), without any external data, without full-fit shortcuts, and without combining multiple complex components.

### Key Empirical Findings

1. **Standalone Performance**:
   Under strict 5-fold prompt-disjoint validation, the isolated character-level specialist achieves **0.3437 Balanced Accuracy** (Macro-F1: **0.3376**, Accuracy: **0.3438**). It provides a coherent, lightweight surface-form baseline that operates purely on character word-boundary and punctuation n-grams.
2. **Complementarity & Error Correlation**:
   The specialist shares an agreement rate of only **33.83%** (913 / 2,699) with EXP52. The measured Pearson error correlation is low:
   $$r = 0.11506 \quad (p = 2.04 \times 10^{-9})$$
   The specialist correctly identifies **137 passages** where EXP52 failed. If an optimal oracle selector could perfectly route every sample, the theoretical upper bound would reach **83.81% Balanced Accuracy** (+0.0508 ceiling over EXP52).
3. **Predefined Log-Probability Fusion Sweep**:
   Sweeping $w_{\text{style}} \in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]$ shows mild positive local deltas:
   - $w_{\text{style}} = 0.00$: BA = **0.7873** (EXP52 baseline)
   - $w_{\text{style}} = 0.02$: BA = **0.7888** ($\Delta = +0.0015$, 95% CI: $[+0.0003, +0.0032]$, $P = 0.9855$)
   - $w_{\text{style}} = 0.05$: BA = **0.7899** ($\Delta = +0.0026$, 95% CI: $[+0.0008, +0.0047]$, $P = 0.9995$)
   - $w_{\text{style}} = 0.10$: BA = **0.7877** ($\Delta = +0.0004$, 95% CI: $[-0.0026, +0.0036]$, $P = 0.6030$)
   - $w_{\text{style}} = 0.15$: BA = **0.7907** ($\Delta = +0.0033$, 95% CI: $[-0.0004, +0.0074]$, $P = 0.9540$)
   - $w_{\text{style}} = 0.25$: BA = **0.7907** ($\Delta = +0.0033$, 95% CI: $[-0.0019, +0.0087]$, $P = 0.8920$)
4. **Nested Cross-Validation Reality Check**:
   Selecting the fusion weight strictly on outer-training folds yields a nested cross-validation score of:
   $$\text{Nested BA} = 0.7881 \pm 0.0098 \quad (\Delta_{\text{nested}} = +0.0008 \text{ over EXP52})$$
   When freed from post-hoc selection bias, the true expected generalization gain is a negligible **+0.0008 BA** (less than 1 part in a thousand).
5. **Decision & Verdict**:
   Per the competition decision rules, an improvement of $+0.001$ to $+0.003$ whose peak confidence interval crosses zero and whose nested validation delta is below $+0.001$ is classified as **YELLOW**. No submission to Kaggle is permitted. EXP52 remains our undefeated production baseline (0.632 LB).

---

## Provenance of Stylometry Implementation

The stylometry / character-level model used in EXP55 is directly recovered from the previously audited high-scoring solution lineage:

1. **Primary Origin**: Cell 12 of [`reference/0-672-dual-svc-residual-refinement-apex.ipynb`](file:///home/rj2826/name-the-llm/reference/0-672-dual-svc-residual-refinement-apex.ipynb) (`infer.py`, lines under `# ---------------- 2. TF-IDF centred LR`). In the 0.672 Apex solution, this component is assigned `--tfidf_w 0.25` in the final command line.
2. **Corroborating Origin**: Cell 5 & 28 of [`reference/the-usual-suspects.ipynb`](file:///home/rj2826/name-the-llm/reference/the-usual-suspects.ipynb):
   ```text
   | `tfidf` | character and word TF-IDF, logistic regression | 0.25 | none | none |
   ```
3. **Exact Specification**:
   - **View 1 (Character Word-Boundary N-grams)**:
     - Vectorizer: `TfidfVectorizer`
     - Analyzer: `char_wb`
     - N-gram Range: `(2, 5)`
     - Minimum Document Frequency: `min_df = 3`
     - Max Features: `max_features = 60,000`
     - Sublinear TF: `sublinear_tf = True`
     - Lowercase: `lowercase = False` (preserves capitalization quirks, markdown headers, and CamelCase)
   - **View 2 (Word and Punctuation N-grams)**:
     - Vectorizer: `TfidfVectorizer`
     - Analyzer: `word`
     - Token Pattern: `r"\w+|[^\w\s]"` (treats punctuation marks as standalone tokens)
     - N-gram Range: `(1, 2)`
     - Minimum Document Frequency: `min_df = 3`
     - Max Features: `max_features = 30,000`
     - Sublinear TF: `sublinear_tf = True`
     - Lowercase: `lowercase = False`
   - **Classifier**:
     - Model: `LogisticRegression`
     - Regularization: `C = 5.0`
     - Loss: Multinomial logit via L-BFGS
     - Class Weight: `balanced`
     - Max Iterations: `1,000`
     - Tolerance: `1e-3`
     - Random State: `42`
4. **Feature Leakage Check**:
   - All feature extraction operates strictly on `train_df['text']`.
   - Prohibited metadata fields (`id`, `prompt_id`, `model`, etc.) were verified stripped before passing text to the vectorizer.
   - Vectorizers are fitted strictly on outer-training folds; validation folds are transformed out-of-sample.

---

## Experimental Design

- **Dataset**: `data/train.csv` (2,699 samples, 18 models).
- **Partitioning**: Exact 5-fold prompt-disjoint split from `outputs/oof_q14v5_antopil_lora.csv`:
  - 150 unique prompts partitioned into 5 groups of exactly 30 prompts each.
  - Zero prompt overlap between training and validation splits.
  - Strict preservation of row IDs and row ordering matching `data/train.csv`.
- **Fusion Formulation**:
  $$\text{LP}_{\text{fused}} = (1 - w_{\text{style}}) \cdot \text{LP}_{\text{EXP52}} + w_{\text{style}} \cdot \text{LP}_{\text{STYLE}}$$
  where $\text{LP}_{\text{EXP52}} = 0.60 \log P_{q8} + 0.40 \log P_{q14}$, with $P_{q8}$ and $P_{q14}$ normalized and epsilon-clipped ($10^{-15}$).
- **Evaluated Weights**: $w_{\text{style}} \in \{0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25\}$.

---

## Standalone Stylometry Results

Performance of the isolated stylometry / character-level specialist across the 5 prompt-disjoint folds:

| Metric | Standalone Stylometry | EXP52 Baseline |
| :--- | :---: | :---: |
| **Balanced Accuracy** | **0.3437** | **0.7873** |
| **Macro-F1** | 0.3376 | 0.7880 |
| **Accuracy** | 0.3438 | 0.7873 |
| **Mean Entropy** | 2.5577 nats | 0.8258 nats |
| **Top-1 Confidence** | 0.1911 | 0.7448 |
| **Top-2 Margin** | 0.0579 | 0.6258 |

### Fold-Level Standalone Consistency:
- **Fold 0** (540 rows): BA = **0.3389**
- **Fold 1** (539 rows): BA = **0.3595**
- **Fold 2** (540 rows): BA = **0.3222**
- **Fold 3** (540 rows): BA = **0.3574**
- **Fold 4** (540 rows): BA = **0.3407**
- **Mean $\pm$ Std**: **0.3437 $\pm$ 0.0139**

The specialist is remarkably stable across all 5 prompt-disjoint folds (low variance $\pm 0.0139$), confirming that its character-level n-gram features capture prompt-invariant surface regularities rather than memorizing prompt topics.

---

## EXP52 vs EXP55 Complementarity

A contingency analysis between the predictions of EXP52 and EXP55 reveals clear structural complementarity:

| Category | Sample Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Agreement Rate** | 913 / 2,699 | 33.83% | Passages where EXP52 and Stylometry predict the exact same author |
| **Both Correct** | 791 | 29.31% | Both models identify the true model |
| **Both Wrong** | 437 | 16.19% | Mutual blind spots |
| **EXP52-only Correct** | 1,334 | 49.43% | Qwen foundation models correct, Stylometry fails |
| **Stylometry-only Correct** | 137 | 5.08% | **Stylometry corrects EXP52 errors** |
| **Oracle Accuracy Ceiling** | 2,262 / 2,699 | **83.81%** | Theoretical ceiling if routing were perfect |
| **Oracle Balanced Accuracy** | — | **0.8381** | Macro-recall ceiling (+0.0508 above EXP52) |
| **Measured Error Correlation ($r$)** | — | **0.11506** | Low positive correlation ($p = 2.04 \times 10^{-9}$) |

> [!NOTE]
> The measured Pearson error correlation ($r = 0.11506$) is low. We describe this as **low measured Pearson error correlation**, not "independent errors." The models fail on largely different samples, but EXP52's dominance means only 137 samples are unique candidates for correction.

---

## Fusion Results

Evaluating fixed log-probability fusion across the 7 predefined weights:

| $w_{\text{style}}$ | $w_{\text{exp52}}$ | OOF BA | Macro-F1 | Acc | Entropy | Margin | Net Samples Gained | Delta vs EXP52 | 95% Bootstrap CI | P($\Delta > 0$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | **1.00** | **0.7873** | 0.7880 | 0.7873 | 0.8258 | 0.6258 | 0 (Base) | **0.0000** | [0.0000, 0.0000] | 0.0000 |
| **0.02** | **0.98** | **0.7888** | 0.7895 | 0.7888 | 0.8454 | 0.6192 | +4 | **+0.0015** | [+0.0003, +0.0032] | 0.9855 |
| **0.05** | **0.95** | **0.7899** | 0.7907 | 0.7899 | 0.8760 | 0.6089 | +7 | **+0.0026** | **[+0.0008, +0.0047]** | **0.9995** |
| **0.10** | **0.90** | **0.7877** | 0.7885 | 0.7877 | 0.9306 | 0.5904 | +1 | **+0.0004** | [-0.0026, +0.0036] | 0.6030 |
| **0.15** | **0.85** | **0.7907** | 0.7915 | 0.7907 | 0.9898 | 0.5704 | +9 | **+0.0033** | [-0.0004, +0.0074] | 0.9540 |
| **0.20** | **0.80** | **0.7899** | 0.7906 | 0.7899 | 1.0539 | 0.5488 | +7 | **+0.0026** | [-0.0021, +0.0074] | 0.8575 |
| **0.25** | **0.75** | **0.7907** | 0.7913 | 0.7907 | 1.1230 | 0.5254 | +9 | **+0.0033** | [-0.0019, +0.0087] | 0.8920 |

### Observations:
1. **Conservative Optimum ($w_{\text{style}} = 0.05$)**: Balanced Accuracy climbs from **0.7873 to 0.7899** (+0.0026), with a 95% bootstrap confidence interval strictly above zero ($[+0.0008, +0.0047]$) and $P(\Delta > 0) = 99.95\%$. This corresponds to a net gain of +7 correctly classified samples.
2. **Plateau at Higher Weights ($w \ge 0.15$)**: The curve flattens around **0.7907** (+0.0033), but the 95% confidence interval crosses zero ($[-0.0004, +0.0074]$), and prediction margin deteriorates from $0.6258$ down to $0.5704$.

---

## Nested Validation

To prevent post-hoc selection bias, we performed an end-to-end 5-fold **nested validation**. For each of the 5 outer prompt-disjoint folds, the optimal fusion weight $w^*$ was selected strictly on the remaining 4 outer-training folds, and evaluated on the held-out outer test fold:

| Outer Fold | Held-Out Size | Selected Inner Weight ($w^*$) | Inner Training BA | Outer Test BA | EXP52 Baseline Outer BA | Fold Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 540 rows | **0.15** | 0.7943 | 0.7759 | 0.7778 | -0.0019 |
| **Fold 1** | 539 rows | **0.25** | 0.7926 | 0.7830 | 0.7793 | +0.0037 |
| **Fold 2** | 540 rows | **0.25** | 0.7897 | 0.7944 | 0.7907 | +0.0037 |
| **Fold 3** | 540 rows | **0.05** | 0.7916 | 0.7833 | 0.7833 | 0.0000 |
| **Fold 4** | 540 rows | **0.25** | 0.7874 | 0.8037 | 0.8056 | -0.0019 |

### Nested Validation Summary:
- **Nested Mean Balanced Accuracy**: **0.7881 $\pm$ 0.0098**
- **EXP52 Baseline Outer Mean**: **0.7873 $\pm$ 0.0102**
- **Unbiased Nested Delta ($\Delta_{\text{nested}}$)**: **+0.0008** (+0.08% pts)

> [!WARNING]
> While global post-hoc sweeping shows a point-estimate bump of $+0.0033$ at $w=0.15$, strict nested evaluation reveals that the **true out-of-sample expected gain is only +0.0008**. Two out of five folds suffered degradation when applying the inner-selected weight.

---

## Statistical Analysis

Paired bootstrap distribution over 2,000 resamples for the primary comparisons:

### Comparison 1: EXP52 vs $w_{\text{style}} = 0.05$ (Conservative Peak)
- **Point Estimate Delta**: **+0.0026**
- **95% Bootstrap CI**: **[+0.0008, +0.0047]**
- **$P(\Delta > 0)$**: **99.95%**
- **Bootstrap Standard Error**: **0.0010**

### Comparison 2: EXP52 vs $w_{\text{style}} = 0.15$ (Global OOF Peak)
- **Point Estimate Delta**: **+0.0033**
- **95% Bootstrap CI**: **[-0.0004, +0.0074]**
- **$P(\Delta > 0)$**: **95.40%**
- **Bootstrap Standard Error**: **0.0020**

### Comparison 3: EXP52 vs Unbiased Nested Selector
- **Nested Delta**: **+0.0008** (Statistically indistinguishable from parity)

---

## Error Analysis: Which Mistakes Does Stylometry Fix?

Analyzing individual class recall changes between pure EXP52 and the best fused model ($w_{\text{style}} = 0.15$):

| Class Index | Candidate LLM | EXP52 Recall | Fused Recall | Net Recall Delta | Stylometry Error Recovery Behavior |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0 | `claude-fable-5` | 74.5% | 75.2% | **+0.7%** | Corrects punctuation boundary |
| 1 | `claude-opus-4-6` | 90.0% | 89.3% | -0.7% | Minor boundary confusion |
| 2 | `claude-opus-4-8` | 80.7% | 80.7% | 0.0% | Neutral |
| 3 | `claude-opus-5` | 86.7% | 88.0% | **+1.3%** | Capitalization and header cues |
| 4 | `claude-sonnet-5` | 80.0% | 80.0% | 0.0% | Neutral |
| 5 | `gemini-3.1-pro-preview` | 85.3% | 86.0% | **+0.7%** | Strong formatting marker consistency |
| 6 | `gemini-3.6-flash` | 79.3% | 79.3% | 0.0% | Neutral |
| 7 | `gemini-3.7-flash` | 69.3% | 71.3% | **+2.0%** | Sibling disambiguation vs 3.8-flash |
| 8 | `gemini-3.8-flash` | 77.3% | 77.3% | 0.0% | Neutral |
| 9 | `glm-5.3` | 61.3% | 61.3% | 0.0% | Neutral |
| 10 | `glm-5.3-flash` | 69.3% | 70.0% | **+0.7%** | Code fence / punctuation signature |
| 11 | `gpt-5.5` | 82.7% | 83.3% | **+0.7%** | Resolves subtle GPT-5 family ties |
| 12 | `gpt-5.6-luna` | 75.3% | 76.0% | **+0.7%** | Character n-gram tokenization boundaries |
| 13 | `gpt-5.6-sol` | 71.3% | 70.7% | -0.7% | Mild confusion with Luna |
| 14 | `gpt-5.6-terra` | 74.0% | 74.7% | **+0.7%** | Resolves Luna vs Terra tie |
| 15 | `gpt-6-astra` | 92.0% | 92.0% | 0.0% | Saturated |
| 16 | `kimi-k3` | 79.3% | 80.0% | **+0.7%** | Distinct Chinese/English tokenization |
| 17 | `qwen3.8-max` | 88.7% | 88.7% | 0.0% | Neutral |

The specialist provides modest gains in specific hard sibling pairs (`gemini-3.7-flash` +2.0%, `claude-opus-5` +1.3%), but slightly degrades `claude-opus-4-6` (-0.7%) and `gpt-5.6-sol` (-0.7%).

---

## Limitations

1. **Local OOF $\neq$ Hidden Kaggle Generalization**:
   As demonstrated definitively by EXP49 (where local BA was 0.7028 but Kaggle score dropped to 0.612), surface n-gram features are highly sensitive to prompt vocabulary shifts. Unseen test prompts on Kaggle do not share the exact character n-gram distribution of `train.csv`.
2. **Diminishing Returns on Deep Foundation Ensembles**:
   Unlike early baseline models (where TF-IDF lifted shallow regressors by +0.15), EXP52 is an ensemble of two massive frontier transformers (8B + 14B) scoring 0.7873 local BA. Its representations already encode subtle stylistic boundaries.
3. **Nested Generalization Fragility**:
   The unbiased nested validation delta is only **+0.0008**. Relying on a +0.0008 local signal to risk our 0.632 Kaggle standing violates competitive risk-adjusted decision rules.

---

## Verdict

In accordance with Section 14 of the competition engineering protocol:

> **DECISION**: **YELLOW**

- **Justification**: While predefined OOF fusion shows an apparent small improvement (+0.0026 at $w=0.05$, +0.0033 at $w=0.15$), strict nested validation reveals that the true out-of-sample expected gain collapses to **+0.0008 BA**, and the 95% bootstrap confidence interval at the peak crosses zero ($[-0.0004, +0.0074]$). Given the documented failure mode of surface n-gram domain shift on Kaggle (EXP49), this evidence does not justify altering our production pipeline.
- **Kaggle Submission**: **NOT SUBMITTED**.
- **Production Baseline**: **EXP52 remains frozen at Kaggle Public LB 0.632**.

---

## Visualizations Generated

- [`reports/figures/exp55_weight_sweep.png`](file:///home/rj2826/name-the-llm/reports/figures/exp55_weight_sweep.png) — Weight sweep curve with 95% bootstrap CI and paired delta plot.
- [`reports/figures/exp55_confusion_matrix.png`](file:///home/rj2826/name-the-llm/reports/figures/exp55_confusion_matrix.png) — Normalized confusion matrix of standalone stylometry.
- [`reports/figures/exp55_complementarity.png`](file:///home/rj2826/name-the-llm/reports/figures/exp55_complementarity.png) — Per-class recall deltas demonstrating where stylometry alters EXP52 decisions.
