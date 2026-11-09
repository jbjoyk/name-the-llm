# EXP56: Qwen3-14B Prompt-Conditioned Residual Specialist Report

**Experiment ID**: `EXP56_q14_residual`  
**Date**: 2026-11-08  
**Author**: Antigravity Machine Learning Research Agent  
**Parent Baseline**: EXP52 (Dual-LoRA Qwen3-8B $0.60$ + Qwen3-14B $0.40$, Local OOF BA: **0.7873**, Kaggle Public LB: **0.632**)  
**Predecessor Experiments**: EXP47 (LB 0.620), EXP50 (OOF BA 0.6102), EXP51A (OOF Provenance Verified), EXP52 (LB 0.632), EXP53 (OOF BA 0.7873), EXP54 (OOF BA 0.7888), EXP55 (OOF BA 0.7881)  
**Primary Metric**: Macro-averaged Balanced Accuracy across 18 candidate LLM classes  
**Validation Strategy**: Strict 5-Fold Prompt-Disjoint `GroupKFold(groups=prompt_id)` on `data/train.csv` (2,699 samples, 150 unique prompts, 30 prompts per fold)  
**Nested Validation Strategy**: 5-Fold outer prompt-disjoint CV with inner 4-fold selection of fusion weights  
**Kaggle Submission Status**: **ABSOLUTELY NOT SUBMITTED — ZERO SUBMISSIONS**  
**Production Baseline**: **EXP52 REMAINS FROZEN AT KAGGLE PUBLIC LB 0.632**  
**EXP56 Verdict**: **RED** (Duplicate of existing `q14v5 head_centered` + No meaningful improvement; nested CV exhibits out-of-sample degradation; zero submissions)

---

## 1. Executive Summary

EXP56 tests a specific architectural hypothesis:
> *"The stronger q14v5 Qwen3-14B representation contains prompt-family information that can be removed through prompt centering, exposing residual model-specific information that may complement the frozen EXP52 ensemble."*

Through forensic codebase auditing, architectural inspection of `models/adapters_v5/q14v5/q14v5/heads.pt`, and rigorous 5-fold prompt-disjoint evaluation across all 2,699 training passages, we arrived at two definitive empirical findings:

1. **The Proposed Hypothesis Is Already Implemented by `q14v5 head_centered`**:
   The existing `q14v5` model family already contains a dedicated, pre-trained prompt-centered classification head (`centered.weight` $[18, 40960]$, `centered.bias` $[18]$). It was trained alongside `plain` using a multi-task joint cross-entropy objective:
   $$\mathcal{L} = \mathcal{L}_{\text{plain}}(z) + \mathcal{L}_{\text{centered}}(z - \bar{z}_{\text{prompt}})$$
   When evaluated standalone on prompt-disjoint out-of-fold folds, `q14v5 head_centered` achieves **0.4842 Balanced Accuracy**. When combined with `head_plain` at the standard twin weight ($w_{\text{twin}}=0.35$), it lifts Qwen3-14B from **0.6346 to 0.6491 BA** (+0.0145 lift). Crucially, **this exact centered twin head is ALREADY active in our frozen EXP52 baseline** (`twin_weight = 0.35` for both Q8 and Q14).

2. **The Training-Cluster Residual Model (Model D) Suffers from a Structural Mismatch**:
   When implementing a "new residual specialist" using training-only prompt cluster centroids (as specified in Sections 3–5), the model suffers a severe conceptual failure under strict prompt-disjoint validation. Because validation prompts are 100% disjoint from training prompts, held-out validation passages do not belong to any training cluster. Only 12.4% (336 / 2,699) of validation rows exhibit TF-IDF cosine similarity $\ge 0.15$ with training centroids, and those matches are spurious lexical artifacts. Subtracting unrelated training cluster means injects topic noise, collapsing standalone Balanced Accuracy to **0.1045** (barely above random chance $1/18 = 0.0556$).
   
3. **Nested Generalization Degradation**:
   While post-hoc OOF sweeping against EXP52 suggests a tiny cosmetic bump (+0.0015 BA at $w=0.20$), selection-safe 5-fold nested cross-validation produces an out-of-sample degradation:
   $$\text{Nested BA} = 0.7866 \pm 0.0113 \quad (\Delta_{\text{nested}} = -0.0007 \text{ vs EXP52 } 0.7873)$$
   The 95% bootstrap confidence interval crosses zero ($[-0.0030, +0.0062]$).

Per the competition decision protocol, EXP56 is classified as **RED**. No modifications will be made to EXP52, and zero submissions are permitted.

---

## 2. Technical Audit of Existing `q14v5` Heads (`head_plain` vs `head_centered`)

In accordance with Section 2, we conducted a line-by-line inspection of the training kernel source (`lucasboesen/name-that-llm-train-lora24-s7.ipynb`), inference scripts (`infer.py`, `src/exp52_dual_lora_inference.py`), and the serialized checkpoint tensors (`models/adapters_v5/q14v5/q14v5/heads.pt`):

| Technical Specification | `head_plain` | `head_centered` |
| :--- | :--- | :--- |
| **Representation Used** | Truncated Qwen3-14B base (27 decoder layers), 8 pooling depths $[2, 4, 6, 9, 13, 17, 22, 27]$, token 0 masked (`m[:, 0] = 0`), masked mean pooling per layer, concatenated: $Z \in \mathbb{R}^{B \times 40960}$. | Identical 40,960-dimensional pooled trunk representation $Z$. |
| **LayerNorm Application** | Dedicated `torch.nn.LayerNorm(5120)` applied to each pooled layer output **before** concatenation. | Identical. Normalization occurs per layer before concatenation into $Z$. |
| **Centering Timing** | No centering. Operates on raw normalized pooled features $Z$. | Centering occurs **after** LayerNorm on the concatenated 40,960-dimensional tensor: $Z_{\text{cent}} = Z - \bar{Z}_{\text{prompt}}$. |
| **Prompt Clustering (Training)** | Ground truth prompt IDs from training metadata (`pid[rows] == p`). | Ground truth prompt IDs. Microbatches of 3 completions from the exact same prompt (`triplets(fit, rng)`). No heuristic clustering needed during training. |
| **Cluster Construction (Training)** | Sampled strictly from `fit = fold != int(job)`. Held-out validation rows completely excluded. | Same microbatches. $\bar{Z}_{\text{batch}} = \frac{1}{3}\sum_{j=1}^3 Z_j$ computed strictly over training microbatch samples. |
| **Validation Leakage** | ZERO. Models trained strictly on `fold != k`. | ZERO. Centering statistics during training were computed strictly inside training microbatches. |
| **Classifier Architecture** | `nn.Linear(40960, 18)` (`plain.weight` $[18, 40960]$, `plain.bias` $[18]$). | `nn.Linear(40960, 18)` (`centered.weight` $[18, 40960]$, `centered.bias` $[18]$). |
| **Training Loss** | Standard Cross-Entropy on trunk predictions: $\mathcal{L}_{\text{plain}} = \text{CE}(\text{plain}(\text{drop}(Z)), y)$. | Cross-Entropy on residual features: $\mathcal{L}_{\text{centered}} = \text{CE}(\text{centered}(\text{drop}(Z - \bar{Z}_{\text{batch}})), y)$. |
| **Joint Optimization** | Jointly trained: $\mathcal{L} = \mathcal{L}_{\text{plain}} + \mathcal{L}_{\text{centered}}$ with AdamW (adapters lr 1e-4, heads lr 1e-3, wd 0.01, 3 epochs). | Jointly trained under the exact same multi-task gradient updates. |
| **Test/Inference Mechanism** | Evaluated on test passages: $\text{logits}_{\text{plain}} = Z W_p^T + b_p$. | Test passages clustered via unsupervised content-word TF-IDF capped agglomerative clustering (`capped_agglo`, thr=0.15, cap=18). For clusters $|g| \ge 2$, $Z_{\text{cent}} = Z - \bar{Z}_g$, and $\text{logits}_{\text{cent}} = Z_{\text{cent}} W_c^T + b_c$. |
| **Integration in EXP52** | Primary member: $0.40 \log P_{q14}$. | Twin head active: $\text{LP}_{\text{total}} = \text{LP}_{\text{plain}} + 0.35 \cdot \text{LP}_{\text{cent}}$ for clustered test passages. |

> [!IMPORTANT]
> **Audit Finding (Section 2 Trigger)**:
> The existing `q14v5 head_centered` **already implements the exact prompt-centering hypothesis**. It was trained explicitly to classify passages from $Z - \mu_{\text{prompt}}$, removing the prompt effect to isolate residual model signatures. Furthermore, `head_centered` is already wired into the production EXP52 inference engine at $w_{\text{twin}}=0.35$.

---

## 3. Detailed Comparison of Required Baselines

Under strict 5-fold prompt-disjoint validation on `data/train.csv` (150 prompts, 30 prompts per fold):

| Model / Architecture | Balanced Accuracy | Macro-F1 | Accuracy | Role / Description |
| :--- | :---: | :---: | :---: | :--- |
| **A. EXP47 Qwen3-8B** | **0.5910** | 0.5915 | 0.5910 | Qwen3-8B LoRA24 with Centred Twin ($w_{\text{twin}}=0.35$) |
| **B. q14v5 Plain Head** | **0.6346** | 0.6338 | 0.6347 | Qwen3-14B LoRA24 standalone plain readout (`plain.weight`) |
| **C. q14v5 Centered Head (Standalone)** | **0.4842** | 0.4801 | 0.4843 | `centered.weight` evaluated on prompt-centered validation features |
| **C+B. q14v5 Plain + Centered Twin** | **0.6491** | **0.6485** | **0.6491** | Plain head fused with centered twin ($w_{\text{twin}}=0.35$) (**+0.0145 over plain**) |
| **D. NEW Residual Specialist (Sections 3-5)** | **0.1045** | 0.0978 | 0.1045 | Training-cluster residual specialist (collapsed due to domain mismatch) |
| **E. EXP52 Production Baseline** | **0.7873** | **0.7880** | **0.7873** | Dual-LoRA ($0.60 \text{EXP47} + 0.40 \text{q14v5}$) — **Current Best** |

### Key Observations:
1. **The Efficacy of Intra-Prompt Centering**: In Baseline C, prompt-centering within the same prompt cluster lifts single-model Q14 performance from **0.6346 to 0.6491 BA** (+1.45 percentage points). This confirms that removing prompt-level bias is highly beneficial when applied to completions of the *same* prompt.
2. **The Collapse of Training-Cluster Matching**: Model D achieves only **0.1045 BA**. Because validation prompts are prompt-disjoint, matching a validation prompt to an unrelated training cluster centroid subtracts arbitrary topic means, destroying discriminative signal.

---

## 4. Complementarity Analysis: Model D vs EXP52

A contingency analysis between the predictions of EXP52 (0.7873 BA) and Model D (0.1045 BA) across all 2,699 samples:

| Category | Sample Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Agreement Rate** | 297 / 2,699 | 11.00% | Samples where EXP52 and Model D predict identical LLMs |
| **Both Correct** | 241 | 8.93% | Mutual true positives |
| **Both Wrong** | 533 | 19.75% | Mutual blind spots |
| **EXP52-only Correct** | 1,884 | 69.80% | Deep foundation models correct, Model D fails |
| **Residual-only Correct** | 41 | 1.52% | Samples where Model D uniquely corrects EXP52 |
| **Oracle Accuracy Ceiling** | 2,166 / 2,699 | **80.25%** | Theoretical ceiling if routing were perfect |
| **Oracle Balanced Accuracy** | — | **0.8025** | Macro-recall ceiling (+0.0152 over EXP52) |
| **Measured Error Correlation ($r$)** | — | **0.05616** | **Low measured Pearson error correlation** ($p = 0.0035$) |

> [!NOTE]
> We report $r = 0.05616$ strictly as **low measured Pearson error correlation**. We do not claim statistical independence. Model D's low correlation arises simply because its error rate is extremely high (~89.5%), providing negligible recovery pool (only 41 samples).

---

## 5. Predefined Log-Probability Fusion Sweep

Evaluating fixed log-probability blending across the 6 predefined weights:
$$\text{LP}_{\text{fused}} = (1 - w) \cdot \text{LP}_{\text{EXP52}} + w \cdot \text{LP}_{\text{RESIDUAL}}$$

| $w_{\text{res}}$ | $w_{\text{exp52}}$ | OOF BA | Macro-F1 | Accuracy | Net Flipped Samples | Delta vs EXP52 | 95% Bootstrap CI | $P(\Delta > 0)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | **1.00** | **0.7873** | 0.7880 | 0.7873 | 0 (Base) | **0.0000** | [0.0000, 0.0000] | 0.0000 |
| **0.02** | **0.98** | **0.7877** | 0.7884 | 0.7877 | +1 | **+0.0004** | [-0.0015, +0.0023] | 0.6235 |
| **0.05** | **0.95** | **0.7873** | 0.7881 | 0.7873 | 0 | **+0.0000** | [-0.0024, +0.0024] | 0.4890 |
| **0.10** | **0.90** | **0.7884** | 0.7892 | 0.7884 | +3 | **+0.0011** | [-0.0023, +0.0047] | 0.7305 |
| **0.15** | **0.85** | **0.7881** | 0.7889 | 0.7881 | +2 | **+0.0007** | [-0.0033, +0.0050] | 0.6250 |
| **0.20** | **0.80** | **0.7888** | 0.7896 | 0.7888 | +4 | **+0.0015** | [-0.0030, +0.0062] | 0.7445 |

### Empirical Findings:
1. Across the entire sweep $w \in [0.00, 0.20]$, the maximum apparent delta is only **+0.0015 BA** (at $w=0.20$), representing a net flip of only 4 samples out of 2,699.
2. In every non-zero weight configuration, the 95% bootstrap confidence interval comfortably includes zero (e.g. $[-0.0030, +0.0062]$ at $w=0.20$).
3. The probability of genuine positive lift $P(\Delta > 0)$ is only 74.45%, falling far short of statistical confidence.

---

## 6. Selection-Safe 5-Fold Nested Cross-Validation

To determine whether the post-hoc OOF delta represents genuine generalization or selection noise, we conducted an end-to-end 5-fold **nested validation**. For each outer fold, the optimal weight $w^*$ was selected strictly on the remaining 4 training folds, then evaluated out-of-sample on the held-out test fold:

| Outer Fold | Held-Out Size | Selected Inner Weight ($w^*$) | Inner Train BA | Outer Test BA | Baseline Outer BA | Outer Fold Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 540 rows | **0.15** | 0.7911 | 0.7759 | 0.7778 | **-0.0019** |
| **Fold 1** | 539 rows | **0.20** | 0.7903 | 0.7831 | 0.7793 | **+0.0038** |
| **Fold 2** | 540 rows | **0.15** | 0.7883 | 0.7870 | 0.7907 | **-0.0037** |
| **Fold 3** | 540 rows | **0.10** | 0.7902 | 0.7815 | 0.7833 | **-0.0019** |
| **Fold 4** | 540 rows | **0.10** | 0.7841 | 0.8056 | 0.8056 | **+0.0000** |

### Nested Validation Summary:
- **Nested Mean Balanced Accuracy**: **0.7866 $\pm$ 0.0113**
- **EXP52 Baseline Outer Mean**: **0.7873 $\pm$ 0.0102**
- **Unbiased Nested Delta ($\Delta_{\text{nested}}$)**: **-0.0007** (-0.07 percentage points)

> [!WARNING]
> In 3 out of 5 outer folds (Folds 0, 2, 3), applying the inner-selected weight resulted in an **out-of-sample degradation** on held-out prompts. The true unbiased generalization delta is negative (-0.0007).

---

## 7. Diagnostic Class-Wise Recall Analysis

Examining individual class recall changes between pure EXP52 and the $w=0.20$ fused configuration:

| Class Index | Candidate LLM | EXP52 Recall | Fused Recall | Recall Delta | Diagnostic Behavior |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0 | `claude-fable-5` | 74.5% | 74.5% | 0.0% | Neutral |
| 1 | `claude-opus-4-6` | 90.0% | 89.3% | -0.7% | Slight degradation |
| 2 | `claude-opus-4-8` | 80.7% | 80.7% | 0.0% | Neutral |
| 3 | `claude-opus-5` | 86.7% | 87.3% | +0.7% | Minor tie-break |
| 4 | `claude-sonnet-5` | 80.0% | 80.0% | 0.0% | Neutral |
| 5 | `gemini-3.1-pro-preview` | 85.3% | 86.0% | +0.7% | Minor tie-break |
| 6 | `gemini-3.6-flash` | 79.3% | 79.3% | 0.0% | Neutral |
| 7 | `gemini-3.7-flash` | 69.3% | 70.0% | +0.7% | Minor tie-break |
| 8 | `gemini-3.8-flash` | 77.3% | 77.3% | 0.0% | Neutral |
| 9 | `glm-5.3` | 61.3% | 61.3% | 0.0% | Neutral |
| 10 | `glm-5.3-flash` | 69.3% | 70.0% | +0.7% | Minor tie-break |
| 11 | `gpt-5.5` | 82.7% | 82.7% | 0.0% | Neutral |
| 12 | `gpt-5.6-luna` | 75.3% | 76.0% | +0.7% | Minor tie-break |
| 13 | `gpt-5.6-sol` | 71.3% | 71.3% | 0.0% | Neutral |
| 14 | `gpt-5.6-terra` | 74.0% | 74.0% | 0.0% | Neutral |
| 15 | `gpt-6-astra` | 92.0% | 92.0% | 0.0% | Saturated |
| 16 | `kimi-k3` | 79.3% | 79.3% | 0.0% | Neutral |
| 17 | `qwen3.8-max` | 88.7% | 88.7% | 0.0% | Neutral |

The residual specialist alters decisions on only 5 classes (+0.7% on 4, -0.7% on 1), reflecting random boundary perturbation rather than systematic structural error correction.

---

## 8. Final Verdict & Decision

In accordance with Section 12 of the competition protocol:

> **DECISION**: **RED**

### Justification:
1. **Duplicate of Existing Architecture (Section 2 Trigger)**:
   The hypothesis that Qwen3-14B representations contain prompt-family information that can be removed via prompt centering is **already covered by `q14v5 head_centered`**. It is physically implemented in `heads.pt`, lifts standalone Q14 BA by +0.0145 to 0.6491, and is already active in the production EXP52 baseline.
2. **Structural Failure of Training-Cluster Matching**:
   Attempting to center validation rows using training-only cluster centroids suffers from prompt-disjoint mismatch (only 12.4% match threshold, producing 0.1045 standalone BA).
3. **Out-of-Sample Nested Degradation**:
   Strict selection-safe nested validation yields $\Delta_{\text{nested}} = -0.0007$, and the 95% bootstrap confidence interval crosses zero ($[-0.0030, +0.0062]$).
4. **Policy Enforcement**:
   - **Kaggle Submission**: **NOT SUBMITTED**.
   - **Production Baseline**: **EXP52 remains frozen at Kaggle Public LB 0.632**.

---

## 9. Visualizations Generated

- [`reports/figures/exp56_weight_sweep.png`](file:///home/rj2826/name-the-llm/reports/figures/exp56_weight_sweep.png) — Fusion weight sweep curve with 95% bootstrap CI and paired delta plot.
- [`reports/figures/exp56_complementarity.png`](file:///home/rj2826/name-the-llm/reports/figures/exp56_complementarity.png) — Per-class recall deltas demonstrating minimal boundary impact.
- [`reports/figures/exp56_confusion_matrix.png`](file:///home/rj2826/name-the-llm/reports/figures/exp56_confusion_matrix.png) — Normalized confusion matrix of Model D.
