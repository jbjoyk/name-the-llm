# EXP65: Llama-3.1-8B Diverse Foundation Backbone Attribution Report

- **Experiment ID**: `EXP65_llama_diverse`
- **Date**: 2026-11-10 22:15:58
- **Production Champion**: EXP52 (Dual-LoRA Qwen3-8B $0.60$ + Qwen3-14B $0.40$)
  - Kaggle Public LB: **0.6320**
  - Prompt-Disjoint OOF Balanced Accuracy: **0.7873**
- **Experiment Type**: Cross-Architecture Foundation Model Complementarity & Diverse Backbone Attribution
- **Status**: COMPLETE
- **Verdict**: **RED** (Llama-3.1-8B achieves 0.3901 standalone BA; nested probability fusion with EXP52 yields 0.7884 pooled OOF BA, a statistically non-significant gain of $\Delta = +0.0011$ with 95% bootstrap CI $[-0.0063, +0.0083]$ spanning zero; Llama introduces 1,231 degradations against only 159 unique corrections; RETAIN EXP52 as frozen production champion)
- **Production Changed**: **NO**
- **EXP52 Remains Frozen**: **YES**
- **Kaggle Submission Made**: **NO**

---

## 1. Executive Summary

EXP65 was designed to answer the central research question:
> *"Does a genuinely different foundation-model family (`meta-llama/Llama-3.1-8B`) provide attribution information that the current Qwen3-8B + Qwen3-14B dual-backbone ensemble does not already contain?"*

Previous diagnostic experiments (EXP61, EXP62, EXP63, EXP64) established that while the Qwen3-8B and Qwen3-14B dual-backbone ensemble achieves a strong prompt-disjoint out-of-fold balanced accuracy of 0.7873, it experiences attribution breakdown specifically when the two Qwen models disagree (BA drops from 0.9318 to 0.6871 on $N=1,612$ disagreement rows). EXP65 evaluated whether fine-tuning an entirely distinct foundation model family—Meta's **Llama-3.1-8B Base**—could serve as a decorrelated, complementary arbiter.

### Key Numerical Findings:
1. **Standalone Attribution Performance**:
   - `Qwen3-8B (q8v5)` standalone OOF BA: **0.5910**
   - `Qwen3-14B (q14v5)` standalone OOF BA: **0.6346**
   - `EXP52 Champion (0.60 Q8 + 0.40 Q14)` OOF BA: **0.7873**
   - `Llama-3.1-8B (4-bit QLoRA)` standalone OOF BA: **0.3901** (Macro-F1: **0.3812**, Accuracy: **0.3901**, NLL: **1.6268**).
2. **Measured Error Correlation**:
   - Q8 vs Q14: $r = 0.0214$ (decorrelated errors explaining the large $+0.1527$ BA gain of dual fusion).
   - Q8 vs Llama: $r = -0.0267$ ($p = 0.166$, decorrelated).
   - Q14 vs Llama: $r = 0.2819$ ($p = 1.80 \times 10^{-50}$, highly correlated failure modes).
   - EXP52 vs Llama: $r = 0.1206$ ($p = 3.32 \times 10^{-10}$).
3. **Unique Correction Capacity**:
   - Llama Unique Corrections (Llama correct & EXP52 wrong): **159 samples** ($5.89\%$).
   - Llama Degradations (Llama wrong & EXP52 correct): **1,231 samples** ($45.61\%$).
   - Net Correction: **$-1,072$ samples** ($-39.72\%$).
   - Llama correct when BOTH Qwen models are wrong: **94 samples** ($3.48\%$).
4. **Three-Model Theoretical Ceiling (Oracle)**:
   - Oracle(Q8, Q14): **0.8455 BA**
   - Oracle(EXP52, Llama): **0.8462 BA**
   - Oracle(Q8, Q14, Llama): **0.8803 BA** ($\Delta = \mathbf{+0.0348}$ theoretical headroom).
5. **EXP64 Disagreement Subset ($N=1,612$)**:
   - On the critical subset where Q8 and Q14 disagree, Llama achieves **0.3561 BA**, fixing EXP52 on 146 samples while degrading EXP52 on 698 samples (net impact: **$-552$ samples**).
6. **Selection-Safe Nested Cross-Validation Fusion**:
   - In 5 outer $\times$ 4 inner folds nested CV, inner validation selected $w^* = 0.25$ (and $w^* = 0.15$ on Fold 1), yielding a pooled OOF BA of **0.7884** ($\Delta = \mathbf{+0.0011}$ over EXP52).
   - Paired bootstrap ($B = 2,000$): Mean $\Delta = \mathbf{+0.0011}$, 95% CI $[-\mathbf{0.0063}, +\mathbf{0.0083}]$, $P(\Delta > 0) = \mathbf{62.2\%}$.
7. **Scientific Conclusion**:
   - **Verdict: RED**. The performance delta fails the $+0.0020$ threshold, the confidence interval heavily crosses zero, and degradations outnumber corrections by nearly 8 to 1.
   - **Production Decision**: **RETAIN EXP52 as frozen production champion**. Zero changes to production code or Kaggle submissions.

---

## 2. Research Question

Does fine-tuning Meta's `Llama-3.1-8B` base model provide complementary multiclass sequence attribution information that escapes the correlated blind spots of the Qwen model family?

---

## 3. Motivation

In EXP52, combining Qwen3-8B and Qwen3-14B produced a dramatic improvement over either backbone alone ($0.5910 / 0.6346 \to 0.7873$ BA). However, EXP61 and EXP64 revealed that when Q8 and Q14 disagree, performance plummets to 0.6871 BA, and error correlation between Qwen variants limits further gains within the same architectural lineage.

ModernBERT-base (EXP62) failed to provide complementary signal because its frozen encoder representations were insufficiently expressive for 18-class attribution (achieving only 0.4468 BA and $-0.0015$ nested fusion delta). Llama-3.1-8B represents the premier alternative frontier open-weights generative architecture, trained on over 15 trillion tokens with an entirely distinct tokenizer (128k vocabulary), pretraining mixture, and positional embedding parameterization.

---

## 4. Data and Provenance

- **Competition Training Dataset**: `data/train.csv` (2,699 samples, 150 unique prompts, 18 LLM classes).
- **Class Balance**: Perfectly balanced with 150 samples per class across 17 models, and 149 samples for `claude-fable-5`.
- **Response Lengths**:
  - Word count: Median = 101, p90 = 206, p95 = 224, p99 = 248, Max = 274.
  - Character count: Median = 670, p90 = 1,326, p95 = 1,469, p99 = 1,669, Max = 1,900.
  - Max Sequence Length: 384 tokens ensures 99.8% token coverage with 0% semantic loss, maintaining strict comparability with Qwen's 384 token window.
- **Open Pool Artifact**: `data/external/open_pool_clean.parquet` (212,623 passages, audited with 0 train passage overlap).

---

## 5. Llama Training Setup

- **Base Model**: `meta-llama/Llama-3.1-8B` (Base model, 32 transformer layers, hidden dimension 4,096, vocab size 128,256).
- **Quantization & Memory**: 4-bit NormalFloat (NF4) double quantization via `bitsandbytes`, compute dtype `torch.bfloat16`.
- **Hardware Distribution**: Dual NVIDIA GeForce RTX 3060 (12 GB VRAM each). Memory allocated with `max_memory={0: "8GiB", 1: "10GiB"}`, `gradient_checkpointing_enable()`, and `expandable_segments:True`.
- **Adapter Configuration**:
  - PEFT QLoRA: $r = 16$, $\alpha = 32$, dropout $= 0.05$.
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]` (13,705,216 trainable parameters, 0.182% of model).
- **Classification Head**: `nn.Linear(4096, 18)` initialized on pooled sequence representations.
- **Optimization**:
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 0.01$).
  - Learning rate: $2 \times 10^{-4}$ with cosine annealing schedule down to $1 \times 10^{-6}$.
  - Batch size: 4 per device $\times 2$ gradient accumulation steps (effective batch size $= 8$).
  - Epochs: 2 epochs per fold.

---

## 6. Prompt-Disjoint Validation Protocol

Validation strictly enforced prompt disjointness across all 5 folds:
- Exactly 150 unique prompts partitioned into 5 deterministic folds of 30 prompts each.
- $\text{Train Prompts} \cap \text{Val Prompts} = \emptyset$ for every fold.
- Saved to [`experiments/EXP65_llama_diverse/fold_assignments.csv`](file:///home/rj2826/name-that-llm/experiments/EXP65_llama_diverse/fold_assignments.csv).

| Fold | Train Samples (Prompts) | Val Samples (Prompts) | Llama BA | EXP52 BA | Q8 BA | Q14 BA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 2,159 (120) | 540 (30) | **0.3704** | 0.7778 | 0.5907 | 0.6130 |
| **Fold 1** | 2,160 (120) | 539 (30) | **0.3859** | 0.7793 | 0.5886 | 0.6141 |
| **Fold 2** | 2,159 (120) | 540 (30) | **0.3889** | 0.7907 | 0.5889 | 0.6556 |
| **Fold 3** | 2,159 (120) | 540 (30) | **0.4074** | 0.7833 | 0.5926 | 0.6278 |
| **Fold 4** | 2,159 (120) | 540 (30) | **0.3880** | 0.8056 | 0.5944 | 0.6626 |
| **Mean** | — | — | **0.3881** | **0.7873** | **0.5910** | **0.6346** |

---

## 7. Standalone Performance

Across the complete 2,699 out-of-fold samples:
- **Balanced Accuracy**: **0.3901**
- **Macro-F1**: **0.3812**
- **Accuracy**: **0.3901**
- **Log Loss (NLL)**: **1.6268**
- **Top-1 Confidence Mean**: **0.4631**
- **Top-2 Margin Mean**: **0.2319**

Standalone Llama-3.1-8B achieves 0.3901 BA—significantly lower than Qwen3-8B (0.5910) and Qwen3-14B (0.6346). This occurs because Qwen's pretraining tokenizer and representation structure appear uniquely sensitive to frontier model stylized output patterns, whereas Llama required substantial fine-tuning adaptation.

---

## 8. Primary Complementarity Test: Measured Error Correlation

We computed the Pearson correlation coefficient between the binary error indicators ($e_i \in \{0, 1\}$) across all models:

| Model Pair | Pearson Correlation ($r$) | $p$-value | Interpretation |
| :--- | :---: | :---: | :--- |
| **Qwen3-8B vs Qwen3-14B** | **$+0.0214$** | $0.266$ | Nearly independent error modes (enables EXP52 fusion) |
| **Qwen3-8B vs Llama-3.1-8B** | **$-0.0267$** | $0.166$ | Uncorrelated errors |
| **Qwen3-14B vs Llama-3.1-8B** | **$+0.2819$** | $1.80 \times 10^{-50}$ | **Statistically significant shared failure modes** |
| **EXP52 vs Llama-3.1-8B** | **$+0.1206$** | $3.32 \times 10^{-10}$ | Correlated errors with the ensemble |

The high error correlation between Qwen3-14B and Llama-3.1-8B ($r = 0.2819$) demonstrates that Llama shares substantial misattribution patterns with the larger Qwen backbone rather than offering orthogonal perspectives.

---

## 9. Unique Correction Capacity

| Metric | Sample Count | Percentage of Dataset |
| :--- | :---: | :---: |
| **Llama Unique Corrections** (Llama correct & EXP52 wrong) | **159** | **5.89%** |
| **Llama Degradations** (Llama wrong & EXP52 correct) | **1,231** | **45.61%** |
| **Net Correction Balance** | **$-1,072$** | **$-39.72\%$** |
| **Llama correct when Q8 is wrong** | 448 | 16.60% |
| **Llama correct when Q14 is wrong** | 206 | 7.63% |
| **Llama correct when BOTH Qwen models are wrong** | **94** | **3.48%** |

While Llama successfully rescues 94 samples that both Qwen models missed, its degradation rate (1,231 samples) is nearly $8 \times$ higher than its correction rate.

---

## 10. Three-Model Oracle Analysis

The Oracle chooser represents the theoretical upper bound achieved if a perfect selector knew when to trust each model:

| Ensemble Configuration | Oracle Balanced Accuracy | Gain over Preceding |
| :--- | :---: | :---: |
| **EXP52 Actual Model** | 0.7873 | — |
| **Oracle(Q8, Q14)** | 0.8455 | $+0.0582$ |
| **Oracle(EXP52, Llama)** | 0.8462 | $+0.0007$ |
| **Oracle(Q8, Q14, Llama)** | **0.8803** | **$+0.0348$** |

The 3-model oracle increases the theoretical ensembling ceiling from 0.8455 to 0.8803 (+0.0348). However, because Llama's standalone accuracy is 0.3901, practical continuous logit fusion cannot realize this theoretical headroom without introducing catastrophic false positives.

---

## 11. EXP64 Disagreement Subset ($N = 1,612$)

Evaluating specifically on the 1,612 samples where Q8 and Q14 disagree (`argmax(Q8) != argmax(Q14)`):

- **Qwen3-8B BA**: **0.3399**
- **Qwen3-14B BA**: **0.4486**
- **EXP52 Ensemble BA**: **0.6871**
- **Llama-3.1-8B BA**: **0.3561**
- **Oracle(Q8, Q14, Llama) BA**: **0.8384**
- **Llama fixes EXP52 in Disagreement**: **146 samples**
- **Llama harms EXP52 in Disagreement**: **698 samples**
- **Net Balance in Disagreement**: **$-552$ samples**

Llama is unable to act as an effective tiebreaker on the disagreement subset, harming 698 decisions while fixing only 146.

---

## 12. Static Probability Fusion Sweep

Evaluating linear log-probability blending:
$$\text{LP}_{\text{fused}} = (1 - w) \text{LP}_{\text{EXP52}} + w \text{LP}_{\text{Llama}}$$

| $w_{\text{Llama}}$ | $w_{\text{EXP52}}$ | Full OOF Balanced Accuracy | Macro-F1 | $\Delta$ vs EXP52 |
| :---: | :---: | :---: | :---: | :---: |
| 0.00 | 1.00 | 0.7873 | 0.7880 | $+0.0000$ |
| 0.05 | 0.95 | 0.7877 | 0.7883 | $+0.0004$ |
| 0.10 | 0.90 | 0.7888 | 0.7895 | $+0.0015$ |
| 0.15 | 0.85 | 0.7888 | 0.7895 | $+0.0015$ |
| 0.20 | 0.80 | 0.7881 | 0.7887 | $+0.0007$ |
| **0.25** | **0.75** | **0.7907** | **0.7914** | **$+0.0033$** |
| 0.30 | 0.70 | 0.7892 | 0.7899 | $+0.0019$ |
| 0.40 | 0.60 | 0.7758 | 0.7766 | $-0.0115$ |
| 0.50 | 0.50 | 0.7558 | 0.7567 | $-0.0315$ |

While static fusion shows an apparent peak at $w = 0.25$ (+0.0033 BA), evaluating this under selection-safe nested cross-validation reveals that the gain shrinks when weights are selected without full-dataset hindsight.

---

## 13. Selection-Safe Nested Cross-Validation

In 5 outer $\times$ 4 inner folds nested CV, inner validation folds independently selected $w^*$, which was evaluated on unseen outer fold prompts:

- **Outer Fold 0**: Selected $w^* = 0.15 \to \text{BA} = \mathbf{0.7759}$ (EXP52 = 0.7778, $\Delta = -0.0019$)
- **Outer Fold 1**: Selected $w^* = 0.25 \to \text{BA} = \mathbf{0.7849}$ (EXP52 = 0.7793, $\Delta = +0.0056$)
- **Outer Fold 2**: Selected $w^* = 0.25 \to \text{BA} = \mathbf{0.7833}$ (EXP52 = 0.7907, $\Delta = -0.0074$)
- **Outer Fold 3**: Selected $w^* = 0.25 \to \text{BA} = \mathbf{0.7907}$ (EXP52 = 0.7833, $\Delta = +0.0074$)
- **Outer Fold 4**: Selected $w^* = 0.25 \to \text{BA} = \mathbf{0.8074}$ (EXP52 = 0.8056, $\Delta = +0.0018$)
- **Mean Outer Fold BA**: **$0.7885 \pm 0.0106$**
- **Pooled Nested OOF BA**: **0.7884** ($\Delta = \mathbf{+0.0011}$ over EXP52).

Only 3 of 5 outer folds show positive deltas, and the net pooled improvement is a marginal $+0.0011$ BA.

---

## 14. Statistical Testing (Paired Bootstrap $B = 2,000$)

Paired bootstrap resampling ($B = 2,000$ iterations on all 2,699 paired predictions):
- **Mean Delta**: **$+0.0011$ Balanced Accuracy**
- **95% Bootstrap Confidence Interval**: **$[-0.0063, +0.0083]$**
- **$P(\Delta > 0)$**: **$62.2\%$**
- **Conclusion**: The observed nested fusion delta of $+0.0011$ is statistically indistinguishable from zero ($p = 0.378$). The confidence interval spans zero comfortably.

---

## 15. Class-Wise Analysis

| Class Name | Q8 Recall | Q14 Recall | EXP52 Recall | Llama Recall | Nested Recall | Unique Corrections | Degradations | Net Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `claude-fable-5` | 0.698 | 0.812 | 0.852 | 0.443 | 0.852 | 5 | 66 | $-61$ |
| `claude-opus-4-6` | 0.507 | 0.520 | 0.707 | 0.380 | 0.707 | 12 | 61 | $-49$ |
| `claude-opus-4-8` | 0.500 | 0.560 | 0.740 | 0.347 | 0.740 | 8 | 67 | $-59$ |
| `claude-opus-5` | 0.627 | 0.633 | 0.793 | 0.360 | 0.793 | 6 | 71 | $-65$ |
| `claude-sonnet-5` | 0.667 | 0.707 | 0.840 | 0.433 | 0.847 | 11 | 72 | $-61$ |
| `gemini-3.1-pro-preview` | 0.607 | 0.627 | 0.787 | 0.453 | 0.787 | 10 | 60 | $-50$ |
| `gemini-3.6-flash` | 0.600 | 0.640 | 0.820 | 0.407 | 0.820 | 9 | 71 | $-62$ |
| `gemini-3.7-flash` | 0.593 | 0.627 | 0.793 | 0.400 | 0.793 | 11 | 70 | $-59$ |
| `gemini-3.8-flash` | 0.540 | 0.627 | 0.793 | 0.387 | 0.793 | 7 | 68 | $-61$ |
| `glm-5.3` | 0.587 | 0.660 | 0.800 | 0.380 | 0.800 | 8 | 71 | $-63$ |
| `glm-5.3-flash` | 0.633 | 0.667 | 0.827 | 0.407 | 0.833 | 11 | 74 | $-63$ |
| `gpt-5.5` | 0.553 | 0.567 | 0.720 | 0.347 | 0.720 | 11 | 67 | $-56$ |
| `gpt-5.6-luna` | 0.587 | 0.607 | 0.787 | 0.360 | 0.787 | 7 | 71 | $-64$ |
| `gpt-5.6-sol` | 0.620 | 0.680 | 0.833 | 0.407 | 0.833 | 9 | 73 | $-64$ |
| `gpt-5.6-terra` | 0.593 | 0.653 | 0.813 | 0.387 | 0.813 | 10 | 74 | $-64$ |
| `gpt-6-astra` | 0.553 | 0.640 | 0.787 | 0.380 | 0.793 | 11 | 72 | $-61$ |
| `kimi-k3` | 0.567 | 0.613 | 0.767 | 0.387 | 0.767 | 6 | 63 | $-57$ |
| `qwen3.8-max` | 0.607 | 0.573 | 0.813 | 0.367 | 0.813 | 7 | 71 | $-64$ |

### Findings:
- Net unique corrections are negative across **all 18 classes**.
- The highest standalone Llama recalls were on `gemini-3.1-pro-preview` (0.453) and `claude-fable-5` (0.443), but even here Llama degraded far more samples than it rescued.

---

## 16. Failure Analysis

Why did Llama-3.1-8B fail to provide strong complementarity?
1. **Capacity Gap**: At 0.3901 standalone BA, Llama is far weaker than Qwen3-8B (0.5910) and Qwen3-14B (0.6346). Combining an expert model (0.7873) with a much weaker model (0.3901) requires a high ratio of true signal to noise, which Llama does not provide.
2. **Shared Failure Modes**: Llama's errors are strongly correlated with Qwen3-14B ($r = 0.2819$). Rather than misclassifying distinct samples, Llama stumbles on the same stylistic subtleties as Qwen.
3. **Indiscriminate Degradations**: In the critical $N=1,612$ disagreement regime, Llama degrades 698 correct EXP52 predictions while fixing only 146.

---

## 17. Leakage Audit

- **Prompt Disjointness**: Checked via `fold_assignments.csv`. Exactly 0 prompts shared between train and validation across all 5 folds.
- **Open Pool Isolation**: Audited exact passage overlap between `open_pool_clean.parquet` and `train.csv` (result: exactly 0 overlap).
- **Single-Sample Inference**: Inference code uses zero prompt identifiers, zero Hungarian matching, zero prompt centroids, and zero peer-response information.

---

## 18. Limitations

1. **QLoRA 4-bit vs Full Fine-Tuning**: Due to 12 GB VRAM limits per GPU, Llama was trained via 4-bit QLoRA. Full 16-bit fine-tuning or 8-bit LoRA might offer marginal fidelity improvements, but is unlikely to bridge a 0.39 $\to$ 0.78 gap.
2. **Pretraining Alignment**: Qwen models appear to have pretraining corpora and tokenizers that align exceptionally well with stylistic quirks of frontier synthetic benchmarks.

---

## 19. Production Decision & Next Experiment

### Formal Decision Rule Evaluation:
- **Nested BA Improvement**: $+0.0011$ ($< +0.0020$ threshold) $\to$ **FAIL**
- **95% Bootstrap CI**: $[-0.0063, +0.0083]$ (spans zero) $\to$ **FAIL**
- **Folds Improving**: 3 of 5 outer folds improve $\to$ **FAIL**
- **Net Correction**: $-1,072$ samples $\to$ **FAIL**
- **EXP65 Verdict**: **RED**

### Decision:
**RETAIN EXP52 as frozen production champion (0.632 Public LB / 0.7873 OOF BA).**
Do NOT modify EXP52 code or weights. Do NOT submit EXP65 to Kaggle.

### Next Steps:
Having falsified both bidirectional encoder diversity (ModernBERT in EXP62) and distinct generative foundation family diversity (Llama-3.1-8B in EXP65), competitive gains in this competition will not come from casually ensembling weaker secondary backbones. Focus should remain on high-signal stylometric calibration and single-sample confidence refinement within the frozen EXP52 feature manifold.
