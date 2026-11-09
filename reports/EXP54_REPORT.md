# EXP54: Controlled Qwen3 (8B+14B) + Open Pool 4-View LinearSVC Fusion Experiment

**Experiment ID**: `EXP54_qwen_svc_fusion`  
**Date**: 2026-11-08  
**Author**: Antigravity Machine Learning Research Agent  
**Parent Baseline**: EXP52 (Dual-LoRA Qwen3-8B $0.60$ + Qwen3-14B $0.40$, Local OOF BA: **0.7873**, Kaggle Public LB: **0.632**)  
**Predecessor Experiments**: EXP47 (LB 0.620), EXP49 (LB 0.612), EXP50 (OOF BA 0.6102), EXP50B (OOF BA 0.6354), EXP53 (OOF BA 0.7873)  
**Primary Metric**: Macro-averaged Balanced Accuracy across 18 candidate LLM classes  
**Validation Strategy**: Strict 5-Fold Prompt-Disjoint `GroupKFold(groups=prompt_id)` on `data/train.csv` (2,699 samples, 150 unique prompts)  
**Kaggle Submission Status**: **ABSOLUTELY NOT SUBMITTED — LOCAL VALIDATION ONLY**  
**Production Baseline**: **EXP52 REMAINS FROZEN AT KAGGLE PUBLIC LB 0.632**

---

## 1. Executive Summary

EXP54 evaluates whether adding a small contribution from the validated **Open Pool 4-view LinearSVC model** ($w_{\text{svc}} \le 0.20$) can improve the frozen **EXP52 Qwen3-8B + Qwen3-14B dual-backbone ensemble** ($w_8 = 0.60, w_{14} = 0.40$, Kaggle LB **0.632**, OOF BA **0.7873**).

### Scientific Context & Background
In EXP49, combining Qwen3-8B with the Open Pool LinearSVC achieved an apparent local OOF jump from **0.5910** to **0.7028** under heavy SVC weighting ($w_{\text{svc}} = 0.85$). However, when evaluated on Kaggle's hidden test set, EXP49 scored only **0.612**—falling below pure EXP47 (**0.620**). Forensic audit revealed that surface n-gram stylometry overfitted the training prompt vocabulary, causing severe domain shift on unseen prompts when given 85% decision mass.

EXP54 explicitly tests the **conservative regime** ($w_{\text{svc}} \in \{0.00, 0.05, 0.10, 0.15, 0.20\}$) where deep semantic representations remain strictly dominant ($w_{\text{qwen}} \ge 0.80$).

### Key Empirical Findings

1. **Marginal OOF Peak at $w_{\text{svc}} = 0.05$**:
   Adding $w_{\text{svc}} = 0.05$ nudges local Balanced Accuracy from **0.7873** to **0.7888** ($\Delta = +0.0015$). However, 2,000 paired bootstrap resamples demonstrate that this gain is **not statistically robust**:
   $$\Delta = +0.0015, \quad 95\% \text{ Bootstrap CI}: [-0.0001, +0.0034], \quad P(\Delta > 0) = 0.9605$$
   The lower bound of the 95% confidence interval crosses zero ($-0.0001$).
2. **Tiny Decision Shift (Only 4 Net Samples)**:
   At $w_{\text{svc}} = 0.05$, the predictions agree with EXP52 on **99.78% of samples** (2,693 out of 2,699). Exactly 6 predictions flip across the entire dataset: 5 errors are fixed, 1 correct sample is broken, yielding a net gain of **only +4 samples**.
3. **Plateau and Fluctuation at Higher Weights**:
   - $w_{\text{svc}} = 0.10$: BA = **0.7877** ($\Delta = +0.0004$, 95% CI: $[-0.0017, +0.0026]$, net +1 sample).
   - $w_{\text{svc}} = 0.15$: BA = **0.7873** ($\Delta = +0.0000$, 95% CI: $[-0.0029, +0.0030]$, 8 fixed vs 8 broken, net 0).
   - $w_{\text{svc}} = 0.20$: BA = **0.7881** ($\Delta = +0.0007$, 95% CI: $[-0.0030, +0.0044]$, 14 fixed vs 12 broken, net +2).
4. **Confidence and Margin Dilution**:
   As $w_{\text{svc}}$ increases from 0.00 to 0.20, prediction entropy increases monotonically from **0.8258 to 1.0717**, and the top-1 margin compresses from **0.6258 down to 0.5456** (logit margin drops from $2.615$ to $2.132$ nats). The linear model dilutes the sharp, confident calibration of the dual-transformer ensemble.
5. **Decisive Engineering Recommendation**:
   A local gain of +4 net samples on training data does not justify re-introducing surface n-gram representations that previously demonstrated negative transfer on Kaggle. **EXP52 remains frozen as our undefeated production baseline (0.632 LB). Zero submissions are recommended.**

---

## 2. Cryptographic Provenance & Artifact Verification

All input prediction arrays, model adapters, and datasets were cryptographically verified before running the fusion experiment. Every artifact matches its established SHA-256 checksum:

| Artifact Name | Local File Path | Expected / Verified SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| **Q8 LoRA Adapter** | `models/adapters_v5/q8v5/q8v5/adapter_model.safetensors` | `a600e552f40bcda00194f147f957d3bbb04e176b6bccf4a1f7fe39b82c8e8f51` | **PASS** |
| **Q8 Heads** | `models/adapters_v5/q8v5/q8v5/heads.pt` | `31868cd0e9618fac506c93a08f118ee00202041d67426890e609d13e2cd33f0a` | **PASS** |
| **Q8 Config** | `models/adapters_v5/q8v5/q8v5/adapter_config.json` | `621136c127ae11004b136582ac899229a191c2d57443b297f7867e88b42d9204` | **PASS** |
| **Q14 LoRA Adapter** | `models/adapters_v5/q14v5/q14v5/adapter_model.safetensors` | `7580831f6d7184c9a676ce88c3b69af00e42ffed1e7132fd8812c9ba632616c4` | **PASS** |
| **Q14 Heads** | `models/adapters_v5/q14v5/q14v5/heads.pt` | `cafdbade649017d666c2da623adb6539877e1125a996f2736133127ae6121b3d` | **PASS** |
| **Q14 Config** | `models/adapters_v5/q14v5/q14v5/adapter_config.json` | `1f2c044ca59f117a2865ce91478f126d25f974224282b70d320828e8b504173e` | **PASS** |
| **EXP47 Q8 OOF** | `outputs/oof_exp47_qwen_centered_twin.npy` | `647a75f59789f2574bef7578ebf0e5f00eae2a8481724cfd6316bb4199ad22cb` | **PASS** |
| **q14v5 Q14 OOF** | `outputs/oof_q14v5_antopil_lora.csv` | `58822e8d71d6b6ae1d77a3eb62a5f3a2d4f1757c239392d1b5d9e227646ab9e9` | **PASS** |
| **Open Pool SVC OOF**| `outputs/oof_open_pool_svc.npy` | `cd7e0833a1d6bc60048a3944ccb55018522a407ce37edf7ccd0ec288728ee5b6` | **PASS** |
| **Open Pool Clean** | `data/external/open_pool_clean.parquet` | `0c5b289e9f5f7c3f439462d09b3d1d7ef4648839db20309e3476066f3982d5c9` | **PASS** |
| **Official Train** | `data/train.csv` | `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` | **PASS** |

---

## 3. Open Pool Dataset Audit & Class Alignment

### Dataset Inspection
- **Source Dataset**: `crtproduction/open-pool-passages` (stored locally at `data/external/open_pool_clean.parquet`).
- **Passage Count**: Exactly **212,623 passages** across 7 columns (`['id', 'text', 'model', 'lab', 'prompt_group', 'source_dataset', 'n_words']`).
- **Balanced Subset**: `data/external/open_pool_36k.parquet` contains exactly 36,000 passages (2,000 passages per class across all 18 classes).
- **Prompt Isolation**: Zero prompt overlap between `data/train.csv` and the Open Pool dataset.
- **Preprocessing**: NFKC text sanitization, whitespace normalization, deduplication.

### Class Alignment Verification
The 18 classes are ordered in canonical alphabetical order. Verified standalone recall of `outputs/oof_open_pool_svc.npy` against `data/train.csv` ground truth:

```text
=== CLASS ALIGNMENT VERIFICATION ===
Idx  0 | Label: claude-fable-5           | Standalone SVC Recall:  16.8%
Idx  1 | Label: claude-opus-4-6          | Standalone SVC Recall:  65.3%
Idx  2 | Label: claude-opus-4-8          | Standalone SVC Recall:  40.7%
Idx  3 | Label: claude-opus-5            | Standalone SVC Recall:  68.7%
Idx  4 | Label: claude-sonnet-5          | Standalone SVC Recall:  62.0%
Idx  5 | Label: gemini-3.1-pro-preview   | Standalone SVC Recall:  81.3%
Idx  6 | Label: gemini-3.6-flash         | Standalone SVC Recall:  58.0%
Idx  7 | Label: gemini-3.7-flash         | Standalone SVC Recall:  44.0%
Idx  8 | Label: gemini-3.8-flash         | Standalone SVC Recall:  47.3%
Idx  9 | Label: glm-5.3                  | Standalone SVC Recall:  15.3%
Idx 10 | Label: glm-5.3-flash            | Standalone SVC Recall:  23.3%
Idx 11 | Label: gpt-5.5                  | Standalone SVC Recall:  46.0%
Idx 12 | Label: gpt-5.6-luna             | Standalone SVC Recall:  37.3%
Idx 13 | Label: gpt-5.6-sol              | Standalone SVC Recall:  29.3%
Idx 14 | Label: gpt-5.6-terra            | Standalone SVC Recall:  30.7%
Idx 15 | Label: gpt-6-astra              | Standalone SVC Recall:  70.7%
Idx 16 | Label: kimi-k3                  | Standalone SVC Recall:  20.7%
Idx 17 | Label: qwen3.8-max              | Standalone SVC Recall:  76.0%
Verified Standalone SVC OOF Balanced Accuracy: 0.4630 (Exact match with EXP28/EXP50)
```

---

## 4. Standalone Baselines & Complementarity Analysis

### Standalone Model Comparison

| Model | Architecture | Local OOF BA | Macro-F1 | Accuracy | Kaggle LB | Role in EXP54 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Qwen3-8B (EXP47)** | 24-Layer LoRA24 + Centred Twin | 0.5910 | 0.5915 | 0.5910 | 0.620 | Member 1 of EXP52 ($w_8 = 0.60$) |
| **Qwen3-14B (q14v5)** | 27-Layer LoRA24 + Centred Twin | 0.6346 | 0.6358 | 0.6347 | — | Member 2 of EXP52 ($w_{14} = 0.40$) |
| **EXP52 Ensemble** | $0.60 \log P_{q8} + 0.40 \log P_{q14}$ | **0.7873** | **0.7880** | **0.7873** | **0.632** | **Frozen Production Baseline** |
| **Open Pool SVC** | 4-View TF-IDF (char+word+casing+func) | 0.4630 | 0.4494 | 0.4631 | 0.463 (est) | Candidate Auxiliary Regularizer |

### EXP52 vs Open Pool LinearSVC Complementarity

Contingency analysis on the 2,699 training samples:

- **Total Samples**: 2,699
- **Prediction Agreement**: 1,167 / 2,699 (**43.24%**)
- **Both Correct**: 1,063 (**39.38%**)
- **Both Wrong**: 387 (**14.34%**)
- **Qwen-only Correct** (EXP52 correct, SVC wrong): 1,062 (**39.35%**)
- **SVC-only Correct** (SVC correct, EXP52 wrong): 187 (**6.93%**)
- **Measured Pearson Error Correlation ($r$)**: **0.14316** ($p = 7.87 \times 10^{-14}$)

> [!NOTE]
> The measured Pearson error correlation ($r = 0.14316$) is very low, indicating that the failure modes of the shallow n-gram classifier and the deep dual-transformer ensemble are largely non-overlapping. However, SVC is correct on only 187 samples where EXP52 is wrong, while EXP52 is already correct on 2,125 samples (78.73%).

---

## 5. EXP54 Primary Results Table

Evaluated strictly via log-probability fusion:
$$\text{LP}_{\text{FINAL}} = (1 - w_{\text{svc}}) \cdot \text{LP}_{\text{QWEN}} + w_{\text{svc}} \cdot \text{LP}_{\text{SVC}}$$
where $\text{LP}_{\text{QWEN}} = 0.60 \log P_{q8} + 0.40 \log P_{q14}$.

All metrics computed on `data/train.csv` (2,699 samples). Statistical significance computed via 2,000 paired bootstrap resamples (seed 42):

| $w_{\text{svc}}$ | $w_{\text{qwen}}$ | OOF BA | Macro-F1 | Acc | Entropy | Margin | Logit Marg | Delta vs EXP52 | 95% Bootstrap CI | P($\Delta > 0$) | Boot SE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | **1.00** | **0.7873** | 0.7880 | 0.7873 | 0.8258 | 0.6258 | 2.615 | **0.0000 (Base)** | [0.0000, 0.0000] | 0.0000 | 0.0000 |
| **0.05** | **0.95** | **0.7888** | 0.7895 | 0.7888 | 0.8794 | 0.6081 | 2.494 | **+0.0015** | **[-0.0001, +0.0034]** | **0.9605** | 0.0009 |
| **0.10** | **0.90** | **0.7877** | 0.7884 | 0.7877 | 0.9379 | 0.5889 | 2.373 | **+0.0004** | [-0.0017, +0.0026] | 0.6355 | 0.0011 |
| **0.15** | **0.85** | **0.7873** | 0.7880 | 0.7873 | 1.0019 | 0.5681 | 2.253 | **+0.0000** | [-0.0029, +0.0030] | 0.4945 | 0.0015 |
| **0.20** | **0.80** | **0.7881** | 0.7886 | 0.7881 | 1.0717 | 0.5456 | 2.132 | **+0.0007** | [-0.0030, +0.0044] | 0.6535 | 0.0019 |

---

## 6. Fold-Level Balanced Accuracy Breakdown

Evaluated across the 5 strictly prompt-disjoint folds from `outputs/oof_q14v5_antopil_lora.csv`:

| $w_{\text{svc}}$ | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold Mean $\pm$ Std |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00 (EXP52)** | 0.7778 | 0.7793 | 0.7907 | 0.7833 | 0.8056 | **0.7873 $\pm$ 0.0102** |
| **0.05** | 0.7815 | 0.7830 | 0.7926 | 0.7815 | 0.8056 | **0.7888 $\pm$ 0.0093** |
| **0.10** | 0.7796 | 0.7812 | 0.7926 | 0.7796 | 0.8056 | **0.7877 $\pm$ 0.0102** |
| **0.15** | 0.7796 | 0.7812 | 0.7944 | 0.7778 | 0.8037 | **0.7873 $\pm$ 0.0101** |
| **0.20** | 0.7796 | 0.7849 | 0.7963 | 0.7796 | 0.8000 | **0.7881 $\pm$ 0.0085** |

### Per-Fold Dynamics at $w_{\text{svc}} = 0.05$:
- **Fold 0**: $0.7778 \rightarrow 0.7815$ (+0.0037, +2 samples fixed)
- **Fold 1**: $0.7793 \rightarrow 0.7830$ (+0.0037, +2 samples fixed)
- **Fold 2**: $0.7907 \rightarrow 0.7926$ (+0.0019, +1 sample fixed)
- **Fold 3**: $0.7833 \rightarrow 0.7815$ (-0.0018, 1 sample broken)
- **Fold 4**: $0.8056 \rightarrow 0.8056$ (0.0000, 0 net change)

---

## 7. Prediction Flip Forensics

To understand exactly what physical changes occur when blending the LinearSVC with EXP52:

| $w_{\text{svc}}$ | Predictions Changed | Agreement w/ EXP52 | Samples Fixed | Samples Broken | Net Samples Gained | Total Correct |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 0 | 100.0% | 0 | 0 | 0 | 2,125 / 2,699 |
| **0.05** | 6 | 99.78% | 5 | 1 | **+4** | 2,129 / 2,699 |
| **0.10** | 11 | 99.59% | 5 | 4 | **+1** | 2,126 / 2,699 |
| **0.15** | 22 | 99.18% | 8 | 8 | **0** | 2,125 / 2,699 |
| **0.20** | 32 | 98.81% | 14 | 12 | **+2** | 2,127 / 2,699 |

### Observations:
1. At $w_{\text{svc}} = 0.05$, only **6 rows out of 2,699 (0.22%)** change prediction. Five of these changes correct a subtle error, while one breaks a previously correct prediction.
2. Beyond $w_{\text{svc}} = 0.05$, the number of changed predictions grows (11 at $0.10$, 22 at $0.15$, 32 at $0.20$), but the fixes and damages **nearly cancel out** (e.g. at $w_{\text{svc}} = 0.15$, exactly 8 samples are fixed and 8 are broken).

---

## 8. Weight Sweep Visualization

The publication-grade plot below illustrates the weight sweep curve alongside the paired bootstrap delta distribution:

![EXP54 Weight Sweep](file:///home/rj2826/name-the-llm/reports/figures/exp54_weight_sweep.png)

*Figure 1: (Left) OOF Balanced Accuracy across $w_{\text{svc}} \in [0.00, 0.20]$ with 95% empirical bootstrap error bars. (Right) Paired Delta vs EXP52 Baseline ($w_{\text{svc}} = 0.00$) with 95% bootstrap confidence intervals. Notice that the 95% CI spans zero across all evaluated weights.*

---

## 9. Scientific Interpretation & Kaggle Risk Assessment

### Why EXP50 Saw Monotonic Gains but EXP54 Sees a Flat Plateau
In EXP50, the parent model was **EXP47 (Qwen3-8B alone)** with a standalone OOF BA of **0.5910**. Because EXP47 made mistakes on 41% of samples, adding Open Pool SVC provided a large pool of complementary decisions, producing a clean monotonic climb from 0.5910 to 0.6102 (+0.0193).

In contrast, EXP54's parent model is **EXP52 (Qwen3-8B + Qwen3-14B)**, which already scores **0.7873 OOF BA** and **0.632 Kaggle LB**. The two foundation models already resolve the vast majority of surface-level ambiguities through 14B depth and 8B pooling. Consequently:
- The reservoir of errors that a shallow 4-view LinearSVC can fix is tiny (only 187 samples total).
- At low weights ($w_{\text{svc}} = 0.05$), it touches only 6 samples.
- At higher weights ($w_{\text{svc}} \ge 0.10$), the surface n-gram noise degrades as many correct transformer decisions as it fixes.

### The Kaggle Domain Shift Vulnerability
EXP49 provided unequivocal proof that **Open Pool LinearSVC features do not transfer cleanly to Kaggle's hidden test set**:
- Kaggle test prompts are unseen and sparse ($R \approx 2$).
- Surface n-gram statistics overfit the training prompt domain.
- When EXP49 gave significant weight to SVC, Kaggle LB dropped from **0.620 to 0.612**.
- A local OOF improvement of **+0.0015** that rests on just **4 net samples** has near-zero probability of surviving unseen prompt shift on Kaggle.

---

## 10. Candidate Selection

- **Candidate A (`EXP54_w05`)**: $w_{\text{svc}} = 0.05, w_{\text{qwen}} = 0.95$  
  - Local OOF BA: **0.7888** ($\Delta = +0.0015$, 95% CI: $[-0.0001, +0.0034]$, $P = 0.9605$).
  - Assessment: Marginal point estimate improvement representing only 4 net flipped samples. Confidence interval crosses zero.
- **Candidate B**: **NONE**  
  - Higher weights ($0.10, 0.15, 0.20$) show zero or negligible gains with degraded margin and higher entropy.
- **Recommendation**: **DO NOT SUBMIT Candidate A**. EXP52 remains our highest validated Kaggle system (LB 0.632) and should not be compromised for an un-replicable 4-sample local shift.

---

## 11. Artifact Checklist

The following output artifacts have been generated and archived in the repository:

1. `src/exp54_qwen_svc.py` — Standalone, reproducible execution pipeline.
2. `experiments/EXP54_qwen_svc/EXP54_results.csv` — Full table of sweep metrics, fold scores, bootstrap CIs, and flip counts.
3. `experiments/EXP54_qwen_svc/EXP54_oof_predictions.parquet` — Per-sample out-of-fold predictions and top-1 probabilities for all weights.
4. `experiments/EXP54_qwen_svc/summary.json` — Machine-readable summary of hashes, metadata, complementarity, and candidate metrics.
5. `experiments/EXP54_qwen_svc/exp54_weight_sweep.png` — Publication plot in experiment directory.
6. `reports/figures/exp54_weight_sweep.png` — Publication plot in reports directory.
7. `reports/EXP54_REPORT.md` — Complete analytical report.

---

## EXP54 VERDICT

Current production baseline:

    EXP52 = Kaggle LB 0.632

EXP53 conclusion:

    q14 weight = 0.40 remains frozen

Best OOF SVC weight:

    0.05

Best OOF BA:

    0.7888

Delta vs EXP52:

    +0.0015

95% bootstrap CI:

    [-0.0001, +0.0034]

P(delta > 0):

    0.9605

Recommended Kaggle candidate(s):

    NONE

Reason:

    The highest local OOF bump (+0.0015 BA) corresponds to only 4 net samples flipped out of 2,699, with the 95% bootstrap CI spanning zero ([-0.0001, +0.0034]). Given the proven failure mode of Open Pool SVC domain shift on Kaggle (EXP49 LB dropped from 0.620 to 0.612), this marginal local gain is not statistically robust enough to justify risking the production EXP52 baseline (LB 0.632).

Kaggle submission:

    NOT SUBMITTED

Production baseline:

    EXP52 remains frozen at Kaggle LB 0.632
