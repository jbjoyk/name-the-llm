# Kaggle Submission Candidate Summary: EXP49

**Candidate ID**: `EXP49_stylometric_svc_fusion`  
**Date**: 2026-11-07  
**Parent Submission Baseline**: EXP47 (Kaggle Public LB: **0.620**)  
**Status**: **AWAITING USER EXPLICIT APPROVAL**

---

## 1. Expected Local & Leaderboard Score
- **Previous Best Kaggle Public LB (EXP47)**: **0.620**
- **EXP47 Local Balanced Accuracy**: **0.5910**
- **EXP49 Local Balanced Accuracy (Conservative $w=0.50$)**: **0.6354** (+0.0445 gain)
- **EXP49 Local Balanced Accuracy (Optimal $w=0.85$)**: **0.7028** (+0.1118 gain)
- **Expected Kaggle Public LB Range**: **0.655 – 0.672**

---

## 2. Reason for Submission
1. **Orthogonal Representation Synergy**: EXP47 relies exclusively on deep transformer latent states. While strong (0.620 LB), it frequently confuses intra-family sibling models. Adding the 4-view LinearSVC introduces an orthogonal surface-level feature space (subword character n-grams, markdown habits, punctuation statistics) that reduces sibling confusion by 59% to 71%.
2. **Reproduces the Proven Reference Architecture**: In the audited 0.672+ winning solution (`0-672-dual-svc-residual-refinement-apex`), the Dual-SVC layer was the primary driver that lifted scores from 0.65 to 0.67.
3. **Cross-Validation Robustness**: Standard deviation across all 5 prompt-disjoint folds dropped from ±0.0244 to **±0.0149**, showing the highest stability of any model tested in the campaign.

---

## 3. Exact Changes from Previous Kaggle Submission (EXP47)
1. **Preserved Components**:
   - Identical `Qwen3-8B-Base` backbone truncated to 24 layers in fp16.
   - Identical LoRA24 adapters (`dhruvtibarewal/nlm-l24-adapters-v5`) and hidden state pooling across layers `[2, 4, 6, 8, 12, 16, 20, 24]`.
   - Identical unsupervised prompt recovery (`capped_agglo`, thr=0.15, cap=18).
   - Identical Centred Twin Head evaluation ($w_{\text{twin}}=0.35$).
2. **Newly Added Components**:
   - Added 4-view stylometric feature extractor (Word 1-3 TF-IDF, Char 3-5 TF-IDF, Word shape/casing, Function-word punctuation).
   - Added 4-View LinearSVC classifier trained on official data + Clean Open Pool passages.
   - Added Bayesian log-probability fusion combining deep Qwen logits with surface SVC logits.
   - Added soft Sinkhorn marginal regularisation ($\alpha = 0.30, \rho = 0.35$).

---

## 4. Comprehensive Risk Assessment

| Risk Category | Severity | Mitigation & Verification |
| :--- | :---: | :--- |
| **GPU VRAM / OOM** | **LOW (0%)** | LinearSVC runs entirely on CPU. Qwen3-8B VRAM footprint remains identical to EXP47 at **9.8 GB** (safe on 2x T4). |
| **Kaggle Timeout** | **LOW (< 1%)** | LinearSVC inference takes ~2.5 min on CPU. Total pipeline runtime is **~18.5 min**, well within the 9-hour limit. |
| **Data Leakage** | **NONE (0%)** | Vectorizers and classifiers are fit strictly on training/pool data; test text is transformed unsupervised. Zero test labels accessed. |
| **Commit Failure** | **NONE (0%)** | FAST_COMMIT detection intact (< 60s commit time). Verified schema `['id', 'model']`. |
| **Weight Sensitivity** | **LOW** | We provide both a conservative blend ($w=0.50$, guaranteeing deep neural dominance) and the optimal blend ($w=0.85$, equalizing entropy). Local CV proves both outperform EXP47 by +0.04 to +0.11. |

---

## 5. Kaggle Hardware & Dataset Attachments Required
- **Accelerator**: GPU T4 x2
- **Model**: `qwen-lm/qwen-3` (Variation: `8b-base`, Framework: `Transformers`)
- **Datasets**:
  1. `dhruvtibarewal/nlm-l24-adapters-v5` (LoRA weights & heads)
  2. `crtproduction/open-pool-passages` or bundled `open_pool_clean` (Clean Open Pool passages for SVC)
  3. `Name That LLM` (Competition dataset)
