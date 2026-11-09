# EXP50 Leakage & Provenance Audit: Controlled Qwen-Dominant Fusion

**Experiment ID**: `EXP50_qwen_dominant_fusion`  
**Date**: 2026-11-07  
**Auditor**: Senior Competitive ML Engineering Agent  
**Status**: **PASSED (100% CLEAN & PROVENANCE-VERIFIED)**

---

## 1. Audit Checkpoints

| Audit Checkpoint | Protocol Requirement | Empirical Implementation & Finding | Status |
| :--- | :--- | :--- | :---: |
| **Prompt Group Leakage** | Grouped evaluation on `prompt_id` to prevent prompt memorization. | Verified 5-Fold `GroupKFold(n_splits=5, groups=prompt_id)`. Zero prompt overlap across fold splits on `data/train.csv` (150 prompts, 2,699 passages). | **PASSED** |
| **Open Pool Contamination** | External Open Pool passages must not share prompts or texts with official validation splits. | Clean Open Pool (`data/external/open_pool_clean.parquet`) was verified against training and validation prompts via TF-IDF cosine similarity filtering (purging any row with similarity $> 0.65$). Exact string matching returned 0 contaminated rows. | **PASSED** |
| **Label Leakage** | Target labels from validation folds must never influence vectorizers, scaling, or classifiers. | Both Qwen3-8B Centred Twin predictions (`oof_exp47_qwen_centered_twin.npy`) and LinearSVC predictions (`oof_reference_svc.npy`, `oof_open_pool_svc.npy`) were generated out-of-fold. Classifiers saw strictly the 4 training folds. | **PASSED** |
| **Architectural Invariance** | EXP47 inference engine and weights must remain 100% bitwise unchanged. | Verified exact SHA256 hashes of base model config, `heads.pt`, and 6 LoRA adapters. Zero modifications made to tokenization, layer truncation (`KEEP_LAYERS=24`), pooling, or twin head debiasing. | **PASSED** |
| **Post-Processing Isolation** | Post-processing must not peek at test distribution. | Primary sweep contains **no Sinkhorn matrix balancing**, no Hungarian matching, and no test-level clustering. Predictions are evaluated strictly sample-by-sample under convex logit fusion. | **PASSED** |

---

## 2. Artifact Provenance & Cryptographic Hashes

All input artifacts were verified via SHA256 checksums prior to evaluation:

| Artifact Name | Exact File Path | File Size | SHA256 Checksum (First 32 Chars) |
| :--- | :--- | :---: | :--- |
| **Official Training Data** | `/home/rj2826/name-that-llm/data/train.csv` | 2,139,418 B | `40a22ce8d88c5f46feb4637a099db815` |
| **Clean Open Pool Data** | `/home/rj2826/name-that-llm/data/external/open_pool_clean.parquet` | 103,375,180 B | `0c5b289e9f5f7c3f439462d09b3d1d7e` |
| **EXP47 OOF Predictions** | `/home/rj2826/name-that-llm/outputs/oof_exp47_qwen_centered_twin.npy` | 194,456 B | `647a75f59789f2574bef7578ebf0e5f0` |
| **EXP28 Pool SVC OOF** | `/home/rj2826/name-that-llm/outputs/oof_open_pool_svc.npy` | 194,456 B | `cd7e0833a1d6bc60048a3944ccb55018` |
| **EXP27 Official SVC OOF** | `/home/rj2826/name-that-llm/outputs/oof_reference_svc.npy` | 194,456 B | `7f0aef0bafca28baac27ff328a4802a2` |

---

## 3. Decision Boundary & Agreement Audit

In EXP49, assigning 85% voting weight to the LinearSVC led to catastrophic decision inversion:
- EXP49 predicted with LinearSVC on **70.36%** of samples, matching Qwen on only **29.64%**.
- In EXP50, enforcing Qwen dominance ($w_{\text{qwen}} \ge 0.75$) restores proper model hierarchy:
  - At $w_{\text{qwen}} = 0.95$: **99.5%** prediction agreement with Qwen (only 13 samples altered).
  - At $w_{\text{qwen}} = 0.90$: **99.1%** prediction agreement with Qwen (only 24 samples altered).
  - At $w_{\text{qwen}} = 0.85$: **98.6%** prediction agreement with Qwen (only 38 samples altered).
  - At $w_{\text{qwen}} = 0.80$: **97.7%** prediction agreement with Qwen (only 62 samples altered).
  - At $w_{\text{qwen}} = 0.75$: **96.9%** prediction agreement with Qwen (only 84 samples altered).

Because Qwen's top-2 logit margin ($1.955$ nats) is $8.24\times$ wider than the SVC's ($0.237$ nats), when Qwen is confident ($\Delta > 1.0$ nats), the auxiliary SVC cannot override it under $w_{\text{svc}} \le 0.25$. The auxiliary model acts strictly as a tie-breaker when Qwen is genuinely undecided.

---

## 4. Final Determination

**LEAKAGE AUDIT RATING: PASSED (GREEN)**  
Zero prompt leakage, zero validation contamination, and strict architectural provenance confirmed.
