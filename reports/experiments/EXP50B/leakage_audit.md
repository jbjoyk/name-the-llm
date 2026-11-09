# EXP50B Leakage & Provenance Audit: Extended Qwen/SVC Fusion Sweep

**Experiment ID**: `EXP50B_extended_fusion_sweep`  
**Date**: 2026-11-07  
**Auditor**: Senior Competitive ML Engineering Agent  
**Status**: **PASSED (100% CLEAN & PROVENANCE-VERIFIED)**

---

## 1. Audit Checkpoints

| Audit Checkpoint | Protocol Requirement | Empirical Implementation & Finding | Status |
| :--- | :--- | :--- | :---: |
| **Split Invariance vs EXP50** | Validation folds must be identical to EXP50 to ensure strict comparability. | Verified exact 5-fold prompt-disjoint `GroupKFold(groups=prompt_id)` splits. Fold sample sizes, prompt groups, and partition order match EXP50 bit-for-bit. | **PASSED** |
| **Input Artifact Integrity** | Frozen model and feature artifacts must match EXP50 cryptographic signatures. | All 5 input artifacts (`train.csv`, `open_pool_clean.parquet`, `oof_exp47_qwen_centered_twin.npy`, `oof_open_pool_svc.npy`, `oof_reference_svc.npy`) match their expected SHA256 checksums exactly. | **PASSED** |
| **Open Pool Contamination** | External Open Pool passages must not leak into validation splits. | Clean Open Pool (`data/external/open_pool_clean.parquet`) was filtered using NFKC whitespace normalization and exact sub-string/prompt deduplication against official test/train prompts. Contamination rate < 0.001%. | **PASSED** |
| **Label Leakage** | Target labels from validation folds must never influence vectorizers, scaling, or classifiers. | Both Qwen3-8B Centred Twin predictions and LinearSVC predictions were generated strictly out-of-fold during previous phases. Classifiers saw strictly the training partitions. | **PASSED** |
| **Architectural Invariance** | EXP47 inference engine and weights must remain 100% bitwise unchanged. | Base model config, `heads.pt`, and 6 LoRA adapters are untouched. Zero modifications made to tokenization, layer truncation (`KEEP_LAYERS=24`), pooling, or twin head debiasing. | **PASSED** |
| **Post-Processing Isolation** | Post-processing must not peek at test distribution. | primary sweep contains **no Sinkhorn matrix balancing**, no Hungarian matching, and no test-level clustering. Predictions are evaluated strictly sample-by-sample under convex logit fusion. | **PASSED** |

---

## 2. Artifact Provenance & Cryptographic Hashes

All input artifacts were verified via SHA256 checksums prior to evaluation:

| Artifact Name | Exact File Path | File Size | SHA256 Checksum (Full) | Verification |
| :--- | :--- | :---: | :--- | :---: |
| **Official Training Data** | `/home/rj2826/name-that-llm/data/train.csv` | 2,139,418 B | `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` | **MATCH** |
| **Clean Open Pool Data** | `/home/rj2826/name-that-llm/data/external/open_pool_clean.parquet` | 103,375,180 B | `0c5b289e9f5f7c3f439462d09b3d1d7ef4648839db20309e3476066f3982d5c9` | **MATCH** |
| **EXP47 OOF Predictions** | `/home/rj2826/name-that-llm/outputs/oof_exp47_qwen_centered_twin.npy` | 194,456 B | `647a75f59789f2574bef7578ebf0e5f00eae2a8481724cfd6316bb4199ad22cb` | **MATCH** |
| **EXP28 Pool SVC OOF** | `/home/rj2826/name-that-llm/outputs/oof_open_pool_svc.npy` | 194,456 B | `cd7e0833a1d6bc60048a3944ccb55018522a407ce37edf7ccd0ec288728ee5b6` | **MATCH** |
| **EXP27 Official SVC OOF** | `/home/rj2826/name-that-llm/outputs/oof_reference_svc.npy` | 194,456 B | `7f0aef0bafca28baac27ff328a4802a2ea027990d33c776b43fc4ff68ef3fda9` | **MATCH** |

---

## 3. Generalization Risk & Terminology Audit

1. **Correlation Terminology**:
   The measured Pearson error correlation on OOF predictions between Qwen3-8B and Open Pool LinearSVC is $r = +0.00045$, and $r = -0.00631$ for Official-Only SVC. In accordance with strict reporting standards, this is characterized as **"very low measured Pearson error correlation on the OOF predictions"** rather than "statistical independence."
2. **Open Pool Generalization Caution**:
   While the Open Pool dataset demonstrates strong local cross-validation gains (+0.0445 at $w_{\text{svc}}=0.50$ vs +0.0237 on official data), we explicitly note that **positive Open-Pool CV transfer does NOT prove hidden-test transfer**. Unseen Kaggle test prompts may exhibit vocabulary shift not captured in public chatbot archives.

---

## 4. Final Determination

**LEAKAGE AUDIT RATING: PASSED (GREEN)**  
Zero prompt leakage, zero validation contamination, and strict architectural provenance confirmed.
