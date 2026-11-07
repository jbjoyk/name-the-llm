# Kaggle Submission Candidate Summary: EXP49 (Validated)

**Candidate ID**: `EXP49_stylometric_svc_fusion`  
**Date**: 2026-11-07  
**Validation Gate**: **PASSED (Selection-Safe 5-Fold Nested GroupKFold)**  
**Parent Submission Baseline**: EXP47 (Kaggle Public LB: **0.620**)  
**Status**: **VALIDATED AND READY FOR EXPLICIT APPROVAL**

---

## 1. Validated Performance Summary
- **Current Verified Kaggle Best (EXP47)**: **0.620 Public LB** (Local: **0.5910**)
- **EXP49 Standalone LinearSVC**: **0.4630 Local BA**
- **EXP49 Selection-Safe Nested CV (No Sinkhorn)**: **0.6976 Local BA** (Mean fold: **0.6977 ± 0.0145**)
- **EXP49 Selection-Safe Nested CV (+ Sinkhorn $\alpha=0.30, \rho=0.35$)**: **0.7054 Local BA** (Mean fold: **0.7055 ± 0.0144**)
- **Final Recommended Fixed Weight**: **$w_{\text{svc}} = 0.850$** (Median across all outer folds)
- **Expected Kaggle Public LB Range**: **0.655 – 0.672**

---

## 2. Selection-Safe Validation Audit
1. **Zero Selection Bias**: Blend weight $w$ was tuned strictly on inner 4-fold CV using only training prompts, frozen, and evaluated once on outer validation prompts.
2. **Outer Fold Stability**:
   - Fold 1: Selected $w=0.87$ $\rightarrow$ Val BA: **0.7037**
   - Fold 2: Selected $w=0.82$ $\rightarrow$ Val BA: **0.6963**
   - Fold 3: Selected $w=0.87$ $\rightarrow$ Val BA: **0.7074**
   - Fold 4: Selected $w=0.82$ $\rightarrow$ Val BA: **0.6704**
   - Fold 5: Selected $w=0.85$ $\rightarrow$ Val BA: **0.7106**
3. **Error Independence**: Pearson error correlation between Qwen3-8B and LinearSVC is **$r = 0.0005$**.
4. **Leakage & Overlap Audit**: Zero string matches and zero normalized whitespace matches between Open Pool and official data.

---

## 3. Exact Kernel Implementation & Attachments
- **Kernel File**: [`submissions/candidates/EXP49/submission_kernel_exp49.ipynb`](file:///home/rj2826/name-the-llm/submissions/candidates/EXP49/submission_kernel_exp49.ipynb)
- **Kaggle Hardware**: GPU T4 x2
- **Required Kaggle Attachments**:
  1. `qwen-lm/qwen-3` (Variation: `8b-base`, Framework: `Transformers`)
  2. `dhruvtibarewal/nlm-l24-adapters-v5` (LoRA weights & heads.pt)
  3. `crtproduction/open-pool-passages` (or bundled `open_pool_clean.parquet`)
  4. `Name That LLM` (Competition dataset)
- **Runtime on 2x T4**: ~18.5 minutes (FAST_COMMIT passes in < 60s).
- **VRAM**: 9.8 GB (peak 5.0 GB per GPU). Zero OOM risk.

---

## 4. Strict Protocol Reminder
This candidate kernel has NOT been submitted to Kaggle. It is staged in the repository awaiting your explicit approval.
