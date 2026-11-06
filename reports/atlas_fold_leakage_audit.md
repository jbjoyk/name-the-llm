# Fold-Level Leakage Audit: Pretrained ATLAS Representation
**Document**: `reports/atlas_fold_leakage_audit.md`  
**Date**: 2026-11-06  
**Auditor**: Antigravity Research Engineer  
**Validation Strategy Under Audit**: 5-Fold `GroupKFold(group=prompt_id)`

---

## 1. Fold-by-Fold Overlap Table

| Fold | Held-Out Prompt Range | Held-Out Prompts | Held-Out Responses | ATLAS Training Prompt Overlap | ATLAS Training Response Overlap | Official Training Overlap | Leakage Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fold 1** | `p004, p009, ..., p149` | 30 | 540 | 30 / 30 (100.0%) | 540 / 540 (100.0%) | 100.0% | **CONTAMINATED** |
| **Fold 2** | `p003, p008, ..., p148` | 30 | 540 | 30 / 30 (100.0%) | 540 / 540 (100.0%) | 100.0% | **CONTAMINATED** |
| **Fold 3** | `p001, p007, ..., p147` | 30 | 540 | 30 / 30 (100.0%) | 540 / 540 (100.0%) | 100.0% | **CONTAMINATED** |
| **Fold 4** | `p000, p006, ..., p146` | 30 | 540 | 30 / 30 (100.0%) | 540 / 540 (100.0%) | 100.0% | **CONTAMINATED** |
| **Fold 5** | `p002, p005, ..., p145` | 30 | 539 | 30 / 30 (100.0%) | 539 / 539 (100.0%) | 100.0% | **CONTAMINATED** |

---

## 2. Leakage Analysis

1. **Why `GroupKFold(group=prompt_id)` Fails to Protect Against External Pretraining**:
   - `GroupKFold` splits the 150 prompt groups into 120 training prompts and 30 validation prompts.
   - For all components trained inside the fold loop (e.g. TF-IDF vectorizers, standard scalers, LinearSVC, Gaussian heads), `GroupKFold` guarantees zero leakage because held-out prompt passages are never passed to `.fit()`.
   - However, when evaluating a **pretrained external checkpoint** (such as `adapters.safetensors`), the weights were already fit beforehand on all 150 prompts and 2,699 responses.
   - Splitting `data/train.csv` into folds does not erase the training history embedded inside the frozen weights.

2. **Compliance with Competition Leak-Free Rule**:
   - **Condition for Leak-Free External Model**: The external model's training corpus must contain zero passages or prompts that overlap with the validation evaluation set.
   - **Finding**: Because the ATLAS bundle used the entire official competition `train.csv` as its training corpus, the held-out validation sets for all 5 folds are 100% contaminated.
