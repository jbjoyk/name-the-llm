# Phase 12 — Out-of-Fold (OOF) Prediction Artifact Audit

**Audit Date**: November 2026  
**Evaluator**: Antigravity Machine Learning Research Agent  
**Dataset Target**: Official `data/train.csv` ($N = 2,699$ rows, 150 prompt groups, 18 LLM classes)  
**Status**: APPROVED & VERIFIED FOR META-ENSEMBLING

---

## 1. Executive Summary

Prior to constructing the soft meta-ensemble in Phase 12, an exhaustive cryptographic, structural, and alignment audit was conducted across all 39 prediction arrays stored in `outputs/`.

The objective is to establish an ironclad roster of **genuine, leak-free, prompt-disjoint out-of-fold (OOF)** prediction arrays to serve as inputs to the meta-model, while strictly identifying and excluding invalid, contaminated, or synthetic artifacts.

---

## 2. Complete Roster of Evaluated OOF Artifacts

| Experiment / Source | File Path | Shape | OOF? | Prompt-Disjoint? | Valid? | Standalone Raw BA | Underlying Signal Type |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP_36 / EXP_38** | `outputs/oof_exp36_predictions.npy` | $(2699, 18)$ | **Yes** | **Yes** | **YES** | **`0.5286`** | 4-View Open Pool SVC + Shared Gauss Head |
| **EXP_40** | `outputs/oof_exp40_gat_predictions.npy` | $(2699, 18)$ | **Yes** | **Yes** | **YES** | `0.3560` | Prompt-Residual Graph Attention Network |
| **EXP_28** | `outputs/oof_open_pool_svc.npy` | $(2699, 18)$ | **Yes** | **Yes** | **YES** | `0.4630` | 4-View Sparse LinearSVC (Open Pool $w=0.35$) |
| **EXP_36 Gauss** | `outputs/oof_shared_prompt_gaussian.npy`| $(2699, 18)$ | **Yes** | **Yes** | **YES** | `0.4679` | Additive Two-Way Gaussian Model (Log-Likelihood) |
| **EXP_41 (Pair 1)**| `outputs/oof_exp41_pair_gemini_*.npy` | $(2699,)$ | **Yes** | **Yes** | **YES** | `0.6000` | Sibling Specialist Log-Odds (Gemini 3.7 vs 3.8) |
| **EXP_41 (Pair 2)**| `outputs/oof_exp41_pair_glm_*.npy` | $(2699,)$ | **Yes** | **Yes** | **YES** | `0.5233` | Sibling Specialist Log-Odds (GLM 5.3 vs 5.3-flash) |
| **EXP_41 (Pair 3)**| `outputs/oof_exp41_pair_gpt_sol_terra*.npy`| $(2699,)$ | **Yes** | **Yes** | **YES** | `0.5900` | Sibling Specialist Log-Odds (GPT-5.6 Sol vs Terra) |
| **EXP_41 (Pair 4)**| `outputs/oof_exp41_pair_gpt_luna_sol*.npy` | $(2699,)$ | **Yes** | **Yes** | **YES** | `0.5900` | Sibling Specialist Log-Odds (GPT-5.6 Luna vs Sol) |
| **EXP_41 (Pair 5)**| `outputs/oof_exp41_pair_gpt_luna_terra*.npy`| $(2699,)$ | **Yes** | **Yes** | **YES** | `0.6000` | Sibling Specialist Log-Odds (GPT-5.6 Luna vs Terra) |
| **EXP_02** | `outputs/oof_style.npy` | $(2699, 18)$ | **Yes** | **Yes** | **YES** | `0.3130` | 31-dim Rule-Based Stylometry |
| **EXP_06** | `outputs/oof_syntax.npy` | $(2699, 18)$ | **Yes** | **Yes** | **YES** | `0.2786` | MiniLM Layer 3 Morpho-Syntax |
| **EXP_07** | `outputs/oof_likelihood.npy` | $(2699, 18)$ | **Yes** | **Yes** | **YES** | `0.2526` | MultiScale Character/Word N-Gram Likelihood |
| *EXP_37 ATLAS* | `outputs/oof_atlas_gated_predictions.npy`| $(2699,)$ | No | No | **DISQUALIFIED** | N/A | 100% In-Sample Leakage (Proven Overlap) |

---

## 3. Row-by-Row Alignment Verification

Every qualified prediction array was checked for mathematical consistency:
1. **Dimension**: Exactly $N = 2,699$ rows matching `data/train.csv` row-for-row.
2. **Column Structure**: For 18-class probability matrices, column indices $0 \dots 17$ strictly correspond to the canonical alphabetical class ordering:
   `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`, `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`, `glm-5.3`, `glm-5.3-flash`, `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`, `kimi-k3`, `qwen3.8-max`.
3. **Probability Bounds**: All probability distributions lie strictly in $[0, 1]$ and sum to $1.0 \pm 10^{-4}$ along axis 1.
4. **Disjointness**: All validated OOF files were generated strictly under outer `GroupKFold(5, group=prompt_id)` where no prompt was ever present in both training and test partitions.

---

## 4. Exclusion of Contaminated Checkpoints (ATLAS Ban)

In accordance with Phase 9 audit protocols (`reports/atlas_data_overlap_audit.md`):
* `outputs/oof_atlas_gated_predictions.npy` and all ATLAS-derived logits are **permanently banned** from this meta-ensemble.
* The underlying model (`josephadamski91/ntl-atlas-qwen3-8b-v1`) was cryptographically verified to have been trained directly on the official competition training set, rendering any local out-of-fold evaluation entirely invalid.
