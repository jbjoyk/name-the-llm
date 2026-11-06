# Reference Sibling-Style Residual Refinement Report
**Experiment ID**: `EXP_31_reference_sibling_style`  
**Date**: 2026-11-05 18:30:37 UTC  
**Validation**: 5-Fold `GroupKFold(group=prompt_id)`  
**Baseline Model**: 4-View LinearSVC (`0.3749` BalAcc)  
**Refined Model**: **`0.3800` BalAcc** (Macro F1 = **`0.3632`**)  
**Net Gain**: **`+0.0052`**  
**Runtime**: 0.3s  

---

## 1. Executive Summary
The reference sibling style residual mechanism re-ranks disputed intra-family candidates using 142d stylometric features:
- **Intra-Family OOF Accuracies**:
  - Claude family (5 models): `42.17%` (vs 20.0% chance)
  - Gemini family (4 models): `48.00%` (vs 25.0% chance)
  - GPT family (5 models): `41.47%` (vs 20.0% chance)
  - GLM family (2 models): `52.67%` (vs 50.0% chance)
- **Conservation of Provider Probability**: By preserving exact provider-level logsumexp, this method improves same-lab resolution without introducing cross-family error cascades.
- **Result**: Boosts 4-View LinearSVC from **0.3749** to **0.3800 Balanced Accuracy**.

## 2. Per-Class Recall Forensics
| Model Class | Recall | F1-Score |
| :--- | :---: | :---: |
| `claude-fable-5` | 12.08% | 0.1513 |
| `claude-opus-4-6` | 49.33% | 0.5086 |
| `claude-opus-4-8` | 22.00% | 0.2739 |
| `claude-opus-5` | 66.67% | 0.5348 |
| `claude-sonnet-5` | 40.67% | 0.3664 |
| `gemini-3.1-pro-preview` | 80.67% | 0.6630 |
| `gemini-3.6-flash` | 42.67% | 0.4507 |
| `gemini-3.7-flash` | 42.00% | 0.4026 |
| `gemini-3.8-flash` | 46.67% | 0.4230 |
| `glm-5.3` | 13.33% | 0.1544 |
| `glm-5.3-flash` | 16.67% | 0.1724 |
| `gpt-5.5` | 39.33% | 0.3894 |
| `gpt-5.6-luna` | 33.33% | 0.3333 |
| `gpt-5.6-sol` | 14.67% | 0.1811 |
| `gpt-5.6-terra` | 24.00% | 0.2687 |
| `gpt-6-astra` | 68.00% | 0.5484 |
| `kimi-k3` | 24.67% | 0.2681 |
| `qwen3.8-max` | 47.33% | 0.4479 |
