# EXP_28: Official + Cleaned Open Pool Multi-View LinearSVC Report
**Experiment ID**: `EXP_28_open_pool_svc`  
**Date**: 2026-11-05 18:50:24 UTC  
**Validation Strategy**: 5-Fold `GroupKFold(group=prompt_id)` (Strict Prompt-Disjoint)  
**Training Data**: Official Data (2,159 rows, w=1.0) + Clean Open Pool (~36,000 rows, w=0.35)  
**Primary Metric**: Balanced Accuracy = **0.4630** (Macro F1 = **0.4494**)  
**Fold Scores**: [0.45, 0.463, 0.4444, 0.4889, 0.4686] (Std: ±0.0156)  
**Runtime**: 402.4s  
**Contamination Filtering**: Purged an average of 0.4 near-prompt rows per fold.  

---

## 1. Executive Summary & Impact Analysis
- **Official 4-View SVC Baseline (`EXP_27`)**: `0.3749` Balanced Accuracy
- **Official + Clean Open Pool 4-View SVC (`EXP_28`)**: **`0.4630`** Balanced Accuracy
- **Net Delta**: **`+0.0881`** (`+8.81%` relative gain)

## 2. Per-Class Recall Forensics
| Model Class | Recall | F1-Score |
| :--- | :---: | :---: |
| `claude-fable-5` | 16.78% | 0.1908 |
| `claude-opus-4-6` | 65.33% | 0.6577 |
| `claude-opus-4-8` | 40.67% | 0.4326 |
| `claude-opus-5` | 68.67% | 0.6338 |
| `claude-sonnet-5` | 62.00% | 0.5196 |
| `gemini-3.1-pro-preview` | 81.33% | 0.7821 |
| `gemini-3.6-flash` | 58.00% | 0.5724 |
| `gemini-3.7-flash` | 44.00% | 0.4204 |
| `gemini-3.8-flash` | 47.33% | 0.4383 |
| `glm-5.3` | 15.33% | 0.1769 |
| `glm-5.3-flash` | 23.33% | 0.2465 |
| `gpt-5.5` | 46.00% | 0.4678 |
| `gpt-5.6-luna` | 37.33% | 0.4179 |
| `gpt-5.6-sol` | 29.33% | 0.3372 |
| `gpt-5.6-terra` | 30.67% | 0.3433 |
| `gpt-6-astra` | 70.67% | 0.6386 |
| `kimi-k3` | 20.67% | 0.2551 |
| `qwen3.8-max` | 76.00% | 0.5588 |
