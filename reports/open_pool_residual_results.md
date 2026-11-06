# EXP_29: Official + Open Pool + 4-View SVC + Prompt Residual Report
**Experiment ID**: `EXP_29_open_pool_residual`  
**Date**: 2026-11-05 18:50:38 UTC  
**Validation Strategy**: 5-Fold `GroupKFold(group=prompt_id)` (Strict Prompt-Disjoint Nested CV)  
**Primary Metric**: Balanced Accuracy = **0.4712** (Macro F1 = **0.4709**)  
**Raw Blended Balanced Accuracy**: **0.4575**  
**Fold Scores (Hungarian)**: [0.4722, 0.4519, 0.5037, 0.4685, 0.4597] (Std: ±0.0177)  
**Runtime**: 0.4s  

---

## 1. Executive Summary & Impact Analysis
- **EXP_27 Official 4-View SVC**: `0.3749` BalAcc
- **EXP_28 Open Pool 4-View SVC**: `0.4630` BalAcc
- **EXP_29 Open Pool + Prompt Residual + Hungarian**: **`0.4712`** BalAcc
- **Net Gain vs Official Baseline**: **`+0.0963`**

## 2. Per-Class Recall Forensics
| Model Class | Recall | F1-Score |
| :--- | :---: | :---: |
| `claude-fable-5` | 21.48% | 0.2140 |
| `claude-opus-4-6` | 59.33% | 0.5914 |
| `claude-opus-4-8` | 49.33% | 0.4868 |
| `claude-opus-5` | 68.00% | 0.6800 |
| `claude-sonnet-5` | 46.67% | 0.4667 |
| `gemini-3.1-pro-preview` | 76.67% | 0.7616 |
| `gemini-3.6-flash` | 59.33% | 0.5953 |
| `gemini-3.7-flash` | 46.00% | 0.4710 |
| `gemini-3.8-flash` | 49.33% | 0.4868 |
| `glm-5.3` | 25.33% | 0.2533 |
| `glm-5.3-flash` | 23.33% | 0.2341 |
| `gpt-5.5` | 46.00% | 0.4539 |
| `gpt-5.6-luna` | 39.33% | 0.4000 |
| `gpt-5.6-sol` | 32.00% | 0.3221 |
| `gpt-5.6-terra` | 35.33% | 0.3569 |
| `gpt-6-astra` | 71.33% | 0.7157 |
| `kimi-k3` | 46.00% | 0.4554 |
| `qwen3.8-max` | 53.33% | 0.5316 |
