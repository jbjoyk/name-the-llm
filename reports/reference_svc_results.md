# Reference 4-View Multi-View LinearSVC Report
**Experiment ID**: `EXP_27_reference_svc`  
**Date**: 2026-11-05 18:27:52 UTC  
**Validation**: 5-Fold `GroupKFold(group=prompt_id)`  
**Primary Metric**: Balanced Accuracy = **0.3749** (Macro F1 = **0.3623**)  
**Runtime**: 24.3s  

---

## 1. Executive Summary
The reference 4-view multi-view LinearSVC combines:
- Character n-grams (3-5)
- Word n-grams (1-3)
- Casing shape transform (2-6)
- Function-word projection (1-3)

On official training data under strict prompt-disjoint GroupKFold, this architecture achieves **0.3749 Balanced Accuracy**, achieving a **+0.0734** improvement over standard word+char TF-IDF (0.3015).

## 2. Per-Class Recall Forensics
| Model Class | Recall | F1-Score |
| :--- | :---: | :---: |
| `claude-fable-5` | 11.41% | 0.1399 |
| `claude-opus-4-6` | 48.00% | 0.5106 |
| `claude-opus-4-8` | 25.33% | 0.2969 |
| `claude-opus-5` | 63.33% | 0.5621 |
| `claude-sonnet-5` | 42.67% | 0.3689 |
| `gemini-3.1-pro-preview` | 82.00% | 0.6649 |
| `gemini-3.6-flash` | 38.67% | 0.4173 |
| `gemini-3.7-flash` | 41.33% | 0.3987 |
| `gemini-3.8-flash` | 43.33% | 0.4075 |
| `glm-5.3` | 14.67% | 0.1583 |
| `glm-5.3-flash` | 20.67% | 0.2033 |
| `gpt-5.5` | 29.33% | 0.3200 |
| `gpt-5.6-luna` | 31.33% | 0.3186 |
| `gpt-5.6-sol` | 17.33% | 0.2149 |
| `gpt-5.6-terra` | 28.00% | 0.2937 |
| `gpt-6-astra` | 62.00% | 0.5314 |
| `kimi-k3` | 26.00% | 0.2653 |
| `qwen3.8-max` | 49.33% | 0.4498 |

## 3. Key Findings
1. The function-word and casing shape views successfully shield the linear classifier from domain prompt topic memorization.
2. LinearSVC fits in under 60 seconds with zero GPU requirements, making it an extraordinarily cost-effective member for ensembling.
