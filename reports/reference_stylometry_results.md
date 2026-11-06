# Reference 142d Stylometry & Marker Evaluation Report
**Experiment ID**: `EXP_28_reference_stylometry`  
**Date**: 2026-11-05 18:28:26 UTC  
**Validation**: 5-Fold `GroupKFold(group=prompt_id)`  
**Primary Metric**: Balanced Accuracy = **0.3086** (Macro F1 = **0.3005**)  
**Runtime**: 4.1s  

---

## 1. Executive Summary
The reference 142d stylometric vector extracted from `sibling_style.py` consists of:
- 16 base text statistics (lengths, casing, lines, whitespace, non-ascii)
- 55 structural punctuation and markdown markers
- 71 function word frequencies

Under strict prompt-disjoint GroupKFold, this purely tabular stylometric representation achieves **0.3086 Balanced Accuracy** standalone with 142 features, completely outperforming 384d pretrained semantic embeddings (0.1408).

## 2. Per-Class Recall Forensics
| Model Class | Recall | F1-Score |
| :--- | :---: | :---: |
| `claude-fable-5` | 12.08% | 0.1385 |
| `claude-opus-4-6` | 40.00% | 0.3704 |
| `claude-opus-4-8` | 21.33% | 0.2278 |
| `claude-opus-5` | 44.00% | 0.3826 |
| `claude-sonnet-5` | 32.00% | 0.3117 |
| `gemini-3.1-pro-preview` | 65.33% | 0.5851 |
| `gemini-3.6-flash` | 29.33% | 0.3056 |
| `gemini-3.7-flash` | 42.00% | 0.4118 |
| `gemini-3.8-flash` | 34.67% | 0.3421 |
| `glm-5.3` | 16.00% | 0.1825 |
| `glm-5.3-flash` | 20.67% | 0.2230 |
| `gpt-5.5` | 36.00% | 0.3333 |
| `gpt-5.6-luna` | 20.00% | 0.2190 |
| `gpt-5.6-sol` | 15.33% | 0.1673 |
| `gpt-5.6-terra` | 17.33% | 0.1831 |
| `gpt-6-astra` | 50.00% | 0.4559 |
| `kimi-k3` | 20.67% | 0.2060 |
| `qwen3.8-max` | 38.67% | 0.3636 |
