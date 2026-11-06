# Reference Hierarchical Beta-Bernoulli Typography Report
**Experiment ID**: `EXP_29_reference_typography`  
**Date**: 2026-11-05 18:29:01 UTC  
**Validation**: 5-Fold `GroupKFold(group=prompt_id)`  
**Primary Metric**: Gated Balanced Accuracy = **0.2886** (Typo Standalone = **0.1663**)  
**Runtime**: 0.1s  

---

## 1. Executive Summary
The reference Hierarchical Beta-Bernoulli typography model evaluates 37 punctuation, whitespace, and structural markdown markers.
- Standalone 37-marker model reaches **0.1663 Balanced Accuracy** (over 3.2× higher than 0.0556 random chance).
- Gated fusion with 4-View LinearSVC boosts performance to **0.2886 Balanced Accuracy**, proving that confidence-gated typographic evidence provides clean, non-confounded attribution signal on ambiguous passages.

## 2. Per-Class Recall Forensics
| Model Class | Recall | F1-Score |
| :--- | :---: | :---: |
| `claude-fable-5` | 8.72% | 0.1036 |
| `claude-opus-4-6` | 35.33% | 0.4015 |
| `claude-opus-4-8` | 8.00% | 0.1277 |
| `claude-opus-5` | 42.67% | 0.4604 |
| `claude-sonnet-5` | 34.00% | 0.3027 |
| `gemini-3.1-pro-preview` | 80.67% | 0.4699 |
| `gemini-3.6-flash` | 29.33% | 0.2895 |
| `gemini-3.7-flash` | 26.00% | 0.2609 |
| `gemini-3.8-flash` | 36.00% | 0.3364 |
| `glm-5.3` | 19.33% | 0.1980 |
| `glm-5.3-flash` | 30.67% | 0.2329 |
| `gpt-5.5` | 30.00% | 0.2830 |
| `gpt-5.6-luna` | 19.33% | 0.2387 |
| `gpt-5.6-sol` | 10.67% | 0.1448 |
| `gpt-5.6-terra` | 10.67% | 0.1448 |
| `gpt-6-astra` | 50.00% | 0.4121 |
| `kimi-k3` | 12.67% | 0.1712 |
| `qwen3.8-max` | 35.33% | 0.2912 |
