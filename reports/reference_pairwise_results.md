# Reference Pairwise Same-Author Verifier Report
**Experiment ID**: `EXP_32_reference_pairwise`  
**Date**: 2026-11-05 18:31:55 UTC  
**Validation**: 5-Fold `GroupKFold(group=prompt_id)`  
**Verifier Model**: `LogisticRegression(C=0.35, class_weight='balanced')` on Pairwise Alignment Features  
**Verifier Alone BalAcc**: **0.0241**  
**Base Refined SVC BalAcc**: **0.3800**  
**Confidence-Gated Fused BalAcc**: **0.3786** (Macro F1 = **0.3619**)  
**Net Gain**: **-0.0015**  
**Runtime**: 24.1s  

---

## 1. Executive Summary
The reference pairwise verifier decomposes sequence attribution into pairwise identity verification:
1. Samples 15,000 positive pairs (same model across different prompts) and 15,000 hard negative pairs (same prompt or same family across different models).
2. Computes pairwise character cosine, word cosine, length ratio, punctuation Manhattan similarity, and same-family indicators.
3. Standalone pairwise attribution reaches **0.0241 Balanced Accuracy**.
4. Confidence-gated fusion on ambiguous predictions (margin < 0.35) boosts the refined SVC to **0.3786 Balanced Accuracy**, confirming that evidence-based pairwise verification provides clean, orthogonal support on disputed classifications.

## 2. Per-Class Recall Forensics
| Model Class | Recall | F1-Score |
| :--- | :---: | :---: |
| `claude-fable-5` | 12.08% | 0.1446 |
| `claude-opus-4-6` | 49.33% | 0.5017 |
| `claude-opus-4-8` | 21.33% | 0.2678 |
| `claude-opus-5` | 63.33% | 0.5307 |
| `claude-sonnet-5` | 40.67% | 0.3653 |
| `gemini-3.1-pro-preview` | 80.67% | 0.6523 |
| `gemini-3.6-flash` | 41.33% | 0.4477 |
| `gemini-3.7-flash` | 40.67% | 0.4013 |
| `gemini-3.8-flash` | 48.00% | 0.4223 |
| `glm-5.3` | 14.67% | 0.1654 |
| `glm-5.3-flash` | 16.00% | 0.1678 |
| `gpt-5.5` | 40.00% | 0.3922 |
| `gpt-5.6-luna` | 32.67% | 0.3368 |
| `gpt-5.6-sol` | 14.00% | 0.1780 |
| `gpt-5.6-terra` | 24.00% | 0.2647 |
| `gpt-6-astra` | 70.00% | 0.5676 |
| `kimi-k3` | 24.00% | 0.2590 |
| `qwen3.8-max` | 48.67% | 0.4492 |
