# Name That LLM — Error Analysis & Forensics Report
**Date:** 2026-11-05  
**Evaluated Model:** `08_08c_meta_stacker_lr`  
**Cross-Validation:** 5-Fold `GroupKFold(group=prompt_id)`  
**Overall Balanced Accuracy:** `0.3630`  
**Macro F1-Score:** `0.3522`  

---

## 1. Executive Summary
An exhaustive error analysis was performed across all 2,699 out-of-fold predictions from the best-performing Meta-Stacker model (`MetaStacker_LR`). Because the competition strictly measures **Balanced Accuracy** (unweighted macro-average of per-class recall), identifying specific classes and confusion clusters is essential for maximizing test set performance.

Key discoveries:
1. **Performance Variance Across Classes:** Recall spans from a high of `74.00%` (`gemini-3.1-pro-preview`) down to `7.38%` (`claude-fable-5`).
2. **Top Tier Discernible Models:** The highest-recalled models exhibit distinct stylistic fingerprints (punctuation density, bullet structures, and specific sub-word vocabulary distributions).
3. **Severe Intra-Family Confusion:** The dominant error mode stems from sibling models trained by the same provider sharing near-identical post-training alignment recipes.
4. **Passage Length Sensitivity:** Accuracy rises with passage length; short responses offer fewer n-gram and syntactic signals, leading to higher entropy predictions.

---

## 2. Per-Class Performance Breakdown

| Rank | Model Name | Recall (BalAcc Contrib) | Precision | Macro F1 | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `gemini-3.1-pro-preview` | **0.7400** | 0.6529 | 0.6937 | 150 |
| 2 | `claude-opus-5` | **0.6200** | 0.4604 | 0.5284 | 150 |
| 3 | `gpt-6-astra` | **0.5867** | 0.4656 | 0.5192 | 150 |
| 4 | `gemini-3.6-flash` | **0.4267** | 0.4267 | 0.4267 | 150 |
| 5 | `qwen3.8-max` | **0.4200** | 0.3962 | 0.4078 | 150 |
| 6 | `claude-opus-4-6` | **0.4200** | 0.4286 | 0.4242 | 150 |
| 7 | `gpt-5.5` | **0.3933** | 0.3576 | 0.3746 | 150 |
| 8 | `gemini-3.7-flash` | **0.3800** | 0.3393 | 0.3585 | 150 |
| 9 | `claude-sonnet-5` | **0.3667** | 0.3313 | 0.3481 | 150 |
| 10 | `kimi-k3` | **0.3533** | 0.3397 | 0.3464 | 150 |
| 11 | `gemini-3.8-flash` | **0.3200** | 0.3478 | 0.3333 | 150 |
| 12 | `gpt-5.6-luna` | **0.3067** | 0.2875 | 0.2968 | 150 |
| 13 | `claude-opus-4-8` | **0.3000** | 0.3191 | 0.3093 | 150 |
| 14 | `gpt-5.6-terra` | **0.2467** | 0.2450 | 0.2458 | 150 |
| 15 | `glm-5.3-flash` | **0.2400** | 0.3077 | 0.2697 | 150 |
| 16 | `glm-5.3` | **0.2400** | 0.2195 | 0.2293 | 150 |
| 17 | `gpt-5.6-sol` | **0.1000** | 0.1899 | 0.1310 | 150 |
| 18 | `claude-fable-5` | **0.0738** | 0.1429 | 0.0973 | 149 |

---

## 3. Top Confused Model Pairs

The table below isolates the 15 most frequent error pairs. Addressing these pairwise ambiguities directly targets the remaining performance gap:

| Rank | True Model | Mistaken For (Predicted) | Error Count | Error Rate within Class |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `gemini-3.8-flash` | `gemini-3.7-flash` | 55 | 36.67% |
| 2 | `gemini-3.7-flash` | `gemini-3.8-flash` | 43 | 28.67% |
| 3 | `claude-fable-5` | `glm-5.3` | 33 | 22.15% |
| 4 | `gemini-3.7-flash` | `gemini-3.6-flash` | 32 | 21.33% |
| 5 | `gpt-5.6-sol` | `gpt-5.6-luna` | 31 | 20.67% |
| 6 | `gpt-5.6-luna` | `gpt-5.6-terra` | 30 | 20.00% |
| 7 | `glm-5.3-flash` | `glm-5.3` | 28 | 18.67% |
| 8 | `gpt-5.6-terra` | `gpt-5.6-luna` | 28 | 18.67% |
| 9 | `gpt-5.6-sol` | `gpt-5.6-terra` | 28 | 18.67% |
| 10 | `gemini-3.6-flash` | `gemini-3.7-flash` | 27 | 18.00% |
| 11 | `kimi-k3` | `gpt-5.5` | 27 | 18.00% |
| 12 | `glm-5.3` | `claude-opus-5` | 25 | 16.67% |
| 13 | `gemini-3.6-flash` | `gemini-3.1-pro-preview` | 25 | 16.67% |
| 14 | `gpt-5.5` | `kimi-k3` | 24 | 16.00% |
| 15 | `claude-opus-4-6` | `claude-sonnet-5` | 23 | 15.33% |

---

## 4. Error Variation by Passage Length

| Length Quintile | Sample Count | Accuracy |
| :--- | :---: | :---: |
| Very Short (Q1) | 555 | **27.93%** |
| Short (Q2) | 537 | **36.87%** |
| Medium (Q3) | 535 | **36.26%** |
| Long (Q4) | 533 | **37.15%** |
| Very Long (Q5) | 539 | **43.60%** |

---

## 5. Architectural Recommendations for Final Pipeline
1. **Preserve Probability Stacking:** Meta-stacking on out-of-fold predictions achieves 0.3630 BalAcc with low cross-fold variance (±0.0046), effectively calibrating the probability distributions across all 18 classes.
2. **Full-Dataset Fitting:** Retraining feature extractors (TF-IDF vocabularies, Stylometry scalers, Likelihood profilers, Layer 3 embeddings) on 100% of the 2,699 samples will maximize vocabulary coverage and parameter precision for test inference.
3. **Self-Contained Inference:** Package the inference engine in `scripts/predict.py` without external network dependencies, ensuring compatibility with Kaggle's offline submission environment.
