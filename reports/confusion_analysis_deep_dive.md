# Exhaustive Confusion Analysis & Attribution Forensics
**Date:** 2026-11-05  
**Evaluated Architecture:** `08_08c_meta_stacker_lr`  
**Cross-Validation:** 5-Fold `GroupKFold(group=prompt_id)`  
**Overall Balanced Accuracy:** `0.3630`  
**Total Errors Analyzed:** `1719` out of `2699` samples  

---

## 1. Provider Isolation & Error Distribution
- **Same-Provider Errors:** **`913`** out of `1719` (**`53.11%`** of all misclassifications!).
- **Cross-Provider Errors:** **`806`** out of `1719` (**`46.89%`**).
- **Core Insight:** Over **half of all classification mistakes** occur among models developed by the same lab. The global model attribution problem decomposes naturally into:
  1. *Cross-provider discrimination* (distinguishing OpenAI vs Google vs Anthropic vs ZAI vs Qwen vs Moonshot), which the model performs with relatively high precision.
  2. *Intra-provider sibling disambiguation* (e.g. `gemini-3.8-flash` vs `gemini-3.7-flash` or `gpt-5.6-luna` vs `gpt-5.6-terra`), which constitutes the primary barrier to 0.50+ and 0.70+ balanced accuracy.

---

## 2. Easiest vs Hardest Models to Attribute

### Easiest Models (High Attribution Recall)
| Model | Provider | Recall | Precision | F1 | Key Stylistic Signature |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `gemini-3.1-pro-preview` | Google | **74.00%** | 65.29% | 0.6937 | Highly structured, distinct rhetorical questions, specific transition tokens |
| `claude-opus-5` | Anthropic | **62.00%** | 46.04% | 0.5284 | Distinctive intellectual tone, rich punctuation, em-dashes |
| `gpt-6-astra` | OpenAI | **58.67%** | 46.56% | 0.5192 | Unique syntactic cadence, advanced vocabulary distributions |
| `gemini-3.6-flash` | Google | **42.67%** | 42.67% | 0.4267 | Conciseness, direct factual tone |
| `qwen3.8-max` | Qwen | **42.00%** | 39.62% | 0.4078 | Distinctive punctuation patterns and lexical diversity |

### Hardest Models (Low Recall / Heavily Confused)
| Model | Provider | Recall | Main Confuser | Confusion Rate | Root Cause |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `claude-fable-5` | Anthropic | **7.38%** | `glm-5.3` | 22.15% | Short passages; lacks strong Anthropic Opus signature |
| `gpt-5.6-sol` | OpenAI | **10.00%** | `gpt-5.6-luna` | 20.67% | Sibling to Luna & Terra; near-identical token distributions |
| `glm-5.3` | ZAI | **24.00%** | `claude-opus-5` | 16.67% | Overlaps with long expository passages |
| `glm-5.3-flash` | ZAI | **24.00%** | `glm-5.3` | 18.67% | Direct sibling confusion with full GLM-5.3 |
| `gpt-5.6-terra` | OpenAI | **24.67%** | `gpt-5.6-luna` | 18.67% | Direct sibling confusion with Luna & Sol |

---

## 3. Top 15 Confusion Pairs Forensics

| True Model (`model_a`) | Mistaken Model (`model_b`) | Errors | Cond. Rate | Same Provider? | Stylistic Forensic Clue |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `gemini-3.8-flash` | `gemini-3.7-flash` | **55** | **36.67%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `gemini-3.7-flash` | `gemini-3.8-flash` | **43** | **28.67%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `claude-fable-5` | `glm-5.3` | **33** | **22.15%** | No | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `gemini-3.7-flash` | `gemini-3.6-flash` | **32** | **21.33%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `gpt-5.6-sol` | `gpt-5.6-luna` | **31** | **20.67%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `gpt-5.6-luna` | `gpt-5.6-terra` | **30** | **20.00%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `glm-5.3-flash` | `glm-5.3` | **28** | **18.67%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `gpt-5.6-terra` | `gpt-5.6-luna` | **28** | **18.67%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `gpt-5.6-sol` | `gpt-5.6-terra` | **28** | **18.67%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `gemini-3.6-flash` | `gemini-3.7-flash` | **27** | **18.00%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `kimi-k3` | `gpt-5.5` | **27** | **18.00%** | No | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `glm-5.3` | `claude-opus-5` | **25** | **16.67%** | No | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `gemini-3.6-flash` | `gemini-3.1-pro-preview` | **25** | **16.67%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `gpt-5.5` | `kimi-k3` | **24** | **16.00%** | No | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |
| `claude-opus-4-6` | `claude-sonnet-5` | **23** | **15.33%** | **YES** | Near-identical surface length and punctuation; differs primarily in sub-word transition vocabulary and phrasing |

---

## 4. Strategic Implications for the Research Roadmap
1. **Targeted Pairwise Attribution (Phase 3):** Focus specialized binary classifiers specifically on the top intra-family pairs (`gemini-3.8-flash` vs `gemini-3.7-flash`, `gpt-5.6-luna` vs `gpt-5.6-sol`, `glm-5.3-flash` vs `glm-5.3`).
2. **Hierarchical Attribution (Phase 4):** A two-stage pipeline (Provider Classifier -> Sibling Disambiguation) has massive potential because provider boundaries are much cleaner than flat 18-class argmax.
3. **Fingerprint Features (Phase 5):** Sibling models share topics and general syntax, but differ in sub-word tokenizer artifacts, parenthetical frequency, and punctuation cadence.
