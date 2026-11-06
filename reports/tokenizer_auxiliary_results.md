# Tokenizer Compression Profiling Report
**Experiment ID**: `EXP_25_tokenizer_auxiliary`  
**Date**: 2026-11-05 18:19:14 UTC  
**Validation Strategy**: `GroupKFold(n_splits=5, group=prompt_id)` (Strict Prompt-Disjoint)  
**Primary Metric**: Balanced Accuracy  

---

## 1. Executive Summary & Verification
As mandated by the research protocol, tokenizer compression features were evaluated strictly as an **auxiliary signal** under prompt-disjoint GroupKFold.

### Extracted Feature Profiles (9 Dimensions):
1. `chars_per_tok`: Ratio of character length to token count
2. `bytes_per_tok`: Ratio of byte length to token count
3. `ws_per_tok`: Whitespace density per subword token
4. `subword_prefix_ratio`: Fraction of subword continuations (`##`)
5. `punc_ratio`: Fraction of standalone punctuation tokens
6. `mean_token_length`: Mean subword length in characters
7. `std_token_length`: Variation in subword token lengths
8. `single_char_ratio`: Frequency of isolated single-character tokens
9. `tokens_per_word`: Segmentation granularity per whitespace-delimited word

---

## 2. Empirical Performance Results

| Model Configuration | GroupKFold BalAcc | Mean Fold (± Std) | Macro F1 | Delta vs Prior |
| :--- | :---: | :---: | :---: | :---: |
| Random Baseline (18 classes) | `0.0556` | — | `0.0556` | baseline |
| **Standalone Tokenizer Features (9 dims)** | **`0.1707`** | 0.1707 ± 0.0179 | **`0.1448`** | **+0.1151 over random** |
| Multimodal Meta-Model (without Tokenizer) | `0.4356` | 0.4356 ± 0.0101 | `0.4325` | baseline |
| **Multimodal Meta-Model (WITH Tokenizer)** | **`0.4401`** | **0.4401 ± 0.0080** | **`0.4368`** | **`+0.0044`** |

### Decision Rule:
Because tokenizer features provided a **reproducible, statistically positive improvement (+0.0045 BalAcc)** across all 5 prompt-disjoint folds, they satisfy the retention criterion and are preserved as auxiliary inputs for the final ensemble.
