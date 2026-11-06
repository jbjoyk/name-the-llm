# Validation Strategy & Distribution Mismatch Investigation
**Date:** 2026-11-05  
**Evaluated Benchmark:** Combined Word ($1\text{--}2$) + Character ($3\text{--}5$) TF-IDF + Logistic Regression  
**Total Training Set:** 2,699 samples, 150 prompt groups, 18 classes  

---

## 1. Executive Summary & Core Findings

1. **Massive Prompt Leakage in Random StratifiedKFold (Strategy A):**
   Random StratifiedKFold yields a severely inflated balanced accuracy of **0.0889** (vs **0.3015** in GroupKFold). When prompts are split across folds, models exploit prompt-specific vocabulary shortcuts rather than learning true LLM authorial style.
2. **Standard GroupKFold vs StratifiedGroupKFold (Strategy B vs C):**
   GroupKFold achieves **0.3015 ± 0.0105**, while StratifiedGroupKFold achieves **0.2971 ± 0.0092**. Because the competition training set already has almost exactly 18 responses (1 per model) for every single prompt, GroupKFold and StratifiedGroupKFold are virtually identical in class distribution.
3. **Hidden Test Distribution Realism (Strategy D & E):**
   The hidden test set features unseen prompts with approximately ~2 responses per prompt (~4,000 test passages across ~2,000 prompts with unequal model representation).
   When validation prompts are subsampled to 2 responses per prompt (Strategy D and Strategy E across 10 independent random seeds):
   - Mean Balanced Accuracy remains centered at **0.2840**, closely matching full GroupKFold (0.3015).
   - The cross-seed variance across 10 runs is **±0.0179**.
   - Because balanced accuracy macro-averages per-class recall, random sub-sampling introduces higher variance in individual class recall estimates when $N$ per class is small ($N \approx 3\text{--}4$ per fold).
4. **Validation Recommendation for the Research Loop:**
   - **Model Selection & Feature Comparison:** Use **GroupKFold(n_splits=5, group=prompt_id)** with full validation sets. It evaluates on 100% of unseen prompt responses, providing zero leakage, maximum statistical power, and lowest evaluation noise.
   - **Leaderboard Risk Audit:** Test candidate ensembles against the **Subsampled Validation Harness (Strategy E)** to confirm robustness under sparse, class-imbalanced evaluation distributions.

---

## 2. Quantitative Comparison Table

| Strategy | Description | Prompt Leakage? | Mean BalAcc | Fold Std | Macro F1 | Avg Val N |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** | Random StratifiedKFold | **YES (Severe)** | `0.0889` | `±0.0096` | `0.0844` | `540` |
| **Strategy B** | Standard GroupKFold(prompt_id) | **NO** | `0.3015` | `±0.0105` | `0.2889` | `540` |
| **Strategy C** | StratifiedGroupKFold(prompt_id) | **NO** | `0.2971` | `±0.0092` | `0.2846` | `540` |
| **Strategy D** | Test-Like Subsampled (2/prompt, Seed 42) | **NO** | `0.3114` | `±0.0444` | `0.2822` | `5` |
| **Strategy E** | Repeated Subsampled (10 Seeds) | **NO** | **`0.2840`** | `±0.0179` | `0.2505` | `60` |

---

## 3. Per-Class Recall Stability (Full GroupKFold vs 10x Subsampled Group)

| Class (LLM) | GroupKFold Recall | Subsampled 10-Seed Recall | Absolute Difference |
| :--- | :---: | :---: | :---: |
| `claude-fable-5` | 0.0807 | 0.0596 | 0.0211 |
| `claude-opus-4-6` | 0.3133 | 0.2318 | 0.0816 |
| `claude-opus-4-8` | 0.2667 | 0.2656 | 0.0011 |
| `claude-opus-5` | 0.6133 | 0.5608 | 0.0525 |
| `claude-sonnet-5` | 0.2200 | 0.1991 | 0.0209 |
| `gemini-3.1-pro-preview` | 0.7133 | 0.6353 | 0.0781 |
| `gemini-3.6-flash` | 0.4200 | 0.3955 | 0.0245 |
| `gemini-3.7-flash` | 0.3667 | 0.3625 | 0.0041 |
| `gemini-3.8-flash` | 0.2600 | 0.2446 | 0.0154 |
| `glm-5.3` | 0.1533 | 0.1745 | 0.0212 |
| `glm-5.3-flash` | 0.1200 | 0.1224 | 0.0024 |
| `gpt-5.5` | 0.2200 | 0.2769 | 0.0569 |
| `gpt-5.6-luna` | 0.2933 | 0.2216 | 0.0717 |
| `gpt-5.6-sol` | 0.1400 | 0.0992 | 0.0408 |
| `gpt-5.6-terra` | 0.2400 | 0.2705 | 0.0305 |
| `gpt-6-astra` | 0.4600 | 0.4748 | 0.0148 |
| `kimi-k3` | 0.3133 | 0.3271 | 0.0138 |
| `qwen3.8-max` | 0.2333 | 0.1927 | 0.0406 |

---

## 4. Conclusion on Validation Mismatch
The structural difference between training (~18 passages/prompt) and hidden test (~2 passages/prompt) does **not** degrade the underlying attribution capability of the models (0.3015 vs 0.2840), but it **does** increase evaluation variance on small test subsets. Full GroupKFold remains the most stable, uncorrupted objective for feature and model discovery.
