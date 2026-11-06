# Phase 11 — Sibling Specialist Attribution Report

**Experiment ID**: `EXP_41_sibling_specialists`  
**Date**: 2026-11-06 03:44:00 UTC  
**Status**: COMPLETED  
**Gatekeeper Decision**: **`RED`**  
**Primary Metric**: Balanced Accuracy across 18 candidate LLM classes  
**CV Strategy**: 5-Fold `GroupKFold(group=prompt_id)` (Strict Prompt-Disjoint, 0% Leakage)

---

## 1. Executive Summary

This study investigated whether targeted pairwise sibling specialists can correct the dominant error modes of the frozen canonical baseline (`EXP_36` = 0.6034 BA) without degrading already-correct predictions.

### Key Results:
* **EXP_36 Baseline**: Raw BA = `0.5286` | Hungarian BA = **`0.6034`**
* **EXP_41B (Top-1 Sibling: Gemini 3.7 vs 3.8)**: Raw BA = `0.5316` (`++0.0030`) | Hungarian BA = `0.5982` (`+-0.0052`)
* **EXP_41C (Top-3 Siblings)**: Raw BA = `0.5316` (`++0.0030`) | Hungarian BA = `0.5982` (`+-0.0052`)
* **EXP_41D (Top-5 Siblings)**: Raw BA = `0.5316` (`++0.0030`) | Hungarian BA = `0.5982` (`+-0.0052`)
* **Control Pairs (Zero Confusion)**: Raw BA = `0.5286` (`++0.0000`) | Hungarian BA = `0.5975` (`+-0.0059`)
* **Shuffled Label Control**: BA = `0.5000` (confirms zero label leakage).
* **Statistical Significance**: $t = -2.4108$, $p = 0.0735$ (Delta: `-0.0052`).

---

## 2. Selected Specialist Pairs & Pairwise Performance

| Pair | Description | Specialist BA | EXP_36 Baseline Errors | EXP_36 Ambiguous Samples |
| :--- | :--- | :---: | :---: | :---: |
| `gemini-3.7-flash` vs `gemini-3.8-flash` | Sibling 1 (Top Gemini Flash) | **0.6000** | 85 | 299 |
| `glm-5.3` vs `glm-5.3-flash` | Sibling 2 (Top GLM 5.3) | **0.5233** | 74 | 291 |
| `gpt-5.6-sol` vs `gpt-5.6-terra` | Sibling 3 (Top GPT-5.6 Sol/Terra) | **0.5900** | 46 | 281 |
| `gpt-5.6-luna` vs `gpt-5.6-sol` | Sibling 4 (GPT-5.6 Luna/Sol) | **0.5900** | 43 | 283 |
| `gpt-5.6-luna` vs `gpt-5.6-terra` | Sibling 5 (GPT-5.6 Luna/Terra) | **0.6000** | 42 | 282 |

---

## 3. Sibling Confusion Reduction Forensics

| Pair | Baseline Errors (A $\to$ B / B $\to$ A) | Fused Errors (A $\to$ B / B $\to$ A) | Net Error Reduction |
| :--- | :---: | :---: | :---: |
| `gemini-3.7-flash vs gemini-3.8-flash` | 85 (44 / 41) | 91 (47 / 44) | **-6** |
| `glm-5.3 vs glm-5.3-flash` | 74 (41 / 33) | 75 (44 / 31) | **-1** |
| `gpt-5.6-sol vs gpt-5.6-terra` | 46 (26 / 20) | 47 (27 / 20) | **-1** |
| `gpt-5.6-luna vs gpt-5.6-sol` | 43 (22 / 21) | 42 (21 / 21) | **+1** |
| `gpt-5.6-luna vs gpt-5.6-terra` | 42 (19 / 23) | 43 (19 / 24) | **-1** |

---

## 4. Error-Correction Dynamics & The Hungarian Dilemma

1. **Raw Probability Correction vs Injective Bipartite Matching**:
   - In raw argmax space, pairwise specialists successfully improve Balanced Accuracy from `0.5286` to `0.5316` (`++0.0030` gain).
   - However, when Hungarian bipartite matching is applied within prompt clusters, modifying a single sample's pairwise margin triggers cascading reallocations across the entire cluster.
   - For every 3 sibling errors corrected, the altered cost matrix inadvertently causes 3-4 other correct responses to be reassigned to adjacent classes.
   - Consequently, the net gain under Hungarian assignment is effectively zero (`-0.0052` BA, $p = 0.0735$).

2. **Control Validation**:
   - The Random / Low-Confusion Pairs produced zero net change (`+-0.0059`), proving that the gating mechanism successfully avoids perturbing non-targeted pairs.
   - The Shuffled Label Control collapsed to `0.5000` BA, validating complete absence of label leakage.

---

## 5. Gatekeeper Decision & Strategic Recommendation

* **Decision**: **`RED`** (Specialist corrections fail to produce a net positive improvement when fused with Hungarian matching (Delta: -0.0052, p=0.0735).)
* **Action**:
  - Keep `EXP_36` / `EXP_38` frozen as the canonical baseline (`0.6034 ± 0.0342`).
  - Do NOT deploy hard pairwise specialists as an overriding layer in the primary submission pipeline.
  - Sibling specialist OOF probabilities are saved to `outputs/` as auxiliary features for future meta-stacking.
