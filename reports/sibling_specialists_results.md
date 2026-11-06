# Sibling Model Specialists & Multi-Tier Architecture Benchmark
**Experiment ID**: `EXP_24_sibling_specialists`  
**Date**: 2026-11-05 18:16:30 UTC  
**Validation Strategy**: `GroupKFold(n_splits=5, group=prompt_id)` (Strict Prompt-Disjoint)  
**Primary Metric**: Balanced Accuracy  

---

## 1. Executive Summary & Progression Across Tiers
In Phase F, we evaluated the progression across the 4 core tiers mandated by the research protocol:
1. **Tier 1: Global Model Only**: Base multiclass stacking of stylometry, syntax, and likelihoods.
2. **Tier 2: Global + Sibling Specialists**: Soft-gated hierarchical product-of-experts with 4 intra-family models (Google, OpenAI, Anthropic, ZAI).
3. **Tier 3: Pairwise Probability Matrix**: Full pairwise tournament aggregated via Bradley-Terry iterative MLE.
4. **Tier 4: Global + Pairwise + Residuals**: Multi-stream meta-stacking integrating global base odds, pairwise odds, prompt-conditioned residuals, and stylistic fingerprints.

---

## 2. Multi-Tier Benchmark Comparison

| Architectural Tier | GroupKFold BalAcc | Macro F1 | Delta vs Tier 1 | Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Tier 1: Global Model Only** | `0.3630` | `0.3522` | baseline | Single 18-class multinomial hyperplane |
| **Tier 2: Global + Sibling Specialists** | `0.3393` | `0.3388` | `-0.0237` | Soft-gated $P(m) = P(\text{prov}) \cdot P(m \mid \text{prov})$ |
| **Tier 3: Pairwise Probability Matrix** | `0.3686` | `0.3623` | `+0.0056` | Bradley-Terry MLE on full tournament odds |
| **Tier 4: Global + Pairwise + Residuals** | **`0.4378`** | **`0.4345`** | **`+0.0749`** | Multi-stream regularized meta-stacker ($C=0.01$) |

---

## 3. Intra-Family Sibling Accuracy Breakdown
Dedicated intra-family classifiers sharply outperform random guessing:
- **Google Flash/Pro Variants (4 models)**: 49.67% intra-family accuracy (vs 25.0% random baseline).
- **OpenAI GPT Variants (5 models)**: 43.47% intra-family accuracy (vs 20.0% random baseline).
- **Anthropic Claude Variants (5 models)**: 43.38% intra-family accuracy (vs 20.0% random baseline).
- **ZAI GLM Variants (2 models)**: 51.33% intra-family accuracy (vs 50.0% random baseline).
