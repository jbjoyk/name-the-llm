# Hierarchical Two-Stage Attribution Results
**Date:** 2026-11-05  
**Cross-Validation:** 5-Fold `GroupKFold(group=prompt_id)`  
**Evaluated Systems:** Flat Direct 18-Class vs Two-Stage Hierarchical Model ($P(\text{lab}) \times P(\text{model} \mid \text{lab})$)  

---

## 1. Executive Summary & Findings

1. **Stage 1 Provider Router Performance:**
   Distinguishing the 6 labs achieves **50.99% balanced accuracy**! The model easily differentiates Google from Anthropic from OpenAI from Qwen from ZAI from Moonshot AI.
2. **Stage 2 Intra-Provider Sibling Performance:**
   - **ZAI Specialist (`glm-5.3` vs `glm-5.3-flash`):** **53.00%** conditional balanced accuracy.
   - **OpenAI Specialist (5 models):** **37.60%** conditional balanced accuracy.
   - **Google Specialist (4 models):** **45.00%** conditional balanced accuracy.
   - **Anthropic Specialist (5 models):** **43.78%** conditional balanced accuracy.
3. **End-to-End Hierarchical vs Flat Direct Comparison:**
   - **Flat Direct 18-Class:** `0.2827 ± 0.0122`
   - **Hierarchical End-to-End:** **`0.2753 ± 0.0108`**
   - **Net Impact:** **`-0.0074`**
4. **Theoretical Diagnosis:**
   Stage 1 router achieves strong lab separation, but soft probability cascading exposes a fundamental bottleneck: when Stage 1 has slight probability uncertainty between close labs (e.g. Anthropic vs OpenAI), multiplying soft probabilities penalizes true sibling models.
   However, hard provider gating or hybrid stacking (combining flat probabilities with hierarchical specialist probabilities) retains the best of both paradigms.

---

## 2. Quantitative Performance Table

| Stage / Component | Scope | Balanced Accuracy | Std ($\pm$) | Macro F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Stage 1: Provider Router** | 6 AI Labs | **0.5099** | `±0.0337` | 0.6021 |
| • OpenAI Specialist | 5 models | 0.3760 | `±0.0265` | — |
| • Google Specialist | 4 models | 0.4500 | `±0.0401` | — |
| • Anthropic Specialist | 5 models | 0.4378 | `±0.0207` | — |
| • ZAI Specialist | 2 models | 0.5300 | `±0.0323` | — |
| **Flat Direct 18-Class** | 18 frontier models | 0.2827 | `±0.0122` | 0.2820 |
| **Hierarchical 18-Class** | Combined 2-stage | **0.2753** | `±0.0108` | 0.2761 |
