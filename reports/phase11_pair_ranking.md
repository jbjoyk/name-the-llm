# Phase 11 — EXP_36 Error Structure & Pairwise Confusion Ranking

**Analysis Date**: November 2026  
**Dataset Source**: Official `train.csv` ($N=2,699$ passages across 150 prompts, 18 LLM classes)  
**Evaluated Predictions**: `outputs/oof_fused_gaussian_hungarian.npy` (Verified Canonical EXP_36)  
**Overall Performance**: Balanced Accuracy = 0.6034, Total Errors = 1,070 / 2,699 (39.64%)

---

## 1. Top Confusing Pairs Ranked by Confusion Strength

$$\text{Confusion Strength} = \text{Errors}(A \to B) + \text{Errors}(B \to A)$$

| Rank | Model A | Model B | A $\to$ B | B $\to$ A | Total Errors | % of Total Errors | $P(B \mid A)$ | $P(A \mid B)$ | Category |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `gemini-3.7-flash` | `gemini-3.8-flash` | 44 | 41 | **85** | 7.94% | 29.33% | 27.33% | Same-Lab Sibling |
| **2** | `glm-5.3` | `glm-5.3-flash` | 41 | 33 | **74** | 6.92% | 27.33% | 22.00% | Same-Lab Sibling |
| **3** | `claude-fable-5` | `glm-5.3-flash` | 26 | 40 | **66** | 6.17% | 17.45% | 26.67% | Cross-Lab Confusion |
| **4** | `claude-fable-5` | `glm-5.3` | 34 | 26 | **60** | 5.61% | 22.82% | 17.33% | Cross-Lab Confusion |
| **5** | `gpt-5.6-sol` | `gpt-5.6-terra` | 26 | 20 | **46** | 4.30% | 17.33% | 13.33% | Same-Lab Sibling |
| **6** | `gpt-5.6-luna` | `gpt-5.6-sol` | 22 | 21 | **43** | 4.02% | 14.67% | 14.00% | Same-Lab Sibling |
| **7** | `gpt-5.6-luna` | `gpt-5.6-terra` | 19 | 23 | **42** | 3.93% | 12.67% | 15.33% | Same-Lab Sibling |
| **8** | `gpt-5.5` | `qwen3.8-max` | 14 | 13 | **27** | 2.52% | 9.33% | 8.67% | Cross-Lab Confusion |
| **9** | `gpt-5.5` | `kimi-k3` | 6 | 20 | **26** | 2.43% | 4.00% | 13.33% | Cross-Lab Confusion |
| **10**| `claude-opus-4-8` | `claude-sonnet-5` | 13 | 12 | **25** | 2.34% | 8.67% | 8.00% | Same-Lab Sibling |
| **11**| `gpt-5.5` | `gpt-5.6-terra` | 11 | 14 | **25** | 2.34% | 7.33% | 9.33% | Same-Lab Sibling |
| **12**| `claude-fable-5` | `claude-opus-4-8` | 13 | 11 | **24** | 2.24% | 8.72% | 7.33% | Same-Lab Sibling |
| **13**| `claude-fable-5` | `claude-sonnet-5` | 12 | 12 | **24** | 2.24% | 8.05% | 8.00% | Same-Lab Sibling |
| **14**| `gemini-3.6-flash` | `gemini-3.7-flash` | 13 | 11 | **24** | 2.24% | 8.67% | 7.33% | Same-Lab Sibling |
| **15**| `claude-opus-4-6` | `claude-opus-4-8` | 9 | 13 | **22** | 2.06% | 6.00% | 8.67% | Same-Lab Sibling |

---

## 2. Key Empirical Findings on Error Structure

1. **Sibling Pair Dominance**:
   - **Gemini Flash Siblings**: `gemini-3.7-flash` $\leftrightarrow$ `gemini-3.8-flash` is the single most confused pair in the entire competition (85 errors, 7.94% of all errors). The confusion is almost perfectly symmetric (44 vs 41).
   - **GLM 5.3 Siblings**: `glm-5.3` $\leftrightarrow$ `glm-5.3-flash` is the second highest pair (74 errors, 6.92% of all errors). Symmetrically confused (41 vs 33).
   - **GPT-5.6 Triad**: The three GPT-5.6 variants (`luna`, `sol`, `terra`) form a tight triangular confusion block accounting for 131 errors (12.24% of all errors):
     - `sol` $\leftrightarrow$ `terra`: 46 errors (4.30%)
     - `luna` $\leftrightarrow$ `sol`: 43 errors (4.02%)
     - `luna` $\leftrightarrow$ `terra`: 42 errors (3.93%)
   - Together, these 5 same-family sibling pairs account for **290 errors (27.10% of ALL errors)**.

2. **Claude Fable vs GLM Outlier Asymmetry**:
   - `claude-fable-5` exhibits severe confusion with `glm-5.3-flash` (66 errors) and `glm-5.3` (60 errors).
   - This cross-lab confusion arises because `claude-fable-5` had the lowest single recall in EXP_36 (~24%) and frequently collapses into default GLM feature margins.

3. **Selected Specialist Pairs for Phase 11**:
   We select the top 5 distinct sibling confusion pairs:
   - **Pair 1**: `gemini-3.7-flash` vs `gemini-3.8-flash` (85 errors, 7.94% of errors)
   - **Pair 2**: `glm-5.3` vs `glm-5.3-flash` (74 errors, 6.92% of errors)
   - **Pair 3**: `gpt-5.6-sol` vs `gpt-5.6-terra` (46 errors, 4.30% of errors)
   - **Pair 4**: `gpt-5.6-luna` vs `gpt-5.6-sol` (43 errors, 4.02% of errors)
   - **Pair 5**: `gpt-5.6-luna` vs `gpt-5.6-terra` (42 errors, 3.93% of errors)

4. **Random Pair Controls (Low/Zero Confusion)**:
   For Section 21 of the protocol:
   - **Control Pair 1**: `gemini-3.1-pro-preview` vs `glm-5.3-flash` (0 errors)
   - **Control Pair 2**: `claude-opus-5` vs `gpt-5.6-terra` (0 errors)
   - **Control Pair 3**: `gemini-3.6-flash` vs `gpt-6-astra` (0 errors)
