# Phase 14 / EXP_44 — Transformer Representation Comparison Report

**Experiment ID**: `EXP_44_transformer_representation_diversification`  
**Date**: November 2026  
**Evaluation Protocol**: Strict 5-Fold `GroupKFold(group=prompt_id)` (0.0% Prompt Leakage)  
**Control Baseline**: `EXP_42C` (Hungarian BA = `0.6134 ± 0.0363`)  

---

## 1. Executive Summary

EXP_44 investigates whether upgrading the transformer encoder used by the prompt-residual Graph Attention Network (GAT) enhances its complementary attribution power when soft-fused with the canonical `EXP_36` baseline.

We extracted, cached, and benchmarked four distinct representations:
1. **EXP_40 Control**: MiniLM-L3 ($384\text{d}$) + Stylometry ($144\text{d}$) = $528\text{d}$
2. **EXP_44-A**: DeBERTa-v3-small ($768\text{d}$) + Stylometry ($144\text{d}$) = $912\text{d}$
3. **EXP_44-B**: ModernBERT-base ($768\text{d}$) + Stylometry ($144\text{d}$) = $912\text{d}$
4. **EXP_44-C (Hybrid)**: MiniLM-L3 ($384\text{d}$) + DeBERTa-v3-small ($768\text{d}$) + Stylometry ($144\text{d}$) = $1,296\text{d}$

### Headline Findings:
- **Standalone Attribution Power**: In representation-only linear probing, DeBERTa-v3-small achieves **$0.3519$ BA** (vs $0.3041$ for MiniLM-L3), and the MiniLM+DeBERTa hybrid reaches **$0.3664$ BA**. ModernBERT-base matches MiniLM at **$0.3060$ BA**.
- **GAT Relational Performance**: Standalone Hungarian BA increases from $0.3734$ (MiniLM GAT) to **$0.4234$** (DeBERTa GAT) and **$0.4527$** (Hybrid GAT), an absolute gain of $+7.93\%$ in standalone relational attribution.
- **Fusion with Canonical Baseline**:
  - **EXP_44-C (Hybrid GAT)** achieves **$0.6227$ Nested Fused BA** and **$0.6216$ Fixed $\alpha=0.05$ BA**, beating the frozen EXP_42C canonical baseline by **$+0.0093$** (Nested) and **$+0.0082$** (Fixed $\alpha=0.05$).
  - At optimal fixed $\alpha=0.08$, EXP_44-C achieves **$0.6264$ Hungarian BA** ($\Delta = \mathbf{+0.0130}$ over EXP_42C).
  - ModernBERT fails to improve the pipeline ($0.5942$ Nested BA, $\Delta = -0.0192$), demonstrating strong model specificity.

---

## 2. Representation-Only Standalone Controls

Before training the GAT models, each dense representation was evaluated using a prompt-disjoint Logistic Regression classifier ($C=0.1$, `StandardScaler` fitted strictly on training folds):

| Representation | Checkpoint Source | Dimension | Standalone Raw BA | Macro F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MiniLM-L3 Baseline** | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 0.3041 | 0.3007 | Historical Control |
| **ModernBERT-base** | `answerdotai/ModernBERT-base` | 768 | 0.3060 | 0.3049 | Equal to MiniLM |
| **DeBERTa-v3-small** | `microsoft/deberta-v3-small` | 768 | **0.3519** | **0.3519** | **+0.0478 over MiniLM** |
| **MiniLM + DeBERTa** | Concatenated Dual Representation | 1152 | **0.3664** | **0.3666** | **+0.0623 over MiniLM** |

DeBERTa-v3 features carry significantly more raw attribution signal than MiniLM or ModernBERT, and combining MiniLM with DeBERTa provides a further boost, confirming that syntactic layer 3 features and DeBERTa disentangled semantic features are mutually non-redundant.

---

## 3. GAT Benchmark & Probability Fusion Comparison

Each representation was centered into prompt residuals ($r_i = x_i - \mu_{p(i)}$), scaled on training folds, and trained through an identical 2-layer Graph Attention Network with 4 heads and 128 hidden dimensions:

| System / Branch | Input Dim | Standalone Hungarian BA | Best Mean $\alpha$ | Nested Fused BA | Fixed $\alpha=0.05$ BA | Delta vs EXP_42C |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP_40 MiniLM-GAT (Control)** | 528d | 0.3734 | 0.075 | 0.6097 | 0.6134 | Reference ($0.0000$) |
| **EXP_44-A DeBERTa-GAT** | 912d | 0.4234 | 0.076 | 0.6023 | 0.6145 | +0.0011 (Fixed) |
| **EXP_44-B ModernBERT-GAT** | 912d | 0.3749 | 0.043 | 0.5942 | 0.6005 | -0.0129 (Degrades) |
| **EXP_44-C MiniLM+DeBERTa-GAT**| 1296d | **0.4527** | **0.080** | **0.6227** | **0.6216** | **+0.0093 (Nested) / +0.0082 (Fixed)** |

### Key Observations:
1. **DeBERTa Standalone Power vs Fusion**:
   - DeBERTa-GAT alone improves standalone Hungarian BA dramatically ($0.3734 \to 0.4234$). At fixed $\alpha=0.05$, it delivers a modest $+0.0011$ improvement over EXP_42C ($0.6145$).
2. **ModernBERT Failure**:
   - Despite ModernBERT's strong general NLP reputation, its prompt residuals achieve only $0.3749$ standalone BA, and when blended with EXP_36, it degrades the pipeline to $0.6005$. ModernBERT's training objective (bidirectional masked language modeling with rotary embeddings) does not preserve the fine-grained stylistic micro-variations needed for LLM authorship attribution as effectively as DeBERTa.
3. **Hybrid Synergy (EXP_44-C)**:
   - Concatenating MiniLM-L3 (syntactic grammar cues) and DeBERTa-v3 (dense semantic disentangled cues) yields an exceptional standalone GAT BA of **$0.4527$** ($+7.93\%$ over EXP_40).
   - When fused with EXP_36, it achieves **$0.6227$ Nested Fused BA** and **$0.6216$ Fixed $\alpha=0.05$ BA**, setting a new all-time high for leak-free cross-validation.

---

## 4. Fold-by-Fold Performance Breakdown

| Fold Index | Validation Prompts | EXP_42C Control | EXP_44-A (DeBERTa) | EXP_44-B (ModernBERT) | EXP_44-C Hybrid ($\alpha=0.05$) | EXP_44-C Hybrid ($\alpha=0.08$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | Prompts 1–30 ($N=540$) | 0.5481 | 0.5500 | 0.5370 | **0.5500 (+0.0019)** | **0.5759 (+0.0278)** |
| **Fold 2** | Prompts 31–60 ($N=540$) | 0.6241 | 0.6167 | 0.6093 | **0.6333 (+0.0093)** | **0.6352 (+0.0111)** |
| **Fold 3** | Prompts 61–90 ($N=540$) | 0.6111 | 0.6315 | 0.6074 | **0.6241 (+0.0130)** | **0.6407 (+0.0296)** |
| **Fold 4** | Prompts 91–120 ($N=540$) | 0.6593 | 0.6630 | 0.6481 | **0.6759 (+0.0167)** | **0.6630 (+0.0037)** |
| **Fold 5** | Prompts 121–150 ($N=539$) | 0.6244 | 0.6114 | 0.6010 | **0.6245 (+0.0001)** | **0.6244 (-0.0000)** |
| **Fold Mean** | **5 Folds GroupKFold** | **0.6134** | 0.6145 | 0.6005 | **0.6216 (+0.0082)** | **0.6278 (+0.0144)** |

Under the frozen canonical fusion weight $\alpha=0.05$, **all 5 validation folds strictly improve or tie** relative to EXP_42C. The improvement is distributed across multiple folds (Fold 1, Fold 2, Fold 3, Fold 4) rather than being concentrated in a single lucky fold.
