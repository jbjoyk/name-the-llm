# Phase 7: Shared-Prompt Gaussian Group Heads & Multi-Modal Fusion Report
**Experiment ID**: `EXP_36_shared_prompt_gaussian`  
**Date**: 2026-11-05 18:59:10 UTC  
**Validation Strategy**: 5-Fold `GroupKFold(group=prompt_id)` (Strict Prompt-Disjoint Leak-Free CV)  
**Feature Representations**: MiniLM Layer 3 syntax (384d) + Reference stylometry (144d) = 528d  
**Group Head Model**: Additive Two-Way Gaussian with WPCA Whitening and Ledoit-Wolf Shrinkage (`src/gauss.py`)  
**Ensemble Partner**: Open Pool 4-View LinearSVC (`outputs/oof_open_pool_svc.npy`, EXP_28)  

---

## 1. Executive Summary & Breakthrough Results

| Evaluation Regime | Prompt-Disjoint BalAcc | Macro F1 | Delta vs Baseline |
| :--- | :---: | :---: | :---: |
| **Regime 1: Baseline Logistic Regression (528d)** | `0.3415` | `0.3407` | *-0.1989* |
| **Regime 2: Gaussian Singleton Likelihood** | `0.3412` | - | *-0.1947* |
| **Regime 3: Gaussian Head (Oracle Prompt Groups)** | `0.4805` | - | *-0.0553* |
| **Regime 4: Gaussian Head (Recovered Clusters Alone)** | `0.4679` | `0.4682` | *-0.0680* |
| **Prior Best Independent: Open Pool 4-View SVC** | `0.4630` | `0.4494` | *reference* |
| **Prior Best Injective: Open Pool SVC + Hungarian** | `0.5364` | `0.5337` | `+0.0734` |
| **Regime 6: Fused SVC + Gaussian Head (Direct Argmax)** | **`0.5286`** | **`0.5185`** | **`+0.0656`** |
| **Regime 7: Fused SVC + Gauss Head + Hungarian Matching** | **`0.6034`** | **`0.6014`** | **`+0.0670`** |

> **Key Discovery**: Integrating the Shared-Prompt Gaussian Head log-probabilities with Open Pool 4-View SVC elevates direct unconstrained attribution to **`0.5286`** (up from 0.4630), and with injective Hungarian bipartite matching on recovered clusters, pushes the competition pipeline past the 0.60 milestone to **`0.6034 ± 0.0342`** Balanced Accuracy (**Macro F1: `0.6014`**).

---

## 2. Fold-by-Fold Performance Progression

| Fold | Baseline LR | Singleton | Recovered EM | Fused Direct | Fused Hungarian |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | `0.3444` | `0.3370` | `0.4593` | `0.5037` | **`0.5407`** |
| Fold 2 | `0.3296` | `0.3537` | `0.4204` | `0.5167` | **`0.6000`** |
| Fold 3 | `0.3389` | `0.3352` | `0.4722` | `0.5278` | **`0.6204`** |
| Fold 4 | `0.3407` | `0.3648` | `0.4963` | `0.5593` | **`0.6426`** |
| Fold 5 | `0.3541` | `0.3149` | `0.4913` | `0.5355` | **`0.6132`** |
| **Mean ± Std** | `0.3415` | `0.3412` | `0.4679` | `0.5286` | **`0.6034 ± 0.0342`** |

---

## 3. Mathematical Mechanisms & Verification

1. **Decoupling Prompt Variance ($S_p$) from Residual Noise ($S_e$)**:
   - The two-way model $Z = \mu_y + P_{p} + E$ isolates the prompt shift vector $P_{p}$.
   - Whitened PCA and Ledoit-Wolf shrinkage ensure stable inversion of covariance matrices even with 150 prompt groups and 18 classes.
   - Eigenvalues $\lambda$ reach percentiles of $[0.20, 0.89, 5.19, 17.84]$, proving that prompt-topic variance dramatically dominates residual noise ($S_p \gg S_e$).

2. **Synergy Between Discrete N-Grams and Continuous Syntax Representations**:
   - The 4-view LinearSVC captures discrete lexical cues, punctuation frequencies, and casing shape templates.
   - The Gaussian Head captures continuous syntactic depth, clause structures, and sentence embeddings from MiniLM Layer 3.
   - Ensembling in log-probability space ($\log p = \log p_{\text{SVC}} + 0.05 \cdot \log p_{\text{Gauss}}$) allows continuous syntactic features to adjust boundary confidences without disrupting strong n-gram matches.

3. **Injective Matching Resolves Sibling Model Ambiguity**:
   - Enforcing bipartite Hungarian matching on recovered prompt clusters guarantees that each candidate LLM is assigned to at most one passage per prompt.
   - This constraint resolves fine-grained sibling confusions (e.g. Claude Opus 4.6 vs 5, GPT-5.6 Luna vs Sol vs Terra).

---

## 4. Per-Class Attribution Forensics

| Model Class | Recall | Precision | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `gemini-3.1-pro-preview` | 93.3% | 89.7% | 0.9150 | 150 |
| `claude-opus-5` | 85.3% | 83.7% | 0.8449 | 150 |
| `claude-opus-4-6` | 83.3% | 81.7% | 0.8251 | 150 |
| `gpt-6-astra` | 82.0% | 80.9% | 0.8146 | 150 |
| `gemini-3.6-flash` | 79.3% | 81.0% | 0.8013 | 150 |
| `qwen3.8-max` | 71.3% | 67.7% | 0.6948 | 150 |
| `claude-sonnet-5` | 64.0% | 58.5% | 0.6115 | 150 |
| `gpt-5.5` | 64.0% | 59.6% | 0.6174 | 150 |
| `claude-opus-4-8` | 64.0% | 63.6% | 0.6379 | 150 |
| `gemini-3.8-flash` | 60.0% | 59.2% | 0.5960 | 150 |
| `gemini-3.7-flash` | 58.7% | 59.9% | 0.5926 | 150 |
| `gpt-5.6-luna` | 53.3% | 56.3% | 0.5479 | 150 |
| `kimi-k3` | 50.0% | 52.4% | 0.5119 | 150 |
| `gpt-5.6-sol` | 48.7% | 51.8% | 0.5017 | 150 |
| `gpt-5.6-terra` | 42.0% | 45.0% | 0.4345 | 150 |
| `glm-5.3` | 30.0% | 31.5% | 0.3072 | 150 |
| `glm-5.3-flash` | 30.0% | 30.4% | 0.3020 | 150 |
| `claude-fable-5` | 26.8% | 27.0% | 0.2694 | 149 |

---

## 5. Experiment Metadata & Replication
- **Script**: `notebooks/36_shared_prompt_gaussian.py`
- **Notebook**: `notebooks/36_shared_prompt_gaussian.ipynb`
- **Runtime**: 20.41 seconds
- **Reproducibility**: Complete deterministic reproduction under fixed random seeds and prompt-disjoint GroupKFold.
