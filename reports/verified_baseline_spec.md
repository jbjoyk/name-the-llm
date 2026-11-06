# Verified Canonical Baseline Specification: EXP_38 (Frozen EXP_36)
**Document**: `reports/verified_baseline_spec.md`  
**Date**: 2026-11-06  
**Auditor / Baseline Architect**: Antigravity Research Engineer  
**Status**: CANONICAL FROZEN BASELINE (Replaces all previous baseline references)  

---

## 1. Executive Baseline Summary

| Specification Attribute | Frozen Configuration Value |
| :--- | :--- |
| **Baseline Identifier** | `EXP_38_baseline_freeze` (derived from `EXP_36_shared_prompt_gaussian`) |
| **Validation Strategy** | Strict 5-Fold `GroupKFold(group=prompt_id)` |
| **Primary Evaluation Metric** | **Balanced Accuracy** (Macro Recall across 18 candidate LLMs) |
| **Official Training Dataset** | `data/train.csv` (2,699 passages, 150 prompts, 18 models) |
| **Training Hash (SHA-256)** | `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` |
| **External Training Corpus** | `data/external/open_pool_clean.parquet` (36,000 balanced passages, $w=0.35$) |
| **External Hash (SHA-256)** | `f94ba05f3fc134b2f48e42b2bb74b0dc34da70557223bdfbc34b3dc04c5a96aa` |
| **Primary 5-Fold Balanced Accuracy** | **`0.6034 ± 0.0342`** (Fold scores: `[0.5407, 0.6000, 0.6204, 0.6426, 0.6132]`) |
| **Primary 5-Fold Macro F1** | **`0.6014`** |
| **5-Seed Shuffle Mean Balanced Accuracy** | **`0.5976 ± 0.0023`** |
| **95% Confidence Interval** | **`[0.5956, 0.5997]`** |
| **R2 Simulated Balanced Accuracy (5 seeds)** | **`0.4771 ± 0.0060`** (Gain over standalone SVC: `+0.0435`) |
| **End-to-End 5-Fold Runtime** | **20.4 seconds** |
| **Peak Memory Footprint** | 4.2 GB System RAM, 0.8 GB GPU VRAM |

---

## 2. Component Inventory & Hyperparameters

### Component 1: 4-View Linear Support Vector Classifier (`LinearSVC`)
- **Loss**: `squared_hinge`
- **Regularization**: $C = 0.5$, `class_weight='balanced'`, `max_iter=2000`, `random_state=42`
- **Training Samples**: 2,699 official rows ($w=1.0$) + 36,000 Open Pool rows ($w=0.35$)
- **Views**:
  1. *Character N-Grams*: Sublinear TF-IDF, n-gram range `[3, 5]`, `min_df=3`, `max_features=120,000`
  2. *Word N-Grams*: Sublinear TF-IDF with punctuation tokens, n-gram range `[1, 3]`, `min_df=2`, `max_features=80,000`
  3. *Casing Shape*: Casing transform (`X` for `[A-Z]`, `x` for `[a-z]`), character n-gram range `[2, 6]`, `min_df=3`, `max_features=40,000`
  4. *Function Words*: 70 stopwords preserved, content words replaced with `CONTENT`, word n-gram range `[1, 3]`, `min_df=2`, `max_features=30,000`
- **Output**: Multi-class decision function mapped to log-probabilities: $\log p_{\text{SVC}} = \text{log\_softmax}(\text{decision\_function})$.

### Component 2: Shared-Prompt Gaussian Group Head (`src/gauss.py`)
- **Input Features**: 528 dimensions
  - 384d MiniLM Layer 3 syntax mean pooling (`sentence-transformers/all-MiniLM-L6-v2`)
  - 144d Stylometry features (16 base text statistics, 55 punctuation markers, 71 function word ratios)
- **Dimensionality Reduction**: Whitened PCA (WPCA) to $d = 128$ dimensions
- **Covariance Shrinkage**: Ledoit-Wolf shrinkage on error covariance $S_e$ and prompt covariance $S_p$
- **Inference Mode**:
  - $n \le 3$: Exact joint log-marginal table with injective mask (`_inj_mask`)
  - $n > 3$: Cavity-EM marginalization (`em_marginals`, 5 iterations, `repel=True`)
- **Output**: Log-posterior matrix: $\log p_{\text{Gauss}}$.

### Component 3: Inference-Time Blind Prompt Recovery
- **Algorithm**: Hierarchical agglomerative clustering with average linkage (`scipy.cluster.hierarchy.linkage`)
- **Feature Extraction**: Content-word TF-IDF (`stop_words='english'`, `min_df=2`, `max_df=0.08`, `sublinear_tf=True`)
- **Distance Metric**: Cosine distance matrix: $D = 1 - S$, $\text{diag}(D) = 0$
- **Cluster Cut**: Distance threshold $t = 1.0 - 0.15 = 0.85$
- **Size Capping**: If any cluster exceeds 18 passages, recursively re-cluster with `criterion='maxclust', t=18`.
- **Clustering Quality**: Mean Adjusted Rand Index (ARI) = **`0.8812`**, V-Measure = **`0.9483`**.

### Component 4: Multi-Modal Score Fusion & Injective Assignment
- **Fusion Formula**: Log-linear ensemble:
  $$\log p_{\text{blend}} = \log p_{\text{SVC}} + 0.05 \cdot \log p_{\text{Gauss}}$$
  $$p_{\text{blend}} = \text{softmax}(\log p_{\text{blend}})$$
- **Bipartite Matching**: Maximum-weight matching (`scipy.optimize.linear_sum_assignment`) inside each recovered prompt cluster:
  $$\text{cost\_matrix} = -p_{\text{blend}}[\text{cluster\_indices}]$$
  $$\text{row\_ind}, \text{col\_ind} = \text{linear\_sum\_assignment}(\text{cost\_matrix})$$
- **Fallback**: For singleton clusters ($n=1$), assign $\text{argmax}(p_{\text{blend}})$.

---

## 3. Clean Environment Reproduction Protocol

To reproduce this exact baseline from a clean checkout:
1. Environment setup:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install torch==2.15.0+cu132 scikit-learn==1.9.1 scipy==1.18.1 numpy==2.5.4 pandas==3.1.0 transformers==4.51.0
   ```
2. Verify input hashes:
   ```bash
   sha256sum data/train.csv
   # 40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707
   sha256sum data/external/open_pool_clean.parquet
   # f94ba05f3fc134b2f48e42b2bb74b0dc34da70557223bdfbc34b3dc04c5a96aa
   ```
3. Run verification execution:
   ```bash
   python notebooks/36_shared_prompt_gaussian.py
   # Expected result: Mean Balanced Accuracy = 0.6034 ± 0.0342
   ```
