# Phase 15 — End-to-End Data Leakage & Selection Audit Report

**Audit Date**: November 2026  
**Investigator**: Antigravity Machine Learning Research Agent  
**Audited System**: `EXP_45` Final Hybrid Ensemble Pipeline  
**Dataset**: Official `data/train.csv` (SHA-256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`)  

---

## 1. Executive Summary & Leakage Verdict

**Overall Pipeline Verdict**: **`PASS (0.0% Validation Leakage)`**

Every component of the final attribution pipeline—from sparse tokenization and dense transformer embedding extraction to prompt clustering, GAT training, probability fusion, and Hungarian assignment—was audited against the strict criterion:
> *"Can information from the validation fold influence this component before prediction?"*

Zero instances of validation contamination, target leakage, or prompt-id leakage were found.

---

## 2. Component-by-Component Leakage Audit

### A. TF-IDF Fitting
- **Audit Question**: Are TF-IDF vocabularies or IDF frequencies fit across validation documents?
- **Finding**: In each of the 5 GroupKFold cross-validation splits, the 4-view TF-IDF vectorizers (`char`, `word`, `casing_shape`, `function_words`) are fitted strictly on the training fold documents ($N \approx 2,159$ official passages plus external Open Pool data). The validation fold texts are transformed using `.transform()` with fixed vocabulary and IDF weights.
- **Verdict**: **`PASS`**

### B. StandardScaler Normalization
- **Audit Question**: Are scaler means or variances computed over the whole dataset before fold splitting?
- **Finding**: In all GAT and linear probing experiments, `StandardScaler` is initialized fresh inside each fold. `scaler.fit_transform(R_tr)` is executed exclusively on training fold residuals. Validation residuals are transformed via `scaler.transform(R_val)`.
- **Verdict**: **`PASS`**

### C. Gaussian Heads
- **Audit Question**: Do class-conditional covariance matrices or mean vectors incorporate validation responses?
- **Finding**: The Shared-Prompt Gaussian Head (`gauss.fit`) is parameterized strictly on `X_dense_train`, `y_train`, and `prompts_train`. The validation fold is scored out-of-sample using `gauss.scores` with prompt clusters recovered blindly from validation text alone.
- **Verdict**: **`PASS`**

### D. MiniLM Embedding Generation
- **Audit Question**: Does frozen encoder extraction leak target labels or cross-fold information?
- **Finding**: `sentence-transformers/all-MiniLM-L6-v2` is a frozen pretrained model. Extraction is purely deterministic row-by-row token representation. Neither class labels (`model`) nor prompt identifiers (`prompt_id`) are passed to the tokenizer or model forward pass.
- **Verdict**: **`PASS`**

### E. DeBERTa Embedding Generation
- **Audit Question**: Does DeBERTa extraction leak target labels or cross-fold information?
- **Finding**: `microsoft/deberta-v3-small` is utilized as a frozen feature extractor without fine-tuning. Tokenization and forward passes run with `torch.no_grad()`. Passages are processed independently in batches. No labels or prompt IDs enter the computation.
- **Verdict**: **`PASS`**

### F. Prompt Residual Calculation
- **Audit Question**: Does computing residuals ($r_i = x_i - \mu_{p(i)}$) leak validation prompt means into training or vice-versa?
- **Finding**: Residuals are computed strictly within partitioned scopes:
  - On the training set: $\mu_p$ is computed from training fold responses belonging to prompt $p$.
  - On the validation/test set: Validation passages never see training means. Instead, validation passages are grouped into clusters blindly using unsupervised agglomerative clustering (`recover_prompt_clusters(val_texts)`), and cluster means $\mu_{\hat{p}}$ are subtracted locally.
- **Verdict**: **`PASS`**

### G. Prompt Clustering (Blind Recovery)
- **Audit Question**: Does prompt clustering utilize true prompt IDs, labels, or cross-fold information?
- **Finding**: `recover_prompt_clusters(texts)` executes content-word TF-IDF extraction (`max_df=0.15`) and agglomerative average linkage clustering solely on the raw string content of the input passages. True `prompt_id` strings and class labels are completely absent.
- **Verdict**: **`PASS`**

### H. GAT Training
- **Audit Question**: Do GAT attention edges cross between training and validation folds?
- **Finding**: Graph attention is constrained to intra-prompt subgraphs ($K \le 18$). Cross-prompt attention edges are explicitly masked to zero (`attn_mask`). Training batches contain only training prompts; validation batches contain only validation prompts. Zero gradient flow or message passing exists across fold boundaries.
- **Verdict**: **`PASS`**

### I. Alpha (Fusion Weight) Selection
- **Audit Question**: Was $\alpha$ optimized on the same validation fold used for final scoring?
- **Finding**:
  - In our pre-specified evaluation ($\alpha=0.05$), $\alpha$ was frozen in Phase 12 before EXP_44 or EXP_45 were conceived (zero degrees of freedom).
  - In our nested evaluation, an inner 4-fold GroupKFold grid search was executed strictly on `tr_idx`. The outer validation fold was never evaluated during hyperparameter selection.
  - When $\alpha=0.08$ is examined globally, it is transparently labeled as an *OOF-selected optimistic estimate* and excluded from primary generalization claims.
- **Verdict**: **`PASS`**

### J. Hungarian Assignment
- **Audit Question**: Do target class labels or ground-truth assignment matrices enter the Hungarian solver?
- **Finding**: `hungarian_assignment(probs, clusters)` takes only predicted probability matrices and integer cluster assignments. It solves $\min \sum -P_{ij}$ via `scipy.optimize.linear_sum_assignment`. Zero ground-truth labels are provided to the solver.
- **Verdict**: **`PASS`**

### K. External Open Pool Data
- **Audit Question**: Does external data contain duplicates or leakage of official validation prompts?
- **Finding**: Open Pool v11 (`crtproduction/name-that-llm-open-pool`, 36,000 samples) was deduplicated against the official competition dataset in Phase 4. It shares 0 prompt overlap and 0 text duplicates with official validation sets.
- **Verdict**: **`PASS`**

### L. Cached OOF Predictions
- **Audit Question**: Were cached arrays generated with data overlap or corrupted indices?
- **Finding**: All arrays (`oof_exp36_predictions.npy`, `oof_exp40_gat_predictions.npy`, `oof_exp44_hybrid_gat_predictions.npy`) have shape $(2,699, 18)$ and correspond 1:1 to official `train.csv` row indices. Each row was generated strictly when that sample was held out in the validation split.
- **Verdict**: **`PASS`**

---

## 3. Summary Table

| Audit Area | Scope | Leakage Potential | Verification Method | Status |
| :--- | :--- | :---: | :--- | :---: |
| **A. TF-IDF** | Feature Extraction | None | Inner-fold `.fit()`, outer `.transform()` | **PASS** |
| **B. Scalers** | Normalization | None | Fit on `R_tr` only, transform on `R_val` | **PASS** |
| **C. Gaussian Heads** | Relational Likelihood | None | Parameterized on train folds only | **PASS** |
| **D. MiniLM** | Dense Embeddings | None | Frozen encoder, no target inputs | **PASS** |
| **E. DeBERTa** | Dense Embeddings | None | Frozen encoder, no target inputs | **PASS** |
| **F. Residuals** | Prompt Centering | None | Train uses train prompts; Val uses blind clusters | **PASS** |
| **G. Clustering** | Prompt Recovery | None | Unsupervised TF-IDF linkage on texts only | **PASS** |
| **H. GAT** | Relational Attention | None | Intra-subgraph attention only, no cross-fold edges | **PASS** |
| **I. Alpha Selection** | Probability Weighting | None | Strict inner GroupKFold selection | **PASS** |
| **J. Hungarian** | Bipartite Matching | None | Solves purely on probability cost matrix | **PASS** |
| **K. Open Pool** | External Data | None | Verified zero prompt/passage overlap | **PASS** |
| **L. Cached OOF** | Array Artifacts | None | SHA-256 verified, strictly held-out slices | **PASS** |
