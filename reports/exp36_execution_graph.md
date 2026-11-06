# End-to-End Execution Graph & Leakage Audit: EXP_36 Baseline
**Document**: `reports/exp36_execution_graph.md`  
**Date**: 2026-11-06  
**Auditor**: Antigravity Research Engineer  
**Audit Target**: `notebooks/36_shared_prompt_gaussian.py` (Source of Canonical 0.6034 BA)

---

## 1. Complete Stage-by-Stage Execution Graph

| Stage # | Stage Name | Input Artifacts | Output Artifacts | Fitting Data | Inference Data | Labels Used? | Prompt IDs Used? | Test Info Leakage? | Random State | Saved Artifact |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Dataset Ingestion** | `data/train.csv` (2,699 rows) | Raw string arrays `texts`, integer labels `y`, `prompts` | None | Entire dataset | Yes (for evaluation targets) | Yes (for fold splitting) | **NO** (unseen prompts in test) | N/A | None |
| **2** | **Validation Split** | `prompts` (150 unique) | 5 fold train/val index pairs `(tr_idx, val_idx)` | GroupKFold on prompt groups | Entire dataset | No | **YES** (strictly groups by prompt_id) | **NO** (folds are strictly disjoint) | Deterministic | Folds in RAM |
| **3** | **Open Pool Curation** | `open_pool_clean.parquet` | 36,000 balanced passages ($w=0.35$) | Filtered for quality and label balance | Training split only | Yes (for class-balanced sampling) | No | **NO** (0 prompt overlap, 0 exact text duplicates) | `seed=42` | `open_pool_clean.parquet` |
| **4** | **4-View Feature Extraction** | `texts[tr_idx]`, `open_pool_texts` | Sparse CSR matrices (270k features total) | Fitted **strictly on training fold** (`tr_idx` + pool) | Transform on `val_idx` | No | No | **NO** (vocabulary fit only on train) | N/A | Cached in memory |
| **5** | **4-View LinearSVC** | 4-View sparse matrices | Decision function scores (18 logits) | Fit on `tr_idx` + pool | Predict on `val_idx` | **YES** (training fold targets only) | No | **NO** (held-out targets never seen) | `seed=42` | `outputs/oof_open_pool_svc.npy` |
| **6** | **Dense Syntax Extraction** | Raw text passages | MiniLM Layer 3 mean embeddings (384d) | Frozen `all-MiniLM-L6-v2` encoder (no training) | Inferred on all passages | **NO** (unsupervised forward pass) | No | **NO** (frozen pretrained weights) | N/A | `outputs/minilm_layer3_mean.npy` |
| **7** | **Dense Stylometry Extraction** | Raw text passages | 144d rule-based stylistic vectors | Deterministic regex / token counts | Inferred on all passages | **NO** (unsupervised rule-based) | No | **NO** (deterministic rules) | N/A | `outputs/features_reference_stylometry.npy` |
| **8** | **Shared-Prompt Gaussian Fit** | Concatenated 528d vectors `X_comb[tr_idx]` | Whitening matrix $W$, author means $\mu$, eigenvalues $\lambda$ | Fit on `tr_idx` via `gauss.fit` | None | **YES** (training fold targets only) | **YES** (training prompt IDs only) | **NO** (val prompts never seen during fit) | N/A | `prm` dict |
| **9** | **Inference Prompt Recovery** | `texts[val_idx]` (unseen held-out text) | Cluster labels `rec_clusters` | Unsupervised average linkage on TF-IDF | Held-out validation text only | **NO** (zero labels used) | **NO** (zero prompt IDs used; blind recovery) | **NO** (operates purely on text overlap) | Deterministic | `rec_clusters` |
| **10** | **Gaussian Scoring** | `X_comb[val_idx]`, `rec_clusters`, `prm` | Gaussian log-posteriors `sc_rec` | Evaluated via `gauss.scores` using recovered clusters | `val_idx` | **NO** (unsupervised posterior evaluation) | **NO** (uses recovered clusters) | **NO** (held-out data evaluated using train prm) | N/A | `outputs/oof_shared_prompt_gaussian.npy` |
| **11** | **Multi-Modal Score Fusion** | `log_p_svc[val_idx]`, `sc_rec` | Blended probabilities `prob_blend` | Log-linear blend: $\log p_{\text{svc}} + 0.05 \cdot \log p_{\text{gauss}}$ | `val_idx` | **NO** (fixed predetermined weight 0.05) | No | **NO** (zero tuning on evaluation fold) | N/A | In RAM |
| **12** | **Hungarian Assignment** | `prob_blend`, `rec_clusters` | Final predicted class indices | Maximum-weight bipartite matching per cluster | `val_idx` | **NO** (bipartite matching on probs) | **NO** (uses recovered clusters) | **NO** (injective optimization only) | Deterministic | `outputs/oof_fused_gaussian_hungarian.npy` |

---

## 2. Leakage Verification Checklist

- [x] **No Target Leakage in Vectorizers**: All 4 TF-IDF vectorizers are fit exclusively on the training fold `tr_idx` and external pool.
- [x] **No Target Leakage in Scalers**: Any standard scaling is fit on `tr_idx` only.
- [x] **No Prompt Leakage in Gaussian Head**: `gauss.fit` estimates prompt covariance $S_p$ and residual covariance $S_e$ strictly using `tr_idx` prompt IDs.
- [x] **Blind Prompt Recovery**: `recover_prompt_clusters` receives only raw validation text strings; ground-truth prompt IDs are completely inaccessible.
- [x] **No Metric Tuning on Test**: The fusion weight ($w_{\text{gauss}} = 0.05$) and clustering distance threshold ($t = 0.85$) are fixed global hyperparameters, never optimized per validation fold.
- [x] **Strict Disjointness**: Held-out prompts in each fold share 0% overlap with the training fold.
