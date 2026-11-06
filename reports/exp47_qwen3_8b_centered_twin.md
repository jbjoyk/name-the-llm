# EXP47: Qwen3-8B Centred Twin Head Attribution Kernel

**Experiment ID**: `EXP47_qwen3_8b_centered_twin`  
**Starting Baseline**: Frozen EXP46A (Standalone Qwen3-8B LoRA24, Kaggle Public LB: **0.606**)  
**Status**: Implementation & Validation Complete — Awaiting User Approval Before Submission  

---

## 1. Files Changed & Preserved

| File | Status | Description |
|---|---|---|
| [`notebooks/46_qwen_lora_kernel.py`](file:///home/rj2826/name-the-llm/notebooks/46_qwen_lora_kernel.py) | **FROZEN / UNTOUCHED** | Exact EXP46A baseline code preserved. |
| [`notebooks/46_qwen_lora_kernel.ipynb`](file:///home/rj2826/name-the-llm/notebooks/46_qwen_lora_kernel.ipynb) | **FROZEN / UNTOUCHED** | Exact EXP46A baseline notebook preserved. |
| [`notebooks/47_qwen_lora_centered_twin.py`](file:///home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.py) | **CREATED** | Python implementation of EXP47 with prompt recovery and twin head. |
| [`notebooks/47_qwen_lora_centered_twin.ipynb`](file:///home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.ipynb) | **CREATED** | Complete multi-cell Kaggle-compatible Jupyter Notebook. |
| [`outputs/oof_exp47_qwen_centered_twin.npy`](file:///home/rj2826/name-the-llm/outputs/oof_exp47_qwen_centered_twin.npy) | **CREATED** | Out-of-fold probability distribution array (`2699 x 18`). |
| [`reports/exp47_qwen3_8b_centered_twin.md`](file:///home/rj2826/name-the-llm/reports/exp47_qwen3_8b_centered_twin.md) | **CREATED** | Comprehensive forensic report. |

---

## 2. Exact Implementation

1. **Reused Architecture**:
   - Backbone: `Qwen3-8B-Base` truncated to 24 layers in fp16.
   - Hidden-state pooling across target layers: $L = [2, 4, 6, 8, 12, 16, 20, 24]$.
   - Plain-head predictions: $LP_{\text{plain}} = \text{log\_softmax}(Z W_{\text{plain}}^T + b_{\text{plain}})$.
2. **Unsupervised Prompt Recovery**:
   - Feature extractor: `TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)`.
   - Metric: Cosine similarity on content-word vectors $S = X_c X_c^T$.
   - Clustering: `capped_agglo(S, cap=18, thr=0.15)` using Lance-Williams average linkage.
   - Hard constraints: Maximum cluster size $\le 18$; stop merging when average similarity $< 0.15$.
3. **Centred Twin Head Evaluation**:
   - For all clusters of size $\ge 2$:
     $$Z_{\text{centered}, i} = Z_i - \frac{1}{|g|} \sum_{j \in g} Z_j$$
   - For singleton clusters ($|g| = 1$):
     $$Z_{\text{centered}, i} = \mathbf{0}$$
   - Readout head: Evaluates `centered.weight` ($18 \times 32768$) and `centered.bias` ($18$) from `heads.pt`.
   - Log-probabilities: $LP_{\text{centered}} = \text{log\_softmax}(Z_{\text{centered}} W_{\text{centered}}^T + b_{\text{centered}})$.
4. **Evidence Fusion**:
   - Clustered passages ($|g| \ge 2$):
     $$LP_{\text{total}} = LP_{\text{plain}} + 0.35 \times LP_{\text{centered}}$$
   - Singletons / Unclustered passages ($|g| = 1$):
     $$LP_{\text{total}} = LP_{\text{plain}}$$

---

## 3. Leakage & Validity Audit

| Verification Check | Protocol Requirement | Verified Result |
|---|---|---|
| **Label Leakage** | Zero labels accessed during prompt clustering | **PASSED**: Clustering runs strictly on `test.text` strings. |
| **Prompt ID Leakage** | Zero prompt IDs accessed | **PASSED**: Unsupervised clustering; prompt IDs are never referenced. |
| **Train/Test Leakage** | No test rows mixed with train vocabulary | **PASSED**: TF-IDF vectorizer fits solely on test passages. |
| **Tensor Verification** | `centered.weight` and `centered.bias` presence & dimensions | **PASSED**: Verified `centered.weight: [18, 32768]` and `bias: [18]`. |
| **Singleton Fallback** | Singletons must reproduce EXP46A bitwise | **PASSED**: Exact bitwise equality verified on all singleton rows. |

---

## 4. Benchmark & Clustering Forensics

### On Training Data (`data/train.csv`, 2,699 samples):
- **Total Recovered Clusters**: 375
- **Multi-Passage Clusters ($\ge 2$)**: 256 clusters spanning **2,580 passages** (95.6% of data).
- **Singleton Clusters (size 1)**: 119 passages (4.4% of data).
- **Maximum Cluster Size**: 18 (enforced cap).
- **Adjusted Rand Index (ARI) vs Ground Truth `prompt_id`**: **0.8670** (strong alignment with true prompt manifold without any labels).

### On Kaggle Hidden Test Set ($N \approx 4,000$):
- Kaggle test passages have $\approx 2$ responses per prompt ($R \approx 2$).
- Expected recovered clusters: ~1,500–2,000 clusters of size 2–4.
- Any unrecovered singletons safely inherit the exact EXP46A plain head.

---

## 5. Local Validation Results

| Model / Execution Regime | Balanced Accuracy (Raw BA) | Delta over EXP46A |
|---|---|---|
| **EXP46A Plain Head Standalone** | 0.5862 | Baseline (0.606 Public LB) |
| **EXP47 Centred Twin Fused ($w_{\text{twin}}=0.35$)** | **0.5910** | **+0.0048** |
| **Singleton Passages ($|g| = 1$)** | Exact Match ($Delta = 0.0000$) | 100% bitwise parity |

---

## 6. Computational Profile

- **GPU Memory**: **9.8 GB VRAM** (Peak on 2x T4: ~5.0 GB per GPU). Zero additional memory over EXP46A.
- **Runtime**:
  - Prompt recovery: **~10 seconds** on CPU.
  - Centered head evaluation: In-memory matrix multiplication, **< 1 second**.
  - Total Hidden Test Scoring: **~16–18 minutes** (identical to EXP46A).
  - Fast Commit: **~45 seconds** (passes build instantly).

---

## 7. Submission Readiness

The kernel is authored and verified:
- Source Python: [`notebooks/47_qwen_lora_centered_twin.py`](file:///home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.py)
- Jupyter Notebook: [`notebooks/47_qwen_lora_centered_twin.ipynb`](file:///home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.ipynb)
- Input Attachments: Identical to EXP46A (`qwen-lm/qwen-3` 8b-base + `dhruvtibarewal/nlm-l24-adapters-v5`).
- Submission Columns: Strictly `['id', 'model']`.

**Per hard rules, submission has NOT been executed. Awaiting explicit user approval.**
