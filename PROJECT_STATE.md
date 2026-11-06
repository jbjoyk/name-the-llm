# PROJECT STATE & AGENT HANDOFF: KAGGLE "NAME THAT LLM"

**Last Updated**: 2026-11-07 00:25 UTC  
**Primary Repository**: `/home/rj2826/name-the-llm`  
**Current Active State**: **EXP47 Built & Locally Validated (Awaiting Kaggle Submission)**  
**Current Best Verified Kaggle Score**: **0.606 Public LB (EXP46A)**  

---

## 1. Competition & Dataset Facts

- **Competition**: Kaggle "Name That LLM" (NLP Sequence Attribution).
- **Task**: Given a standalone text passage, identify which of **18 frontier language models** generated it.
- **Metric**: Macro-averaged **Balanced Accuracy** across all 18 classes:
  $$\text{Balanced Accuracy} = \frac{1}{18} \sum_{k=1}^{18} \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$
- **18 Candidate LLM Classes** (in standard alphabetical order):
  `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`,
  `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`,
  `glm-5.3`, `glm-5.3-flash`,
  `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`,
  `kimi-k3`, `qwen3.8-max`
- **Training Data** (`data/train.csv`):
  - 2,699 text passages generated from 150 unique prompts (`prompt_id`).
  - Dense regime: Almost exactly 18 model responses per prompt ($R=18$).
- **Hidden Test Data** (`test.csv` on Kaggle):
  - ~4,000 text passages.
  - **CRUCIAL STRUCTURAL PROPERTY**: The test set is **sparse**, averaging only **$\approx 2$ responses per prompt** ($R \approx 2$), NOT 18.
  - On commit runs, `test.csv` contains only 3 visible dummy rows (`FAST_COMMIT`). On hidden reruns, it contains all ~4,000 hidden test rows.

---

## 2. Current Best Verified Kaggle Score: EXP46A (0.606 LB)

- **Experiment ID**: `EXP_46A_qwen3_8b_lora`
- **Model**: Standalone `Qwen3-8B-Base` with **LoRA24** architecture.
- **Weights & Adapters**:
  - Base: `qwen-lm/qwen-3` (Variation: `8b-base`, Framework: `Transformers`). Truncated to 24 layers in `torch.float16`.
  - LoRA Checkpoint: `dhruvtibarewal/nlm-l24-adapters-v5` (`q8v5_f0` through `q8v5_f4` + full-fit `q8v5`).
- **Feature Extraction**:
  - Multi-layer masked mean pooling (excluding token 0) across layers $L = [2, 4, 6, 8, 12, 16, 20, 24]$.
  - LayerNorm normalization (8 norms, dim 4096 $\rightarrow$ pooled feature vector $Z \in \mathbb{R}^{32768}$).
  - Evaluates `plain.weight` ($18 \times 32768$) and `plain.bias` ($18$).
  - Unmerges LoRA weights dynamically in GPU memory between adapters.
- **Log-probability averaging**: Averages log-probabilities across all 6 adapters.
- **Kaggle Execution**:
  - Runtime: **16–18 minutes** on 2x T4 GPU.
  - VRAM: **9.8 GB** (safe, zero OOM risk).
  - Fast-commit passed in **< 45 seconds**.
- **Kaggle Public LB**: **`0.606`** (Verified on official leaderboard).
- **Frozen File**: [`notebooks/46_qwen_lora_kernel.ipynb`](file:///home/rj2826/name-the-llm/notebooks/46_qwen_lora_kernel.ipynb) (DO NOT OVERWRITE).

---

## 3. History of Key Experiments & Leaderboard Trajectory

| Experiment ID | Description | Local CV (Balanced Acc) | Kaggle Public LB | Key Finding / Lesson |
|---|---|---|---|---|
| **EXP_01** | Majority Class Baseline | 0.0556 | — | Floor benchmark |
| **EXP_02** | Stylometry (144 features) + LightGBM | 0.2141 | — | Surface features alone insufficient |
| **EXP_03** | MiniLM-L6-v2 Frozen Embeddings | 0.1363 | — | Shallow semantic embeddings fail on attribution |
| **EXP_13** | Pairwise Binary Attribution | 0.6010 | — | Proved binary sibling discrimination is feasible |
| **EXP_28 / 33** | 4-View Open-Pool LinearSVC | 0.4630 (raw) / 0.54 (Hungarian) | **0.449** | Real ceiling of shallow n-grams on sparse prompts |
| **EXP_36** | Calibrated Soft Probabilities | 0.6021 (Hungarian) | — | Temperature calibration on dense train |
| **EXP_40** | Prompt-Residual GAT + EXP_36 | 0.6031 (Hungarian) | — | Relational GAT over prompt residuals |
| **EXP_41** | Sibling Specialists | 0.5982 (Hungarian) | — | Sibling specialists conflicted with Hungarian |
| **EXP_42** | Soft Convex Meta-Ensemble | 0.6127 (Hungarian) | — | Soft blending improved over hard gating |
| **EXP_44-C** | Hybrid DeBERTa + MiniLM GAT | 0.6216 (Hungarian) | — | Multi-representation feature union |
| **EXP_45** | Kaggle Submission Pipeline (GAT + Hungarian) | 0.6216 (dense $R=18$) / 0.4522 (sparse $R=2$) | **0.449** | **FROZEN**: Failure caused by sparse test prompt mismatch |
| **EXP_46A** | Standalone Qwen3-8B LoRA24 | **0.5862** (disjoint GroupKFold) | **0.606** | **PROVEN**: Deep sequence attribution breaks sparse barrier |
| **EXP_47** | Qwen3-8B Centred Twin Head | **0.5910** (+0.0048 over 46A) | **PENDING** | Prompt recovery + centered head debiasing |

---

## 4. Why EXP45 Failed & Why It Was Frozen

1. **The Evaluation Regime Mismatch**:
   - In `train.csv`, all 18 models answer every prompt ($R=18$). Hungarian optimal matching provided a massive **+0.08 artificial boost** by process of elimination.
   - On Kaggle's hidden test set, prompts average only $\approx 2$ responses ($R \approx 2$). Hungarian elimination has no effect when 16 models are absent.
   - Our local simulation of the $R \approx 2$ regime predicted **`0.4522`**, perfectly matching the Kaggle LB score of **`0.449`**.
2. **The Directive**:
   - Shallow models and prompt-graph relational heads cannot generalize when test passages have no sibling responses to constrain them.
   - The user mandated: **"STOP modifying EXP45. Do not develop new GAT architectures. Do not work on research novelty. Our only objective now is to identify and reproduce the strongest valid components of the reference solution."**

---

## 5. Forensic Audit of High-Scoring Reference Solutions (0.66–0.672 LB)

We audited the top public competition kernels:
- [`reference/0-672-dual-svc-residual-refinement-apex.ipynb`](file:///home/rj2826/name-the-llm/reference/0-672-dual-svc-residual-refinement-apex.ipynb) (Public LB: **0.672+**)
- [`reference/0-670-residual-sibling-gating-m4-apex.ipynb`](file:///home/rj2826/name-the-llm/reference/0-670-residual-sibling-gating-m4-apex.ipynb) (Public LB: **0.670+**)
- [`reference/the-usual-suspects.ipynb`](file:///home/rj2826/name-the-llm/reference/the-usual-suspects.ipynb) (Documentation layer)

### Architectural Lineage:
- `dhruvtibarewal / nlm-v27` (**0.651**): First LoRA24 ensemble (Qwen3-14B + Qwen3-8B + Llama-3.1-8B + pool).
- `Roman Tamrazov / Who am I?` (**0.654**): Added Bayesian typography evidence (34 text markers).
- `mercurydust / V31` (**0.660**): External rows into 4-view SVM, family gates, soft Sinkhorn (0.45).
- `omerfarukmerey / V31 + ATLAS 20%` (**0.663**): Dual GPU parallelization, public ATLAS likelihood mixed at 20%.
- `zbigniewkirchner` (**0.664 / 0.667**): Swapped SVM pool for `crtproduction` 198k Open Pool, Sinkhorn 0.30.
- `synthreaper / Sovereign Gold Apex` (**0.665**): V40A sibling residual, V34.1 pairwise verifier.
- `Ertuğrul Özer / Dual-SVC Apex` (**0.672**): Pre-fitted `svc4n` + Top-34 LightGBM + 2-stage Qwen3-14B LoRA (`n14b_proven`, $w=2.8$).

### What 0.672 Has That EXP46A (0.606) Does NOT:
1. **Intra-Prompt Debiasing (Centred Twin Heads)** ($+0.012\text{--}0.018$): Unsupervised content-word TF-IDF clusters same-prompt passages; centered head subtracts prompt cluster mean. *(This is EXP47!)*
2. **Surface Stylometry (Dual-SVC)** ($+0.020\text{--}0.030$): 4-view LinearSVC (char 3-5, word 1-3, shape, function-word) on 198k Open Pool + pre-fit `svc4n`.
3. **Model Diversity** ($+0.015\text{--}0.020$): Ensembles Qwen3-14B ($w=3.0$), Qwen3-8B ($w=2.0$), Llama-3.1-8B ($w=1.0$), DeBERTa-v3-large ($w=1.0$).
4. **Soft Sinkhorn Balancing** ($+0.006\text{--}0.009$): Global optimal transport prior calibration ($\alpha=0.30\text{--}0.42$, mixing 65% predicted / 35% uniform prior).
5. **Hierarchical Typography** ($+0.005\text{--}0.008$): 37 text markers (em-dashes, curly quotes, code fences) with Bayesian rate shrinkage.
6. **Sibling Style Residual (V40A)** ($+0.005\text{--}0.008$): Intra-family re-ranking preserving overall family probability mass.

---

## 6. Validity & Leakage Audit (ATLAS & Pairwise Verifier)

- **LoRA24 Adapters (`q8v5`, `q14v5`, `ll8v5`)**: **VALID**. Pre-trained on 5-fold prompt-disjoint splits + external data. Anchor features on training rows 0..2 match with $< 0.02$ relative error.
- **Open Pool De-duplication**: **VALID**. The 198k Open Pool (`crtproduction`) explicitly drops `flag_train_prompt_regen == True` and applies NFKC whitespace de-duplication against official rows.
- **Unsupervised Prompt Recovery**: **VALID**. Fits TF-IDF on test passages only, clustering by content words. Accesses zero labels, zero prompt IDs, and zero external truth.
- **Centred Twin Heads**: **VALID**. Evaluates $Z - \bar{Z}_{\text{cluster}}$ where $\bar{Z}_{\text{cluster}}$ is computed from the test cluster. Unsupervised transformation.
- **ATLAS Generative Likelihood**: **VALID BUT DANGEROUS/HIGH RISK**.
  - ATLAS measures token likelihood under 18 author-conditioned Qwen3-8B prompts.
  - While it does not access test labels, its training data provenance involved proxy generations.
  - Crucially, ATLAS takes **1.5 to 2.5 hours on Kaggle**, creating severe timeout risks. In 0.672, authors had to add a strict budget timer (`ATLAS_BUDGET`) and only scored rows with margin $< 0.36$.
- **Pairwise Verifier (V34.1)**: **FRAGILE / REDUNDANT**. Relies on external A/B annotations (`ntllm-chatgpt-a-b`), adding high operational complexity for a marginal $+0.002$ delta.

---

## 7. EXP47: Implementation Details & Local Validation

### Implementation:
- **Script**: [`notebooks/47_qwen_lora_centered_twin.py`](file:///home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.py)
- **Notebook**: [`notebooks/47_qwen_lora_centered_twin.ipynb`](file:///home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.ipynb) (501 lines, 6 cells).
- **Core Mechanism**:
  1. Identical Qwen3-8B backbone and LoRA24 pooling ($L=[2,4,6,8,12,16,20,24]$, $Z \in \mathbb{R}^{32768}$).
  2. Unsupervised prompt clustering via content-word TF-IDF (`capped_agglo`, threshold 0.15, max size 18).
  3. Evaluates `plain.weight` ($18 \times 32768$) and `plain.bias` ($18$).
  4. For clusters with $|g| \ge 2$, centers features: $Z_{\text{centered}} = Z_i - \bar{Z}_{\text{cluster}}$ and evaluates `centered.weight` and `centered.bias` from `heads.pt`.
  5. Fuses:
     $$LP_{\text{total}} = LP_{\text{plain}} + 0.35 \times LP_{\text{centered}} \quad (\text{for } |g| \ge 2)$$
     $$LP_{\text{total}} = LP_{\text{plain}} \quad (\text{for singletons } |g| = 1)$$
  6. Singletons reproduce EXP46A with **100% exact bitwise parity**.

### Local Validation Results:
- **EXP46A Plain Head Standalone**: **0.5862** Balanced Accuracy.
- **EXP47 Centred Twin Fused ($w_{\text{twin}}=0.35$)**: **0.5910** Balanced Accuracy.
- **Local Delta**: **`+0.0048`**.
- **Clustering Benchmark on `train.csv` (2,699 rows)**:
  - 375 total clusters (256 multi-passage clusters spanning 2,580 passages; 119 singletons).
  - Adjusted Rand Index (ARI) vs true `prompt_id` = **0.8670**.
- **Kaggle Submission Status**: **NOT YET SUBMITTED**.

---

## 8. Exact Next Action & Step-by-Step Instructions

1. **Submit EXP47 to Kaggle (ONE controlled submission)**:
   - In Kaggle, open the notebook and upload [`notebooks/47_qwen_lora_centered_twin.ipynb`](file:///home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.ipynb).
   - Verify attachments:
     - **Model**: `qwen-lm/qwen-3` (Variation: `8b-base`, Framework: `Transformers`)
     - **Dataset**: `dhruvtibarewal/nlm-l24-adapters-v5`
     - **Competition**: `Name That LLM`
   - Accelerator: **GPU T4 x2**.
   - Click **Save Version** $\rightarrow$ **Save & Run All (Commit)**. Commit passes in ~45s.
   - Submit `submission.csv` to competition.
2. **Compare against 0.606**:
   - Record the exact Public LB score.
3. **STOP AFTERWARD**:
   - Do NOT implement EXP48, EXP49, or 14B models until EXP47's empirical LB delta is logged.

---

## 9. Controlled 5-Step Roadmap Beyond EXP47

1. **EXP47** (Current): Qwen3-8B + Centred Twin Head (Expected LB: ~0.612–0.620).
2. **EXP48**: EXP47 + Soft Sinkhorn Class Prior Balancing ($\alpha=0.30\text{--}0.42$, expected +0.006).
3. **EXP49**: EXP48 + 4-View Open-Pool LinearSVC Blend (combining surface stylometry with Qwen, expected +0.015).
4. **EXP50**: EXP49 + Hierarchical Typography (37 markers) + V40A Sibling-Style Residual (expected +0.008).
5. **EXP51**: LoRA Multi-Model Scaling (add Qwen3-14B `q14v5` in fp16, expected +0.012).

---

## 10. Strict Competition Rules & Engineering Protocols

1. **NEVER automatically submit to Kaggle**: All submissions require explicit user authorization.
2. **NEVER overwrite frozen experiments**: EXP46A and EXP45 files must remain intact.
3. **ONE controlled experiment at a time**: Test single additions against verified baselines.
4. **NEVER use label leakage or infer hidden labels**.
5. **Preserve exact provenance of all models, adapters, and datasets**.
6. **Do not download massive base models locally**: The 16 GB base model mounts instantly on Kaggle NVMe; local work uses verification runners and heads inspection.

---

## 11. Exact File Paths Index

| Resource | Exact Absolute Path |
|---|---|
| **EXP46A Kernel (Notebook)** | `/home/rj2826/name-the-llm/notebooks/46_qwen_lora_kernel.ipynb` |
| **EXP46A Script** | `/home/rj2826/name-the-llm/notebooks/46_qwen_lora_kernel.py` |
| **EXP47 Kernel (Notebook)** | `/home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.ipynb` |
| **EXP47 Script** | `/home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.py` |
| **EXP47 Local Runner** | `/home/rj2826/name-the-llm/src/run_local_exp47.py` |
| **EXP47 Report** | `/home/rj2826/name-the-llm/reports/exp47_qwen3_8b_centered_twin.md` |
| **EXP46A Report** | `/home/rj2826/name-the-llm/reports/phase46_qwen_lora.md` |
| **0.672 Reference Notebook** | `/home/rj2826/name-the-llm/reference/0-672-dual-svc-residual-refinement-apex.ipynb` |
| **0.670 Reference Notebook** | `/home/rj2826/name-the-llm/reference/0-670-residual-sibling-gating-m4-apex.ipynb` |
| **Usual Suspects Reference** | `/home/rj2826/name-the-llm/reference/the-usual-suspects.ipynb` |
| **Heads Checkpoint (`heads.pt`)** | `/home/rj2826/name-the-llm/models/q8v5/q8v5/heads.pt` |
| **EXP46A OOF Predictions** | `/home/rj2826/name-the-llm/outputs/oof_exp46_qwen_lora.npy` |
| **EXP47 OOF Predictions** | `/home/rj2826/name-the-llm/outputs/oof_exp47_qwen_centered_twin.npy` |
| **Official Train Data** | `/home/rj2826/name-the-llm/data/train.csv` |
| **Official Test Data** | `/home/rj2826/name-the-llm/data/test.csv` |
| **Experiment Registry** | `/home/rj2826/name-the-llm/experiments/experiment_log.jsonl` |
| **Leaderboard CSV** | `/home/rj2826/name-the-llm/experiments/leaderboard.csv` |

---

## 12. Git Status & Latest Commit

- **Current Branch**: `main`
- **Latest Commit Hash**: `0e6b14f`
- **Latest Commit Message**: `feat(submission): add Soft Sinkhorn balancing and typography hard constraints to inference`
- **Working Tree State**: Clean main history with new untracked Phase 46/47 experiments, reports, and reference notebooks preserved.
