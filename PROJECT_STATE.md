# PROJECT STATE & AGENT HANDOFF: KAGGLE "NAME THAT LLM"

**Last Updated**: 2026-11-07 11:52 UTC  
**Primary Repository**: `/home/rj2826/name-the-llm`  
**Current Active State**: **EXP49 Candidate Ready (Awaiting User Submission Approval)**  
**Current Best Verified Kaggle Score**: **0.620 Public LB (EXP47)**  

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

## 2. Best Verified Kaggle Scores: EXP46A (0.606) & EXP47 (0.620)

- **EXP46A (Standalone Qwen3-8B LoRA24)**:
  - Kaggle Public LB: **`0.606`**
  - Architecture: Qwen3-8B base truncated to 24 layers + 6 merged `q8v5` LoRA adapters + multi-layer pooled hidden states `[2,4,6,8,12,16,20,24]`.
- **EXP47 (Qwen3-8B Centred Twin Head)**:
  - Kaggle Public LB: **`0.620`** (+0.014 over EXP46A)
  - Local BA: **0.5910** (baseline plain: 0.5862, delta: +0.0048)
  - Mechanism: Unsupervised content-word TF-IDF prompt clustering (`capped_agglo`, thr=0.15, cap=18) + centered head debiasing ($Z - \bar{Z}_{\text{cluster}}$) with $w_{\text{twin}}=0.35$.

---

## 3. History of Key Experiments & Leaderboard Trajectory

| Experiment ID | Description | Local CV (Balanced Acc) | Kaggle Public LB | Decision / Status |
|---|---|---|---|---|
| **EXP_45** | Relational GAT + Hungarian Matching | 0.4522 (sparse sim) | **0.449** | **FROZEN** (Sparse prompt breakdown) |
| **EXP_46A** | Standalone Qwen3-8B LoRA24 | **0.5862** | **0.606** | **FROZEN BASELINE** |
| **EXP_47** | Qwen3-8B Centred Twin Head | **0.5910** | **0.620** | **VERIFIED TOP ON KAGGLE** |
| **EXP_48** | EXP47 + Damped Sinkhorn Calibration | **0.5910** (+0.0000) | *Not Submitted* | **YELLOW** (Ensemble regularizer) |
| **EXP_49** | EXP47 + 4-View Open-Pool LinearSVC | **0.6354–0.7028** (+0.0445–0.1118) | **CANDIDATE READY** | **GREEN (Awaiting User Approval)** |

---

## 4. Forensic Audit of High-Scoring Reference Solutions (0.66–0.672 LB)

Audited top public competition kernels:
- [`reference/0-672-dual-svc-residual-refinement-apex.ipynb`](file:///home/rj2826/name-the-llm/reference/0-672-dual-svc-residual-refinement-apex.ipynb) (Public LB: **0.672+**)
- [`reference/0-670-residual-sibling-gating-m4-apex.ipynb`](file:///home/rj2826/name-the-llm/reference/0-670-residual-sibling-gating-m4-apex.ipynb) (Public LB: **0.670+**)
- [`reference/the-usual-suspects.ipynb`](file:///home/rj2826/name-the-llm/reference/the-usual-suspects.ipynb) (Documentation layer)

### What 0.672 Has That EXP46A/EXP47 Do NOT:
1. **Surface Stylometry (Dual-SVC)** ($+0.020\text{--}0.030$): 4-view LinearSVC on 198k Open Pool + pre-fit `svc4n`. *(Implemented in EXP49: achieves +0.0445 to +0.1118 local gain!)*
2. **Soft Sinkhorn Balancing** ($+0.006\text{--}0.009$): Global optimal transport prior calibration ($\alpha=0.30, \rho=0.35$). *(Validated in EXP48 as ensemble regularizer)*.
3. **Model Diversity** ($+0.015\text{--}0.020$): Ensembles Qwen3-14B ($w=3.0$), Qwen3-8B ($w=2.0$), Llama-3.1-8B ($w=1.0$), DeBERTa-v3-large ($w=1.0$). *(Target of EXP51)*.
4. **Hierarchical Typography** ($+0.005\text{--}0.008$): 37 text markers with Bayesian shrinkage. *(Target of EXP50)*.
5. **Sibling Style Residual (V40A)** ($+0.005\text{--}0.008$): Intra-family re-ranking preserving family probability mass. *(Target of EXP50)*.

---

## 5. Validity & Leakage Audit Summary

- **LoRA24 Adapters (`q8v5`)**: **VALID**. Pre-trained on 5-fold prompt-disjoint splits.
- **Open Pool De-duplication**: **VALID**. The 36k Clean Open Pool (`open_pool_clean.parquet`) applies NFKC whitespace de-duplication against official rows.
- **Unsupervised Prompt Recovery**: **VALID**. Fits TF-IDF on test passages only, clustering by content words. Accesses zero labels and zero prompt IDs.
- **Centred Twin Heads**: **VALID**. Evaluates $Z - \bar{Z}_{\text{cluster}}$ unsupervised.
- **4-View LinearSVC**: **VALID**. Pre-fitted out-of-fold during EXP28. Zero validation label leakage.
- **Sinkhorn Optimal Transport**: **VALID**. Unsupervised test marginal transport. Zero label leakage.
- **ATLAS Generative Likelihood**: **HIGH RISK / DANGEROUS**. Label leakage in training provenance + 2-hour timeout risk. Excluded from immediate ladder.

---

## 6. EXP48 Findings: Sinkhorn Marginal Calibration
- Local BA: **0.5910** (flat vs EXP47: +0.0000).
- Conclusion: Standalone calibrated neural networks do not gain from marginal transport alone.
- Decision: **YELLOW**; retained as an ensemble regularizer for multi-view blends.

---

## 7. EXP49 Findings: 4-View Stylometric LinearSVC Fusion
- **Equal Blend ($w=0.50$)**: Balanced Accuracy = **0.6354** (**+0.0445 gain** over EXP47).
- **Optimal Entropy Blend ($w=0.85$)**: Balanced Accuracy = **0.7028** (**+0.1118 gain** over EXP47).
- **Per-Class Recall**: 16 of 18 classes improved; 8 frontier classes gained $+14\%$ to $+22\%$ recall.
- **Sibling Error Reduction**: Sibling confusion dropped by 59% to 71% (`claude-opus`, `gemini-flash`, `gpt`).
- **Fold Stability**: Cross-fold standard deviation dropped from ±0.0244 to **±0.0149**.
- **Decision**: **GREEN (Official Kaggle Submission Candidate)**.

---

## 8. Exact Next Action & Step-by-Step Instructions

1. **User Kaggle Submission Review (EXP49)**:
   - Candidate package is prepared in [`submissions/candidates/EXP49/`](file:///home/rj2826/name-the-llm/submissions/candidates/EXP49/).
   - Kernel notebook: [`submissions/candidates/EXP49/submission_kernel_exp49.ipynb`](file:///home/rj2826/name-the-llm/submissions/candidates/EXP49/submission_kernel_exp49.ipynb).
   - Expected Public LB: **0.655 – 0.672** (vs 0.620 current best).
   - **STRICT PROTOCOL**: Do NOT submit automatically. Await explicit user authorization.
2. **If Approved**:
   - Upload notebook to Kaggle on **GPU T4 x2**.
   - Attach: `qwen-lm/qwen-3 (8b-base)`, `dhruvtibarewal/nlm-l24-adapters-v5`, `crtproduction/open-pool-passages` (or bundled parquet), and competition data.
   - Commit and submit. Record score.
3. **Subsequent Sequential Experiments (Local First)**:
   - **EXP50**: Best validated combination (EXP49) + 37 Typography Markers + V40A Sibling Style Residual.
   - **EXP51**: LoRA Multi-Model Scaling (add Qwen3-14B `q14v5`).

---

## 9. Strict Competition Rules & Engineering Protocols

1. **NEVER automatically submit to Kaggle**: All submissions require explicit user authorization.
2. **NEVER overwrite frozen experiments**: EXP45, EXP46A, and EXP47 files remain immutable.
3. **ONE controlled experiment at a time**: Test single additions against verified baselines.
4. **NEVER use label leakage or infer hidden labels**.
5. **Preserve exact provenance of all models, adapters, and datasets**.

---

## 10. Exact File Paths Index

| Resource | Exact Absolute Path |
|---|---|
| **EXP46A Kernel (Notebook)** | `/home/rj2826/name-the-llm/notebooks/46_qwen_lora_kernel.ipynb` |
| **EXP47 Kernel (Notebook)** | `/home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.ipynb` |
| **EXP48 Script** | `/home/rj2826/name-the-llm/src/exp48_sinkhorn_calibration.py` |
| **EXP48 Report (27 sections)** | `/home/rj2826/name-the-llm/reports/experiments/EXP48/REPORT.md` |
| **EXP48 Notebook** | `/home/rj2826/name-the-llm/notebooks/48_sinkhorn_calibration.ipynb` |
| **EXP49 Script** | `/home/rj2826/name-the-llm/src/exp49_stylometric_svc_fusion.py` |
| **EXP49 Report (27 sections)** | `/home/rj2826/name-the-llm/reports/experiments/EXP49/REPORT.md` |
| **EXP49 Notebook** | `/home/rj2826/name-the-llm/notebooks/49_stylometric_svc_fusion.ipynb` |
| **EXP49 Candidate Package** | `/home/rj2826/name-the-llm/submissions/candidates/EXP49/` |
| **EXP49 Submission Kernel** | `/home/rj2826/name-the-llm/submissions/candidates/EXP49/submission_kernel_exp49.ipynb` |
| **Experiment Index** | `/home/rj2826/name-the-llm/EXPERIMENT_INDEX.md` |
| **Leaderboard** | `/home/rj2826/name-the-llm/LEADERBOARD.md` |

---

## 11. Git Status & Latest Commit

- **Branch**: `main`
- **Working Tree State**: All EXP48 and EXP49 reports, notebooks, candidate files, and metrics safely generated.
