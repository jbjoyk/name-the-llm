# PROJECT STATE & AGENT HANDOFF: KAGGLE "NAME THAT LLM"

**Last Updated**: 2026-11-07 13:45 UTC  
**Primary Repository**: `/home/rj2826/name-the-llm`  
**Current Active State**: **EXP49 Evaluated on Kaggle (0.612 Public LB) | Best Remains EXP47 (0.620)**  
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
| **EXP_49** | EXP47 + 4-View Open-Pool LinearSVC ($w=0.85$) + SK | **0.6976** (nested raw) / **0.7054** (nested + SK) | **0.612** | **EVALUATED (Surface Domain Shift)** |

---

## 4. EXP49 Validation Gate Findings (Selection-Safe Nested CV)
1. **Nested CV Result**: Outer validation score across 5 prompt-disjoint folds is **0.6976** (0.6977 ± 0.0145). With Sinkhorn ($\alpha=0.30, \rho=0.35$), it reaches **0.7054** (0.7055 ± 0.0144).
2. **Selected Weights**: Outer fold weights chosen on inner training splits: `[0.87, 0.82, 0.87, 0.82, 0.85]` (Median: **0.850**).
3. **Representation Orthogonality**: Error correlation between Qwen3-8B and LinearSVC is **$r = 0.0005$** (pure statistical independence). Prediction agreement is only 29.64%.
4. **Leakage & Contamination Audit**: Zero string overlap and zero normalized whitespace matches between Open Pool and official data.
5. **Expected Public LB**: **0.655 – 0.672** (anticipating minor domain shift and sparse prompt regime).

---

## 5. Kaggle Evaluation & Exact Next Campaign Actions

1. **EXP49 Evaluated on Kaggle & Forensic Audit Completed**:
   - **Official Public LB**: **0.612**
   - **Baseline Comparison**: Outperforms standalone EXP46A (0.606), but falls slightly below pure EXP47 (0.620).
   - **Forensic Audit Report**: [`reports/experiments/EXP49/KAGGLE_0612_FORENSIC_AUDIT.md`](file:///home/rj2826/name-the-llm/reports/experiments/EXP49/KAGGLE_0612_FORENSIC_AUDIT.md).
   - **Key Findings**:
     - Qwen3-8B LoRA24 inference engine, 24-layer truncation, manual LoRA injection, LayerNorms, and prompt clustering are **100% bitwise identical** between EXP47 and EXP49.
     - **Decision Mass Inversion**: In EXP49, $w_{\text{svc}} = 0.850$ gave 85% linear voting weight to the shallow LinearSVC, reducing the 8B Qwen model to 15%.
     - **Logit Scale Mismatch**: Qwen's logit margins are $8.2\times$ sharper than the LinearSVC's flat softmax logits (mean top-2 gap: 1.955 nats vs 0.237 nats). Local tuning pushed $w_{\text{svc}}=0.85$ to balance scales on `train.csv`, but on unseen test passages, surface n-grams suffer domain shift.
     - **Reference Ensemble Contrast**: In the 0.672 reference notebook, 6 transformer backbones totaled $>10.5$ weight while the SVM represented only $\approx 12\%$ of total mass. EXP49 inverted this by making the SVM 85% of the ensemble.
   - **Protocol Update**: Moving forward, deep sequence models must constitute $\ge 75\%$ of voting mass ($w_{\text{deep}} \ge 0.75$), with surface features acting strictly as conservative regularizers ($\le 25\%$).
2. **Next Sequential Experiment (EXP50 - Local First)**:
   - Combine verified EXP47 (deep anchor) with audited typography markers and V40A sibling style residuals.
   - Enforce **deep-dominant conservative weighting** ($w_{\text{deep}} \ge 0.75$, $w_{\text{surface}} \le 0.25$).
   - Run local 5-fold prompt-disjoint `GroupKFold` cross-validation sweeps before proposing any Kaggle candidate.
3. **Subsequent Sequential Experiments**:
   - **EXP51**: LoRA Multi-Model Scaling (add Qwen3-14B `q14v5`).
   - **EXP52+**: Multi-model deep ensemble + conservative regularizers.

---

## 6. Strict Competition Rules & Engineering Protocols

1. **NEVER automatically submit to Kaggle**: All submissions require explicit user authorization.
2. **NEVER overwrite frozen experiments**: EXP45, EXP46A, and EXP47 files remain immutable.
3. **ONE controlled experiment at a time**: Test single additions against verified baselines.
4. **NEVER use label leakage or infer hidden labels**.
5. **Preserve exact provenance of all models, adapters, and datasets**.

---

## 7. Exact File Paths Index

| Resource | Exact Absolute Path |
|---|---|
| **EXP46A Kernel (Notebook)** | `/home/rj2826/name-the-llm/notebooks/46_qwen_lora_kernel.ipynb` |
| **EXP47 Kernel (Notebook)** | `/home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.ipynb` |
| **EXP48 Script** | `/home/rj2826/name-the-llm/src/exp48_sinkhorn_calibration.py` |
| **EXP48 Report (27 sections)** | `/home/rj2826/name-the-llm/reports/experiments/EXP48/REPORT.md` |
| **EXP49 Script** | `/home/rj2826/name-the-llm/src/exp49_stylometric_svc_fusion.py` |
| **EXP49 Report (Extended)** | `/home/rj2826/name-the-llm/reports/experiments/EXP49/REPORT.md` |
| **EXP49 Notebook** | `/home/rj2826/name-the-llm/notebooks/49_stylometric_svc_fusion.ipynb` |
| **EXP49 Candidate Package** | `/home/rj2826/name-the-llm/submissions/candidates/EXP49/` |
| **EXP49 Submission Kernel** | `/home/rj2826/name-the-llm/submissions/candidates/EXP49/submission_kernel_exp49.ipynb` |
| **EXP49 Kaggle 0.612 Forensic Audit** | `/home/rj2826/name-the-llm/reports/experiments/EXP49/KAGGLE_0612_FORENSIC_AUDIT.md` |
| **Experiment Index** | `/home/rj2826/name-the-llm/EXPERIMENT_INDEX.md` |
| **Leaderboard** | `/home/rj2826/name-the-llm/LEADERBOARD.md` |
