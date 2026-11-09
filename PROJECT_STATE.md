# PROJECT STATE & AGENT HANDOFF: KAGGLE "NAME THAT LLM"

**Last Updated**: 2026-11-09 13:52 UTC
**Primary Repository**: `/home/rj2826/name-that-llm`
**Current Active State**: **EXP62 Cross-Architecture Complementarity Test Completed (Verdict: RED; Retain EXP52 as frozen production champion at 0.632 LB)**
**Current Production Baseline**: **EXP52**
- **Kaggle Public LB**: **0.632**
- **Prompt-Disjoint OOF BA**: **0.7873**
- **Status**: **EXP52 frozen**
- **Latest Completed Experiment**: **EXP62**
- **Latest Experiment Verdict**: **RED**
- **Next Research Priority**: **Stop un-fine-tuned cross-architecture model stacking. Focus on prompt-invariant calibration or sequence fine-tuning.**
- **Kaggle Submissions**: **No Kaggle submission after EXP52.**

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
| **EXP_47** | Qwen3-8B Centred Twin Head | **0.5910** | **0.620** | **FORMER TOP ON KAGGLE** |
| **EXP_48** | EXP47 + Damped Sinkhorn Calibration | **0.5910** (+0.0000) | *Not Submitted* | **YELLOW** (Ensemble regularizer) |
| **EXP_49** | EXP47 + 4-View Open-Pool LinearSVC ($w=0.85$) + SK | **0.6976** (nested raw) / **0.7054** (nested + SK) | **0.612** | **EVALUATED (Surface Domain Shift)** |
| **EXP_50** | Controlled Qwen-Dominant Fusion ($w_{\text{qwen}}=0.75, w_{\text{svc}}=0.25$) | **0.6102** (+0.0193 vs EXP47) | *Not Submitted* | **GREEN (Validated Candidate)** |
| **EXP_50B** | Extended Sweep (0.20--0.50): Max at Boundary ($w=0.50$) | **0.6354** (+0.0445 vs EXP47) | *Not Submitted* | **ANALYSIS (Boundary Unresolved)** |
| **EXP_51** | Qwen3-14B Two-Stage LoRA Technical Audit | **0.6346** (14B OOF) / **0.7873** (Dual Fusion) | *Not Submitted* | **GREEN (Stop & Report / Complementary)** |
| **EXP_51A** | Forensic Provenance Audit of Qwen3-14B OOF | **0.6346** (Verified Dhruv `q14v5`) | *Not Submitted* | **GREEN (Provenance Verified)** |
| **EXP_52** | Dual-LoRA Qwen3-8B (0.60) + Qwen3-14B (0.40) Kernel | **0.7873** (OOF Reference BA) | **0.632** | **CURRENT PRODUCTION BASELINE (TOP)** |
| **EXP_53** | Dual-LoRA Fusion Weight Sweep (w14: 0.00–0.60) | **0.7873** (Peak at w14=0.40) | *Not Submitted* | **FROZEN (0.40 Confirmed Optimal)** |
| **EXP_54** | EXP52 + Open Pool 4-View LinearSVC Sweep ($w_{\text{svc}} \le 0.20$) | **0.7888** (+0.0015 at w=0.05) | *Not Submitted* | **FROZEN (EXP52 Retained at 0.632)** |
| **EXP_55** | Isolated Stylometry / Char TF-IDF Specialist Sweep ($w \le 0.25$) | **0.7881** (nested) / **0.7899** ($w=0.05$) | *Not Submitted* | **YELLOW (EXP52 Retained at 0.632)** |
| **EXP_56** | Qwen3-14B Prompt-Conditioned Residual Specialist ($w \le 0.20$) | **0.7866** (nested) / **0.7888** ($w=0.20$) | *Not Submitted* | **RED (Duplicate of head_cent; Retain EXP52)** |
| **EXP_57A** | Weight-Space Programming Diagnosis for NTL | **0.7873** (Baseline Inventory) | *Not Submitted* | **READ-ONLY DIAGNOSIS (Formulated Hypotheses)** |
| **EXP_57B** | Class-Head Norm Equalization Validation | **0.7847** ($\Delta = -0.0026$ in EXP52) | *Not Submitted* | **RED (Refuted Causal Hypothesis; Retain EXP52)** |
| **EXP_58** | Gemini Sibling Early-Layer Discriminator Falsification | **0.5367** (Layer 4 vs 0.6067 full) | *Not Submitted* | **RED (Refuted Early-Layer Hypothesis; Retain EXP52)** |
| **EXP_59** | OOF vs Kaggle Hidden-Surface Forensic Audit | **0.7873** (100% Verified) | **0.6320** | **GREEN (Audit Complete; Formulated Hypotheses)** |
| **EXP_60** | Sparse Test-Regime Calibration ($R=2$ vs $R=18$) | **0.7774** (Sparse R=2) / **0.7873** (Dense R=18) | *Not Submitted* | **YELLOW / PARTIALLY SUPPORTED (Density $\Delta\Delta=+0.0057$, Explains 6.4% Gap)** |
| **EXP_61** | Hidden-Surface / Representation-Shift Forensics | **0.7858** (Matched $R=2$) / **0.7873** (Dense OOF) | *Not Submitted* | **YELLOW / PARTIALLY SUPPORTED (Manifold distance $\rho=+0.0071$, error correlation drives gap)** |
| **EXP_62** | Cross-Architecture Complementarity Test (ModernBERT-base) | **0.7858** (Nested CV) / **0.3093** (ModernBERT standalone) | *Not Submitted* | **RED (Refuted cross-architecture hypothesis, nested $\Delta=-0.0015$; Retain EXP52)** |

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
2. **EXP50 Controlled Qwen-Dominant Sweep Completed (Local First)**:
   - Evaluated 6 weights ($w_{\text{qwen}} \in [0.75, 1.00]$, $w_{\text{svc}} \in [0.00, 0.25]$) across Official Only and Official + 36k Open Pool.
   - **Local BA**: Climbed monotonically from 0.5910 to **0.6102** (+0.0193) under Open Pool, with zero fold degradation across all 5 folds.
   - **Open Pool Lift**: Open Pool roughly doubles the lift (+0.0193 vs +0.0078 Official Only).
   - **Anchor Stability**: Retains 96.9%–99.8% prediction agreement with Qwen, restricting SVC strictly to an auxiliary tie-breaker.
   - **Decision**: **GREEN**. Retained locally; awaiting user approval before any submission.
3. **EXP50B Extended Fusion Weight Sweep Completed (0.20 to 0.50)**:
   - Evaluated 31 fine-grained weights ($w_{\text{svc}} \in [0.20, 0.50]$ at 0.01 steps) across Official Only and Open Pool.
   - **Key Finding**: Local Balanced Accuracy rises monotonically across the entire window without developing an interior peak. Maximum in range is at boundary $w_{\text{svc}} = 0.50$ (**0.6354 BA** Open Pool, **0.6147 BA** Official Only).
   - **Boundary Status**: The local CV boundary has **NOT been resolved** within $[0.20, 0.50]$.
   - **Epistemic Warning**: High SVC weights overfit local prompt vocabulary and margin mismatch ($8.24\times$ wider in Qwen). Candidate recommendation prioritizes conservative weights ($w_{\text{svc}} = 0.25–0.35$) with $>95\%$ Qwen agreement to prevent unseen test domain shift.
4. **EXP51 Qwen3-14B Two-Stage LoRA Technical Audit Complete**:
   - Comprehensive audit of 11 components from `two-stage-lora-part-two.ipynb`.
   - Dataset `nishantdahal/ntl-handwriting-blend` downloaded and verified (`lora_qwen3_14b`).
   - STOP & REPORT condition triggered: reference release includes strictly a single full-fit checkpoint without fold-specific adapters or OOF arrays. Forward-passing `train.csv` would be in-sample evaluation. Base model not present on local disk.
   - Complementarity with EXP47: Near-zero error correlation ($r=0.0214$), 40.27% agreement, Oracle BA = **0.8455**. Optimal fusion ($w_{14\text{B}}=0.40, w_{\text{EXP47}}=0.60$) achieves **0.7873 Balanced Accuracy** (+0.1963 over EXP47).
5. **EXP51A Qwen3-14B OOF Forensic Provenance Audit Complete**:
   - Analyzed `outputs/oof_q14v5_antopil_lora.csv` (SHA-256: `58822e8d...`).
   - Prompt-disjointness: VERIFIED VALID. Zero prompts span multiple folds; exactly 30 prompts per fold.
   - Training row exclusion: VERIFIED VALID via LoRA24 worker training code (`fit = fold != int(job)`).
   - Lineage: Proved that artifact belongs to Dhruv Tibarewal / Anto Pil's `q14v5` model family (trained on 120k pool + train), NOT Nishant Dahal's `lora_qwen3_14b` (which never released fold adapters).
   - Reproducibility: Exactly 0.634609 Balanced Accuracy directly reproduced.
   - Final Verdict: VALID (as `q14v5` LoRA24 OOF) / INVALID (as Nishant Dahal reproduction).
6. **Subsequent Sequential Experiments**:
   - **EXP52**: Dual-LoRA Kaggle Submission Kernel assembly (Qwen3-8B $w=0.60$ + Qwen3-14B $w=0.40$ + Centred Twin) using verified `dhruvtibarewal/nlm-l24-adapters-v5` adapters. Validated on Kaggle at **0.632 Public LB** (undefeated current top baseline).
   - **EXP53**: Dual-LoRA Fusion Weight Sweep over $w_{14} \in [0.00, 0.60]$. Confirmed $w_{14}=0.40$ is OOF-optimal (0.7873 BA) and frozen.
   - **EXP54**: Controlled Qwen3 (8B+14B) + Open Pool 4-View LinearSVC Sweep over $w_{\text{svc}} \in [0.00, 0.20]$. Marginal OOF peak at $w_{\text{svc}}=0.05$ (0.7888 BA, $\Delta = +0.0015$) represents only 4 net samples with 95% bootstrap CI spanning zero ([-0.0001, +0.0034]). Frozen at EXP52 (0.632 LB); zero submissions recommended.

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
| **EXP50 Script** | `/home/rj2826/name-the-llm/src/exp50_qwen_dominant_fusion.py` |
| **EXP50 Notebook** | `/home/rj2826/name-the-llm/notebooks/50_qwen_dominant_fusion.ipynb` |
| **EXP50 Report** | `/home/rj2826/name-the-llm/reports/experiments/EXP50/REPORT.md` |
| **EXP50 Leakage Audit** | `/home/rj2826/name-the-llm/reports/experiments/EXP50/leakage_audit.md` |
| **EXP50 Candidate Config** | `/home/rj2826/name-the-llm/submissions/candidates/EXP50/config.json` |
| **EXP50 OOF Predictions** | `/home/rj2826/name-the-llm/outputs/oof_exp50_qwen_dominant_fusion.npy` |
| **EXP50B Script** | `/home/rj2826/name-the-llm/src/exp50b_extended_fusion_sweep.py` |
| **EXP50B Notebook** | `/home/rj2826/name-the-llm/notebooks/50b_extended_fusion_sweep.ipynb` |
| **EXP50B Report** | `/home/rj2826/name-the-llm/reports/experiments/EXP50B/REPORT.md` |
| **EXP50B Leakage Audit** | `/home/rj2826/name-the-llm/reports/experiments/EXP50B/leakage_audit.md` |
| **EXP50B Candidate Config** | `/home/rj2826/name-the-llm/submissions/candidates/EXP50B/config.json` |
| **EXP50B OOF Predictions** | `/home/rj2826/name-the-llm/outputs/oof_exp50b_qwen_svc_fusion.npy` |
| **EXP51 Report** | `/home/rj2826/name-the-llm/reports/EXP51_qwen3_14b_two_stage_audit.md` |
| **EXP51 Audit Metrics** | `/home/rj2826/name-the-llm/reports/experiments/EXP51/metrics.json` |
| **EXP51 Checkpoints** | `/home/rj2826/name-the-llm/models/ntl-handwriting-blend/lora_qwen3_14b/` |
| **EXP51A Report** | `/home/rj2826/name-the-llm/reports/EXP51A_qwen3_14b_oof_provenance.md` |
| **EXP51A Persisted OOF** | `/home/rj2826/name-the-llm/outputs/oof_q14v5_antopil_lora.csv` |
| **EXP52 Script** | `/home/rj2826/name-the-llm/src/exp52_dual_lora_inference.py` |
| **EXP52 Notebook** | `/home/rj2826/name-the-llm/notebooks/52_dual_lora_kernel.ipynb` |
| **EXP52 Report** | `/home/rj2826/name-the-llm/reports/EXP52_dual_lora_inference.md` |
| **EXP52 Metrics** | `/home/rj2826/name-the-llm/experiments/EXP52_dual_lora/metrics.json` |
| **EXP52 Runtime** | `/home/rj2826/name-the-llm/experiments/EXP52_dual_lora/runtime.json` |
| **EXP52 Candidate Submission** | `/home/rj2826/name-the-llm/submissions/candidates/EXP52/submission.csv` |
| **EXP53 Script** | `/home/rj2826/name-the-llm/src/exp53_fusion_weight_sweep.py` |
| **EXP53 Report** | `/home/rj2826/name-the-llm/reports/EXP53_REPORT.md` |
| **EXP53 Results** | `/home/rj2826/name-the-llm/experiments/EXP53_fusion_weight/EXP53_results.csv` |
| **EXP53 Plot** | `/home/rj2826/name-the-llm/reports/figures/exp53_weight_sweep.png` |
| **EXP54 Script** | `/home/rj2826/name-the-llm/src/exp54_qwen_svc.py` |
| **EXP54 Report** | `/home/rj2826/name-the-llm/reports/EXP54_REPORT.md` |
| **EXP54 Results** | `/home/rj2826/name-the-llm/experiments/EXP54_qwen_svc/EXP54_results.csv` |
| **EXP54 Predictions** | `/home/rj2826/name-the-llm/experiments/EXP54_qwen_svc/EXP54_oof_predictions.parquet` |
| **EXP54 Plot** | `/home/rj2826/name-the-llm/reports/figures/exp54_weight_sweep.png` |
| **EXP55 Script** | `/home/rj2826/name-the-llm/src/exp55_stylometry.py` |
| **EXP55 Report** | `/home/rj2826/name-the-llm/reports/EXP55_REPORT.md` |
| **EXP55 Results** | `/home/rj2826/name-the-llm/experiments/EXP55_stylometry/EXP55_results.csv` |
| **EXP55 Predictions** | `/home/rj2826/name-the-llm/experiments/EXP55_stylometry/EXP55_oof_predictions.parquet` |
| **EXP55 Summary** | `/home/rj2826/name-the-llm/experiments/EXP55_stylometry/summary.json` |
| **EXP55 OOF Parquet** | `/home/rj2826/name-the-llm/outputs/oof_stylometry_exp55.parquet` |
| **EXP55 Plot (Sweep)** | `/home/rj2826/name-the-llm/reports/figures/exp55_weight_sweep.png` |
| **EXP55 Plot (Confusion)** | `/home/rj2826/name-the-llm/reports/figures/exp55_confusion_matrix.png` |
| **EXP55 Plot (Complement)** | `/home/rj2826/name-the-llm/reports/figures/exp55_complementarity.png` |
| **EXP56 Script** | `/home/rj2826/name-the-llm/src/exp56_q14_residual.py` |
| **EXP56 Report** | `/home/rj2826/name-the-llm/reports/EXP56_REPORT.md` |
| **EXP56 Results** | `/home/rj2826/name-the-llm/experiments/EXP56_q14_residual/EXP56_results.csv` |
| **EXP56 Predictions** | `/home/rj2826/name-the-llm/experiments/EXP56_q14_residual/EXP56_oof_predictions.parquet` |
| **EXP56 Summary** | `/home/rj2826/name-the-llm/experiments/EXP56_q14_residual/summary.json` |
| **EXP56 Plot (Sweep)** | `/home/rj2826/name-the-llm/reports/figures/exp56_weight_sweep.png` |
| **EXP56 Plot (Confusion)** | `/home/rj2826/name-the-llm/reports/figures/exp56_confusion_matrix.png` |
| **EXP56 Plot (Complement)** | `/home/rj2826/name-the-llm/reports/figures/exp56_complementarity.png` |
| **Experiment Index** | `/home/rj2826/name-the-llm/EXPERIMENT_INDEX.md` |
| **Leaderboard** | `/home/rj2826/name-the-llm/LEADERBOARD.md` |
| **EXP57A Report** | `/home/rj2826/name-the-llm/reports/EXP57A_REPORT.md` |
| **EXP57A Directory** | `/home/rj2826/name-the-llm/experiments/EXP57A_weight_space_diagnosis/` |
| **EXP57B Script** | `/home/rj2826/name-the-llm/src/exp57b_norm_equalization.py` |
| **EXP57B Report** | `/home/rj2826/name-the-llm/reports/EXP57B_REPORT.md` |
| **EXP57B Directory** | `/home/rj2826/name-the-llm/experiments/EXP57B_norm_equalization/` |
| **EXP57B Gamma Sweep** | `/home/rj2826/name-the-llm/experiments/EXP57B_norm_equalization/gamma_sweep.csv` |
| **EXP57B Fold Results** | `/home/rj2826/name-the-llm/experiments/EXP57B_norm_equalization/fold_results.csv` |
| **EXP57B EXP52 Integration** | `/home/rj2826/name-the-llm/experiments/EXP57B_norm_equalization/exp52_integration.csv` |
| **EXP58 Script** | `/home/rj2826/name-the-llm/src/exp58_gemini_layer_falsification.py` |
| **EXP58 Report** | `/home/rj2826/name-the-llm/reports/EXP58_REPORT.md` |
| **EXP58 Directory** | `/home/rj2826/name-the-llm/experiments/EXP58_gemini_early_layer_falsification/` |
| **EXP58 Layer Probes** | `/home/rj2826/name-the-llm/experiments/EXP58_gemini_early_layer_falsification/layer_probe_results.csv` |
| **EXP58 Fold Results** | `/home/rj2826/name-the-llm/experiments/EXP58_gemini_early_layer_falsification/fold_results.csv` |
| **EXP58 Distance Analysis** | `/home/rj2826/name-the-llm/experiments/EXP58_gemini_early_layer_falsification/distance_analysis.csv` |
| **EXP58 Centroid Analysis** | `/home/rj2826/name-the-llm/experiments/EXP58_gemini_early_layer_falsification/centroid_analysis.csv` |
| **EXP58 Random Permutations** | `/home/rj2826/name-the-llm/experiments/EXP58_gemini_early_layer_falsification/random_label_control.csv` |
| **EXP58 BA Plot** | `/home/rj2826/name-the-llm/experiments/EXP58_gemini_early_layer_falsification/layer_probe_ba.png` |
| **EXP58 Separation Plot** | `/home/rj2826/name-the-llm/experiments/EXP58_gemini_early_layer_falsification/layer_separation.png` |
| **EXP59 Script** | `/home/rj2826/name-that-llm/src/exp59_forensic_audit.py` |
| **EXP59 Report** | `/home/rj2826/name-that-llm/reports/EXP59_REPORT.md` |
| **EXP59 Directory** | `/home/rj2826/name-that-llm/experiments/EXP59_forensic_audit/` |
| **EXP60 Script** | `/home/rj2826/name-that-llm/src/exp60_sparse_validation.py` |
| **EXP60 Report** | `/home/rj2826/name-that-llm/reports/EXP60_REPORT.md` |
| **EXP60 Directory** | `/home/rj2826/name-that-llm/experiments/EXP60_sparse_validation/` |
| **EXP60 Results CSV** | `/home/rj2826/name-that-llm/experiments/EXP60_sparse_validation/results.csv` |
| **EXP60 Seed Results** | `/home/rj2826/name-that-llm/experiments/EXP60_sparse_validation/seed_results.csv` |
| **EXP60 Cluster Diag** | `/home/rj2826/name-that-llm/experiments/EXP60_sparse_validation/cluster_diagnostics.csv` |
| **EXP60 Density Curve** | `/home/rj2826/name-that-llm/experiments/EXP60_sparse_validation/density_curve.csv` |
| **EXP60 Config JSON** | `/home/rj2826/name-that-llm/experiments/EXP60_sparse_validation/config.json` |
| **EXP61 Script** | `/home/rj2826/name-that-llm/src/exp61_hidden_surface_forensics.py` |
| **EXP61 Report** | `/home/rj2826/name-that-llm/reports/EXP61_REPORT.md` |
| **EXP61 Directory** | `/home/rj2826/name-that-llm/experiments/EXP61_hidden_surface_forensics/` |
| **EXP61 Results CSV** | `/home/rj2826/name-that-llm/experiments/EXP61_hidden_surface_forensics/results.csv` |
| **EXP61 Domain Classifier** | `/home/rj2826/name-that-llm/experiments/EXP61_hidden_surface_forensics/domain_classifier_results.csv` |
| **EXP61 Rep Shift** | `/home/rj2826/name-that-llm/experiments/EXP61_hidden_surface_forensics/representation_shift.csv` |
| **EXP61 Pred Shift** | `/home/rj2826/name-that-llm/experiments/EXP61_hidden_surface_forensics/prediction_shift.csv` |
| **EXP61 Matched Surface** | `/home/rj2826/name-that-llm/experiments/EXP61_hidden_surface_forensics/matched_surface_results.csv` |
| **EXP61 Distance Error** | `/home/rj2826/name-that-llm/experiments/EXP61_hidden_surface_forensics/distance_error_results.csv` |
| **EXP61 Config JSON** | `/home/rj2826/name-that-llm/experiments/EXP61_hidden_surface_forensics/config.json` |
| **EXP61 README** | `/home/rj2826/name-that-llm/experiments/EXP61_hidden_surface_forensics/README.md` |
| **EXP62 Script** | `/home/rj2826/name-that-llm/src/exp62_cross_architecture.py` |
| **EXP62 Report** | `/home/rj2826/name-that-llm/reports/EXP62_REPORT.md` |
| **EXP62 Directory** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/` |
| **EXP62 Results CSV** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/results.csv` |
| **EXP62 Fold Results** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/fold_results.csv` |
| **EXP62 Complementarity** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/complementarity.csv` |
| **EXP62 Oracle Results** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/oracle_results.csv` |
| **EXP62 Fusion Results** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/fusion_results.csv` |
| **EXP62 Nested Results** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/nested_results.csv` |
| **EXP62 Prediction Disag** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/prediction_disagreement.csv` |
| **EXP62 Config JSON** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/config.json` |
| **EXP62 README** | `/home/rj2826/name-that-llm/experiments/EXP62_cross_architecture/README.md` |
