# LEADERBOARD: KAGGLE "NAME THAT LLM"

Official project scoreboard tracking local 5-fold prompt-disjoint GroupKFold validation alongside verified Kaggle Public Leaderboard scores.

> [!IMPORTANT]
> **Scientific Discipline Principle**: Local Balanced Accuracy is evaluated on strict prompt-disjoint partitions (`GroupKFold(groups=prompt_id)`). Selection-safe nested validation is required before declaring candidate readiness. Public Leaderboard scores are recorded only after explicit user submission.

---

| Experiment ID | Description | Local BA (Nested CV) | Local Macro-F1 | Kaggle Public LB | Delta vs Baseline (Local) | Delta vs EXP46A (Local) | Runtime | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP_45** | Relational GAT + Hungarian Matching (Dense Assumption) | 0.4522 (sparse sim) | 0.4410 | **0.449** | — | -0.1340 | ~12 min | **FROZEN** |
| **EXP_46A** | Standalone Qwen3-8B LoRA24 (6 Merged Adapters) | **0.5862** | 0.5866 | **0.606** | +0.1340 | Baseline | ~16–18 min | **FROZEN BASELINE** |
| **EXP_47** | Qwen3-8B + Centred Twin Head ($w_{\text{twin}}=0.35$) | **0.5910** | 0.5915 | **0.620** | +0.0048 | +0.0048 | ~16–18 min | **FORMER TOP** |
| **EXP_48** | EXP47 + Damped Sinkhorn Marginal Calibration | **0.5910** | 0.5915 | *Not Submitted* | +0.0000 | +0.0048 | 2.4s (CPU) | **VALID REGULARIZER** |
| **EXP_49** | EXP47 + 4-View Open-Pool LinearSVC ($w=0.85$) + SK | **0.7054** | **0.7042** | **0.612** | **+0.1144** | **+0.1192** | ~18.5 min | **EVALUATED (Over-weighted SVC)** |
| **EXP_50** | Controlled Qwen-Dominant Fusion ($w_{\text{qwen}}=0.75, w_{\text{svc}}=0.25$) | **0.6102** | **0.6107** | *Not Submitted* | **+0.0193** | **+0.0240** | 0.26s (CPU) | **VALIDATED CANDIDATE (GREEN)** |
| **EXP_50B** | Extended Sweep (0.20--0.50): Max at Boundary ($w=0.50$) | **0.6354** | **0.6359** | *Not Submitted* | **+0.0445** | **+0.0492** | 0.66s (CPU) | **UNRESOLVED BOUNDARY (ANALYSIS)** |
| **EXP_51** | Qwen3-14B Two-Stage LoRA Audit | **0.6346** (14B) / **0.7873** (fused) | 0.6338 / 0.7880 | *Not Submitted* | **+0.1963** (fused) | **+0.2011** | Audit | **STOP & REPORT (GREEN)** |
| **EXP_51A** | Forensic Provenance Audit of Qwen3-14B OOF | **0.6346** (verified OOF) | 0.6338 | *Not Submitted* | **+0.0436** | **+0.0484** | Audit | **VALID (q14v5 OOF verified)** |
| **EXP_52** | Dual-LoRA Qwen3-8B (0.60) + Qwen3-14B (0.40) Kernel | **0.7873** | **0.7880** | **0.632** | **+0.1963** | **+0.2011** | ~116 min (Kaggle) | **VERIFIED TOP (CURRENT BASELINE)** |
| **EXP_53** | Dual-LoRA Weight Sweep (0.00--0.60, 2000 Bootstrap CIs) | **0.7873** | **0.7880** | *Not Submitted* | **+0.1963** | **+0.2011** | 2.5 min (OOF) | **FROZEN (0.40 Confirmed Optimal)** |
| **EXP_54** | EXP52 + Open Pool 4-View LinearSVC Sweep ($w_{\text{svc}} \le 0.20$) | **0.7888** | **0.7895** | *Not Submitted* | **+0.1978** | **+0.2026** | 13.5s (OOF) | **FROZEN (EXP52 Retained at 0.632)** |
| **EXP_55** | Isolated Stylometry / Char TF-IDF Specialist Sweep ($w \le 0.25$) | **0.7881** (nested) / **0.7899** ($w=0.05$) | **0.7907** | *Not Submitted* | **+0.0008** (nested) / **+0.0026** ($w=0.05$) | **+0.2019** | 54.5s (OOF) | **YELLOW (EXP52 Retained at 0.632)** |
| **EXP_56** | Qwen3-14B Prompt-Conditioned Residual Specialist ($w \le 0.20$) | **0.7866** (nested) / **0.7888** ($w=0.20$) | **0.7896** | *Not Submitted* | **-0.0007** (nested) / **+0.0015** ($w=0.20$) | **+0.2004** | 21.0s (OOF) | **RED (Duplicate of head_cent; Retain EXP52)** |
| **EXP_57A** | Weight-Space Programming Diagnosis for NTL | *Read-only diagnosis* | — | *Not Submitted* | — | — | 12.8s | **READ-ONLY DIAGNOSIS** |
| **EXP_57B** | Class-Head Norm Equalization Validation | **0.7847** in EXP52 ($\gamma=1.00$) | 0.7849 | *Not Submitted* | **-0.0026** (vs EXP52) | **+0.1985** | 8.4s | **RED (Refuted causal hypothesis)** |
| **EXP_58** | Gemini Sibling Early-Layer Discriminator Falsification | **0.5367** (Layer 4 vs 0.6067 full) | 0.5366 | *Not Submitted* | **-0.0700** (vs binary) | — | 3.2s | **RED (Falsified early-layer signal)** |

---

## Metric Definitions:
- **Local BA (Nested CV)**: Macro-averaged Balanced Accuracy across all 18 classes on `data/train.csv` (2,699 samples) under strict 5-fold outer / 4-fold inner prompt-disjoint nested cross-validation.
- **Local Macro-F1**: Macro-averaged F1 score across all 18 classes.
- **Kaggle Public LB**: Official Kaggle score evaluated on ~4,000 hidden test passages (sparse prompt regime, $R \approx 2$).
