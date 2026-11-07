# LEADERBOARD: KAGGLE "NAME THAT LLM"

Official project scoreboard tracking local 5-fold prompt-disjoint GroupKFold validation alongside verified Kaggle Public Leaderboard scores.

> [!IMPORTANT]
> **Scientific Discipline Principle**: Local Balanced Accuracy is evaluated on strict prompt-disjoint partitions (`GroupKFold(groups=prompt_id)`). Selection-safe nested validation is required before declaring candidate readiness. Public Leaderboard scores are recorded only after explicit user submission.

---

| Experiment ID | Description | Local BA (Nested CV) | Local Macro-F1 | Kaggle Public LB | Delta vs Baseline (Local) | Delta vs EXP46A (Local) | Runtime | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP_45** | Relational GAT + Hungarian Matching (Dense Assumption) | 0.4522 (sparse sim) | 0.4410 | **0.449** | — | -0.1340 | ~12 min | **FROZEN** |
| **EXP_46A** | Standalone Qwen3-8B LoRA24 (6 Merged Adapters) | **0.5862** | 0.5866 | **0.606** | +0.1340 | Baseline | ~16–18 min | **FROZEN BASELINE** |
| **EXP_47** | Qwen3-8B + Centred Twin Head ($w_{\text{twin}}=0.35$) | **0.5910** | 0.5915 | **0.620** | +0.0048 | +0.0048 | ~16–18 min | **VERIFIED TOP** |
| **EXP_48** | EXP47 + Damped Sinkhorn Marginal Calibration | **0.5910** | 0.5915 | *Not Submitted* | +0.0000 | +0.0048 | 2.4s (CPU) | **VALID REGULARIZER** |
| **EXP_49** | EXP47 + 4-View Open-Pool LinearSVC ($w=0.85$) | **0.6976** | 0.6964 | **CANDIDATE READY** | **+0.1066** | **+0.1114** | ~18.5 min | **VALIDATED GATE PASS** |
| **EXP_49 + SK** | EXP49 + Soft Sinkhorn ($\alpha=0.30, \rho=0.35$) | **0.7054** | **0.7042** | **CANDIDATE READY** | **+0.1144** | **+0.1192** | ~18.5 min | **VALIDATED GATE PASS** |

---

## Metric Definitions:
- **Local BA (Nested CV)**: Macro-averaged Balanced Accuracy across all 18 classes on `data/train.csv` (2,699 samples) under strict 5-fold outer / 4-fold inner prompt-disjoint nested cross-validation.
- **Local Macro-F1**: Macro-averaged F1 score across all 18 classes.
- **Kaggle Public LB**: Official Kaggle score evaluated on ~4,000 hidden test passages (sparse prompt regime, $R \approx 2$).
