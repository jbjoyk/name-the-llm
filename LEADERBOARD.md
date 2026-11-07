# LEADERBOARD: KAGGLE "NAME THAT LLM"

Official project scoreboard tracking local 5-fold prompt-disjoint GroupKFold validation alongside verified Kaggle Public Leaderboard scores.

> [!IMPORTANT]
> **Scientific Discipline Principle**: Local Balanced Accuracy is evaluated on strict prompt-disjoint partitions (`GroupKFold(groups=prompt_id)`). Public Leaderboard scores are recorded only after explicit user submission. Never optimize repeatedly against the Public LB.

---

| Experiment ID | Description | Local BA | Local Macro-F1 | Kaggle Public LB | Delta vs Prev (Local) | Delta vs EXP46A (Local) | Runtime | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP_45** | Relational GAT + Hungarian Matching (Dense Assumption) | 0.4522 (sparse sim) | 0.4410 | **0.449** | — | -0.1340 | ~12 min | **FROZEN** |
| **EXP_46A** | Standalone Qwen3-8B LoRA24 (6 Merged Adapters) | **0.5862** | 0.5866 | **0.606** | +0.1340 | Baseline | ~16–18 min | **FROZEN BASELINE** |
| **EXP_47** | Qwen3-8B + Centred Twin Head ($w_{\text{twin}}=0.35$) | **0.5910** | 0.5915 | **0.620** | +0.0048 | +0.0048 | ~16–18 min | **VERIFIED TOP** |
| **EXP_48** | EXP47 + Damped Sinkhorn Marginal Calibration | **0.5910** | 0.5915 | *Not Submitted* | +0.0000 | +0.0048 | 2.4s (CPU) | **VALID REGULARIZER** |
| **EXP_49** | EXP47 + 4-View Open-Pool LinearSVC ($w=0.50\text{--}0.85$) | **0.6354–0.7028** | **0.6359–0.7011** | **CANDIDATE READY** | **+0.0445–0.1118** | **+0.0492–0.1166** | ~18.5 min | **AWAITING APPROVAL** |

---

## Metric Definitions:
- **Local BA**: Macro-averaged Balanced Accuracy across all 18 classes on `data/train.csv` (2,699 samples) under strict 5-fold prompt-isolated cross-validation.
- **Local Macro-F1**: Macro-averaged F1 score across all 18 classes.
- **Kaggle Public LB**: Official Kaggle score evaluated on ~4,000 hidden test passages (sparse prompt regime, $R \approx 2$).
