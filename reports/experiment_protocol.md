# Kaggle "Name That LLM" — Official Experiment & Validation Protocol

**Document Version**: 1.0.0  
**Effective Date**: November 2026  
**Status**: APPROVED & MANDATORY  
**Baseline Anchor**: EXP_36 / EXP_38 Verified Canonical Baseline (0.5976 ± 0.0023 5-seed BA)

---

## 1. Executive Summary & Purpose

The purpose of this protocol is to enforce strict empirical rigor, eliminate data leakage, guarantee mathematical reproducibility, and govern all model development, validation, ensembling, and offline submission packaging for the Kaggle "Name That LLM" competition.

Following the forensic audit of external checkpoints (notably `EXP_37_atlas` / `ntl-atlas-qwen3-8b-v1`, which was proven to be trained on 100% of the competition's official training set, yielding catastrophic in-sample leakage), **this protocol serves as the ironclad gatekeeper for all future experimental work.**

---

## 2. Seven Mandatory Tenets of Validation

### Tenet 1: Leak-Free Prompt-Disjoint Validation Mandate
* **Official Data Evaluation Only**: All cross-validation metrics reported for model selection and leaderboard comparison must be calculated exclusively on the official competition training set ($N=2,699$ passages across 150 unique prompts).
* **Prompt Disjointness**: Splits must be strictly disjoint by `prompt_id` using `GroupKFold(n_splits=5)` or `LeaveOneGroupOut`. Under no circumstances may passages originating from the same prompt appear across both training and validation folds.
* **External Data Usage**: External datasets (e.g. `crtproduction/name-that-llm-open-pool`) may **ONLY** be added to the training partition of each fold. External data must never be evaluated in the validation partition, nor may external data contain validation prompt texts or near-duplicate prompt variations.
* **Feature Leakage Prevention**: All transformations, scalers, TF-IDF vectorizers, SVD/PCA projectors, label encoders, and target statistics must be fitted solely on the training fold (plus external training pool) and applied out-of-fold (`transform()`).

### Tenet 2: Total Ban on In-Sample Checkpoints
* No external model, LoRA adapter, fine-tuned transformer, classifier, or precomputed embedding that was trained or tuned on the official `train.csv` (or any subset containing competition test/validation prompts) may be evaluated locally.
* Any third-party checkpoint from Hugging Face or Kaggle Models must undergo a cryptographic and string-level data overlap audit (`reports/atlas_data_overlap_audit.md` standard) before inclusion in any local pipeline.
* If training data overlap cannot be definitively disproven, the checkpoint is classified as **UNVERIFIED / LEAKED** and is banned from local CV reporting.

### Tenet 3: Metric Fidelity & Ranking Standard
* **Primary Metric**: **Balanced Accuracy (Macro Recall)** across all 18 candidate LLM classes is the sole metric governing experiment rankings and advancement:
  $$\text{Balanced Accuracy} = \frac{1}{18} \sum_{c=1}^{18} \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
* **Secondary Logging Metrics**: Every experiment must log:
  1. Macro F1 score
  2. Top-3 accuracy
  3. Per-class recall breakdown for all 18 classes
  4. Standard deviation across the 5 validation folds
  5. Sibling-pair confusion rate (e.g., Qwen2.5 vs Qwen3, Llama-3-8B vs Llama-3-70B, Gemma vs Gemma-2)
* Any metric claiming superiority based on raw accuracy (micro-accuracy) is rejected, as class distributions or prompt structures can introduce severe evaluation skew.

### Tenet 4: Scale Invariance & Sparsity Testing ($R=18$ vs $R=2$)
* The competition test set exhibits prompt grouping characteristics where unseen prompts may contain fewer than the full complement of 18 models (estimated test average: $R \approx 2$ passages per prompt group).
* Every candidate pipeline or post-processing head must be evaluated under two distinct structural regimes:
  1. **Full Set ($R=18$)**: The canonical 18-class Hungarian matching regime.
  2. **Sparse Pairwise ($R=2$)**: Multi-seed prompt-pruned simulation where only 2 random candidate models compete per prompt.
* A technique that achieves high gains under $R=18$ by relying exclusively on greedy exhaustive elimination (e.g., hard assignment assuming all 18 classes exist) but collapses under $R=2$ cannot be deployed without graceful Bayesian relaxation or fallback thresholding.

### Tenet 5: Mathematical Reproducibility Guarantee
* Every experiment directory (`experiments/EXP_*`) must contain:
  1. Exact configuration files (`config.json`, `feature_config.json`, `environment.json`).
  2. Pinned random seeds across NumPy, PyTorch, Scikit-Learn, and Python `random`.
  3. Preprocessing hashes and input dataset SHA-256 checksums.
  4. Explicit library dependency specifications (pip freeze / environment capture).
* **Two-Run Equivalence**: Before an experiment is marked canonical, running inference twice from a cold process on a fixed test batch must produce bit-exact identical predictions:
  $$\text{Hash}(\text{Preds}_{\text{Run 1}}) == \text{Hash}(\text{Preds}_{\text{Run 2}})$$

### Tenet 6: The Gatekeeper Rule for Ensemble Entry
* No new model family, feature block, or post-processing layer may be merged into the ensemble or canonical submission pipeline unless it satisfies **both**:
  1. An absolute mean Balanced Accuracy gain of at least $\mathbf{+0.0050}$ (+0.50% BA) over the frozen baseline (`EXP_36` = 0.5976).
  2. Statistical significance at $p < 0.05$ via a paired two-tailed $t$-test across 5 folds and/or non-overlapping 95% bootstrap confidence intervals.
* Out-Of-Fold (OOF) prediction probability vectors must be saved to disk (`outputs/oof_exp*.npy`) alongside gold labels to enable leak-free stacking, blending, and error correlation analysis.

### Tenet 7: Zero Kaggle Submission Rule
* Local offline validation strictly precedes any Kaggle submission.
* Submissions to Kaggle are prohibited unless:
  1. The candidate pipeline has passed the full suite of 15 offline edge-case tests (`scripts/test_offline_submission.py`).
  2. Runtime, memory, and disk footprint have been verified to stay well within Kaggle kernel constraints (CPU: < 9 hours, GPU: < 2 hours, RAM: < 30 GB, Disk: < 19 GB).
  3. The user explicitly directs a Kaggle submission to be prepared or executed.
* The public leaderboard is strictly an external sanity check and must **never** be used as a hill-climbing or training target.

---

## 3. Standard Operating Procedures (SOP) for New Experiments

```
[Proposal] → [Data Audit] → [5-Fold Disjoint CV] → [Statistical Test] → [Offline Sanity Check] → [Leaderboard Log]
```

1. **Naming Convention**: Experiments must follow sequential directory naming: `experiments/EXP_{ID}_{short_slug}/`.
2. **Execution Script**: Each experiment must be driven by an autonomous, reproducible script or modular class residing in `src/` or `experiments/`.
3. **Artifact Retention**:
   - `predictions_oof.npy`: Shape $(2699, 18)$, float32 class probability distributions.
   - `metrics.json`: Primary BA, fold scores, macro F1, confusion matrix.
   - `summary.md`: Brief human-readable post-mortem detailing hypothesis, methodology, results, and decision.
4. **Leaderboard Tracking**: Append verified results to `experiments/leaderboard.csv` and `experiments/experiment_log.jsonl`.
5. **No Destructive Edits**: Never overwrite, delete, or re-run historical experiment folders. New ideas require a new experiment ID.

---

## 4. Current Hierarchy of Verified Benchmarks

| Exp ID | Architecture Description | 5-Fold Grouped CV BA | Status | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **EXP_36 / EXP_38** | 4-View SVC + Open Pool (0.35) + Shared-Prompt Gaussian Head + Hungarian Assignment | **0.5976 ± 0.0023** (5-seed) | **CANONICAL BASELINE** | Certified leak-free, reproducible, production ready |
| EXP_35 | 4-View SVC + Open Pool (0.35) + Shared-Prompt Gaussian Head (Raw) | 0.4410 ± 0.0210 | ARCHIVED | Pre-Hungarian baseline |
| EXP_30 | 4-View SVC + Open Pool (0.35) + Prompt-Disjoint K-Means Embeddings | 0.4280 ± 0.0180 | ARCHIVED | Feature engineering |
| EXP_28 | 4-View SVC + Open Pool (0.35) | 0.4120 ± 0.0220 | ARCHIVED | Pure Open Pool scaling |
| EXP_08 | Multi-view Stacking Ensemble (Official data only) | 0.3630 ± 0.0240 | ARCHIVED | Pre-Open Pool baseline |
| *EXP_37* | *ATLAS adapter (`ntl-atlas-qwen3-8b-v1`)* | *[INVALID]* | **DISQUALIFIED** | *Proven 100% in-sample training data leakage* |

---

## 5. Review & Compliance

Any PR, commit, or experiment script failing to abide by this protocol is immediately flagged as non-compliant and cannot be merged into `main` or included in the final ensemble. All future iterations must benchmark strictly against `EXP_38` using the canonical OOF predictions in `outputs/oof_exp36_predictions.npy`.
