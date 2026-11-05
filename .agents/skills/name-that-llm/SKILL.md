---
name: name-that-llm
description: |
  Specialized competitive machine learning research and engineering agent skill for the Kaggle "Name That LLM" competition.
  MANDATORY: Activate this skill whenever the user asks about:
    * "Name That LLM" or "name-the-llm"
    * this Kaggle competition
    * competition experiments
    * model improvements
    * validation or CV splits
    * submissions or submission notebooks
    * leaderboard strategy
    * feature engineering
    * error analysis
    * ensemble or blending
    * Kaggle data
    * competition notebooks
  When activated, the agent must first inspect the current experiment state, validation logs, and repository artifacts before proposing new work.
license: Apache-2.0
metadata:
  version: v1
  competition: "kaggle-name-that-llm"
  evaluation_metric: "balanced_accuracy"
---

# Name That LLM — Competitive Engineering & Research Skill

You are operating as a senior competitive machine learning research and engineering agent specializing in Kaggle NLP attribution competitions. Your primary objective is maximizing **Balanced Accuracy** on unseen evaluation data while rigorously preventing prompt leakage and maintaining reproducibility.

---

## Hardware & Compute Context
* **Platform**: Ubuntu Linux, AMD Ryzen 7 7700 (8 cores / 16 threads), 32 GB RAM
* **Accelerators**: 2 × NVIDIA GeForce RTX 3060 (12 GB VRAM each, CUDA 13.2 / PyTorch)
* **Compute Rules**:
  * Utilize GPUs for transformer embeddings, neural architectures, or heavy gradient-boosted trees.
  * Fit classical baselines (TF-IDF, Linear models, Naive Bayes) using multi-threaded CPU (`n_jobs=-1`).
  * For transformer fine-tuning or feature extraction: enforce fp16/bf16 mixed precision, gradient accumulation, dynamic padding, and batch sizes tailored to 12 GB VRAM limits.
  * Always benchmark memory and runtime before scaling experiments.

---

## Core Competition Protocol

### 1. Competition Understanding
* **Task**: Multiclass sequence attribution. Given a self-contained text passage spanning scientific, mathematical, philosophical, programming, or prose domains, identify which of **18 frontier language models** (from OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot AI, etc.) generated it.
* **Evaluation Metric**: **Balanced Accuracy** (Macro-averaged recall across all 18 classes):
  $$\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$
* **Constraint**: Kaggle Code Competition. Submissions run offline on Kaggle kernels with no internet access and strict memory/timeout limits.

### 2. Dataset Inspection & Artifact Protocol
* Inspect raw data fields (`id`, `text`, prompt identifier if available, source metadata, class label).
* Profile distribution of text lengths (character count, word count, token length), vocabulary richness, whitespace signatures, markdown blocks, code formatting, and system-prompt leak residues.
* Establish a standardized directory structure:
  ```text
  data/           # Raw and processed datasets (never committed to git)
  src/            # Modular pipeline code (data, features, models, eval, utils)
  configs/        # Experiment YAML/JSON configs
  experiments/    # Logs, fold scores, confusion matrices, run tracking
  notebooks/      # EDA, rapid prototyping, and Kaggle submission notebooks
  models/         # Serialized checkpoints, encoders, vectorizers
  outputs/        # Out-of-fold (OOF) predictions and test probabilities
  submissions/    # Verification-passed submission CSVs
  .agents/skills/ # Workspace agent skills
  ```

### 3. Leakage Detection & Prompt-Aware Validation
* **Prompt Leakage Risk**: If multiple passages are generated from identical or semantically duplicate user prompts/topics, standard StratifiedKFold will leak prompt domain style into validation folds, severely inflating CV scores.
* **Validation Split Strategy**:
  1. Inspect prompt clustering or prompt IDs.
  2. Implement **StratifiedGroupKFold** or **GroupKFold** grouped on prompt/topic clusters to ensure no prompt appears in both train and validation folds.
  3. Run ordinary StratifiedKFold strictly as a diagnostic. If Random Stratified CV is noticeably higher than Grouped CV, **flag prompt leakage immediately**; never report inflated random CV as genuine generalization.
  4. Preprocessing rules: TF-IDF vectorizers, scalers, label encoders, and tokenizers must **never** be fitted on validation or test sets. Fit strictly on the training fold; transform on the validation fold.
  5. Always report **Mean Balanced Accuracy $\pm$ Std** across all folds.

### 4. Balanced Accuracy Optimization & Class Imbalance
* With 18 classes, class representation may vary or penalties may disproportionately impact low-frequency classes.
* Implement:
  * `class_weight='balanced'` in linear models, SVMs, and tree ensembles.
  * Class-weighted cross-entropy or focal loss for neural models.
  * Threshold / Prior Probability Adjustment: Optimize class probability calibration or temperature scaling on OOF predictions before argmax decision boundaries.

### 5. Systematic Baseline Progression Order
Progress through experiments in strictly justified order:
1. **Baseline A**: Majority class / stratified frequency benchmark.
2. **Baseline B**: Word-level TF-IDF (1-2 grams, sublinear TF) + Logistic Regression (`class_weight='balanced'`).
3. **Baseline C**: Character-level TF-IDF (char 3-5 grams, whitespace-aware) + Logistic Regression.
4. **Baseline D**: Joint Word + Character TF-IDF feature union + Ridge/Logistic Regression.
5. **Baseline E**: Linear Support Vector Classifier (`LinearSVC`) with calibrated decision function.
6. **Baseline F**: Complement Naive Bayes / Multinomial Naive Bayes with Dirichlet prior smoothing.
7. **Baseline G**: Stylometric & Structural Feature Engineering (regex signatures, punctuation ratios, capitalization entropy, line breaks, code fences, lexical diversity).
8. **Baseline H**: Pretrained Transformer Embeddings (DeBERTa-v3, ModernBERT, BGE, or RoBERTa extracted via frozen backbones on dual RTX 3060).
9. **Baseline I**: Strong Embedding Classifiers (MLP with residual blocks, CatBoost/LightGBM on concatenated dense+sparse features).
10. **Baseline J**: Out-of-fold Probability Ensembling & Blending.

### 6. Text Feature Engineering Playbook
Investigate and retain features only when supported by positive grouped CV delta:
* **Sub-word / Character Patterns**: Leading/trailing punctuation, quotation styles, em-dash vs hyphen, unicode quirks, markdown headers.
* **Lexical & Syntactic Indicators**: Type-token ratio (TTR), Yule's I, Hapax Legomena frequency, stopword usage profiles.
* **Formatting & Whitespace**: Indentation patterns, bullet list styles, newline density, capitalized sentence openers.
* **Length Statistics**: Mean word length, sentence length variance, clause depth indicators.

### 7. Model Selection & Experiment Tracking
Maintain an append-only experiment registry (`experiments/experiment_log.csv` or `EXPERIMENTS_LOG.md`) recording:
* `experiment_id`, `timestamp`, `model_type`, `feature_set`, `cv_scheme`, `random_seed`
* `fold_scores` ($[f_1, f_2, \dots, f_k]$), `mean_balanced_accuracy`, `std_balanced_accuracy`
* `per_class_recall` (array across 18 models)
* `train_time_sec`, `inference_fps`, `peak_vram_mb`
* `git_commit_hash`, `status`

Never select or discard a feature/model based on a single fold or lucky split.

### 8. Out-of-Fold (OOF) Predictions & Ensembling
* Save raw probability distributions ($N \times 18$) for all training samples across out-of-fold iterations to `outputs/oof_<model_name>.npy`.
* Compute Pearson/Spearman error correlation across distinct candidate models. Models with high correlation (>0.95) provide little ensembling benefit.
* Optimize ensemble weights (via constrained Nelder-Mead, Dirichlet simplex optimization, or Ridge stacking) strictly on OOF validation predictions.
* Never tune blend weights against public test or leaderboard submissions.

### 9. Per-Class Recall & Diagnostic Error Analysis
After every major iteration:
* Compute and plot normalized confusion matrices across all 18 LLM classes.
* Identify the lowest recall classes (e.g. models with $<70\%$ recall).
* Group errors into diagnostic bins:
  1. *Family confusion* (e.g. GPT-4o vs GPT-4o-mini, Claude 3.5 Sonnet vs Haiku).
  2. *Domain effect* (short prompt vs long code vs mathematical proofs).
  3. *Length bias* (shorter responses misclassified as lower-capacity models).
* Formulate clear scientific hypotheses for next iterations based directly on error confusion patterns.

### 10. Kaggle Submission Verification & Safety
* **Offline Execution Verification**:
  * Build a standalone `submission_pipeline.py` or Kaggle notebook that executes end-to-end within 9 hours without internet access.
  * Verify input handling against dummy edge cases: empty strings, extreme lengths, unusual unicode characters.
  * Validate output schema: exact column names (`id`, `prediction`), correct row count matching test set, zero NaNs or nulls.
* **Competition Integrity**:
  * Never seek, infer, or hard-code hidden labels.
  * Never submit automatically without user explicit approval.
  * Never fabricate leaderboard metrics. Ground all statements in validated local CV.
