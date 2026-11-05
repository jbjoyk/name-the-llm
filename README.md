# Name That LLM — Competitive Machine Learning Pipeline

An end-to-end, reproducible, leakage-free research and engineering system for the Kaggle **[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)** competition.

---

## 1. Competition Task & Primary Objective
* **Goal:** Correctly identify which of **18 frontier Large Language Models** (spanning OpenAI, Anthropic, Google, Alibaba Qwen, Zhipu AI, and Moonshot AI) generated a given text passage.
* **Evaluation Metric:** **Balanced Accuracy Score** (unweighted macro-average of per-class recall across all 18 classes):
  $$\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^K \text{Recall}_k = \frac{1}{K} \sum_{k=1}^K \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$
* **Critical Dataset Constraint:** The training set contains multiple LLM responses to the same 150 prompts ($2,699$ total samples), while the test set comprises unseen prompts. Random cross-validation leaks prompt topical content and inflates scores. **Strict leakage-safe prompt grouping via `GroupKFold(n_splits=5, group=prompt_id)` is enforced across all experiments.**

---

## 2. Research Progression & Cross-Validation Leaderboard

All experiments below were evaluated on identical 5-fold prompt-isolated splits with zero data leakage.

| Notebook / Experiment ID | Model Architecture & Linguistic Representations | Mean BalAcc | Fold Std | Macro F1 | Delta vs Prior |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `01_A_majority` | Majority Class Baseline | `0.0556` | `±0.0000` | `0.0058` | — |
| `01_B_word_tfidf_lr` | Word TF-IDF ($1\text{--}2$ n-grams) + Logistic Regression | `0.2671` | `±0.0177` | `0.2520` | `+0.2115` |
| `01_C_char_tfidf_lr` | Character TF-IDF ($3\text{--}5$ char-wb n-grams) + Logistic Regression | `0.2908` | `±0.0087` | `0.2798` | `+0.0237` |
| `01_D_combined_tfidf_lr` | Combined Word + Char TF-IDF + Logistic Regression | `0.3015` | `±0.0105` | `0.2889` | `+0.0107` |
| `02_02d_full_fusion_lr` | Word+Char TF-IDF + 31 Handcrafted Stylometric Features | `0.3082` | `±0.0088` | `0.2882` | `+0.0067` |
| `03_03a_minilm_lr` | Pretrained Dense Embeddings (`all-MiniLM-L6-v2`, Layer 6 Semantic) | `0.1363` | `±0.0102` | `0.1305` | `-0.1719` *(Semantic Collapse)* |
| `04_04b_full_fusion_dense` | Style (Char+Word+Stylo) + Scaled Dense Embeddings Fusion | `0.3178` | `±0.0106` | `0.3000` | `+0.0096` |
| `05_05a_supcon` | Supervised Contrastive Learning (SupCon) Projection Head | `0.1311` | `±0.0113` | `0.1264` | `—` |
| `06_06b_layer3_syntax` | **Intermediate Transformer Layer 3 (Morpho-Syntax)** | `0.2786` | `±0.0112` | `0.2761` | `+0.1423` *(vs Layer 6)* |
| `07_07c_l3_likelihood_tfidf_lr` | Layer 3 Syntax + MultiScale Likelihood Profiles + TF-IDF | `0.3282` | `±0.0187` | `0.3094` | `+0.0104` |
| `08_08a_equal_blend` | 4-Model Equal Weight Probability Average | `0.3286` | `±0.0120` | `0.3211` | `+0.0004` |
| `08_08b_opt_simplex_blend` | Optimal Fold-Isolated Simplex Weighted Blending | `0.3545` | `±0.0134` | `0.3386` | `+0.0259` |
| **`08_08c_meta_stacker_lr`** | **Multi-Paradigm Meta-Stacking (Style + Syntax + Likelihood + Fused)** | **`0.3630`** | **`±0.0046`** | **`0.3508`** | **`+0.0348`** |
| `10_final_pipeline` | Production Meta-Stacker Refit on 100% Data ($2,699$ Samples) | `0.3630` | `±0.0046` | `0.3522` | Verified |

---

## 3. Key Research Discoveries & Theoretical Insights

1. **The Semantic Collapse Phenomenon:**
   Standard off-the-shelf sentence transformer embeddings (e.g. `all-MiniLM-L6-v2` Layer 6) collapse to an abysmal $0.1363$ balanced accuracy. Because all 18 models respond to the identical prompt, semantic encoders map passages to the shared prompt topic rather than authorial style.
2. **Intermediate Transformer Layer Advantage (Layer 3 over Layer 6):**
   Probing representations across the transformer hierarchy revealed that **Layer 3 (Syntax & Morphology)** achieves **$0.2786$ BalAcc**—more than double the performance of Layer 6 ($0.1349$). Deeper transformer layers abstract away formatting and surface structure, whereas intermediate layers preserve authorial syntactic habits.
3. **MultiScale Likelihood Profiling:**
   Class-conditional character ($n=3$) and word ($n=1$) n-gram language models generate a compact $36$-dimensional log-likelihood feature vector that yields $0.2756$ BalAcc on its own and adds orthogonal signal when fused with neural and TF-IDF representations.
4. **Meta-Stacker Variance Reduction:**
   Stacking out-of-fold probability vectors from the 4 distinct model families via balanced logistic regression improved performance to **$0.3630$**, while cutting cross-fold variance down to **$\pm 0.0046$**.
5. **Length Dynamics & Intra-Family Confusion:**
   Attribution accuracy rises monotonically with passage length ($27.9\%$ on Q1 short text vs $43.6\%$ on Q5 long text). Top misclassifications occur strictly within developer families (`gemini-3.8-flash` vs `gemini-3.7-flash`, `gpt-5.6-sol` vs `gpt-5.6-luna`, and `glm-5.3-flash` vs `glm-5.3`).

---

## 4. Repository Structure

```text
├── .agents/skills/name-that-llm/  # Antigravity Competition Skill (YAML + instructions)
├── configs/                       # Experiment and model hyperparameters
├── data/                          # Competition data (train.csv, test.csv, sample_submission.parquet)
├── experiments/                   # Automated experiment tracker (experiment_log.jsonl & leaderboard.csv)
├── models/                        # Serialized production models & scalers (12 joblib artifacts)
├── notebooks/                     # Self-contained experiments in both .ipynb and .py format:
│   ├── 00_data_audit.ipynb
│   ├── 01_leakage_safe_baselines.ipynb
│   ├── 02_stylometry.ipynb
│   ├── 03_semantic_embeddings.ipynb
│   ├── 04_style_semantic_fusion.ipynb
│   ├── 05_contrastive_learning.ipynb
│   ├── 06_transformer_layers.ipynb
│   ├── 07_model_scoring.ipynb
│   ├── 08_ensemble.ipynb
│   ├── 09_error_analysis.ipynb
│   └── 10_final_pipeline.ipynb
├── outputs/                       # Cached embeddings (MiniLM layers 1, 3, 6) & OOF probability matrices
├── reports/                       # Visualizations (confusion_matrix.png, per_class_recall.png) & error report
├── scripts/                       # Production prediction CLI (scripts/predict.py)
├── src/                           # Reusable core modules (experiment_tracker.py, validation.py)
└── submissions/                   # Verified submission files (submission.parquet, submission.csv)
```

---

## 5. Quickstart: Running Inference & Verification

### Running Standalone Offline Inference
The standalone inference engine operates completely offline using the pre-fitted production artifacts in `models/`:

```bash
# Generate verified Kaggle submission in parquet or csv format
python scripts/predict.py \
  --test-path data/test.csv \
  --models-dir models \
  --output-path submissions/submission.parquet
```

### Reproducing the Experiments
Each notebook in `notebooks/` can be executed independently. To re-run the entire pipeline:

```bash
python notebooks/00_data_audit.py
python notebooks/01_leakage_safe_baselines.py
python notebooks/02_stylometry.py
python notebooks/03_semantic_embeddings.py
python notebooks/04_style_semantic_fusion.py
python notebooks/05_contrastive_learning.py
python notebooks/06_transformer_layers.py
python notebooks/07_model_scoring.py
python notebooks/08_ensemble.py
python notebooks/09_error_analysis.py
python notebooks/10_final_pipeline.py
```

To display the live project leaderboard:
```bash
python src/experiment_tracker.py
```

---

## 6. Submission Verification Checklist

Every candidate submission generated by this repository is validated against the official 6-point verification suite:
- [x] **Check 1: File Existence & Size:** File is present on disk and non-empty.
- [x] **Check 2: Exact Row Count Match:** Test sample row count ($N$) matches $N$ predictions exactly.
- [x] **Check 3: Header Conformance:** Column schema is strictly `['id', 'model']`.
- [x] **Check 4: Null & NaN Free:** Zero NaN, null, or missing string values.
- [x] **Check 5: Identifier Alignment:** Identifiers match `test['id']` 1-to-1 in identical row order.
- [x] **Check 6: Valid Class Set Inclusion:** All predicted values belong to the 18 valid frontier LLM classes.
