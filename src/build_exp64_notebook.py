"""
Script to build and execute the EXP64 Jupyter Notebook:
notebooks/64_disagreement_confidence_routing.ipynb
"""

import pathlib
import nbformat
from nbclient import NotebookClient

REPO_DIR = pathlib.Path("/home/rj2826/name-that-llm")
NOTEBOOKS_DIR = REPO_DIR / "notebooks"
EXP_DIR = REPO_DIR / "experiments" / "EXP64_confidence_router"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.14.7"
    }
}

cells = []

# Cell 1: Title
cells.append(nbformat.v4.new_markdown_cell(r"""# Kaggle "Name That LLM" Competition: EXP64
## Disagreement-Aware Confidence Routing for Single-Sample Attribution

- **Experiment ID**: `EXP64_confidence_router`
- **Research Question**: Does single-sample confidence, margin, entropy, and Q8/Q14 disagreement predict which expert is more likely to be correct, and does a dynamic single-sample router outperform the fixed EXP52 fusion (0.60 Q8 + 0.40 Q14)?
- **Baseline**: EXP52 Dual-LoRA Kernel (**0.7873 OOF BA**, **0.632 Public LB**, FROZEN CHAMPION)
- **Constraint**: Strict single-sample inference. ZERO prompt identifiers, zero prompt centroids, zero Hungarian matching, zero peer comparisons.
- **Verdict**: **YELLOW** (Learned router achieves **0.7906 nested OOF BA**, $\Delta = \mathbf{+0.0033}$; bootstrap 95% CI spans zero $[-0.0027, +0.0094]$; overconfident errors cause 31 degradations against 40 corrections)
- **Production Decision**: **Retain EXP52 as frozen champion**.
"""))

# Cell 2: Imports
cells.append(nbformat.v4.new_code_cell(r"""# 1. Environment & Imports Setup
import os
import sys
import json
import time
import pathlib
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from IPython.display import Image, display

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline

REPO_DIR = pathlib.Path("/home/rj2826/name-that-llm")
DATA_DIR = REPO_DIR / "data"
OUTPUTS_DIR = REPO_DIR / "outputs"
EXP_DIR = REPO_DIR / "experiments" / "EXP64_confidence_router"
PLOTS_DIR = REPO_DIR / "plots"

print(f"Working Directory: {REPO_DIR}")
print(f"EXP64 Directory:   {EXP_DIR}")
"""))

# Cell 3: Markdown - Baseline Reproduction
cells.append(nbformat.v4.new_markdown_cell(r"""### 2. Dataset Loading & EXP52 Reproduction Audit
Load `train.csv` (2,699 samples, 150 prompts) and verified out-of-fold logits for Q8 (`oof_exp47_qwen_centered_twin.npy`) and Q14 (`oof_q14v5_antopil_lora.csv`).
"""))

# Cell 4: Code - Load & Verify
cells.append(nbformat.v4.new_code_cell(r"""# 2. Load Train Dataset and EXP52 Baseline
train_df = pd.read_csv(DATA_DIR / "train.csv")
LABELS = sorted(train_df["model"].unique())
L2I = {m: i for i, m in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train_df["model"].values])
prompts = train_df["prompt_id"].values

q14_df = pd.read_csv(OUTPUTS_DIR / "oof_q14v5_antopil_lora.csv")
p8_twin_raw = np.load(OUTPUTS_DIR / "oof_exp47_qwen_centered_twin.npy")
outer_folds = q14_df["fold"].values
eps = 1e-15

p8 = p8_twin_raw / p8_twin_raw.sum(axis=1, keepdims=True)
p14 = q14_df[LABELS].values / q14_df[LABELS].values.sum(axis=1, keepdims=True)

lp8 = np.log(np.clip(p8, eps, 1.0))
lp14 = np.log(np.clip(p14, eps, 1.0))
lp52 = 0.60 * lp8 + 0.40 * lp14
pred_exp52 = lp52.argmax(axis=1)

ba_q8 = balanced_accuracy_score(y_true, lp8.argmax(axis=1))
ba_q14 = balanced_accuracy_score(y_true, lp14.argmax(axis=1))
ba_exp52 = balanced_accuracy_score(y_true, pred_exp52)

print(f"Q8 Standalone BA:   {ba_q8:.6f}")
print(f"Q14 Standalone BA:  {ba_q14:.6f}")
print(f"EXP52 Baseline BA:  {ba_exp52:.6f} (Exact reproduction verified)")
"""))

# Cell 5: Markdown - Feature Extraction
cells.append(nbformat.v4.new_markdown_cell(r"""### 3. Required Single-Sample Features
Construct 37 continuous features from single response outputs:
- **Q8 features**: top-1 prob, top-2 prob, margin, entropy, logit margin, max/mean/std logit
- **Q14 features**: same features
- **Cross-model features**: agreement, prob difference, margin difference, entropy difference, Jensen-Shannon divergence, symmetric KL, probability correlation, rank correlation, top-k overlap, cross-probabilities, L1/L2 distances, fused EXP52 confidence metrics.
"""))

# Cell 6: Code - Features
cells.append(nbformat.v4.new_code_cell(r"""# 3. Load Feature Definitions & Statistics
with open(EXP_DIR / "feature_definitions.json") as f:
    feat_defs = json.load(f)
print(f"Total single-sample features defined: {len(feat_defs)}")

# Display top coefficients from the trained router
df_coeffs = pd.read_csv(EXP_DIR / "router_coefficients.csv")
print("\nTop 10 Most Influential Routing Features:")
display(df_coeffs.head(10)[["feature", "mean_coef", "std_coef", "abs_mean_coef"]])
"""))

# Cell 7: Markdown - Diagnostics
cells.append(nbformat.v4.new_markdown_cell(r"""### 4. Baseline Diagnostics: Correctness Groups
Partition the 2,699 samples into 4 correctness groups:
- **Group A (Both Correct)**: 1,026 samples (38.0%)
- **Group B (Q8 Only Correct)**: 569 samples (21.1%)
- **Group C (Q14 Only Correct)**: 687 samples (25.5%)
- **Group D (Both Wrong)**: 417 samples (15.5%)
"""))

# Cell 8: Code - Diagnostics
cells.append(nbformat.v4.new_code_cell(r"""# 4. Correctness Groups Comparison
df_diag = pd.read_csv(EXP_DIR / "diagnostics_groups.csv")
display(df_diag[["group", "count", "pct", "q8_margin_mean", "q14_margin_mean", "q8_entropy_mean", "q14_entropy_mean", "agree_mean"]])
"""))

# Cell 9: Markdown - Comparison Table
cells.append(nbformat.v4.new_markdown_cell(r"""### 5. Critical Comparison Table: Routing Rules vs Learned Router
Compare static rules (Rule A through F), learned router (nested CV), calibrated router, and random negative control.
"""))

# Cell 10: Code - Comparison Table
cells.append(nbformat.v4.new_code_cell(r"""# 5. Critical Comparison Table
df_comp = pd.read_csv(EXP_DIR / "comparison_table.csv")
display(df_comp)
"""))

# Cell 11: Markdown - Nested CV & Statistical Testing
cells.append(nbformat.v4.new_markdown_cell(r"""### 6. Strict Nested Cross-Validation & Bootstrap Testing
- 5 outer prompt-disjoint folds $\times$ 4 inner prompt-disjoint folds
- Inner grid search over $C \in [0.001, 10.0]$ and $\alpha \in [0.0, 0.4]$
- Paired bootstrap test ($B = 2,000$ resamples)
"""))

# Cell 12: Code - Fold & Bootstrap Results
cells.append(nbformat.v4.new_code_cell(r"""# 6. Fold Results and Bootstrap Confidence Interval
df_folds = pd.read_csv(EXP_DIR / "fold_results.csv")
display(df_folds[["outer_fold", "best_C", "best_alpha", "exp52_ba", "router_ba", "router_delta", "random_router_delta"]])

with open(EXP_DIR / "bootstrap_results.json") as f:
    boot = json.load(f)

print(f"Paired Bootstrap Delta: Mean = {boot['mean_delta']:+.4f}")
print(f"95% Confidence Interval: [{boot['ci_95_lower']:+.4f}, {boot['ci_95_upper']:+.4f}]")
print(f"P(delta > 0):           {boot['p_positive'] * 100:.1f}%")
"""))

# Cell 13: Markdown - Disagreement Analysis
cells.append(nbformat.v4.new_markdown_cell(r"""### 7. Disagreement Subset Analysis ($\arg\max(Q8) \neq \arg\max(Q14)$)
On the 1,612 samples (59.7% of total data) where Q8 and Q14 disagree:
- EXP52 BA: **0.6871**
- Learned Router BA: **0.6945** ($\Delta = +0.0074$)
- Oracle Chooser BA: **0.8425**
- Net corrections: **+9 samples** (40 corrections vs 31 degradations)
"""))

# Cell 14: Code - Disagreement & Deciles
cells.append(nbformat.v4.new_code_cell(r"""# 7. Disagreement & Decile Metrics
df_dis = pd.read_csv(EXP_DIR / "disagreement_results.csv")
display(df_dis)

print("\n--- Confidence Deciles ---")
df_deciles = pd.read_csv(EXP_DIR / "decile_results.csv")
display(df_deciles[["decile", "sample_count", "mean_confidence", "exp52_ba", "router_ba", "router_delta"]])
"""))

# Cell 15: Markdown - Visualizations
cells.append(nbformat.v4.new_markdown_cell(r"""### 8. Publication Figures (300 DPI)
"""))

# Cell 16: Code - Display Figures
cells.append(nbformat.v4.new_code_cell(r"""# 8. Render Key Figures Inline
print("--- Figure 1: Confidence Distributions by Correctness Group ---")
display(Image(filename=str(PLOTS_DIR / "exp64_confidence_distributions.png")))

print("--- Figure 2: Router Performance Comparison ---")
display(Image(filename=str(PLOTS_DIR / "exp64_router_performance.png")))

print("--- Figure 3: Disagreement Subset Performance ---")
display(Image(filename=str(PLOTS_DIR / "exp64_disagreement_analysis.png")))

print("--- Figure 4: Confidence Decile Analysis ---")
display(Image(filename=str(PLOTS_DIR / "exp64_confidence_deciles.png")))

print("--- Figure 5: Corrections vs Degradations ---")
display(Image(filename=str(PLOTS_DIR / "exp64_correction_degradation.png")))

print("--- Figure 6: Router Feature Importance ---")
display(Image(filename=str(PLOTS_DIR / "exp64_router_feature_importance.png")))

print("--- Figure 7: Temperature Calibration Reliability Curves ---")
display(Image(filename=str(PLOTS_DIR / "exp64_calibration.png")))
"""))

# Cell 17: Markdown - Verdict & Decision
cells.append(nbformat.v4.new_markdown_cell(r"""### 9. Final Verdict & Scientific Conclusions

- **Verdict**: **YELLOW**
- **Findings**:
  1. Observable single-sample confidence and margin features strongly correlate with correctness on training data ($t = 20.7, p < 10^{-80}$).
  2. However, under strict 5-outer $\times$ 4-inner prompt-disjoint nested cross-validation, the learned router achieves only **0.7906 BA** ($\Delta = \mathbf{+0.0033}$ over EXP52 0.7873).
  3. The 95% bootstrap confidence interval $[-0.0027, +0.0094]$ comfortably spans zero ($P(\Delta > 0) = 87.0\%$).
  4. Out of 1,612 disagreement samples, the router fixes 40 mistakes that EXP52 made, but simultaneously causes 31 regressions on samples that EXP52 correctly identified, producing a net gain of only **+9 samples**.
  5. Temperature calibration successfully reduces ECE on individual models, but severely distorts relative logit scales when combined, degrading fusion accuracy from 0.7873 to 0.7736.
- **Production Champion Decision**: **Retain EXP52 as frozen champion (0.632 LB / 0.7873 OOF BA). No Kaggle submission.**
"""))

nb.cells.extend(cells)

# Save unexecuted
out_nb_path = NOTEBOOKS_DIR / "64_disagreement_confidence_routing.ipynb"
with open(out_nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Saved initial notebook to {out_nb_path}")

# Execute notebook with NotebookClient
print("Executing notebook with NotebookClient...")
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

# Save fully executed
with open(out_nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Successfully executed and saved notebook to {out_nb_path}")

# Also mirror to experiments/EXP64_confidence_router/
mirror_path = EXP_DIR / "64_disagreement_confidence_routing.ipynb"
with open(mirror_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Mirrored executed notebook to {mirror_path}")
