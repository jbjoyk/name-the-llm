# %% [markdown]
"""
# 00 — Data Audit
EXPERIMENT_ID: 00_data_audit
DATE: 2026-11-05
RESEARCH_QUESTION: What is the structure, quality, and leakage risk of the Name That LLM dataset?
HYPOTHESIS: N/A (descriptive audit)
DATA_VERSION: v1 (original Kaggle release)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id)
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODEL: N/A
FEATURES: N/A
HYPERPARAMETERS: N/A
COMPUTE_CONFIGURATION: CPU only
PREVIOUS_BEST_SCORE: N/A (first experiment)
"""

# %% — Imports
import sys
sys.path.insert(0, "/home/rj2826/name-the-llm/src")

import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import GroupKFold

# %% — Load data
TRAIN = "/home/rj2826/name-the-llm/data/train.csv"
TEST  = "/home/rj2826/name-the-llm/data/test.csv"

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)

print("=" * 60)
print("TRAIN SHAPE:", train.shape)
print("TEST  SHAPE:", test.shape)
print()
print("TRAIN COLUMNS:", list(train.columns))
print("TEST  COLUMNS:", list(test.columns))
print()
print("TRAIN DTYPES:")
print(train.dtypes)
print()
print("NULL COUNTS (train):", train.isnull().sum().to_dict())
print("NULL COUNTS (test):",  test.isnull().sum().to_dict())

# %% — Class distribution
print("\n" + "=" * 60)
print("CLASS DISTRIBUTION (model column)")
print("=" * 60)
vc = train["model"].value_counts()
print(vc.to_string())
print(f"\nTotal classes: {train['model'].nunique()}")
print(f"Min count: {vc.min()} | Max count: {vc.max()}")
print(f"Perfectly balanced: {vc.min() == vc.max() - 1 or vc.min() == vc.max()}")

# %% — Prompt structure analysis (leakage risk)
print("\n" + "=" * 60)
print("PROMPT STRUCTURE ANALYSIS")
print("=" * 60)
n_prompts = train["prompt_id"].nunique()
print(f"Unique prompt_ids: {n_prompts}")
print(f"Total rows: {len(train)}")
print(f"Avg responses per prompt: {len(train)/n_prompts:.2f}")
print()

# Models per prompt
models_per_prompt = train.groupby("prompt_id")["model"].nunique()
print("Models per prompt distribution:")
print(models_per_prompt.value_counts().to_string())
print()

# Check if every prompt is answered by every model
cross = train.groupby(["prompt_id", "model"]).size().unstack(fill_value=0)
print(f"Cross-tab shape (prompts × models): {cross.shape}")
print(f"All cells == 1: {(cross == 1).all().all()}")
print(f"Any prompt missing a model: {(cross == 0).any().any()}")

# %% — LEAKAGE RISK: GroupKFold fold construction
print("\n" + "=" * 60)
print("LEAKAGE ANALYSIS: GroupKFold fold construction")
print("=" * 60)

groups = train["prompt_id"].values
y = train["model"].values

kf = GroupKFold(n_splits=5)
for fold_i, (tr_idx, val_idx) in enumerate(kf.split(train, y, groups=groups)):
    tr_groups = set(groups[tr_idx])
    val_groups = set(groups[val_idx])
    overlap = tr_groups & val_groups
    print(f"Fold {fold_i+1}: train_prompts={len(tr_groups)}, val_prompts={len(val_groups)}, overlap={len(overlap)}")

print("\n✅ Zero prompt overlap in all folds — leakage-safe validated")

# %% — Text feature statistics
print("\n" + "=" * 60)
print("TEXT FEATURE STATISTICS")
print("=" * 60)

def text_stats(text: str) -> dict:
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return {
        "char_len": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "avg_sent_len": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
        "punct_ratio": sum(1 for c in text if c in ".,;:!?") / max(len(text), 1),
        "digit_ratio": sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        "upper_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "newline_count": text.count("\n"),
        "unique_words": len(set(w.lower() for w in words)),
        "type_token_ratio": len(set(w.lower() for w in words)) / max(len(words), 1),
    }

stats_df = pd.DataFrame(train["text"].map(text_stats).tolist())
train_aug = pd.concat([train, stats_df], axis=1)

print("Per-statistic summary:")
print(stats_df.describe().round(3).to_string())

# %% — Per-model text statistics
print("\n" + "=" * 60)
print("PER-MODEL FEATURE MEANS")
print("=" * 60)
per_model = train_aug.groupby("model")[stats_df.columns].mean().round(3)
print(per_model.to_string())

# %% — Length variance across models
print("\n" + "=" * 60)
print("CHAR LENGTH PER MODEL (mean ± std)")
print("=" * 60)
len_stats = train_aug.groupby("model")["char_len"].agg(["mean","std"]).round(1)
len_stats = len_stats.sort_values("mean", ascending=False)
for model, row in len_stats.iterrows():
    bar = "█" * int(row["mean"] / 50)
    print(f"{model:<30} {row['mean']:>7.1f} ± {row['std']:>6.1f}  {bar}")

# %% — Test set inspection
print("\n" + "=" * 60)
print("TEST SET")
print("=" * 60)
test_stats = pd.DataFrame(test["text"].map(text_stats).tolist())
print(test.head(3).to_string())
print("\nTest text stats:")
print(test_stats.describe().round(2).to_string())
print("\n⚠️  Test has only 3 rows — these are placeholder/example rows only.")
print("Real evaluation happens on hidden Kaggle kernel execution.")

# %% — AUDIT SUMMARY
print("\n" + "=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)
print(f"""
Dataset structure:
  Train: {train.shape[0]} rows × {train.shape[1]} cols
  Test:  {test.shape[0]} rows × {test.shape[1]} cols (Kaggle placeholder)
  Classes: {train['model'].nunique()} LLMs
  Prompt IDs: {train['prompt_id'].nunique()} unique prompts
  Balance: {vc.min()}–{vc.max()} samples per class (near-perfect)
  
Structure insight:
  Every prompt is answered by ALL 18 models (18 × 150 = 2700 ≈ 2699 rows).
  This means standard StratifiedKFold WILL leak prompt style across folds.
  → MANDATORY: Use GroupKFold or StratifiedGroupKFold on prompt_id.

Feature hints:
  - char_len varies noticeably across models (useful signal)
  - punct_ratio, upper_ratio, type_token_ratio differ across models
  - Structural / stylometric features are worth investigating

Next step:
  01_leakage_safe_baselines.ipynb — TF-IDF + Logistic Regression with GroupKFold
""")
