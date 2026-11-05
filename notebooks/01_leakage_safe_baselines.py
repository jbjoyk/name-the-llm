# %% [markdown]
"""
# 01 — Leakage-Safe Baselines
EXPERIMENT_ID: 01_baselines
DATE: 2026-11-05
RESEARCH_QUESTION: What do TF-IDF + linear classifiers achieve under strict leakage-safe GroupKFold?
HYPOTHESIS: Word TF-IDF + LR with class_weight=balanced will beat majority baseline
            significantly. Character n-gram TF-IDF adds complementary signal.
DATA_VERSION: v1 (original Kaggle release)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id)
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODELS: MajorityClass, LogisticRegression(word-tfidf), LogisticRegression(char-tfidf),
        LogisticRegression(word+char), LinearSVC, ComplementNaiveBayes
FEATURES: TF-IDF variants
HYPERPARAMETERS: See per-model config blocks
COMPUTE_CONFIGURATION: CPU (n_jobs=-1)
PREVIOUS_BEST_SCORE: N/A (first real experiment)
"""

# %% — Imports & paths
import sys, time, json, datetime
sys.path.insert(0, "/home/rj2826/name-the-llm/src")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.sparse import hstack

TRAIN_PATH = "/home/rj2826/name-the-llm/data/train.csv"
EXPERIMENTS_DIR = "/home/rj2826/name-the-llm/experiments"
RANDOM_SEED = 42
N_SPLITS = 5

# %% — Load data
train = pd.read_csv(TRAIN_PATH)
X_text = train["text"].values
y = train["model"].values
groups = train["prompt_id"].values
classes = sorted(np.unique(y))
print(f"Train: {len(train)} rows | {len(classes)} classes | {len(np.unique(groups))} prompt groups")

kf = GroupKFold(n_splits=N_SPLITS)
splits = list(kf.split(train, y, groups=groups))

# %% — Helper: cross-validate a pipeline function
def cross_validate_fn(pipeline_fn, splits, X_text, y, groups, label):
    """Run cross-validation; return result dict."""
    fold_bal_accs, fold_f1s = [], []
    t0 = time.time()
    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X_text[tr_idx], X_text[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        y_pred = pipeline_fn(X_tr, y_tr, X_val)
        ba = balanced_accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        fold_bal_accs.append(ba)
        fold_f1s.append(f1)
        print(f"  [{label}] Fold {fold_i+1}/{N_SPLITS} | BalAcc={ba:.4f} | MacroF1={f1:.4f}")
    elapsed = time.time() - t0
    result = {
        "label": label,
        "fold_scores": fold_bal_accs,
        "mean_balanced_accuracy": float(np.mean(fold_bal_accs)),
        "std_balanced_accuracy": float(np.std(fold_bal_accs)),
        "macro_f1": float(np.mean(fold_f1s)),
        "training_time_sec": round(elapsed, 1),
    }
    print(f"  → {label}: BalAcc = {result['mean_balanced_accuracy']:.4f} ± {result['std_balanced_accuracy']:.4f} | F1={result['macro_f1']:.4f} | {elapsed:.1f}s")
    return result

results = {}

# %% ─────────────────────────────────────────────────
# BASELINE A: Majority classifier
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE A — Majority Class")
print("="*60)

def majority_fn(X_tr, y_tr, X_val):
    majority = Counter(y_tr).most_common(1)[0][0]
    return np.array([majority] * len(X_val))

from collections import Counter
results["A_majority"] = cross_validate_fn(majority_fn, splits, X_text, y, groups, "MajorityClass")

# %% ─────────────────────────────────────────────────
# BASELINE B: Word TF-IDF + Logistic Regression
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE B — Word TF-IDF + Logistic Regression")
print("="*60)

def word_tfidf_lr_fn(X_tr, y_tr, X_val):
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=100_000, sublinear_tf=True,
        min_df=2, strip_accents="unicode"
    )
    X_tr_v = vec.fit_transform(X_tr)
    X_val_v = vec.transform(X_val)
    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="saga", n_jobs=-1, random_state=RANDOM_SEED
    )
    clf.fit(X_tr_v, y_tr)
    return clf.predict(X_val_v)

results["B_word_tfidf_lr"] = cross_validate_fn(word_tfidf_lr_fn, splits, X_text, y, groups, "WordTFIDF_LR")

# %% ─────────────────────────────────────────────────
# BASELINE C: Character TF-IDF + Logistic Regression
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE C — Character TF-IDF + Logistic Regression")
print("="*60)

def char_tfidf_lr_fn(X_tr, y_tr, X_val):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=200_000, sublinear_tf=True,
        min_df=2
    )
    X_tr_v = vec.fit_transform(X_tr)
    X_val_v = vec.transform(X_val)
    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="saga", n_jobs=-1, random_state=RANDOM_SEED
    )
    clf.fit(X_tr_v, y_tr)
    return clf.predict(X_val_v)

results["C_char_tfidf_lr"] = cross_validate_fn(char_tfidf_lr_fn, splits, X_text, y, groups, "CharTFIDF_LR")

# %% ─────────────────────────────────────────────────
# BASELINE D: Word + Character TF-IDF fusion
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE D — Word + Character TF-IDF + LR")
print("="*60)

def combined_tfidf_lr_fn(X_tr, y_tr, X_val):
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=100_000, sublinear_tf=True, min_df=2
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=200_000, sublinear_tf=True, min_df=2
    )
    X_tr_w = word_vec.fit_transform(X_tr)
    X_tr_c = char_vec.fit_transform(X_tr)
    X_val_w = word_vec.transform(X_val)
    X_val_c = char_vec.transform(X_val)
    X_tr_all = hstack([X_tr_w, X_tr_c])
    X_val_all = hstack([X_val_w, X_val_c])
    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="saga", n_jobs=-1, random_state=RANDOM_SEED
    )
    clf.fit(X_tr_all, y_tr)
    return clf.predict(X_val_all)

results["D_combined_tfidf_lr"] = cross_validate_fn(combined_tfidf_lr_fn, splits, X_text, y, groups, "Combined_TFIDF_LR")

# %% ─────────────────────────────────────────────────
# BASELINE E: LinearSVC
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE E — LinearSVC (char+word TF-IDF)")
print("="*60)

def linear_svc_fn(X_tr, y_tr, X_val):
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=100_000, sublinear_tf=True, min_df=2
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=200_000, sublinear_tf=True, min_df=2
    )
    X_tr_all = hstack([word_vec.fit_transform(X_tr), char_vec.fit_transform(X_tr)])
    X_val_all = hstack([word_vec.transform(X_val), char_vec.transform(X_val)])
    clf = LinearSVC(C=0.5, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(X_tr_all, y_tr)
    return clf.predict(X_val_all)

results["E_linear_svc"] = cross_validate_fn(linear_svc_fn, splits, X_text, y, groups, "LinearSVC")

# %% ─────────────────────────────────────────────────
# BASELINE F: Complement Naive Bayes
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE F — Complement Naive Bayes")
print("="*60)

def cnb_fn(X_tr, y_tr, X_val):
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=100_000, sublinear_tf=True, min_df=2
    )
    X_tr_v = vec.fit_transform(X_tr)
    X_val_v = vec.transform(X_val)
    # CNB needs non-negative; use TF (not sublinear TF) for counts
    vec2 = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=100_000, use_idf=False, min_df=2
    )
    X_tr_v2 = vec2.fit_transform(X_tr)
    X_val_v2 = vec2.transform(X_val)
    clf = ComplementNB(alpha=0.1)
    clf.fit(X_tr_v2, y_tr)
    return clf.predict(X_val_v2)

results["F_cnb"] = cross_validate_fn(cnb_fn, splits, X_text, y, groups, "ComplementNB")

# %% — Summary table
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Model':<30} {'BalAcc':>8} {'±':>7}  {'MacroF1':>8}  {'Time(s)':>8}")
print("-" * 60)
for k, r in results.items():
    print(f"{r['label']:<30} {r['mean_balanced_accuracy']:>8.4f} ±{r['std_balanced_accuracy']:.4f}  {r['macro_f1']:>8.4f}  {r['training_time_sec']:>7.1f}")

best_key = max(results, key=lambda k: results[k]["mean_balanced_accuracy"])
best = results[best_key]
print(f"\n🏆 Best so far: {best['label']} | BalAcc={best['mean_balanced_accuracy']:.4f} ± {best['std_balanced_accuracy']:.4f}")
majority_ba = results["A_majority"]["mean_balanced_accuracy"]
gain = best["mean_balanced_accuracy"] - majority_ba
print(f"   Gain over majority: +{gain:.4f}")

# %% — Save experiment records
import os
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

experiment_records = []
for key, r in results.items():
    record = {
        "experiment_id": f"01_{key}",
        "date": datetime.datetime.utcnow().isoformat(),
        "research_question": "What do TF-IDF + linear classifiers achieve under strict leakage-safe GroupKFold?",
        "hypothesis": "TF-IDF with class_weight=balanced beats majority baseline",
        "research_reference": "Classical stylometry; Argamon & Juola (2011)",
        "dataset_version": "v1",
        "feature_method": r["label"],
        "model": r["label"],
        "hyperparameters": {"n_splits": N_SPLITS, "random_seed": RANDOM_SEED},
        "cv_strategy": "GroupKFold(n_splits=5, group=prompt_id)",
        "group_column": "prompt_id",
        "random_seed": RANDOM_SEED,
        "fold_scores": r["fold_scores"],
        "mean_balanced_accuracy": r["mean_balanced_accuracy"],
        "std_balanced_accuracy": r["std_balanced_accuracy"],
        "macro_f1": r["macro_f1"],
        "per_class_recall": None,
        "confusion_matrix": None,
        "training_time_sec": r["training_time_sec"],
        "inference_time_sec": None,
        "gpu_memory_mb": None,
        "feature_dimensionality": "varies",
        "num_parameters": None,
        "status": "completed",
        "conclusion": f"BalAcc={r['mean_balanced_accuracy']:.4f} ± {r['std_balanced_accuracy']:.4f}",
        "next_recommendation": "If best, run stylometry features next (02_stylometry.ipynb)",
    }
    experiment_records.append(record)

log_path = os.path.join(EXPERIMENTS_DIR, "experiment_log.jsonl")
existing_ids = set()
if os.path.exists(log_path):
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                existing_ids.add(json.loads(line).get("experiment_id", ""))

saved = 0
with open(log_path, "a") as f:
    for rec in experiment_records:
        if rec["experiment_id"] not in existing_ids:
            f.write(json.dumps(rec) + "\n")
            saved += 1

print(f"\n✅ Saved {saved} experiment records → {log_path}")

# %% — Next steps analysis
print("""
NEXT STEPS:
  1. The best baseline sets the bar for future experiments.
  2. → 02_stylometry.ipynb: Add hand-crafted stylometric features
       (sentence length, punctuation ratios, TTR, etc.) 
  3. → If stylometry helps: fuse with TF-IDF
  4. → 03_semantic_embeddings.ipynb: Test sentence-transformers (BGE, MiniLM, etc.)
  5. Research question for next: Do stylistic fingerprints outperform lexical TF-IDF?
""")
