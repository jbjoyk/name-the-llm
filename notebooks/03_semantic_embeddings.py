# %% [markdown]
"""
# 03 — Semantic Embeddings: Pretrained Transformer Representations
EXPERIMENT_ID: 03_semantic_embeddings
DATE: 2026-11-05
RESEARCH_QUESTION: Can dense representations from pretrained sentence encoders capture
                   LLM-specific signatures that generalize to unseen prompts under GroupKFold?
HYPOTHESIS: Transformer embeddings encode latent structural, stylistic, and syntactic
            characteristics in addition to semantic content. A regularized linear/MLP classifier
            on dense sentence embeddings will establish a strong semantic baseline and
            reveal whether sentence-level representations generalize across unseen prompts.
DATA_VERSION: v1 (original Kaggle release)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id)
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODEL: LogisticRegression, LinearSVC, RidgeClassifier, MLPClassifier
FEATURES: Pretrained SentenceTransformer ('all-MiniLM-L6-v2', 384-dim, CUDA fp16)
HYPERPARAMETERS: class_weight='balanced', C=1.0 / alpha=1.0, 5-fold prompt-grouped
COMPUTE_CONFIGURATION: GPU (RTX 3060 12GB) for inference + CPU multi-core for classifiers
PREVIOUS_BEST_SCORE: 0.3082 (FullFusion_LR, 02_stylometry)
"""

# %% — Imports & setup
import sys, time, json, datetime, os
sys.path.insert(0, "/home/rj2826/name-the-llm/src")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix

TRAIN_PATH = "/home/rj2826/name-the-llm/data/train.csv"
EXPERIMENTS_DIR = "/home/rj2826/name-the-llm/experiments"
OUTPUTS_DIR = "/home/rj2826/name-the-llm/outputs"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
PREV_BEST = 0.3082

# %% — Load data
train = pd.read_csv(TRAIN_PATH)
X_text = train["text"].tolist()
y = train["model"].values
groups = train["prompt_id"].values
classes = sorted(np.unique(y))
print(f"Train samples: {len(train)} | Classes: {len(classes)} | Previous Best: {PREV_BEST:.4f}")

# %% — Dense Embedding Extraction (GPU accelerated, cached)
EMB_CACHE_PATH = os.path.join(OUTPUTS_DIR, "embeddings_minilm.npy")

if os.path.exists(EMB_CACHE_PATH):
    print(f"Loading cached embeddings from {EMB_CACHE_PATH}...")
    embeddings = np.load(EMB_CACHE_PATH)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
else:
    print("Extracting sentence embeddings using all-MiniLM-L6-v2 on GPU...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    t0 = time.time()
    embeddings = model.encode(
        X_text,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    extract_time = time.time() - t0
    np.save(EMB_CACHE_PATH, embeddings)
    print(f"Extraction completed in {extract_time:.2f}s | Shape: {embeddings.shape} | Saved → {EMB_CACHE_PATH}")

# %% — Evaluation Helper
kf = GroupKFold(n_splits=N_SPLITS)
splits = list(kf.split(train, y, groups=groups))

def cross_validate_emb_clf(clf_factory, label):
    """Run GroupKFold on dense embeddings with fold-isolated scaling."""
    fold_bal_accs, fold_f1s = [], []
    t0 = time.time()
    all_cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = embeddings[tr_idx], embeddings[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        # Scaling fitted strictly on training fold
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        clf = clf_factory()
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_val_s)
        
        ba = balanced_accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_val, y_pred, labels=classes)
        
        fold_bal_accs.append(ba)
        fold_f1s.append(f1)
        all_cm += cm
        print(f"  [{label}] Fold {fold_i+1}/{N_SPLITS} | BalAcc={ba:.4f} | MacroF1={f1:.4f}")
        
    elapsed = time.time() - t0
    result = {
        "label": label,
        "fold_scores": fold_bal_accs,
        "mean_balanced_accuracy": float(np.mean(fold_bal_accs)),
        "std_balanced_accuracy": float(np.std(fold_bal_accs)),
        "macro_f1": float(np.mean(fold_f1s)),
        "training_time_sec": round(elapsed, 1),
        "confusion_matrix": all_cm.tolist()
    }
    delta = result["mean_balanced_accuracy"] - PREV_BEST
    sign = "+" if delta >= 0 else ""
    print(f"  → {label}: BalAcc={result['mean_balanced_accuracy']:.4f} ± {result['std_balanced_accuracy']:.4f} | F1={result['macro_f1']:.4f} | {sign}{delta:.4f} vs prev best | {elapsed:.1f}s\n")
    return result

results = {}

# %% ─────────────────────────────────────────────────
# EXP 03a: MiniLM + Logistic Regression (balanced)
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 03a — MiniLM (384-d) + Logistic Regression (balanced)")
print("=" * 60)

def make_lr():
    return LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced",
        solver="lbfgs", random_state=RANDOM_SEED
    )

results["03a_minilm_lr"] = cross_validate_emb_clf(make_lr, "MiniLM_LR")

# %% ─────────────────────────────────────────────────
# EXP 03b: MiniLM + LinearSVC (balanced)
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 03b — MiniLM (384-d) + LinearSVC (balanced)")
print("=" * 60)

def make_svc():
    return LinearSVC(
        C=0.5, class_weight="balanced", max_iter=4000,
        random_state=RANDOM_SEED
    )

results["03b_minilm_svc"] = cross_validate_emb_clf(make_svc, "MiniLM_LinearSVC")

# %% ─────────────────────────────────────────────────
# EXP 03c: MiniLM + RidgeClassifier (balanced)
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 03c — MiniLM (384-d) + RidgeClassifier (balanced)")
print("=" * 60)

def make_ridge():
    return RidgeClassifier(
        alpha=1.0, class_weight="balanced", random_state=RANDOM_SEED
    )

results["03c_minilm_ridge"] = cross_validate_emb_clf(make_ridge, "MiniLM_Ridge")

# %% ─────────────────────────────────────────────────
# EXP 03d: MiniLM + MLPClassifier (2-layer non-linear head)
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 03d — MiniLM (384-d) + MLPClassifier (hidden=(256, 128))")
print("=" * 60)

def make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=0.01,
        early_stopping=True,
        max_iter=300,
        random_state=RANDOM_SEED
    )

results["03d_minilm_mlp"] = cross_validate_emb_clf(make_mlp, "MiniLM_MLP")

# %% — Results Summary
print("=" * 75)
print("RESULTS SUMMARY (03_semantic_embeddings)")
print("=" * 75)
print(f"{'Model':<30} {'BalAcc':>8} {'±':>7}  {'MacroF1':>8}  {'Time(s)':>8}  {'vs_prev':>8}")
print("-" * 75)
for k, r in results.items():
    delta = r["mean_balanced_accuracy"] - PREV_BEST
    sign = "+" if delta >= 0 else ""
    print(f"{r['label']:<30} {r['mean_balanced_accuracy']:>8.4f} ±{r['std_balanced_accuracy']:.4f}  "
          f"{r['macro_f1']:>8.4f}  {r['training_time_sec']:>7.1f}  {sign}{delta:.4f}")

best_key = max(results, key=lambda k: results[k]["mean_balanced_accuracy"])
best = results[best_key]
print(f"\n🏆 Best in 03: {best['label']} | BalAcc={best['mean_balanced_accuracy']:.4f} ± {best['std_balanced_accuracy']:.4f}")
print(f"   vs Prev best ({PREV_BEST:.4f}): {'+' if best['mean_balanced_accuracy'] > PREV_BEST else ''}{best['mean_balanced_accuracy'] - PREV_BEST:.4f}")

# %% — Append to Experiment Registry
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
    for key, r in results.items():
        exp_id = f"03_{key}"
        if exp_id not in existing_ids:
            record = {
                "experiment_id": exp_id,
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "research_question": "Can dense pretrained sentence embeddings identify LLM authors on unseen prompts?",
                "hypothesis": "Sentence embeddings capture latent syntax and style that generalize to unseen prompts",
                "research_reference": "Reimers & Gurevych (2019); Uchendu et al. (2020) Authorship Attribution",
                "dataset_version": "v1",
                "feature_method": f"all-MiniLM-L6-v2 (384-d) + {r['label']}",
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
                "confusion_matrix": r.get("confusion_matrix"),
                "training_time_sec": r["training_time_sec"],
                "status": "completed",
                "conclusion": f"BalAcc={r['mean_balanced_accuracy']:.4f} ± {r['std_balanced_accuracy']:.4f}",
                "next_recommendation": "Next: 04_style_semantic_fusion.py to combine dense embeddings with char/word TF-IDF and stylometry.",
            }
            f.write(json.dumps(record) + "\n")
            saved += 1

print(f"\n✅ Saved {saved} experiment records → {log_path}")
