# %% [markdown]
"""
# 04 — Style-Semantic Fusion
EXPERIMENT_ID: 04_style_semantic_fusion
DATE: 2026-11-05
RESEARCH_QUESTION: Does fusing dense semantic representations with stylometric and
                   lexical/character TF-IDF features improve balanced accuracy over
                   TF-IDF + Stylometry alone (0.3082)?
HYPOTHESIS: Dense semantic embeddings alone suffer from topic-dependency under GroupKFold,
            but when fused with high-dimensional character/word n-grams and stylometric
            metrics, they can provide complementary contextual information about discourse
            coherence and sentence structure, improving balanced accuracy if properly regularized.
DATA_VERSION: v1 (original Kaggle release)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id)
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODELS: LogisticRegression (L2, tuned C), CalibratedClassifier, Probability Blend
FEATURES: Word TF-IDF + Char TF-IDF (2-6) + 31 Stylometric + 384-d MiniLM (L2 normalized)
HYPERPARAMETERS: class_weight='balanced', C in [0.5, 1.0, 2.0]
COMPUTE_CONFIGURATION: CPU multi-core (cached embeddings from GPU)
PREVIOUS_BEST_SCORE: 0.3082 (02_02d_full_fusion_lr)
"""

# %% — Imports & Setup
import sys, time, json, datetime, os
sys.path.insert(0, "/home/rj2826/name-the-llm/src")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

# Define feature extractor directly for self-contained notebook
import re
def extract_stylometric_features(texts):
    records = []
    for text in texts:
        words = text.split()
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()] or [text]
        n_words = max(len(words), 1)
        n_sents = max(len(sents), 1)
        n_chars = max(len(text), 1)
        word_lens = [len(w) for w in words]
        sent_word_lens = [len(s.split()) for s in sents]
        word_lower = [w.lower().strip(".,;:!?\"'") for w in words]
        unique_words = set(w for w in word_lower if w)
        ttr = len(unique_words) / n_words
        
        from collections import Counter
        freq = Counter(word_lower)
        M1 = sum(freq.values())
        M2 = sum(v**2 for v in freq.values())
        yules_k = 10000 * (M2 - M1) / max(M1**2, 1)
        complex_words = sum(1 for w in words if len(w) >= 3 and sum(1 for c in w if c in 'aeiouAEIOU') >= 3)
        
        feat = {
            "char_count": n_chars, "word_count": n_words, "sent_count": n_sents,
            "char_per_word": n_chars / n_words,
            "avg_word_len": np.mean(word_lens) if word_lens else 0,
            "std_word_len": np.std(word_lens) if word_lens else 0,
            "short_word_ratio": sum(1 for w in words if len(w) <= 3) / n_words,
            "long_word_ratio": sum(1 for w in words if len(w) >= 7) / n_words,
            "avg_sent_len": np.mean(sent_word_lens) if sent_word_lens else 0,
            "std_sent_len": np.std(sent_word_lens) if sent_word_lens else 0,
            "min_sent_len": np.min(sent_word_lens) if sent_word_lens else 0,
            "max_sent_len": np.max(sent_word_lens) if sent_word_lens else 0,
            "type_token_ratio": ttr, "unique_word_count": len(unique_words), "yules_k": yules_k,
            "comma_rate": 1000 * text.count(",") / n_chars,
            "semicolon_rate": 1000 * text.count(";") / n_chars,
            "colon_rate": 1000 * text.count(":") / n_chars,
            "exclamation_rate": 1000 * text.count("!") / n_chars,
            "question_rate": 1000 * text.count("?") / n_chars,
            "period_rate": 1000 * text.count(".") / n_chars,
            "em_dash_rate": 1000 * (text.count("—") + text.count("–")) / n_chars,
            "paren_rate": 1000 * (text.count("(") + text.count(")")) / n_chars,
            "quote_rate": 1000 * (text.count('"') + text.count('"')) / n_chars,
            "total_punct_rate": 1000 * sum(1 for c in text if c in ".,;:!?—–()\"'") / n_chars,
            "upper_ratio": sum(1 for c in text if c.isupper()) / max(sum(1 for c in text if c.isalpha()), 1),
            "digit_ratio": sum(1 for c in text if c.isdigit()) / n_chars,
            "newline_count": text.count("\n"),
            "gunning_fog": 0.4 * ((n_words / n_sents) + 100 * (complex_words / n_words)),
            "complex_word_ratio": complex_words / n_words,
        }
        records.append(feat)
    return pd.DataFrame(records)

TRAIN_PATH = "/home/rj2826/name-the-llm/data/train.csv"
EXPERIMENTS_DIR = "/home/rj2826/name-the-llm/experiments"
OUTPUTS_DIR = "/home/rj2826/name-the-llm/outputs"
RANDOM_SEED = 42
N_SPLITS = 5
PREV_BEST = 0.3082

train = pd.read_csv(TRAIN_PATH)
X_text = train["text"].values
y = train["model"].values
groups = train["prompt_id"].values
classes = sorted(np.unique(y))

# Load cached MiniLM embeddings
dense_emb = np.load(os.path.join(OUTPUTS_DIR, "embeddings_minilm.npy"))
print(f"Loaded {len(train)} samples. Dense embeddings shape: {dense_emb.shape}")

kf = GroupKFold(n_splits=N_SPLITS)
splits = list(kf.split(train, y, groups=groups))

def cross_validate_pipeline(pipeline_fn, label):
    fold_bal_accs, fold_f1s = [], []
    t0 = time.time()
    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X_text[tr_idx], X_text[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        emb_tr, emb_val = dense_emb[tr_idx], dense_emb[val_idx]
        
        y_pred = pipeline_fn(X_tr, y_tr, emb_tr, X_val, emb_val)
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
    delta = result["mean_balanced_accuracy"] - PREV_BEST
    sign = "+" if delta >= 0 else ""
    print(f"  → {label}: BalAcc={result['mean_balanced_accuracy']:.4f} ± {result['std_balanced_accuracy']:.4f} | F1={result['macro_f1']:.4f} | {sign}{delta:.4f} vs prev best | {elapsed:.1f}s\n")
    return result

results = {}

# %% ─────────────────────────────────────────────────
# EXP 04a: Expanded Char n-grams (2, 6) + Word TF-IDF + Stylometry
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 04a — Expanded Char n-grams (2, 6) + Word TF-IDF + Stylometry")
print("=" * 60)

def exp_04a_fn(X_tr, y_tr, emb_tr, X_val, emb_val):
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=100_000, sublinear_tf=True, min_df=2
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 6),
        max_features=250_000, sublinear_tf=True, min_df=2
    )
    X_tr_w = word_vec.fit_transform(X_tr)
    X_tr_c = char_vec.fit_transform(X_tr)
    X_val_w = word_vec.transform(X_val)
    X_val_c = char_vec.transform(X_val)
    
    feats_tr = extract_stylometric_features(X_tr).fillna(0).values.astype(float)
    feats_val = extract_stylometric_features(X_val).fillna(0).values.astype(float)
    scaler = StandardScaler()
    feats_tr_s = scaler.fit_transform(feats_tr)
    feats_val_s = scaler.transform(feats_val)
    
    X_tr_all = hstack([X_tr_w, X_tr_c, csr_matrix(feats_tr_s)])
    X_val_all = hstack([X_val_w, X_val_c, csr_matrix(feats_val_s)])
    
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             solver="saga", random_state=RANDOM_SEED)
    clf.fit(X_tr_all, y_tr)
    return clf.predict(X_val_all)

results["04a_expanded_char_word_stylo"] = cross_validate_pipeline(exp_04a_fn, "ExpandedCharWord_Stylo_LR")

# %% ─────────────────────────────────────────────────
# EXP 04b: Full Fusion: TF-IDF + Stylometry + L2-Dense Embeddings
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 04b — Full Fusion: TF-IDF + Stylometry + MiniLM (L2-normed)")
print("=" * 60)

def exp_04b_fn(X_tr, y_tr, emb_tr, X_val, emb_val):
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
    
    feats_tr = extract_stylometric_features(X_tr).fillna(0).values.astype(float)
    feats_val = extract_stylometric_features(X_val).fillna(0).values.astype(float)
    scaler_stylo = StandardScaler()
    feats_tr_s = scaler_stylo.fit_transform(feats_tr)
    feats_val_s = scaler_stylo.transform(feats_val)
    
    # Scale dense embeddings to unit norm
    emb_tr_n = normalize(emb_tr, norm="l2")
    emb_val_n = normalize(emb_val, norm="l2")
    
    X_tr_all = hstack([X_tr_w, X_tr_c, csr_matrix(feats_tr_s), csr_matrix(emb_tr_n)])
    X_val_all = hstack([X_val_w, X_val_c, csr_matrix(feats_val_s), csr_matrix(emb_val_n)])
    
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             solver="saga", random_state=RANDOM_SEED)
    clf.fit(X_tr_all, y_tr)
    return clf.predict(X_val_all)

results["04b_full_fusion_dense"] = cross_validate_pipeline(exp_04b_fn, "FullFusion_with_Dense_LR")

# %% ─────────────────────────────────────────────────
# EXP 04c: Probability-Level Blend: Style Model + Semantic Model
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 04c — Soft Probability Blending: (0.85 Style + 0.15 Semantic)")
print("=" * 60)

def exp_04c_fn(X_tr, y_tr, emb_tr, X_val, emb_val):
    # Model 1: Style model (TF-IDF + Stylometry)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=100_000, sublinear_tf=True, min_df=2)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200_000, sublinear_tf=True, min_df=2)
    X_tr_w = word_vec.fit_transform(X_tr)
    X_tr_c = char_vec.fit_transform(X_tr)
    X_val_w = word_vec.transform(X_val)
    X_val_c = char_vec.transform(X_val)
    
    feats_tr = extract_stylometric_features(X_tr).fillna(0).values.astype(float)
    feats_val = extract_stylometric_features(X_val).fillna(0).values.astype(float)
    scaler = StandardScaler()
    feats_tr_s = scaler.fit_transform(feats_tr)
    feats_val_s = scaler.transform(feats_val)
    
    X_tr_style = hstack([X_tr_w, X_tr_c, csr_matrix(feats_tr_s)])
    X_val_style = hstack([X_val_w, X_val_c, csr_matrix(feats_val_s)])
    
    clf_style = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="saga", random_state=RANDOM_SEED)
    clf_style.fit(X_tr_style, y_tr)
    probs_style = clf_style.predict_proba(X_val_style)
    
    # Model 2: Dense Semantic Model
    scaler_emb = StandardScaler()
    emb_tr_s = scaler_emb.fit_transform(emb_tr)
    emb_val_s = scaler_emb.transform(emb_val)
    clf_sem = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_SEED)
    clf_sem.fit(emb_tr_s, y_tr)
    probs_sem = clf_sem.predict_proba(emb_val_s)
    
    # Blend: 85% Style + 15% Semantic
    probs_blend = 0.85 * probs_style + 0.15 * probs_sem
    pred_idx = np.argmax(probs_blend, axis=1)
    return clf_style.classes_[pred_idx]

results["04c_prob_blend_style_semantic"] = cross_validate_pipeline(exp_04c_fn, "ProbBlend_Style_Semantic")

# %% — Summary & Record Saving
print("=" * 75)
print("RESULTS SUMMARY (04_style_semantic_fusion)")
print("=" * 75)
print(f"{'Model':<35} {'BalAcc':>8} {'±':>7}  {'MacroF1':>8}  {'Time(s)':>8}  {'vs_prev':>8}")
print("-" * 75)
for k, r in results.items():
    delta = r["mean_balanced_accuracy"] - PREV_BEST
    sign = "+" if delta >= 0 else ""
    print(f"{r['label']:<35} {r['mean_balanced_accuracy']:>8.4f} ±{r['std_balanced_accuracy']:.4f}  "
          f"{r['macro_f1']:>8.4f}  {r['training_time_sec']:>7.1f}  {sign}{delta:.4f}")

best_key = max(results, key=lambda k: results[k]["mean_balanced_accuracy"])
best = results[best_key]
print(f"\n🏆 Best in 04: {best['label']} | BalAcc={best['mean_balanced_accuracy']:.4f} ± {best['std_balanced_accuracy']:.4f}")
print(f"   vs Prev best ({PREV_BEST:.4f}): {'+' if best['mean_balanced_accuracy'] > PREV_BEST else ''}{best['mean_balanced_accuracy'] - PREV_BEST:.4f}")

# Append to experiment_log.jsonl
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
        exp_id = f"04_{key}"
        if exp_id not in existing_ids:
            record = {
                "experiment_id": exp_id,
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "research_question": "Does fusing dense semantic embeddings with stylometry and TF-IDF improve balanced accuracy?",
                "hypothesis": "Dense semantic representations offer complementary discourse structure when fused with n-gram and stylometric features",
                "research_reference": "Stamatatos (2009); Biba et al. (2024) Multi-modal style-semantic fusion",
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
                "training_time_sec": r["training_time_sec"],
                "status": "completed",
                "conclusion": f"BalAcc={r['mean_balanced_accuracy']:.4f} ± {r['std_balanced_accuracy']:.4f}",
                "next_recommendation": "If blend/expanded char wins → proceed to contrastive learning and fine-grained error analysis.",
            }
            f.write(json.dumps(record) + "\n")
            saved += 1

print(f"\n✅ Saved {saved} experiment records → {log_path}")
