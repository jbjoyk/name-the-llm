# %% [markdown]
"""
# 02 — Stylometry: Hand-crafted Stylometric Features
EXPERIMENT_ID: 02_stylometry
DATE: 2026-11-05
RESEARCH_QUESTION: Do hand-crafted stylometric features (sentence stats, punctuation,
                   lexical diversity, readability) meaningfully improve balanced accuracy
                   over TF-IDF when applied to LLM authorship attribution?
HYPOTHESIS: LLMs exhibit systematic stylistic fingerprints (verbosity, sentence length,
            punctuation style, lexical diversity) that survive prompt variation.
            Adding stylometric features to TF-IDF should improve balanced accuracy
            on unseen prompts (GroupKFold validation).
DATA_VERSION: v1 (original Kaggle release)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id)
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODEL: LogisticRegression, LightGBM
FEATURES: Stylometric features alone; Stylometry + TF-IDF fusion
HYPERPARAMETERS: See per-model blocks
COMPUTE_CONFIGURATION: CPU (n_jobs=-1)
PREVIOUS_BEST_SCORE: 0.3015 (Combined_TFIDF_LR, 01_baselines)
"""

# %% — Imports
import sys, time, json, datetime, re, os
sys.path.insert(0, "/home/rj2826/name-the-llm/src")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

TRAIN_PATH = "/home/rj2826/name-the-llm/data/train.csv"
EXPERIMENTS_DIR = "/home/rj2826/name-the-llm/experiments"
RANDOM_SEED = 42
N_SPLITS = 5
PREV_BEST = 0.3015

# %% — Load data
train = pd.read_csv(TRAIN_PATH)
X_text = train["text"].values
y = train["model"].values
groups = train["prompt_id"].values
classes = sorted(np.unique(y))

kf = GroupKFold(n_splits=N_SPLITS)
splits = list(kf.split(train, y, groups=groups))
print(f"Train: {len(train)} rows | {len(classes)} classes | previous best: {PREV_BEST}")

# %% — Stylometric feature extractor
def extract_stylometric_features(texts):
    """
    Extract 40+ stylometric features per text.
    All features are computed without any fitted state — safe to apply per-fold.
    """
    records = []
    for text in texts:
        words = text.split()
        # Tokenize sentences (simple regex)
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sents:
            sents = [text]
        n_words = max(len(words), 1)
        n_sents = max(len(sents), 1)
        n_chars = max(len(text), 1)
        
        word_lens = [len(w) for w in words]
        sent_word_lens = [len(s.split()) for s in sents]
        
        # Lexical diversity
        word_lower = [w.lower().strip(".,;:!?\"'") for w in words]
        unique_words = set(w for w in word_lower if w)
        ttr = len(unique_words) / n_words  # Type-token ratio
        
        # Punctuation features
        commas = text.count(",")
        semicolons = text.count(";")
        colons = text.count(":")
        exclamations = text.count("!")
        questions = text.count("?")
        periods = text.count(".")
        em_dashes = text.count("—") + text.count("–")
        parens = text.count("(") + text.count(")")
        quotes = text.count('"') + text.count('"') + text.count('"')
        
        # Case features
        upper_chars = sum(1 for c in text if c.isupper())
        alpha_chars = sum(1 for c in text if c.isalpha())
        
        # Digit/special features
        digits = sum(1 for c in text if c.isdigit())
        
        # Short word ratio (<=3 chars) — function word proxy
        short_words = sum(1 for w in words if len(w) <= 3)
        
        # Long word ratio (>=7 chars) — vocabulary sophistication
        long_words = sum(1 for w in words if len(w) >= 7)
        
        # Whitespace / newlines
        newlines = text.count("\n")
        double_newlines = text.count("\n\n")
        
        # Yule's K (vocabulary richness measure)
        from collections import Counter
        freq = Counter(word_lower)
        M1 = sum(freq.values())
        M2 = sum(v**2 for v in freq.values())
        yules_k = 10000 * (M2 - M1) / max(M1**2, 1)
        
        # Avg characters per word
        avg_word_len = np.mean(word_lens) if word_lens else 0
        std_word_len = np.std(word_lens) if word_lens else 0
        
        # Avg sentence length
        avg_sent_len = np.mean(sent_word_lens) if sent_word_lens else 0
        std_sent_len = np.std(sent_word_lens) if sent_word_lens else 0
        min_sent_len = np.min(sent_word_lens) if sent_word_lens else 0
        max_sent_len = np.max(sent_word_lens) if sent_word_lens else 0
        
        # Gunning Fog readability (approximation)
        complex_words = sum(1 for w in words if len(w) >= 3 and sum(1 for c in w if c in 'aeiouAEIOU') >= 3)
        gunning_fog = 0.4 * ((n_words / n_sents) + 100 * (complex_words / n_words))
        
        feat = {
            # Length
            "char_count": n_chars,
            "word_count": n_words,
            "sent_count": n_sents,
            "char_per_word": n_chars / n_words,
            
            # Word-level
            "avg_word_len": avg_word_len,
            "std_word_len": std_word_len,
            "short_word_ratio": short_words / n_words,
            "long_word_ratio": long_words / n_words,
            
            # Sentence-level
            "avg_sent_len": avg_sent_len,
            "std_sent_len": std_sent_len,
            "min_sent_len": min_sent_len,
            "max_sent_len": max_sent_len,
            
            # Lexical diversity
            "type_token_ratio": ttr,
            "unique_word_count": len(unique_words),
            "yules_k": yules_k,
            
            # Punctuation ratios (per 1000 chars)
            "comma_rate": 1000 * commas / n_chars,
            "semicolon_rate": 1000 * semicolons / n_chars,
            "colon_rate": 1000 * colons / n_chars,
            "exclamation_rate": 1000 * exclamations / n_chars,
            "question_rate": 1000 * questions / n_chars,
            "period_rate": 1000 * periods / n_chars,
            "em_dash_rate": 1000 * em_dashes / n_chars,
            "paren_rate": 1000 * parens / n_chars,
            "quote_rate": 1000 * quotes / n_chars,
            "total_punct_rate": 1000 * (commas + semicolons + colons + exclamations + questions + periods) / n_chars,
            
            # Case
            "upper_ratio": upper_chars / max(alpha_chars, 1),
            "digit_ratio": digits / n_chars,
            
            # Whitespace
            "newline_count": newlines,
            "double_newline_count": double_newlines,
            
            # Readability
            "gunning_fog": gunning_fog,
            "complex_word_ratio": complex_words / n_words,
        }
        records.append(feat)
    
    return pd.DataFrame(records)

# Verify feature extractor
sample_feats = extract_stylometric_features(X_text[:3])
print(f"Feature count: {len(sample_feats.columns)}")
print(sample_feats.head(2).T.to_string())

# %% — Helper: cross-validate
def cross_validate_fn(pipeline_fn, splits, X_text, y, label):
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
        print(f"  [{label}] Fold {fold_i+1}/{N_SPLITS} | BalAcc={ba:.4f} | F1={f1:.4f}")
    elapsed = time.time() - t0
    result = {
        "label": label,
        "fold_scores": fold_bal_accs,
        "mean_balanced_accuracy": float(np.mean(fold_bal_accs)),
        "std_balanced_accuracy": float(np.std(fold_bal_accs)),
        "macro_f1": float(np.mean(fold_f1s)),
        "training_time_sec": round(elapsed, 1),
    }
    delta = result['mean_balanced_accuracy'] - PREV_BEST
    sign = "+" if delta >= 0 else ""
    print(f"  → {label}: BalAcc={result['mean_balanced_accuracy']:.4f} ± {result['std_balanced_accuracy']:.4f} | {sign}{delta:.4f} vs prev best")
    return result

results = {}

# %% ─────────────────────────────────────────────────
# EXP 02a: Stylometry-only + LR
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP 02a — Stylometry only + Logistic Regression")
print("="*60)

def stylo_lr_fn(X_tr, y_tr, X_val):
    feats_tr = extract_stylometric_features(X_tr).fillna(0).values.astype(float)
    feats_val = extract_stylometric_features(X_val).fillna(0).values.astype(float)
    scaler = StandardScaler()
    feats_tr = scaler.fit_transform(feats_tr)
    feats_val = scaler.transform(feats_val)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                             solver="lbfgs", random_state=RANDOM_SEED)
    clf.fit(feats_tr, y_tr)
    return clf.predict(feats_val)

results["02a_stylo_lr"] = cross_validate_fn(stylo_lr_fn, splits, X_text, y, "Stylo_LR")

# %% ─────────────────────────────────────────────────
# EXP 02b: Stylometry only + LightGBM
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP 02b — Stylometry only + LightGBM")
print("="*60)

def stylo_lgb_fn(X_tr, y_tr, X_val):
    feats_tr = extract_stylometric_features(X_tr).fillna(0).values.astype(float)
    feats_val = extract_stylometric_features(X_val).fillna(0).values.astype(float)
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=RANDOM_SEED,
        verbose=-1, n_jobs=-1
    )
    clf.fit(feats_tr, y_tr_enc)
    preds_enc = clf.predict(feats_val)
    return le.inverse_transform(preds_enc)

results["02b_stylo_lgb"] = cross_validate_fn(stylo_lgb_fn, splits, X_text, y, "Stylo_LGB")

# %% ─────────────────────────────────────────────────
# EXP 02c: Char TF-IDF + Stylometry fused → LR
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP 02c — Char TF-IDF + Stylometry → LR (Fusion)")
print("="*60)

def char_stylo_lr_fn(X_tr, y_tr, X_val):
    # TF-IDF component
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=200_000, sublinear_tf=True, min_df=2
    )
    X_tr_c = char_vec.fit_transform(X_tr)
    X_val_c = char_vec.transform(X_val)
    
    # Stylometric component (dense, scaled)
    feats_tr = extract_stylometric_features(X_tr).fillna(0).values.astype(float)
    feats_val = extract_stylometric_features(X_val).fillna(0).values.astype(float)
    scaler = StandardScaler()
    feats_tr_s = scaler.fit_transform(feats_tr)
    feats_val_s = scaler.transform(feats_val)
    
    # Fuse (sparse + dense via hstack)
    X_tr_all = hstack([X_tr_c, csr_matrix(feats_tr_s)])
    X_val_all = hstack([X_val_c, csr_matrix(feats_val_s)])
    
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             solver="saga", random_state=RANDOM_SEED)
    clf.fit(X_tr_all, y_tr)
    return clf.predict(X_val_all)

results["02c_char_stylo_lr"] = cross_validate_fn(char_stylo_lr_fn, splits, X_text, y, "CharTFIDF+Stylo_LR")

# %% ─────────────────────────────────────────────────
# EXP 02d: Word+Char TF-IDF + Stylometry → LR (Full fusion)
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXP 02d — Word+Char TF-IDF + Stylometry → LR (Full Fusion)")
print("="*60)

def full_fusion_lr_fn(X_tr, y_tr, X_val):
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
    scaler = StandardScaler()
    feats_tr_s = scaler.fit_transform(feats_tr)
    feats_val_s = scaler.transform(feats_val)
    
    X_tr_all = hstack([X_tr_w, X_tr_c, csr_matrix(feats_tr_s)])
    X_val_all = hstack([X_val_w, X_val_c, csr_matrix(feats_val_s)])
    
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             solver="saga", random_state=RANDOM_SEED)
    clf.fit(X_tr_all, y_tr)
    return clf.predict(X_val_all)

results["02d_full_fusion_lr"] = cross_validate_fn(full_fusion_lr_fn, splits, X_text, y, "FullFusion_LR")

# %% — Summary
print("\n" + "="*60)
print("RESULTS SUMMARY (02_stylometry)")
print("="*60)
print(f"{'Model':<35} {'BalAcc':>8} {'±':>7}  {'MacroF1':>8}  {'Time(s)':>8}  {'vs_prev':>8}")
print("-" * 75)
for k, r in results.items():
    delta = r["mean_balanced_accuracy"] - PREV_BEST
    sign = "+" if delta >= 0 else ""
    print(f"{r['label']:<35} {r['mean_balanced_accuracy']:>8.4f} ±{r['std_balanced_accuracy']:.4f}  "
          f"{r['macro_f1']:>8.4f}  {r['training_time_sec']:>7.1f}  {sign}{delta:.4f}")

best_key = max(results, key=lambda k: results[k]["mean_balanced_accuracy"])
best = results[best_key]
print(f"\n🏆 Best in 02: {best['label']} | BalAcc={best['mean_balanced_accuracy']:.4f}")
print(f"   vs Prev best ({PREV_BEST:.4f}): {'+' if best['mean_balanced_accuracy'] > PREV_BEST else ''}{best['mean_balanced_accuracy'] - PREV_BEST:.4f}")

# %% — Save experiment records
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
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
        exp_id = f"02_{key}"
        if exp_id not in existing_ids:
            record = {
                "experiment_id": exp_id,
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "research_question": "Do stylometric features improve LLM attribution over TF-IDF alone?",
                "hypothesis": "LLMs have systematic stylistic fingerprints that survive prompt variation",
                "research_reference": "Argamon & Juola (2011); Koppel et al. (2011)",
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
                "next_recommendation": "If fusion helps → 03_semantic_embeddings.ipynb",
            }
            f.write(json.dumps(record) + "\n")
            saved += 1

print(f"\n✅ Saved {saved} experiment records → {log_path}")

# %% — Research-driven next step decision
print("""
RESEARCH ANALYSIS:
- Stylometry-only: Tells us whether structural fingerprints alone identify LLMs
- Fusion with TF-IDF: Tells us whether stylometry is complementary to lexical features
- LightGBM on stylometry: Tells us if non-linear interactions matter

DECISION RULE:
  If 02c or 02d > 0.31 → Stylometry is complementary → keep in pipeline
  If Stylo_LGB >> Stylo_LR → non-linear interactions matter → use tree models
  If no improvement → Stylometry is absorbed by TF-IDF → skip in final pipeline

NEXT UNCONDITIONALLY:
  03_semantic_embeddings.ipynb
  - Test sentence-transformers (all-MiniLM-L6-v2 as cheap pilot)
  - Research: Can pretrained embedding spaces capture LLM-specific style?
  - Hypothesis: Embeddings trained on diverse text may capture semantic+stylistic
    fingerprints that generalize to unseen prompts better than TF-IDF
""")
