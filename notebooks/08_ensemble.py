# %% [markdown]
"""
# 08 — Ensemble Construction: OOF Diversity, Blending & Stacking
EXPERIMENT_ID: 08_ensemble
DATE: 2026-11-05
RESEARCH_QUESTION: How much does combining distinct model paradigms (n-gram stylometry,
                   intermediate transformer syntax, and class-conditional likelihood profiles)
                   improve balanced accuracy via OOF probability blending and stacking?
HYPOTHESIS: Model A (Expanded TF-IDF + Stylometry), Model B (Layer 3 Transformer Syntax),
            and Model C (MultiScale Likelihood Profiles) make partially uncorrelated errors
            because they capture orthogonal linguistic dimensions (character tokens vs deep syntax
            vs n-gram transitions). Optimizing ensemble weights on out-of-fold predictions
            will yield complementary error reduction and exceed individual model performance.
DATA_VERSION: v1 (original Kaggle release)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id)
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODELS: Equal Averaging, Optimal Simplex Blending, Meta-Stacking (Ridge & Logistic Regression)
PREVIOUS_BEST_SCORE: 0.3282 (07_07c_l3_likelihood_tfidf_lr)
"""

# %% — Imports & Setup
import sys, time, json, datetime, os, math
sys.path.insert(0, "/home/rj2826/name-the-llm/src")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import defaultdict, Counter
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

TRAIN_PATH = "/home/rj2826/name-the-llm/data/train.csv"
EXPERIMENTS_DIR = "/home/rj2826/name-the-llm/experiments"
OUTPUTS_DIR = "/home/rj2826/name-the-llm/outputs"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
PREV_BEST = 0.3282

train = pd.read_csv(TRAIN_PATH)
X_text = train["text"].values
y = train["model"].values
groups = train["prompt_id"].values
classes = sorted(np.unique(y))
n_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"Loaded {len(train)} samples across {n_classes} classes. Previous Best: {PREV_BEST:.4f}")

# Load Layer 3 features from Exp 06
feat_l3 = np.load(os.path.join(OUTPUTS_DIR, "minilm_layer3_mean.npy"))

# Define Stylometry Extractor
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

class NgramLikelihoodProfiler:
    def __init__(self, n=3, alpha=0.05, char_level=True):
        self.n = n
        self.alpha = alpha
        self.char_level = char_level
        self.vocab = set()
        self.class_counts = defaultdict(Counter)
        self.class_totals = defaultdict(int)
        
    def _get_ngrams(self, text):
        if self.char_level:
            return [text[i:i + self.n] for i in range(len(text) - self.n + 1)]
        else:
            words = text.split()
            return [" ".join(words[i:i + self.n]) for i in range(len(words) - self.n + 1)]
        
    def fit(self, texts, labels):
        self.vocab = set()
        self.class_counts = defaultdict(Counter)
        self.class_totals = defaultdict(int)
        for text, label in zip(texts, labels):
            ngrams = self._get_ngrams(text)
            self.vocab.update(ngrams)
            for ng in ngrams:
                self.class_counts[label][ng] += 1
                self.class_totals[label] += 1
        self.vocab_size = len(self.vocab)
        return self

    def transform(self, texts):
        profiles = np.zeros((len(texts), n_classes), dtype=np.float32)
        log_alpha_denom = {c: math.log(self.class_totals[c] + self.alpha * self.vocab_size) for c in classes}
        for i, text in enumerate(texts):
            ngrams = self._get_ngrams(text)
            n_tokens = max(len(ngrams), 1)
            for c_idx, c in enumerate(classes):
                counts = self.class_counts[c]
                denom = log_alpha_denom[c]
                log_prob = 0.0
                for ng in ngrams:
                    cnt = counts.get(ng, 0)
                    log_prob += math.log(cnt + self.alpha) - denom
                profiles[i, c_idx] = log_prob / n_tokens
        return profiles

# %% — Generate Out-of-Fold (OOF) Probabilities for Candidate Models
kf = GroupKFold(n_splits=N_SPLITS)
splits = list(kf.split(train, y, groups=groups))

oof_style = np.zeros((len(train), n_classes))      # Model 1: Expanded TF-IDF + Stylometry
oof_syntax = np.zeros((len(train), n_classes))     # Model 2: Layer 3 Transformer Syntax
oof_likelihood = np.zeros((len(train), n_classes)) # Model 3: MultiScale Likelihood
oof_fusion = np.zeros((len(train), n_classes))     # Model 4: L3 + Likelihood + TFIDF Fused

print("Generating Out-of-Fold (OOF) prediction probabilities for all 4 distinct candidate models...")
t0 = time.time()

for fold_i, (tr_idx, val_idx) in enumerate(splits):
    X_tr, X_val = X_text[tr_idx], X_text[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    l3_tr, l3_val = feat_l3[tr_idx], feat_l3[val_idx]
    
    # ── Model 1: Expanded TF-IDF + Stylometry ──
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=150_000, sublinear_tf=True, min_df=2)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=75_000, sublinear_tf=True, min_df=2)
    X_tr_tfidf = hstack([char_vec.fit_transform(X_tr), word_vec.fit_transform(X_tr)])
    X_val_tfidf = hstack([char_vec.transform(X_val), word_vec.transform(X_val)])
    
    s_tr = StandardScaler().fit_transform(extract_stylometric_features(X_tr).fillna(0).values)
    s_val = StandardScaler().fit(extract_stylometric_features(X_tr).fillna(0).values).transform(extract_stylometric_features(X_val).fillna(0).values)
    
    m1 = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="saga", random_state=RANDOM_SEED)
    m1.fit(hstack([X_tr_tfidf, csr_matrix(s_tr)]), y_tr)
    oof_style[val_idx] = m1.predict_proba(hstack([X_val_tfidf, csr_matrix(s_val)]))
    
    # ── Model 2: Layer 3 Transformer Syntax ──
    scaler_l3 = StandardScaler()
    l3_tr_s = scaler_l3.fit_transform(l3_tr)
    l3_val_s = scaler_l3.transform(l3_val)
    m2 = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    m2.fit(l3_tr_s, y_tr)
    oof_syntax[val_idx] = m2.predict_proba(l3_val_s)
    
    # ── Model 3: MultiScale Likelihood ──
    p_c3 = NgramLikelihoodProfiler(n=3, alpha=0.05, char_level=True).fit(X_tr, y_tr)
    p_w1 = NgramLikelihoodProfiler(n=1, alpha=0.05, char_level=False).fit(X_tr, y_tr)
    prof_tr = np.hstack([p_c3.transform(X_tr), p_w1.transform(X_tr)])
    prof_val = np.hstack([p_c3.transform(X_val), p_w1.transform(X_val)])
    
    scaler_p = StandardScaler()
    m3 = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    m3.fit(scaler_p.fit_transform(prof_tr), y_tr)
    oof_likelihood[val_idx] = m3.predict_proba(scaler_p.transform(prof_val))
    
    # ── Model 4: L3 + Likelihood + TFIDF Fused ──
    m4 = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="saga", random_state=RANDOM_SEED)
    m4.fit(hstack([X_tr_tfidf, csr_matrix(scaler_p.fit_transform(prof_tr)), csr_matrix(l3_tr_s)]), y_tr)
    oof_fusion[val_idx] = m4.predict_proba(hstack([X_val_tfidf, csr_matrix(scaler_p.transform(prof_val)), csr_matrix(l3_val_s)]))
    
    print(f"  Fold {fold_i+1}/{N_SPLITS} complete.")

print(f"OOF generation completed in {time.time() - t0:.1f}s.")
np.save(os.path.join(OUTPUTS_DIR, "oof_style.npy"), oof_style)
np.save(os.path.join(OUTPUTS_DIR, "oof_syntax.npy"), oof_syntax)
np.save(os.path.join(OUTPUTS_DIR, "oof_likelihood.npy"), oof_likelihood)
np.save(os.path.join(OUTPUTS_DIR, "oof_fusion.npy"), oof_fusion)

# Helper function to evaluate OOF predictions by fold
def score_oof_by_folds(oof_probs, label):
    fold_scores, fold_f1s = [], []
    for tr_idx, val_idx in splits:
        preds = [classes[idx] for idx in np.argmax(oof_probs[val_idx], axis=1)]
        ba = balanced_accuracy_score(y[val_idx], preds)
        f1 = f1_score(y[val_idx], preds, average="macro", zero_division=0)
        fold_scores.append(ba)
        fold_f1s.append(f1)
    result = {
        "label": label,
        "fold_scores": fold_scores,
        "mean_balanced_accuracy": float(np.mean(fold_scores)),
        "std_balanced_accuracy": float(np.std(fold_scores)),
        "macro_f1": float(np.mean(fold_f1s)),
    }
    print(f"  → {label}: BalAcc={result['mean_balanced_accuracy']:.4f} ± {result['std_balanced_accuracy']:.4f} | F1={result['macro_f1']:.4f}")
    return result

results = {}
print("\n" + "=" * 60)
print("INDIVIDUAL OOF MODEL BASELINES")
print("=" * 60)
score_oof_by_folds(oof_style, "OOF_Model1_Style")
score_oof_by_folds(oof_syntax, "OOF_Model2_Syntax")
score_oof_by_folds(oof_likelihood, "OOF_Model3_Likelihood")
score_oof_by_folds(oof_fusion, "OOF_Model4_Fused")

# %% ─────────────────────────────────────────────────
# EXP 08a: Equal Weight Probability Blending (Style + Syntax + Likelihood + Fusion)
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXP 08a — Equal Weight Average (M1 + M2 + M3 + M4)")
print("=" * 60)
oof_equal = (oof_style + oof_syntax + oof_likelihood + oof_fusion) / 4.0
results["08a_equal_blend"] = score_oof_by_folds(oof_equal, "EqualWeight_Blend_4Models")

# %% ─────────────────────────────────────────────────
# EXP 08b: Optimal Simplex Weighted Blend
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXP 08b — Optimal Fold-Isolated Simplex Weighted Blend")
print("=" * 60)

# Optimize weights fold-by-fold strictly on training indices
oof_opt_blend = np.zeros_like(oof_fusion)
for fold_i, (tr_idx, val_idx) in enumerate(splits):
    # Optimize weights on training folds
    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        comb_tr = (w[0] * oof_style[tr_idx] + 
                   w[1] * oof_syntax[tr_idx] + 
                   w[2] * oof_likelihood[tr_idx] + 
                   w[3] * oof_fusion[tr_idx])
        preds = [classes[idx] for idx in np.argmax(comb_tr, axis=1)]
        # Negative balanced accuracy
        return -balanced_accuracy_score(y[tr_idx], preds)
        
    res = minimize(loss_func, [0.2, 0.1, 0.2, 0.5], bounds=[(0, 1)] * 4, method="SLSQP")
    opt_w = res.x / np.sum(res.x)
    print(f"  Fold {fold_i+1} optimal weights (Style, Syntax, Likelihood, Fused): {np.round(opt_w, 3)}")
    
    # Apply optimal weights to validation fold
    oof_opt_blend[val_idx] = (opt_w[0] * oof_style[val_idx] + 
                              opt_w[1] * oof_syntax[val_idx] + 
                              opt_w[2] * oof_likelihood[val_idx] + 
                              opt_w[3] * oof_fusion[val_idx])

results["08b_opt_simplex_blend"] = score_oof_by_folds(oof_opt_blend, "OptSimplex_Blend")

# %% ─────────────────────────────────────────────────
# EXP 08c: Meta-Stacking with Ridge Classifier
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXP 08c — Meta-Stacking Classifier (Ridge on OOF Probabilities)")
print("=" * 60)

oof_meta_stack = np.zeros_like(oof_fusion)
# Concatenate all 4 probability matrices (72 features per sample)
meta_X = np.hstack([oof_style, oof_syntax, oof_likelihood, oof_fusion])

for fold_i, (tr_idx, val_idx) in enumerate(splits):
    meta_tr, meta_val = meta_X[tr_idx], meta_X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    stacker = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    stacker.fit(meta_tr, y_tr)
    oof_meta_stack[val_idx] = stacker.predict_proba(meta_val)

results["08c_meta_stacker_lr"] = score_oof_by_folds(oof_meta_stack, "MetaStacker_LR")

# %% — Results Summary
print("\n" + "=" * 75)
print("RESULTS SUMMARY (08_ensemble)")
print("=" * 75)
print(f"{'Ensemble Model':<35} {'BalAcc':>8} {'±':>7}  {'MacroF1':>8}  {'vs_prev':>8}")
print("-" * 75)
for k, r in results.items():
    delta = r["mean_balanced_accuracy"] - PREV_BEST
    sign = "+" if delta >= 0 else ""
    print(f"{r['label']:<35} {r['mean_balanced_accuracy']:>8.4f} ±{r['std_balanced_accuracy']:.4f}  "
          f"{r['macro_f1']:>8.4f}  {sign}{delta:.4f}")

best_key = max(results, key=lambda k: results[k]["mean_balanced_accuracy"])
best = results[best_key]
print(f"\n🏆 Best in 08: {best['label']} | BalAcc={best['mean_balanced_accuracy']:.4f} ± {best['std_balanced_accuracy']:.4f}")
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
        exp_id = f"08_{key}"
        if exp_id not in existing_ids:
            record = {
                "experiment_id": exp_id,
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "research_question": "Does multi-paradigm OOF ensembling improve balanced accuracy on unseen prompts?",
                "hypothesis": "Orthogonal linguistic models (style, syntax, likelihood, fusion) provide complementary errors",
                "research_reference": "Wolpert (1992) Stacked Generalization; Caruana et al. (2004) Ensemble Selection",
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
                "training_time_sec": None,
                "status": "completed",
                "conclusion": f"BalAcc={r['mean_balanced_accuracy']:.4f} ± {r['std_balanced_accuracy']:.4f}",
                "next_recommendation": "Perform detailed error analysis on best ensemble in 09_error_analysis.py",
            }
            f.write(json.dumps(record) + "\n")
            saved += 1

print(f"\n✅ Saved {saved} experiment records → {log_path}")
