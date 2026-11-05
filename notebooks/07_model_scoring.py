# %% [markdown]
"""
# 07 — Model-Specific Scoring: Perplexity & Likelihood Profiles
EXPERIMENT_ID: 07_model_scoring
DATE: 2026-11-05
RESEARCH_QUESTION: Can class-conditional n-gram likelihood profiles (pseudo-perplexity under
                   each candidate LLM's language model) and tokenizer compression profiles
                   provide a discriminative scoring mechanism for the 18 frontier models?
HYPOTHESIS: Closed frontier LLMs (GPT-5.x, Claude-5.x, Gemini-3.x, Qwen, GLM, Kimi) cannot
            be queried directly during offline evaluation. However, fitting 18 class-conditional
            smoothed character and word language models strictly on training folds yields an
            18-dimensional log-likelihood score vector $\\log P(T \\mid \\text{Model}_k)$ that
            acts as a direct likelihood ratio test, providing a specialized inductive bias.
DATA_VERSION: v1 (original Kaggle release)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id)
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODELS: Class-Conditional Likelihood Scorer (Multinomial / Naive Bayes Likelihood),
        Likelihood Profile + Logistic Regression, Likelihood + TF-IDF Fusion
FEATURES: 18-class Word & Char Log-Likelihood profiles + Perplexity ratios
HYPERPARAMETERS: Lidstone smoothing alpha=0.01, GroupKFold(n_splits=5)
COMPUTE_CONFIGURATION: CPU multi-core
PREVIOUS_BEST_SCORE: 0.3178 (04_04b_full_fusion_dense)
"""

# %% — Imports & Setup
import sys, time, json, datetime, os, math
sys.path.insert(0, "/home/rj2826/name-the-llm/src")

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.sparse import hstack, csr_matrix

TRAIN_PATH = "/home/rj2826/name-the-llm/data/train.csv"
EXPERIMENTS_DIR = "/home/rj2826/name-the-llm/experiments"
OUTPUTS_DIR = "/home/rj2826/name-the-llm/outputs"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
PREV_BEST = 0.3178

train = pd.read_csv(TRAIN_PATH)
X_text = train["text"].values
y = train["model"].values
groups = train["prompt_id"].values
classes = sorted(np.unique(y))
n_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"Loaded {len(train)} samples across {n_classes} classes. Previous Best: {PREV_BEST:.4f}")

# %% — N-gram Likelihood Profiler (Fold-Isolated)
class NgramLikelihoodProfiler:
    """
    Fits 18 smoothed n-gram language models (one per candidate LLM).
    Computes an 18-dimensional log-likelihood profile vector for each test passage.
    Strictly fold-isolated: fitted only on training fold.
    """
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
        log_alpha_denom = {
            c: math.log(self.class_totals[c] + self.alpha * self.vocab_size)
            for c in classes
        }
        
        for i, text in enumerate(texts):
            ngrams = self._get_ngrams(text)
            n_tokens = max(len(ngrams), 1)
            for c_idx, c in enumerate(classes):
                counts = self.class_counts[c]
                denom = log_alpha_denom[c]
                # Sum log probabilities (with Lidstone smoothing)
                log_prob = 0.0
                for ng in ngrams:
                    cnt = counts.get(ng, 0)
                    log_prob += math.log(cnt + self.alpha) - denom
                # Normalize by passage length to get average log-likelihood (per-token cross entropy proxy)
                profiles[i, c_idx] = log_prob / n_tokens
                
        return profiles

# %% — Evaluate Likelihood Profiling via GroupKFold
kf = GroupKFold(n_splits=N_SPLITS)
splits = list(kf.split(train, y, groups=groups))

# Load Layer 3 representations from Exp 06 (discovered as best layer: 0.2786)
feat_l3 = np.load(os.path.join(OUTPUTS_DIR, "minilm_layer3_mean.npy"))

results = {}

# %% ─────────────────────────────────────────────────
# EXP 07a: Standalone Char-3 Likelihood Profiles + LR
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 07a — Char-3 Likelihood Profiles (18-d) + Logistic Regression")
print("=" * 60)

fold_bal_accs_a, fold_f1s_a = [], []
t0 = time.time()

for fold_i, (tr_idx, val_idx) in enumerate(splits):
    X_tr, X_val = X_text[tr_idx], X_text[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    profiler = NgramLikelihoodProfiler(n=3, alpha=0.05, char_level=True)
    profiler.fit(X_tr, y_tr)
    
    prof_tr = profiler.transform(X_tr)
    prof_val = profiler.transform(X_val)
    
    scaler = StandardScaler()
    prof_tr_s = scaler.fit_transform(prof_tr)
    prof_val_s = scaler.transform(prof_val)
    
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(prof_tr_s, y_tr)
    y_pred = clf.predict(prof_val_s)
    
    ba = balanced_accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    fold_bal_accs_a.append(ba)
    fold_f1s_a.append(f1)
    print(f"  [Char3_Likelihood_LR] Fold {fold_i+1}/{N_SPLITS} | BalAcc={ba:.4f} | MacroF1={f1:.4f}")

elapsed_a = time.time() - t0
results["07a_char3_likelihood_lr"] = {
    "label": "Char3_Likelihood_LR",
    "fold_scores": fold_bal_accs_a,
    "mean_balanced_accuracy": float(np.mean(fold_bal_accs_a)),
    "std_balanced_accuracy": float(np.std(fold_bal_accs_a)),
    "macro_f1": float(np.mean(fold_f1s_a)),
    "training_time_sec": round(elapsed_a, 1)
}
print(f"  → Char3_Likelihood_LR: BalAcc={results['07a_char3_likelihood_lr']['mean_balanced_accuracy']:.4f} ± {results['07a_char3_likelihood_lr']['std_balanced_accuracy']:.4f} | {elapsed_a:.1f}s\n")

# %% ─────────────────────────────────────────────────
# EXP 07b: Multi-Scale Likelihood (Char-3 + Char-4 + Word-1) + LR
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 07b — Multi-Scale Likelihood (Char-3 + Char-4 + Word-1, 54-d) + LR")
print("=" * 60)

fold_bal_accs_b, fold_f1s_b = [], []
t0 = time.time()

for fold_i, (tr_idx, val_idx) in enumerate(splits):
    X_tr, X_val = X_text[tr_idx], X_text[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    p_c3 = NgramLikelihoodProfiler(n=3, alpha=0.05, char_level=True).fit(X_tr, y_tr)
    p_c4 = NgramLikelihoodProfiler(n=4, alpha=0.05, char_level=True).fit(X_tr, y_tr)
    p_w1 = NgramLikelihoodProfiler(n=1, alpha=0.05, char_level=False).fit(X_tr, y_tr)
    
    prof_tr = np.hstack([p_c3.transform(X_tr), p_c4.transform(X_tr), p_w1.transform(X_tr)])
    prof_val = np.hstack([p_c3.transform(X_val), p_c4.transform(X_val), p_w1.transform(X_val)])
    
    scaler = StandardScaler()
    prof_tr_s = scaler.fit_transform(prof_tr)
    prof_val_s = scaler.transform(prof_val)
    
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(prof_tr_s, y_tr)
    y_pred = clf.predict(prof_val_s)
    
    ba = balanced_accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    fold_bal_accs_b.append(ba)
    fold_f1s_b.append(f1)
    print(f"  [MultiScale_Likelihood_LR] Fold {fold_i+1}/{N_SPLITS} | BalAcc={ba:.4f} | MacroF1={f1:.4f}")

elapsed_b = time.time() - t0
results["07b_multiscale_likelihood_lr"] = {
    "label": "MultiScale_Likelihood_LR",
    "fold_scores": fold_bal_accs_b,
    "mean_balanced_accuracy": float(np.mean(fold_bal_accs_b)),
    "std_balanced_accuracy": float(np.std(fold_bal_accs_b)),
    "macro_f1": float(np.mean(fold_f1s_b)),
    "training_time_sec": round(elapsed_b, 1)
}
print(f"  → MultiScale_Likelihood_LR: BalAcc={results['07b_multiscale_likelihood_lr']['mean_balanced_accuracy']:.4f} ± {results['07b_multiscale_likelihood_lr']['std_balanced_accuracy']:.4f} | {elapsed_b:.1f}s\n")

# %% ─────────────────────────────────────────────────
# EXP 07c: Full System: Layer 3 Syntactic Embeddings + MultiScale Likelihood + TF-IDF Fusion
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 07c — High-Performance Fusion: Layer 3 Syntax + Multi-Likelihood + TF-IDF")
print("=" * 60)

fold_bal_accs_c, fold_f1s_c = [], []
t0 = time.time()

for fold_i, (tr_idx, val_idx) in enumerate(splits):
    X_tr, X_val = X_text[tr_idx], X_text[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    l3_tr, l3_val = feat_l3[tr_idx], feat_l3[val_idx]
    
    # 1. TF-IDF
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=150_000, sublinear_tf=True, min_df=2)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=75_000, sublinear_tf=True, min_df=2)
    X_tr_tfidf = hstack([char_vec.fit_transform(X_tr), word_vec.fit_transform(X_tr)])
    X_val_tfidf = hstack([char_vec.transform(X_val), word_vec.transform(X_val)])
    
    # 2. MultiScale Likelihood (54-d)
    p_c3 = NgramLikelihoodProfiler(n=3, alpha=0.05, char_level=True).fit(X_tr, y_tr)
    p_w1 = NgramLikelihoodProfiler(n=1, alpha=0.05, char_level=False).fit(X_tr, y_tr)
    prof_tr = np.hstack([p_c3.transform(X_tr), p_w1.transform(X_tr)])
    prof_val = np.hstack([p_c3.transform(X_val), p_w1.transform(X_val)])
    
    scaler_prof = StandardScaler()
    prof_tr_s = scaler_prof.fit_transform(prof_tr)
    prof_val_s = scaler_prof.transform(prof_val)
    
    # 3. Layer 3 Syntactic Embedding (384-d, standardized)
    scaler_l3 = StandardScaler()
    l3_tr_s = scaler_l3.fit_transform(l3_tr)
    l3_val_s = scaler_l3.transform(l3_val)
    
    # Fuse all representations
    X_tr_all = hstack([X_tr_tfidf, csr_matrix(prof_tr_s), csr_matrix(l3_tr_s)])
    X_val_all = hstack([X_val_tfidf, csr_matrix(prof_val_s), csr_matrix(l3_val_s)])
    
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="saga", random_state=RANDOM_SEED)
    clf.fit(X_tr_all, y_tr)
    y_pred = clf.predict(X_val_all)
    
    ba = balanced_accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    fold_bal_accs_c.append(ba)
    fold_f1s_c.append(f1)
    print(f"  [L3_Syntax_Likelihood_TFIDF_LR] Fold {fold_i+1}/{N_SPLITS} | BalAcc={ba:.4f} | MacroF1={f1:.4f}")

elapsed_c = time.time() - t0
results["07c_l3_likelihood_tfidf_lr"] = {
    "label": "L3_Syntax_Likelihood_TFIDF_LR",
    "fold_scores": fold_bal_accs_c,
    "mean_balanced_accuracy": float(np.mean(fold_bal_accs_c)),
    "std_balanced_accuracy": float(np.std(fold_bal_accs_c)),
    "macro_f1": float(np.mean(fold_f1s_c)),
    "training_time_sec": round(elapsed_c, 1)
}
print(f"  → L3_Syntax_Likelihood_TFIDF_LR: BalAcc={results['07c_l3_likelihood_tfidf_lr']['mean_balanced_accuracy']:.4f} ± {results['07c_l3_likelihood_tfidf_lr']['std_balanced_accuracy']:.4f} | {elapsed_c:.1f}s\n")

# %% — Summary & Record Saving
print("=" * 75)
print("RESULTS SUMMARY (07_model_scoring)")
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
print(f"\n🏆 Best in 07: {best['label']} | BalAcc={best['mean_balanced_accuracy']:.4f} ± {best['std_balanced_accuracy']:.4f}")
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
        exp_id = f"07_{key}"
        if exp_id not in existing_ids:
            record = {
                "experiment_id": exp_id,
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "research_question": "Can class-conditional n-gram likelihood profiles provide model scoring for unseen prompts?",
                "hypothesis": "Class-conditional character and word language models act as direct likelihood ratio tests",
                "research_reference": "Goodman (2001) Language modeling smoothing; Stamatatos (2009)",
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
                "next_recommendation": "Use top diverse models for probability-level stacking in 08_ensemble.py",
            }
            f.write(json.dumps(record) + "\n")
            saved += 1

print(f"\n✅ Saved {saved} experiment records → {log_path}")
