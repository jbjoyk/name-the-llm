# %% [markdown]
"""
# 10 — Final Production Pipeline & Offline Submission Verification
EXPERIMENT_ID: 10_final_pipeline
DATE: 2026-11-05
RESEARCH_QUESTION: How does the complete meta-stacking ensemble perform when trained on 100% of the
                   annotated competition data, and can it generate a strictly verified, leakage-free
                   offline submission file conforming to Kaggle requirements?
HYPOTHESIS: Refitting the winning 4-model ensemble (Style + Syntax + Likelihood + Fused) and Meta-Stacker
            on all 2,699 samples will maximize vocabulary coverage, stabilizer parameters, and class
            separation on unseen prompts, enabling a compliant offline submission pipeline.
DATA_VERSION: v1 (full dataset)
CV_STRATEGY: Evaluated via 5-Fold GroupKFold(group=prompt_id) (0.3630 ± 0.0046 BalAcc); refit on 100% train data
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODEL: Full Production Meta-Stacker Ensemble (Style + Syntax + Likelihood + Fused + Meta-LR)
OUTPUTS: models/*.joblib, submissions/submission.parquet, submissions/submission.csv, scripts/predict.py
"""

# %% — Imports & Setup
import os, sys, time, json, math, re, datetime
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, "/home/rj2826/name-the-llm/src")

BASE_DIR = "/home/rj2826/name-the-llm"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)

RANDOM_SEED = 42

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_parquet(os.path.join(DATA_DIR, "sample_submission.parquet"))

classes = sorted(np.unique(train["model"].values))
n_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"Loaded {len(train)} training samples across {n_classes} classes.")
print(f"Loaded {len(test)} test samples to score.")

# %% — Feature Extraction Classes and Functions
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
        self.classes = None
        
    def _get_ngrams(self, text):
        if self.char_level:
            return [text[i:i + self.n] for i in range(len(text) - self.n + 1)]
        else:
            words = text.split()
            return [" ".join(words[i:i + self.n]) for i in range(len(words) - self.n + 1)]
        
    def fit(self, texts, labels):
        self.classes = sorted(np.unique(labels))
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
        profiles = np.zeros((len(texts), len(self.classes)), dtype=np.float32)
        log_alpha_denom = {c: math.log(self.class_totals[c] + self.alpha * self.vocab_size) for c in self.classes}
        for i, text in enumerate(texts):
            ngrams = self._get_ngrams(text)
            n_tokens = max(len(ngrams), 1)
            for c_idx, c in enumerate(self.classes):
                counts = self.class_counts[c]
                denom = log_alpha_denom[c]
                log_prob = 0.0
                for ng in ngrams:
                    cnt = counts.get(ng, 0)
                    log_prob += math.log(cnt + self.alpha) - denom
                profiles[i, c_idx] = log_prob / n_tokens
        return profiles

# %% — Fit Full Production Models on 100% Training Data
print("\n" + "=" * 60)
print("TRAINING PRODUCTION ENSEMBLE ON 100% OF DATA (2,699 SAMPLES)")
print("=" * 60)
t_start = time.time()

X_text_full = train["text"].values
y_full = train["model"].values

# 1. Fit TF-IDF Vectorizers
print("1/6: Fitting TF-IDF Vectorizers (Char 3-5 + Word 1-2)...")
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=150_000, sublinear_tf=True, min_df=2)
word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=75_000, sublinear_tf=True, min_df=2)
char_feat_full = char_vec.fit_transform(X_text_full)
word_feat_full = word_vec.fit_transform(X_text_full)
X_tfidf_full = hstack([char_feat_full, word_feat_full])

# 2. Fit Stylometric Scaler
print("2/6: Extracting & Scaling Stylometric Features...")
df_stylo_full = extract_stylometric_features(X_text_full).fillna(0)
scaler_stylo = StandardScaler()
stylo_feat_full = scaler_stylo.fit_transform(df_stylo_full.values)

# Fit Model 1: Style
print("     Fitting Base Model 1 (Style: TF-IDF + Stylometry)...")
m1_style = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="saga", random_state=RANDOM_SEED)
m1_style.fit(hstack([X_tfidf_full, csr_matrix(stylo_feat_full)]), y_full)

# 3. Fit Transformer Syntax Model
print("3/6: Fitting Base Model 2 (Syntax: Layer 3 MiniLM)...")
feat_l3_full = np.load(os.path.join(OUTPUTS_DIR, "minilm_layer3_mean.npy"))
scaler_l3 = StandardScaler()
l3_scaled_full = scaler_l3.fit_transform(feat_l3_full)

m2_syntax = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
m2_syntax.fit(l3_scaled_full, y_full)

# 4. Fit Likelihood Profilers
print("4/6: Fitting MultiScale Likelihood Profilers (Char-3 + Word-1)...")
p_c3 = NgramLikelihoodProfiler(n=3, alpha=0.05, char_level=True).fit(X_text_full, y_full)
p_w1 = NgramLikelihoodProfiler(n=1, alpha=0.05, char_level=False).fit(X_text_full, y_full)
prof_full = np.hstack([p_c3.transform(X_text_full), p_w1.transform(X_text_full)])
scaler_prof = StandardScaler()
prof_scaled_full = scaler_prof.fit_transform(prof_full)

m3_likelihood = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
m3_likelihood.fit(prof_scaled_full, y_full)

# 5. Fit Model 4: Full Fusion
print("5/6: Fitting Base Model 4 (High-Performance Fusion: L3 + Likelihood + TF-IDF)...")
X_fused_full = hstack([X_tfidf_full, csr_matrix(prof_scaled_full), csr_matrix(l3_scaled_full)])
m4_fused = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="saga", random_state=RANDOM_SEED)
m4_fused.fit(X_fused_full, y_full)

# 6. Fit Meta-Stacker on Full OOF Probabilities
print("6/6: Training Meta-Stacker on 4-Model OOF Probability Matrices...")
oof_style = np.load(os.path.join(OUTPUTS_DIR, "oof_style.npy"))
oof_syntax = np.load(os.path.join(OUTPUTS_DIR, "oof_syntax.npy"))
oof_likelihood = np.load(os.path.join(OUTPUTS_DIR, "oof_likelihood.npy"))
oof_fusion = np.load(os.path.join(OUTPUTS_DIR, "oof_fusion.npy"))
meta_X_full = np.hstack([oof_style, oof_syntax, oof_likelihood, oof_fusion])

meta_stacker = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
meta_stacker.fit(meta_X_full, y_full)

print(f"Full ensemble training complete in {time.time() - t_start:.1f}s.")

# %% — Persist All Production Artifacts
print("\nPersisting all model checkpoints and feature scalers to models/...")
joblib.dump(char_vec, os.path.join(MODELS_DIR, "char_vectorizer.joblib"))
joblib.dump(word_vec, os.path.join(MODELS_DIR, "word_vectorizer.joblib"))
joblib.dump(scaler_stylo, os.path.join(MODELS_DIR, "scaler_stylometry.joblib"))
joblib.dump(m1_style, os.path.join(MODELS_DIR, "m1_style.joblib"))
joblib.dump(scaler_l3, os.path.join(MODELS_DIR, "scaler_l3.joblib"))
joblib.dump(m2_syntax, os.path.join(MODELS_DIR, "m2_syntax.joblib"))
joblib.dump(p_c3, os.path.join(MODELS_DIR, "profiler_c3.joblib"))
joblib.dump(p_w1, os.path.join(MODELS_DIR, "profiler_w1.joblib"))
joblib.dump(scaler_prof, os.path.join(MODELS_DIR, "scaler_profiler.joblib"))
joblib.dump(m3_likelihood, os.path.join(MODELS_DIR, "m3_likelihood.joblib"))
joblib.dump(m4_fused, os.path.join(MODELS_DIR, "m4_fused.joblib"))
joblib.dump(meta_stacker, os.path.join(MODELS_DIR, "meta_stacker.joblib"))
np.save(os.path.join(MODELS_DIR, "label_classes.npy"), np.array(classes))
print("✅ Successfully persisted all 12 model and transformer components to models/.")

# %% — End-to-End Inference Verification on test.csv
print("\n" + "=" * 60)
print("RUNNING END-TO-END INFERENCE ON TEST SET")
print("=" * 60)

test_texts = test["text"].values
test_ids = test["id"].values
n_test = len(test)

# A. Extract TF-IDF & Stylometry for Test
print("A. Extracting Test TF-IDF & Stylometry...")
test_char = char_vec.transform(test_texts)
test_word = word_vec.transform(test_texts)
test_tfidf = hstack([test_char, test_word])
test_stylo = scaler_stylo.transform(extract_stylometric_features(test_texts).fillna(0).values)

prob_m1 = m1_style.predict_proba(hstack([test_tfidf, csr_matrix(test_stylo)]))

# B. Extract Test Layer 3 Transformer Syntax
print("B. Extracting Test Transformer Layer 3 Hidden States...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
embedder.eval()

inputs = tokenizer(list(test_texts), padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
with torch.no_grad():
    out = embedder(**inputs, output_hidden_states=True)
    test_l3_raw = out.hidden_states[3].mean(dim=1).cpu().numpy()

test_l3_scaled = scaler_l3.transform(test_l3_raw)
prob_m2 = m2_syntax.predict_proba(test_l3_scaled)

# C. Extract Test Likelihood Profiles
print("C. Computing Test MultiScale Likelihood Profiles...")
test_prof_raw = np.hstack([p_c3.transform(test_texts), p_w1.transform(test_texts)])
test_prof_scaled = scaler_prof.transform(test_prof_raw)
prob_m3 = m3_likelihood.predict_proba(test_prof_scaled)

# D. Base Model 4 Fused Probabilities
print("D. Computing Base Model 4 Fused Probabilities...")
test_fused_feat = hstack([test_tfidf, csr_matrix(test_prof_scaled), csr_matrix(test_l3_scaled)])
prob_m4 = m4_fused.predict_proba(test_fused_feat)

# E. Meta-Stacker Final Ensemble Prediction
print("E. Running Meta-Stacker on Test Combined Probabilities...")
test_meta_X = np.hstack([prob_m1, prob_m2, prob_m3, prob_m4])
final_test_probs = meta_stacker.predict_proba(test_meta_X)
test_pred_idx = np.argmax(final_test_probs, axis=1)
test_preds = [classes[idx] for idx in test_pred_idx]

# %% — Formulate Submission DataFrames
sub_df = pd.DataFrame({"id": test_ids, "model": test_preds})
parquet_path = os.path.join(SUBMISSIONS_DIR, "submission.parquet")
csv_path = os.path.join(SUBMISSIONS_DIR, "submission.csv")

sub_df.to_parquet(parquet_path, index=False)
sub_df.to_csv(csv_path, index=False)
print(f"\nSaved submission to:\n  • {parquet_path}\n  • {csv_path}")

# %% — Rigorous 6-Point Kaggle Submission Verification Suite
print("\n" + "=" * 60)
print("KAGGLE SUBMISSION RIGOROUS VERIFICATION SUITE")
print("=" * 60)

verifications = []

# Check 1: File Existence & Non-empty
exists_p = os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0
exists_c = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
verifications.append(("Check 1: File existence and non-zero byte size", exists_p and exists_c))

# Check 2: Row Count Match
sub_read = pd.read_parquet(parquet_path)
row_match = len(sub_read) == n_test
verifications.append((f"Check 2: Exact row count match (expected {n_test}, got {len(sub_read)})", row_match))

# Check 3: Column Header Conformance
cols_match = list(sub_read.columns) == ["id", "model"]
verifications.append(("Check 3: Exact column header schema (['id', 'model'])", cols_match))

# Check 4: Absence of Null / NaN / Empty
null_free = sub_read["id"].isna().sum() == 0 and sub_read["model"].isna().sum() == 0
verifications.append(("Check 4: Zero null or NaN values in predictions", null_free))

# Check 5: Identifier Alignment
id_align = (sub_read["id"].values == test_ids).all()
verifications.append(("Check 5: 1-to-1 exact ID order match with test.csv", id_align))

# Check 6: Valid Class Set Inclusion
valid_classes = set(classes)
pred_set = set(sub_read["model"].values)
classes_valid = pred_set.issubset(valid_classes)
verifications.append((f"Check 6: All predicted labels are valid LLM classes ({pred_set})", classes_valid))

all_passed = all(v[1] for v in verifications)
for desc, passed in verifications:
    status_icon = "PASS [OK]" if passed else "FAIL [X]"
    print(f"  {status_icon} : {desc}")

print("-" * 60)
if all_passed:
    print("🏆 ALL 6 SUBMISSION VERIFICATION CHECKS PASSED PERFECTLY!")
else:
    raise RuntimeError("Submission verification check failed!")

print("\nSample Submission Output:")
print(sub_read)

# %% — Produce Standalone Prediction CLI Script: scripts/predict.py
predict_script_content = '''#!/usr/bin/env python3
"""
Standalone Kaggle Offline Inference Script for 'Name That LLM'.
Loads fitted production models from models/ and generates verified submission predictions.
"""

import os, sys, math, re, argparse
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import hstack, csr_matrix
import torch
from transformers import AutoTokenizer, AutoModel

def extract_stylometric_features(texts):
    records = []
    for text in texts:
        words = text.split()
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', text) if s.strip()] or [text]
        n_words = max(len(words), 1)
        n_sents = max(len(sents), 1)
        n_chars = max(len(text), 1)
        word_lens = [len(w) for w in words]
        sent_word_lens = [len(s.split()) for s in sents]
        word_lower = [w.lower().strip(".,;:!?\\"'") for w in words]
        unique_words = set(w for w in word_lower if w)
        ttr = len(unique_words) / n_words
        
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
            "total_punct_rate": 1000 * sum(1 for c in text if c in ".,;:!?—–()\\"'") / n_chars,
            "upper_ratio": sum(1 for c in text if c.isupper()) / max(sum(1 for c in text if c.isalpha()), 1),
            "digit_ratio": sum(1 for c in text if c.isdigit()) / n_chars,
            "newline_count": text.count("\\n"),
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
        self.class_counts = {}
        self.class_totals = {}
        self.classes = None
        
    def _get_ngrams(self, text):
        if self.char_level:
            return [text[i:i + self.n] for i in range(len(text) - self.n + 1)]
        else:
            words = text.split()
            return [" ".join(words[i:i + self.n]) for i in range(len(words) - self.n + 1)]

    def transform(self, texts):
        profiles = np.zeros((len(texts), len(self.classes)), dtype=np.float32)
        log_alpha_denom = {c: math.log(self.class_totals[c] + self.alpha * self.vocab_size) for c in self.classes}
        for i, text in enumerate(texts):
            ngrams = self._get_ngrams(text)
            n_tokens = max(len(ngrams), 1)
            for c_idx, c in enumerate(self.classes):
                counts = self.class_counts[c]
                denom = log_alpha_denom[c]
                log_prob = 0.0
                for ng in ngrams:
                    cnt = counts.get(ng, 0)
                    log_prob += math.log(cnt + self.alpha) - denom
                profiles[i, c_idx] = log_prob / n_tokens
        return profiles

def main():
    parser = argparse.ArgumentParser(description="Predict LLM source from text")
    parser.add_argument("--test-path", type=str, default="/home/rj2826/name-the-llm/data/test.csv")
    parser.add_argument("--models-dir", type=str, default="/home/rj2826/name-the-llm/models")
    parser.add_argument("--output-path", type=str, default="/home/rj2826/name-the-llm/submissions/submission.parquet")
    args = parser.parse_args()

    test = pd.read_csv(args.test_path)
    test_texts = test["text"].values
    test_ids = test["id"].values
    
    print(f"Loaded {len(test)} test samples from {args.test_path}")
    print(f"Loading trained models from {args.models_dir}...")
    
    char_vec = joblib.load(os.path.join(args.models_dir, "char_vectorizer.joblib"))
    word_vec = joblib.load(os.path.join(args.models_dir, "word_vectorizer.joblib"))
    scaler_stylo = joblib.load(os.path.join(args.models_dir, "scaler_stylometry.joblib"))
    m1_style = joblib.load(os.path.join(args.models_dir, "m1_style.joblib"))
    scaler_l3 = joblib.load(os.path.join(args.models_dir, "scaler_l3.joblib"))
    m2_syntax = joblib.load(os.path.join(args.models_dir, "m2_syntax.joblib"))
    p_c3 = joblib.load(os.path.join(args.models_dir, "profiler_c3.joblib"))
    p_w1 = joblib.load(os.path.join(args.models_dir, "profiler_w1.joblib"))
    scaler_prof = joblib.load(os.path.join(args.models_dir, "scaler_profiler.joblib"))
    m3_likelihood = joblib.load(os.path.join(args.models_dir, "m3_likelihood.joblib"))
    m4_fused = joblib.load(os.path.join(args.models_dir, "m4_fused.joblib"))
    meta_stacker = joblib.load(os.path.join(args.models_dir, "meta_stacker.joblib"))
    classes = list(np.load(os.path.join(args.models_dir, "label_classes.npy")))

    print("Extracting features and generating predictions...")
    test_char = char_vec.transform(test_texts)
    test_word = word_vec.transform(test_texts)
    test_tfidf = hstack([test_char, test_word])
    test_stylo = scaler_stylo.transform(extract_stylometric_features(test_texts).fillna(0).values)
    prob_m1 = m1_style.predict_proba(hstack([test_tfidf, csr_matrix(test_stylo)]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    embedder.eval()
    inputs = tokenizer(list(test_texts), padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        out = embedder(**inputs, output_hidden_states=True)
        test_l3_raw = out.hidden_states[3].mean(dim=1).cpu().numpy()
    test_l3_scaled = scaler_l3.transform(test_l3_raw)
    prob_m2 = m2_syntax.predict_proba(test_l3_scaled)

    test_prof_raw = np.hstack([p_c3.transform(test_texts), p_w1.transform(test_texts)])
    test_prof_scaled = scaler_prof.transform(test_prof_raw)
    prob_m3 = m3_likelihood.predict_proba(test_prof_scaled)

    test_fused_feat = hstack([test_tfidf, csr_matrix(test_prof_scaled), csr_matrix(test_l3_scaled)])
    prob_m4 = m4_fused.predict_proba(test_fused_feat)

    test_meta_X = np.hstack([prob_m1, prob_m2, prob_m3, prob_m4])
    final_test_probs = meta_stacker.predict_proba(test_meta_X)
    test_preds = [classes[idx] for idx in np.argmax(final_test_probs, axis=1)]

    sub = pd.DataFrame({"id": test_ids, "model": test_preds})
    if args.output_path.endswith(".parquet"):
        sub.to_parquet(args.output_path, index=False)
    else:
        sub.to_csv(args.output_path, index=False)
    print(f"Successfully wrote {len(sub)} predictions to {args.output_path}")

if __name__ == "__main__":
    main()
'''

predict_script_path = os.path.join(SCRIPTS_DIR, "predict.py")
with open(predict_script_path, "w") as f:
    f.write(predict_script_content)
os.chmod(predict_script_path, 0o755)
print(f"\nGenerated production prediction CLI → {predict_script_path}")

# %% — Log Experiment to experiment_log.jsonl
log_path = os.path.join(EXPERIMENTS_DIR, "experiment_log.jsonl")
record = {
    "experiment_id": "10_final_pipeline",
    "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "research_question": "Can the winning meta-stacking ensemble be refit on 100% of data and pass all Kaggle submission checks?",
    "hypothesis": "Refitting on full dataset maximizes parameter precision and passes all verification checks",
    "research_reference": "End-to-End Ensemble Production Pipeline",
    "dataset_version": "v1",
    "feature_method": "Full_Production_MetaStacker_Ensemble",
    "model": "MetaStacker_LR_100pct_Train",
    "hyperparameters": {"n_splits": 5, "random_seed": RANDOM_SEED},
    "cv_strategy": "GroupKFold(n_splits=5, group=prompt_id) -> Refit 100%",
    "group_column": "prompt_id",
    "random_seed": RANDOM_SEED,
    "fold_scores": None,
    "mean_balanced_accuracy": 0.3630,
    "std_balanced_accuracy": 0.0046,
    "macro_f1": 0.3522,
    "per_class_recall": None,
    "training_time_sec": None,
    "status": "completed",
    "conclusion": "Ensemble refit on 100% data. Checkpoints saved to models/. All 6 Kaggle submission checks passed.",
    "next_recommendation": "Ready for user review and optional manual Kaggle submission.",
}
with open(log_path, "a") as f:
    f.write(json.dumps(record) + "\n")

print("\n✅ Successfully completed Notebook 10: Final Pipeline!")
