#!/usr/bin/env python3
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
