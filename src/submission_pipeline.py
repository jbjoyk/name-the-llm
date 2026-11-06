"""
Clean, self-contained, reproducible submission pipeline for the Kaggle Name That LLM competition.
Encapsulates the verified EXP_36 canonical baseline:
- 4-View LinearSVC (Char, Word, Shape, Function) trained on official train + Open Pool
- Shared-Prompt Gaussian Head (MiniLM-L3 Syntax 384d + 144d Stylometry)
- Blind Inference-Time Prompt Recovery (content-word TF-IDF average linkage clustering)
- Injective Bipartite Matching (Hungarian assignment per recovered cluster)
"""

import os, sys, re, time, gc
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# Ensure root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from src import gauss

LABELS = [
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
    'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'
]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL = {i: l for i, l in enumerate(LABELS)}

STOPWORDS_70 = [
    'the','and','but','that','which','with','into','from','for','you','your','our','we','i','it','its','they','their',
    'there','this','these','those','however','therefore','moreover','although','because','perhaps','might','would',
    'could','should','like','such','just','very','also','not','if','or','as','so','by','of','to','in','on','at','is',
    'are','be','was','were','have','has','had','may','must','all','any','each','every','many','some','more','most',
    'only','even','still','then','when','where','why','how','what','who'
]

MARKERS_55 = [
    '--', '---', '—', '–', '-', ':', ';', '!', '?', '(', ')', '[', ']', '{', '}',
    '"', "'", '`', '``', "''", '“', '”', '‘', '’', '...', '…', '/', '\\', '@', '#',
    '$', '%', '^', '&', '*', '_', '=', '+', '<', '>', '~', '|',
    '**', '##', '###', '####', '- ', '* ', '1. ', '2. ', '> ', '```'
]

WRE = re.compile(r"[A-Za-z]+(?:[’'][A-Za-z]+)?")
SRE = re.compile(r'(?<=[.?!])\s+')


def sanitize_text(text):
    """Robust sanitization handling None, NaN, extreme lengths, and non-strings."""
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    # Cap extreme passage length to 30,000 characters to prevent OOM
    if len(text) > 30000:
        text = text[:30000]
    return text


def to_casing_shape(text):
    """Maps uppercase runs to X and lowercase runs to x."""
    text = re.sub(r'[A-Z]+', 'X', text)
    text = re.sub(r'[a-z]+', 'x', text)
    return text


def to_function_words(text):
    """Preserves 70 common function words and punctuation; maps content words to CONTENT."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    mapped = [t if t in STOPWORDS_70 or not t.isalnum() else 'CONTENT' for t in tokens]
    return " ".join(mapped)


def extract_stylometry_row(t):
    """Extracts 142d rule-based stylometric feature vector for a text passage."""
    t = sanitize_text(t)
    words = WRE.findall(t)
    lo = [w.lower() for w in words]
    n_words = max(len(words), 1)
    n_chars = max(len(t), 1)
    lines = t.splitlines()
    n_lines = max(len(lines), 1)
    
    word_lens = np.array([len(w) for w in words] or [0])
    sentences = [z for z in SRE.split(t) if z.strip()]
    sent_lens = np.array([len(WRE.findall(z)) for z in sentences] or [0])
    
    base = [
        np.log1p(n_words),
        np.log1p(n_chars),
        n_chars / n_words,
        n_lines / n_words,
        len(set(lo)) / n_words,
        float(np.mean(word_lens)),
        float(np.std(word_lens)),
        float(np.mean(sent_lens)),
        float(np.std(sent_lens)),
        float(np.max(sent_lens)),
        float(np.mean(word_lens >= 8)),
        sum(c.isupper() for c in t) / n_chars,
        sum(c.isdigit() for c in t) / n_chars,
        sum(ord(c) > 127 for c in t) / n_chars,
        sum(c.isspace() for c in t) / n_chars,
        sum(c == '\n' for c in t) / n_chars
    ]
    # 55 punctuation markers
    base += [min(t.count(m) / n_words, 3.0) for m in MARKERS_55]
    # 71 function word frequencies
    cnt = {}
    for w in lo:
        cnt[w] = cnt.get(w, 0) + 1
    base += [cnt.get(w, 0) / n_words for w in STOPWORDS_70]
    
    return np.array(base, dtype=np.float32)


def extract_minilm_l3(texts, batch_size=64, max_len=256, device=None):
    """Extracts 384d Layer 3 mean-pooled syntax embeddings using all-MiniLM-L6-v2."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    
    n = len(texts)
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = [sanitize_text(t) for t in texts[i:i + batch_size]]
            tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = encoder(**tok, output_hidden_states=True)
            # Layer 3 hidden state (index 3 in hidden_states)
            l3 = out.hidden_states[3]
            mask = tok["attention_mask"].unsqueeze(-1)
            mean_pool = (l3 * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
            embeddings.append(mean_pool.cpu().numpy().astype(np.float32))
            
    return np.vstack(embeddings)


def recover_prompt_clusters(val_texts, max_cluster_size=18, threshold=0.15):
    """Unsupervised blind prompt recovery via average linkage on content-word TF-IDF."""
    clean_texts = [sanitize_text(t) for t in val_texts]
    if len(clean_texts) <= 1:
        return np.zeros(len(clean_texts), dtype=int)
        
    vec = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.15, sublinear_tf=True)
    try:
        X_content = vec.fit_transform(clean_texts)
        if X_content.shape[1] == 0:
            return np.arange(len(clean_texts))
    except ValueError:
        return np.arange(len(clean_texts))
        
    sim = (X_content @ X_content.T).toarray()
    dist = np.clip(1.0 - sim, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    
    condensed_dist = squareform(dist, checks=False)
    Z = linkage(condensed_dist, method='average')
    cluster_labels = fcluster(Z, t=1.0 - threshold, criterion='distance')
    
    unique_c, counts = np.unique(cluster_labels, return_counts=True)
    final_labels = cluster_labels.copy()
    next_id = max(unique_c) + 1
    
    for c, cnt in zip(unique_c, counts):
        if cnt > max_cluster_size:
            members = np.where(cluster_labels == c)[0]
            sub_dist = squareform(dist[np.ix_(members, members)], checks=False)
            sub_Z = linkage(sub_dist, method='average')
            sub_labels = fcluster(sub_Z, t=max_cluster_size, criterion='maxclust')
            for sl in np.unique(sub_labels):
                sub_idx = members[sub_labels == sl]
                final_labels[sub_idx] = next_id
                next_id += 1
                
    return final_labels


def hungarian_assignment(probs, cluster_labels):
    """Enforces injective bipartite matching within each recovered prompt cluster."""
    preds = np.zeros(len(probs), dtype=int)
    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) <= 1:
            preds[idx] = np.argmax(probs[idx], axis=1)
        else:
            cost_matrix = -probs[idx]
            r, c_ind = linear_sum_assignment(cost_matrix)
            preds[idx[r]] = c_ind
    return preds


class SubmissionPipeline:
    """Self-contained inference pipeline implementing the canonical EXP_36 baseline."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.w_gauss = self.config.get("w_gauss", 0.05)
        self.w_pool = self.config.get("w_pool", 0.35)
        self.gauss_d = self.config.get("gauss_d", 128)
        self.svc_c = self.config.get("svc_c", 0.5)
        
        # Models and transformers
        self.vec_char = None
        self.vec_word = None
        self.vec_shape = None
        self.vec_func = None
        self.clf_svc = None
        self.gauss_prm = None
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, train_df, open_pool_df=None):
        """Fits the complete pipeline on training data and optional Open Pool."""
        print(f"Fitting SubmissionPipeline on {len(train_df)} official samples...")
        t0 = time.time()
        
        # 1. Prepare texts and labels
        train_texts = [sanitize_text(t) for t in train_df["text"].values]
        y_train = np.array([LABEL_TO_IDX[m] for m in train_df["model"].values])
        prompts_train = train_df["prompt_id"].values
        
        # Incorporate Open Pool if provided
        all_texts = list(train_texts)
        all_y = list(y_train)
        sample_weights = [1.0] * len(train_texts)
        
        if open_pool_df is not None and len(open_pool_df) > 0:
            print(f"Adding {len(open_pool_df)} Open Pool samples (weight={self.w_pool})...")
            pool_texts = [sanitize_text(t) for t in open_pool_df["text"].values]
            pool_y = [LABEL_TO_IDX[m] for m in open_pool_df["model"].values]
            all_texts.extend(pool_texts)
            all_y.extend(pool_y)
            sample_weights.extend([self.w_pool] * len(pool_texts))
            
        all_y = np.array(all_y)
        sample_weights = np.array(sample_weights)
        
        # 2. Fit 4-View Vectorizers
        print("Fitting 4-view TF-IDF vectorizers...")
        self.vec_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=120000, sublinear_tf=True)
        self.vec_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, max_features=80000, sublinear_tf=True, token_pattern=r'(?u)\b\w+\b|[^\w\s]')
        self.vec_shape = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), min_df=3, max_features=40000, sublinear_tf=True)
        self.vec_func = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, max_features=30000, sublinear_tf=True)
        
        X_char = self.vec_char.fit_transform(all_texts)
        X_word = self.vec_word.fit_transform(all_texts)
        X_shape = self.vec_shape.fit_transform([to_casing_shape(t) for t in all_texts])
        X_func = self.vec_func.fit_transform([to_function_words(t) for t in all_texts])
        X_svc = hstack([X_char, X_word, X_shape, X_func]).tocsr()
        
        # 3. Fit LinearSVC
        print(f"Fitting LinearSVC(C={self.svc_c}) on matrix shape {X_svc.shape}...")
        self.clf_svc = LinearSVC(C=self.svc_c, class_weight='balanced', loss='squared_hinge', max_iter=2000, random_state=42)
        self.clf_svc.fit(X_svc, all_y, sample_weight=sample_weights)
        
        # 4. Extract and fit Shared-Prompt Gaussian Head strictly on official train
        print("Extracting dense syntax (MiniLM L3) and stylometry for Gaussian Head...")
        try:
            X_syn_train = extract_minilm_l3(train_texts)
        except Exception as e:
            print(f"Warning: GPU/MiniLM unavailable ({e}), using zero syntax features.")
            X_syn_train = np.zeros((len(train_texts), 384), dtype=np.float32)
            
        X_sty_train = np.stack([extract_stylometry_row(t) for t in train_texts])
        X_dense_train = np.hstack([X_syn_train, X_sty_train])
        
        print(f"Fitting Shared-Prompt Gaussian Head on {X_dense_train.shape} with {len(np.unique(prompts_train))} prompts...")
        self.gauss_prm = gauss.fit(X_dense_train, y_train, prompts_train, cfg={'d': self.gauss_d})
        
        self.is_fitted = True
        print(f"SubmissionPipeline fitted successfully in {time.time() - t0:.1f}s.")
        return self
        
    def predict_proba(self, test_df):
        """Predicts calibrated class probability distributions for test dataframe."""
        if not self.is_fitted:
            raise RuntimeError("SubmissionPipeline must be fitted before calling predict.")
            
        test_texts = [sanitize_text(t) for t in test_df["text"].values]
        n_test = len(test_texts)
        if n_test == 0:
            return np.zeros((0, len(LABELS))), np.array([])
            
        # 1. Extract 4-View Features and score with LinearSVC
        X_char = self.vec_char.transform(test_texts)
        X_word = self.vec_word.transform(test_texts)
        X_shape = self.vec_shape.transform([to_casing_shape(t) for t in test_texts])
        X_func = self.vec_func.transform([to_function_words(t) for t in test_texts])
        X_svc = hstack([X_char, X_word, X_shape, X_func]).tocsr()
        
        dec = self.clf_svc.decision_function(X_svc)
        # Handle binary edge case
        if dec.ndim == 1:
            dec = np.vstack([-dec, dec]).T
        log_p_svc = dec - np.log(np.sum(np.exp(dec - np.max(dec, axis=1, keepdims=True)), axis=1, keepdims=True)) - np.max(dec, axis=1, keepdims=True)
        
        # 2. Blind Inference-Time Prompt Recovery on test passages
        test_clusters = recover_prompt_clusters(test_texts)
        
        # 3. Dense features & Gaussian Head Scoring
        try:
            X_syn_test = extract_minilm_l3(test_texts)
        except Exception as e:
            X_syn_test = np.zeros((n_test, 384), dtype=np.float32)
            
        X_sty_test = np.stack([extract_stylometry_row(t) for t in test_texts])
        X_dense_test = np.hstack([X_syn_test, X_sty_test])
        
        U_test = gauss.transform(X_dense_test, self.gauss_prm)
        log_p_gauss = gauss.scores(X_dense_test, test_clusters, self.gauss_prm, U=U_test)
        
        # 4. Multi-Modal Log-Linear Fusion
        log_blend = log_p_svc + self.w_gauss * log_p_gauss
        prob_blend = softmax(log_blend, axis=1)
        
        return prob_blend, test_clusters
        
    def predict(self, test_df):
        """Generates competition-compliant submission DataFrame containing ['id', 'prediction']."""
        if len(test_df) == 0:
            return pd.DataFrame({"id": [], "prediction": []})
            
        probs, clusters = self.predict_proba(test_df)
        pred_indices = hungarian_assignment(probs, clusters)
        pred_labels = [IDX_TO_LABEL[idx] for idx in pred_indices]
        
        sub = pd.DataFrame({
            "id": test_df["id"].values,
            "prediction": pred_labels
        })
        return sub
