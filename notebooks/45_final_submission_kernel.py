"""
# ==============================================================================
# EXP_45: Final Offline Submission Kernel for Kaggle "Name That LLM"
# Self-contained, leak-free production inference pipeline.
# Architecture: EXP_44-C Hybrid Prompt-Residual GAT + EXP_36 Soft Convex Fusion
# Post-Processing: Injective Bipartite Matching (Hungarian Assignment)
# Output Schema: ['id', 'model']
# ==============================================================================
"""

import os
import sys
import gc
import re
import time
import math
import random
import itertools
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
from sklearn.covariance import ledoit_wolf_shrinkage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel

# Pinned Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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


def resolve_model_path(model_identifier):
    """Resolves model path across Kaggle offline datasets and local HuggingFace cache."""
    short_name = model_identifier.split('/')[-1].lower()

    # 1. Search inside /kaggle/input if mounted on Kaggle
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files:
                root_lower = root.lower()
                if ("minilm" in short_name and ("minilm" in root_lower or "all-mini" in root_lower)) or \
                   ("deberta" in short_name and ("deberta" in root_lower or "v3-small" in root_lower)):
                    print(f">> [Offline Model Found] {model_identifier} -> {root}")
                    return root

    # 2. Check direct local directories
    candidates = [
        f"/kaggle/input/{model_identifier.split('/')[-1]}",
        f"/kaggle/input/{short_name}",
        f"/kaggle/input/{model_identifier.replace('/', '-')}"
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "config.json")):
            return c

    # 3. If running inside Kaggle environment and no local folder exists, return None (avoids DNS timeout)
    if os.path.exists("/kaggle"):
        return None

    return model_identifier


def sanitize_text(text):
    """Robust sanitization handling None, NaN, extreme lengths, and non-strings."""
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    if len(text) > 30000:
        text = text[:30000]
    return text


def to_casing_shape(text):
    text = re.sub(r'[A-Z]+', 'X', text)
    text = re.sub(r'[a-z]+', 'x', text)
    return text


def to_function_words(text):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    mapped = [t if t in STOPWORDS_70 or not t.isalnum() else 'CONTENT' for t in tokens]
    return " ".join(mapped)


def extract_stylometry_row(t):
    """Extracts 144d rule-based stylometric feature vector for a text passage."""
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
        sum(c == '\n' for c in t) / n_chars,
        sum(c in MARKERS_55 for c in t) / n_chars,
        len(sentences) / n_words
    ]
    base += [min(t.count(m) / n_words, 3.0) for m in MARKERS_55]
    cnt = {}
    for w in lo:
        cnt[w] = cnt.get(w, 0) + 1
    base += [cnt.get(w, 0) / n_words for w in STOPWORDS_70]
    
    arr = np.array(base, dtype=np.float32)
    if len(arr) < 144:
        arr = np.pad(arr, (0, 144 - len(arr)))
    elif len(arr) > 144:
        arr = arr[:144]
    return arr


def extract_minilm_l3(texts, batch_size=64, max_len=256, device=None):
    """Extracts 384d Layer 3 mean-pooled syntax embeddings using all-MiniLM-L6-v2."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model_path("sentence-transformers/all-MiniLM-L6-v2")
    if model_path is None:
        raise FileNotFoundError("MiniLM offline weights not attached under /kaggle/input.")
    is_local = os.path.isdir(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=is_local)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=is_local).to(device)
    encoder.eval()
    
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [sanitize_text(t) for t in texts[i:i + batch_size]]
            tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = encoder(**tok, output_hidden_states=True)
            l3 = out.hidden_states[3]
            mask = tok["attention_mask"].unsqueeze(-1)
            mean_pool = (l3 * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
            embeddings.append(mean_pool.cpu().numpy().astype(np.float32))
    return np.vstack(embeddings)


def extract_deberta_v3(texts, batch_size=32, max_len=512, device=None):
    """Extracts 768d mean-pooled semantic embeddings using microsoft/deberta-v3-small."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model_path("microsoft/deberta-v3-small")
    if model_path is None:
        raise FileNotFoundError("DeBERTa offline weights not attached under /kaggle/input.")
    is_local = os.path.isdir(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=is_local)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=is_local).to(device)
    encoder.eval()
    
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [sanitize_text(t) for t in texts[i:i + batch_size]]
            tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            mask = tok["attention_mask"].unsqueeze(-1)
            out = encoder(**tok)
            last_hidden = out.last_hidden_state
            mean_pool = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
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


# ==============================================================================
# Inlined Shared-Prompt Gaussian Group Head
# ==============================================================================
GAUSS_DEFAULT = {
    "red": "wpca", "d": 256, "se_shrink": "lw", "sp_shrink": "lw",
    "lam_floor": 1e-4, "n_exact": 3, "inj": True, "iters": 5, "repel": True, "sp_scale": 1.0
}
_MASK = {}

def _lse(A, axis=None, keepdims=False):
    m = np.max(A, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m
    if keepdims: return out
    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())

def _log_softmax(L): return L - _lse(L, axis=1, keepdims=True)

def _lw_shrinkage(X):
    return float(ledoit_wolf_shrinkage(X, assume_centered=True))

def _shrunk(S, a):
    d = S.shape[0]
    return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)

def _two_way(Z, y, pinv, G, iters=4):
    P = np.zeros((G, Z.shape[1]))
    cnt_p = np.bincount(pinv, minlength=G).astype(float)
    cls = np.unique(y)
    mu = np.zeros((18, Z.shape[1]))
    for _ in range(iters):
        R = Z - P[pinv]
        for c in cls: mu[c] = R[y == c].mean(0)
        R = Z - mu[y]
        P = np.zeros_like(P)
        np.add.at(P, pinv, R)
        P /= np.maximum(cnt_p[:, None], 1.0)
    return mu, P, cnt_p

def gauss_fit(X, y, prompts, cfg=None):
    c = dict(GAUSS_DEFAULT); c.update(cfg or {})
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y); N, D = X.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True); G = len(pu)
    mx = X.mean(0); sx = X.std(0); sx = np.where(sx > 1e-8, sx, 1.0); Xs = (X - mx) / sx
    Pm = np.zeros((G, D))
    np.add.at(Pm, pinv, Xs)
    Pm /= np.maximum(np.bincount(pinv)[:, None], 1.0)
    R = Xs - Pm[pinv]
    d = int(min(c["d"], N - G - 1, D))
    if N < D:
        w, U = np.linalg.eigh(R @ R.T)
        o = np.argsort(w)[::-1][:d]
        w, U = np.clip(w[o], 1e-12, None), U[:, o]
        V = (R.T @ U) / np.sqrt(w)
    else:
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        V = Vt[:d].T
    Z = Xs @ V; d = Z.shape[1]
    mu, P, ng = _two_way(Z, y, pinv, G)
    E = Z - mu[y] - P[pinv]
    dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = E.T @ E / dof
    ae = _lw_shrinkage(E) if c["se_shrink"] == "lw" else float(c["se_shrink"])
    Se = _shrunk(Se_raw, ae)
    Pc = P - P.mean(0)
    Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / np.maximum(ng, 1.0)) * Se_raw
    ap = _lw_shrinkage(Pc) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se))
    Sp = (1 - ap) * Sp_raw + ap * (tr / d) * np.eye(d)
    Sp *= c["sp_scale"]
    L = np.linalg.cholesky(Se + 1e-6 * np.eye(d))
    Li = np.linalg.inv(L)
    M = Li @ Sp @ Li.T
    M = 0.5 * (M + M.T)
    lam, Q = np.linalg.eigh(M)
    lam = np.clip(lam, c["lam_floor"], None)
    A = Q.T @ Li
    W = V @ A.T
    return {
        "mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": 1.0, "Tg": 1.0, "cfg": c
    }

def gauss_transform(X, prm):
    return ((np.asarray(X, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]

def _singleton_ll(U, m, lam):
    R = U[:, None, :] - m[None]
    return -0.5 * (R ** 2 / (1 + lam)).sum(-1)

def _group_logpost(U, prm, cfg=None):
    n = len(U)
    m, lam = prm["m"], prm["lam"]
    if n == 1:
        return _log_softmax(_singleton_ll(U, m, lam))
    v = lam / (1 + (n - 1) * lam)
    L = _singleton_ll(U, m, lam)
    q = np.exp(_log_softmax(L))
    for _ in range(5):
        s = U - q @ m
        S = s.sum(0)
        Pm = (S[None] - s) * v
        R = U[:, None, :] - Pm[:, None, :] - m[None]
        L = -0.5 * (R ** 2 / (1 + v)).sum(-1)
        lo = np.log(np.clip(1 - q, 1e-6, None))
        L = L + (lo.sum(0)[None] - lo)
        q = np.exp(_log_softmax(L))
    return _log_softmax(L)

def gauss_scores(X, lab, prm, U=None):
    U = gauss_transform(X, prm) if U is None else U
    n = len(U)
    out = np.empty((n, prm["m"].shape[0]))
    pu, pinv = np.unique(lab, return_inverse=True)
    sizes = np.bincount(pinv)
    for g in np.unique(pinv):
        ii = np.where(pinv == g)[0]
        out[ii] = _group_logpost(U[ii], prm)
    return out


# ==============================================================================
# Relational Graph Attention Network (GAT)
# ==============================================================================
def compute_prompt_residuals(X, prompt_labels):
    R = np.zeros_like(X)
    for p in np.unique(prompt_labels):
        mask = (prompt_labels == p)
        mean_p = np.mean(X[mask], axis=0, keepdims=True)
        R[mask] = X[mask] - mean_p
    return R


def pack_prompt_graph_batch(node_features, prompt_labels, node_targets=None, max_k=18):
    unique_prompts = np.unique(prompt_labels)
    n_prompts = len(unique_prompts)
    D = node_features.shape[1]

    batch_x = np.zeros((n_prompts, max_k, D), dtype=np.float32)
    batch_mask = np.zeros((n_prompts, 1, max_k, max_k), dtype=bool)
    batch_y = np.full((n_prompts, max_k), -1, dtype=np.int64)
    batch_valid = np.zeros((n_prompts, max_k), dtype=bool)
    index_mapping = []

    for p_idx, p in enumerate(unique_prompts):
        orig_indices = np.where(prompt_labels == p)[0]
        k = min(len(orig_indices), max_k)
        batch_x[p_idx, :k] = node_features[orig_indices[:k]]
        batch_mask[p_idx, 0, :k, :k] = True
        batch_valid[p_idx, :k] = True
        if node_targets is not None:
            batch_y[p_idx, :k] = node_targets[orig_indices[:k]]
        index_mapping.append((p, orig_indices[:k]))

    return (
        torch.tensor(batch_x, dtype=torch.float32),
        torch.tensor(batch_mask, dtype=torch.bool),
        torch.tensor(batch_y, dtype=torch.long),
        torch.tensor(batch_valid, dtype=torch.bool),
        index_mapping
    )


class PromptGATLayer(nn.Module):
    def __init__(self, d_in, d_out, n_heads=4, dropout=0.15):
        super().__init__()
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.q_proj = nn.Linear(d_in, d_out, bias=False)
        self.k_proj = nn.Linear(d_in, d_out, bias=False)
        self.v_proj = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.skip_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.ffn = nn.Sequential(
            nn.Linear(d_out, d_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
            nn.Dropout(dropout)
        )
        self.dropout = dropout

    def forward(self, x, mask=None):
        B, K, _ = x.shape
        q = self.q_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, self.d_out)
        h = self.norm1(self.skip_proj(x) + self.out_proj(attn_out))
        out = self.norm2(h + self.ffn(h))
        return out


class PromptResidualGAT(nn.Module):
    def __init__(self, in_features=1296, hidden_dim=128, num_classes=18, n_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gat1 = PromptGATLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.gat2 = PromptGATLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.gat1(h, mask=mask)
        h = self.gat2(h, mask=mask)
        logits = self.classifier(h)
        return logits


class FinalSubmissionPipeline:
    """Production Hybrid Pipeline: EXP_36 (SVC + Gauss) + EXP_44-C Hybrid GAT + Hungarian."""

    def __init__(self, alpha=0.05, w_gauss=0.05, w_pool=0.35, device=None):
        self.alpha = alpha
        self.w_gauss = w_gauss
        self.w_pool = w_pool
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vec_char = None
        self.vec_word = None
        self.vec_shape = None
        self.vec_func = None
        self.clf_svc = None
        self.gauss_prm = None
        self.gat_model = None
        self.scaler = None
        self.is_fitted = False

    def fit(self, train_df, open_pool_df=None):
        print(f">> Fitting FinalSubmissionPipeline on {len(train_df)} official samples on {self.device}...")
        t0 = time.time()

        train_texts = [sanitize_text(t) for t in train_df["text"].values]
        y_train = np.array([LABEL_TO_IDX[m] for m in train_df["model"].values])
        prompts_train = train_df["prompt_id"].values

        all_texts = list(train_texts)
        all_y = list(y_train)
        sample_weights = [1.0] * len(train_texts)

        if open_pool_df is not None and len(open_pool_df) > 0:
            pool_texts = [sanitize_text(t) for t in open_pool_df["text"].values]
            pool_y = [LABEL_TO_IDX[m] for m in open_pool_df["model"].values]
            all_texts.extend(pool_texts)
            all_y.extend(pool_y)
            sample_weights.extend([self.w_pool] * len(pool_texts))

        all_y = np.array(all_y)
        sample_weights = np.array(sample_weights)

        # 4-View Vectorizers
        print(">> Fitting 4-view TF-IDF vectorizers...")
        self.vec_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=120000, sublinear_tf=True)
        self.vec_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, max_features=80000, sublinear_tf=True, token_pattern=r'(?u)\b\w+\b|[^\w\s]')
        self.vec_shape = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), min_df=3, max_features=40000, sublinear_tf=True)
        self.vec_func = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, max_features=30000, sublinear_tf=True)

        X_char = self.vec_char.fit_transform(all_texts)
        X_word = self.vec_word.fit_transform(all_texts)
        X_shape = self.vec_shape.fit_transform([to_casing_shape(t) for t in all_texts])
        X_func = self.vec_func.fit_transform([to_function_words(t) for t in all_texts])
        X_svc = hstack([X_char, X_word, X_shape, X_func]).tocsr()

        print(f">> Fitting LinearSVC(C=0.5) on shape {X_svc.shape}...")
        self.clf_svc = LinearSVC(C=0.5, class_weight='balanced', loss='squared_hinge', max_iter=2000, random_state=42)
        self.clf_svc.fit(X_svc, all_y, sample_weight=sample_weights)

        print(">> Extracting syntax, semantic, and stylometric features for Gaussian Head & GAT...")
        self.has_minilm = False
        try:
            X_syn_train = extract_minilm_l3(train_texts, device=self.device)
            self.has_minilm = True
        except Exception as e:
            print(f">> [Notice] MiniLM offline weights not detected ({e}). Using fallback.")
            X_syn_train = np.zeros((len(train_texts), 384), dtype=np.float32)

        self.has_deberta = False
        try:
            X_deb_train = extract_deberta_v3(train_texts, device=self.device)
            self.has_deberta = True
        except Exception as e:
            print(f">> [Notice] DeBERTa offline weights not detected ({e}). Using fallback.")
            X_deb_train = np.zeros((len(train_texts), 768), dtype=np.float32)

        X_sty_train = np.stack([extract_stylometry_row(t) for t in train_texts])
        self.has_gat = self.has_minilm or self.has_deberta

        # Shared-Prompt Gaussian Head
        X_dense_train = np.hstack([X_syn_train, X_sty_train]) if self.has_minilm else X_sty_train
        print(f">> Fitting Shared-Prompt Gaussian Head on {X_dense_train.shape}...")
        self.gauss_prm = gauss_fit(X_dense_train, y_train, prompts_train, cfg={'d': 128})

        # Hybrid Prompt-Residual GAT (1296d)
        if self.has_gat:
            X_hybrid_train = np.hstack([X_syn_train, X_deb_train, X_sty_train]).astype(np.float32)
            R_train = compute_prompt_residuals(X_hybrid_train, prompts_train)
            self.scaler = StandardScaler()
            R_train_s = self.scaler.fit_transform(R_train)

            bx_tr, bmask_tr, by_tr, bvalid_tr, _ = pack_prompt_graph_batch(R_train_s, prompts_train, y_train)
            bx_tr, bmask_tr, by_tr, bvalid_tr = bx_tr.to(self.device), bmask_tr.to(self.device), by_tr.to(self.device), bvalid_tr.to(self.device)

            print(f">> Training Hybrid GAT (1296d -> 128d, 4 heads) for 40 epochs on {len(np.unique(prompts_train))} prompt graphs...")
            torch.manual_seed(SEED)
            self.gat_model = PromptResidualGAT(in_features=1296, hidden_dim=128, num_classes=18, n_heads=4, dropout=0.2).to(self.device)
            optimizer = AdamW(self.gat_model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

            self.gat_model.train()
            for epoch in range(40):
                optimizer.zero_grad()
                logits = self.gat_model(bx_tr, bmask_tr)
                loss = F.cross_entropy(logits[bvalid_tr], by_tr[bvalid_tr])
                loss.backward()
                optimizer.step()
                scheduler.step()

            self.gat_model.eval()
        else:
            print(">> [Notice] Operating on 4-View LinearSVC + 144d Stylometry Gaussian Head + Hungarian Assignment.")

        self.is_fitted = True
        print(f">> Pipeline successfully fitted in {time.time() - t0:.1f}s.")
        return self

    def predict_proba(self, test_df):
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling predict.")

        test_texts = [sanitize_text(t) for t in test_df["text"].values]
        n_test = len(test_texts)
        if n_test == 0:
            return np.zeros((0, len(LABELS))), np.array([])

        # 1. LinearSVC scoring
        X_char = self.vec_char.transform(test_texts)
        X_word = self.vec_word.transform(test_texts)
        X_shape = self.vec_shape.transform([to_casing_shape(t) for t in test_texts])
        X_func = self.vec_func.transform([to_function_words(t) for t in test_texts])
        X_svc = hstack([X_char, X_word, X_shape, X_func]).tocsr()

        dec = self.clf_svc.decision_function(X_svc)
        if dec.ndim == 1:
            dec = np.vstack([-dec, dec]).T
        log_p_svc = dec - np.log(np.sum(np.exp(dec - np.max(dec, axis=1, keepdims=True)), axis=1, keepdims=True)) - np.max(dec, axis=1, keepdims=True)

        # 2. Prompt recovery
        test_clusters = recover_prompt_clusters(test_texts)

        # 3. Dense features extraction
        if self.has_minilm:
            try:
                X_syn_test = extract_minilm_l3(test_texts, device=self.device)
            except Exception:
                X_syn_test = np.zeros((n_test, 384), dtype=np.float32)
        else:
            X_syn_test = np.zeros((n_test, 384), dtype=np.float32)

        if self.has_deberta:
            try:
                X_deb_test = extract_deberta_v3(test_texts, device=self.device)
            except Exception:
                X_deb_test = np.zeros((n_test, 768), dtype=np.float32)
        else:
            X_deb_test = np.zeros((n_test, 768), dtype=np.float32)

        X_sty_test = np.stack([extract_stylometry_row(t) for t in test_texts])

        # 4. Gaussian Head scoring
        X_dense_test = np.hstack([X_syn_test, X_sty_test]) if self.has_minilm else X_sty_test
        log_p_gauss = gauss_scores(X_dense_test, test_clusters, self.gauss_prm)

        # Base EXP_36 blend
        log_blend_36 = log_p_svc + self.w_gauss * log_p_gauss
        p_exp36 = softmax(log_blend_36, axis=1)

        # 5. Hybrid Prompt-Residual GAT scoring if active
        if self.has_gat:
            X_hybrid_test = np.hstack([X_syn_test, X_deb_test, X_sty_test]).astype(np.float32)
            R_test = compute_prompt_residuals(X_hybrid_test, test_clusters)
            R_test_s = self.scaler.transform(R_test)

            bx_te, bmask_te, _, bvalid_te, mapping_te = pack_prompt_graph_batch(R_test_s, test_clusters)
            bx_te, bmask_te = bx_te.to(self.device), bmask_te.to(self.device)

            with torch.no_grad():
                logits_gat = self.gat_model(bx_te, bmask_te)
                probs_gat_tensor = F.softmax(logits_gat, dim=-1)

            p_gat = np.zeros((n_test, 18), dtype=np.float32)
            for p_idx, (_, orig_idx) in enumerate(mapping_te):
                k = len(orig_idx)
                p_gat[orig_idx] = probs_gat_tensor[p_idx, :k].cpu().numpy()

            p_final = (1.0 - self.alpha) * p_exp36 + self.alpha * p_gat
        else:
            p_final = p_exp36

        return p_final, test_clusters

    def predict(self, test_df):
        if len(test_df) == 0:
            return pd.DataFrame({"id": [], "model": []})

        probs, clusters = self.predict_proba(test_df)
        pred_indices = hungarian_assignment(probs, clusters)
        pred_labels = [IDX_TO_LABEL[idx] for idx in pred_indices]

        sub = pd.DataFrame({
            "id": test_df["id"].values,
            "model": pred_labels
        })
        return sub


def main():
    print("=" * 80)
    print("EXP_45: KAGGLE OFFLINE SUBMISSION EXECUTION")
    print("=" * 80)

    # 1. Discover Data Paths (Kaggle mount vs local repo)
    if os.path.exists("/kaggle/input/name-that-llm/test.csv"):
        data_dir = "/kaggle/input/name-that-llm"
        work_dir = "/kaggle/working"
    elif os.path.exists("/kaggle/input/name-the-llm/test.csv"):
        data_dir = "/kaggle/input/name-the-llm"
        work_dir = "/kaggle/working"
    elif os.path.exists("/kaggle/input"):
        # Search /kaggle/input for test.csv
        found_dir = None
        for root, dirs, files in os.walk("/kaggle/input"):
            if "test.csv" in files:
                found_dir = root
                break
        data_dir = found_dir if found_dir else "/kaggle/input/name-that-llm"
        work_dir = "/kaggle/working"
    else:
        # Local environment (safe against Jupyter NameError)
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            cwd = os.getcwd()
            if os.path.exists(os.path.join(cwd, "data", "test.csv")):
                base_dir = cwd
            elif os.path.exists(os.path.join(os.path.dirname(cwd), "data", "test.csv")):
                base_dir = os.path.dirname(cwd)
            else:
                base_dir = cwd
        data_dir = os.path.join(base_dir, "data")
        work_dir = os.path.join(base_dir, "submissions")

    os.makedirs(work_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    pool_path = os.path.join(data_dir, "external/open_pool.parquet")

    print(f"Data Directory: {data_dir}")
    print(f"Working Directory: {work_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    pool_df = pd.read_parquet(pool_path) if os.path.exists(pool_path) else None

    # 2. Fit and Predict
    pipeline = FinalSubmissionPipeline(alpha=0.05, w_gauss=0.05, w_pool=0.35)
    pipeline.fit(train_df, pool_df)

    print(f"\n>> Scoring {len(test_df)} test samples...")
    t0 = time.time()
    sub_df = pipeline.predict(test_df)
    print(f">> Inference finished in {time.time() - t0:.2f}s.")

    # 3. Output files for Kaggle submission
    out_parquet = os.path.join(work_dir, "submission.parquet")
    out_csv = os.path.join(work_dir, "submission.csv")
    sub_df.to_parquet(out_parquet, index=False)
    sub_df.to_csv(out_csv, index=False)
    print(f">> Saved: {out_parquet}")
    print(f">> Saved: {out_csv}")

    # Also ensure present in current working directory
    if not os.path.exists("submission.parquet"):
        sub_df.to_parquet("submission.parquet", index=False)
    if not os.path.exists("submission.csv"):
        sub_df.to_csv("submission.csv", index=False)

    print("\nSubmission Preview:")
    print(sub_df.head(10))
    print("=" * 80)
    print("SUBMISSION READY FOR KAGGLE")
    print("=" * 80)


if __name__ == "__main__":
    main()
