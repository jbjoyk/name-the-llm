"""
================================================================================
KAGGLE "NAME THAT LLM" COMPETITION: EXP49 STYLOMETRIC LINEARSVC FUSION KERNEL
================================================================================
Experiment: EXP49_stylometric_svc_fusion
Architecture:
1. Qwen3-8B LoRA24 (24 layers, multi-layer hidden pooling across 8 layers)
2. Unsupervised Content-Word Prompt Recovery + Centred Twin Head (twin_weight=0.35)
3. 4-View Multi-Granularity LinearSVC (Word 1-3, Char 3-5, Shape, Function Words)
   trained on official train.csv (w=1.0) + Clean Open Pool passages (w=0.35)
4. Bayesian Log-Probability Fusion (optimal fixed weight w_svc=0.850, w_qwen=0.150)
5. Damped Sinkhorn-Knopp Global Marginal Calibration (alpha=0.30, rho=0.35)
Fault-Tolerance:
- Fully self-contained: Gracefully degrades to EXP47 if external data/SVC fails.
- Never throws an unhandled exception: Guarantees submission.csv is always generated.
Hardware Target: Kaggle GPU T4 x2 or P100 / A100 (runs offline, <= 25 mins)
================================================================================
"""

import os
import sys
import gc
import json
import time
import glob
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.special
import torch

# ==============================================================================
# 1. Candidate LLM Class Definitions & Hyperparameters
# ==============================================================================
LABELS = [
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'glm-5.3', 'glm-5.3-flash',
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'kimi-k3', 'qwen3.8-max'
]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for i, l in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# Architecture configuration
LORA24_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]
KEEP_LAYERS = 24
MAX_SEQ_LEN = 384
BATCH_SIZE = 16

# Prompt recovery and twin head parameters
CLUSTER_THRESHOLD = 0.15
CLUSTER_CAP = 18
TWIN_WEIGHT = 0.35

# Ensembling & Calibration parameters (Validated via Selection-Safe 5-Fold Nested CV)
WEIGHT_SVC = 0.850       # Validated optimal weight for 4-View LinearSVC
WEIGHT_QWEN = 0.150      # Weight for Qwen3-8B LoRA24 Centred Twin Head
SINKHORN_ALPHA = 0.30    # Damped optimal transport regularisation strength
SINKHORN_RHO = 0.35      # Target mix toward training prior


def log(*args):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)


# ==============================================================================
# 2. Path Discovery & Strict Pre-Flight Verification
# ==============================================================================
def resolve_data_dir():
    candidates = [
        "/kaggle/input/competitions/name-that-llm",
        "/kaggle/input/name-that-llm",
        "/kaggle/input/name-the-llm",
        "data"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "test.csv")):
            return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if "test.csv" in files:
            return root
    raise FileNotFoundError("test.csv not found in any standard Kaggle or local path!")


def resolve_base_model_path():
    """Finds Qwen3-8B-Base model checkpoint."""
    cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
    candidates = [os.path.dirname(x) for x in cfgs if 'qwen' in x.lower() and '8b' in x.lower() and 'base' in x.lower()]
    if not candidates:
        candidates = [os.path.dirname(x) for x in cfgs if 'qwen' in x.lower() and '8b' in x.lower()]
    if candidates:
        return candidates[0]
    
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots")
    if os.path.exists(hf_cache):
        snaps = os.listdir(hf_cache)
        if snaps:
            return os.path.join(hf_cache, snaps[0])

    return None


def resolve_adapters_dirs():
    """Finds all fold and full-fit adapters for q8v5 in nlm-l24-adapters-v5."""
    pattern = '/kaggle/input/**/heads.pt'
    heads = glob.glob(pattern, recursive=True)
    
    q8_dirs = [os.path.dirname(h) for h in heads if 'q8v5' in h.lower()]
    if q8_dirs:
        fold_dirs = [d for d in q8_dirs if any(f'q8v5_f{k}' in d.lower() for k in range(5))]
        full_dirs = [d for d in q8_dirs if d not in fold_dirs]
        sorted_dirs = sorted(fold_dirs) + full_dirs
        return sorted_dirs

    if heads:
        return sorted([os.path.dirname(h) for h in heads])

    local_candidates = [
        "models/q8v5/q8v5",
        "models/q8v5",
        "/tmp/q8_test/q8v5"
    ]
    for lc in local_candidates:
        if os.path.exists(os.path.join(lc, "heads.pt")):
            return [lc]

    return None


def resolve_open_pool_path():
    """Finds Open Pool dataset parquet or csv file on Kaggle or locally."""
    candidates = [
        "/kaggle/input/open-pool-passages/open_pool_clean.parquet",
        "/kaggle/input/open-pool-passages/open_pool_36k.parquet",
        "/kaggle/input/open-pool-passages/open_pool_clean.csv",
        "/kaggle/input/crtproduction-open-pool/open_pool_clean.parquet",
        "/kaggle/input/crtproduction-open-pool/open_pool_36k.parquet",
        "/kaggle/input/ntl-open-pool/ntl_open_pool.parquet",
        "/kaggle/input/ntl-open-pool/open_pool_clean.parquet",
        "data/external/open_pool_36k.parquet",
        "data/external/open_pool_clean.parquet",
        "data/external/open_pool_clean.csv",
        "data/external/open_pool/open_pool_clean.parquet",
        "data/external/open_pool/open_pool_clean.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if "open_pool" in f.lower() and (f.endswith(".parquet") or f.endswith(".csv")):
                return os.path.join(root, f)
    return None


# ==============================================================================
# 3. Unsupervised Prompt Recovery (Lance-Williams Capped Agglomerative)
# ==============================================================================
def capped_agglo(S, cap=18, thr=0.15):
    """
    Lance-Williams Average Linkage Agglomerative Clustering:
    - Never merges clusters past maximum size `cap` (18).
    - Stops merging when maximum pairwise cosine similarity falls below `thr` (0.15).
    - Strict unsupervised clustering: Zero labels, zero prompt IDs, zero test leakage.
    """
    n = len(S)
    A = S.astype(np.float32).copy()
    np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32)
    alive = np.ones(n, bool)
    lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf)
        k = int(np.argmax(M))
        i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new
        A[:, i] = new
        A[i, i] = -np.inf
        A[j] = -np.inf
        A[:, j] = -np.inf
        size[i] += size[j]
        alive[j] = False
        lab[lab == j] = i
    return lab


def recover_prompt_clusters(texts, thr=0.15, cap=18):
    """
    Recovers prompt clusters from raw test texts using content-word TF-IDF.
    Falls back gracefully to individual singletons on small commit test sets.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    if n < 50:
        log("Notice: Test set size < 50 rows (commit run). Assigning singletons.")
        return np.arange(n)

    try:
        t0 = time.time()
        v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
        Xc = v.fit_transform(texts)
        if Xc.shape[1] == 0:
            return np.arange(n)
        S = (Xc @ Xc.T).toarray()
        groups = capped_agglo(S, cap=cap, thr=thr)
        log(f"Prompt recovery completed in {time.time() - t0:.2f}s:")
        cluster_sizes = pd.Series(groups).value_counts()
        n_clusters = len(cluster_sizes)
        n_multi = (cluster_sizes >= 2).sum()
        n_single = (cluster_sizes == 1).sum()
        log(f"  Total clusters: {n_clusters} ({n_multi} multi-passage, {n_single} singletons)")
        log(f"  Passages in multi-passage clusters: {cluster_sizes[cluster_sizes >= 2].sum()} / {n}")
        return groups
    except Exception as e:
        log(f"Warning: Prompt clustering fallback encountered: {e}. Defaulting to singletons.")
        return np.arange(n)


# ==============================================================================
# 4. LoRA24 In-Memory Prediction Engine with Centred Twin Head Fusion
# ==============================================================================
@torch.no_grad()
def predict_qwen_lora24_with_twin(base_path, adapter_dirs, texts, groups, twin_w=0.35, max_len=384, bs=16):
    """
    Executes multi-adapter LoRA24 inference with Centred Twin Head:
    1. Loads truncated base model (keep=24 layers) ONCE into GPU VRAM in fp16.
    2. Featurizes passages across target layers [2, 4, 6, 8, 12, 16, 20, 24].
    3. For each adapter:
       - Merges LoRA weights into base model linear layers.
       - Extracts LayerNorm-pooled features Z (shape: N x 32768).
       - Evaluates Plain Head: LP_plain = log_softmax(Z @ W_plain.T + b_plain).
       - Centers features per recovered prompt cluster:
         Z_centered = Z - mean(Z_cluster) for clusters of size >= 2
       - Evaluates Centred Head: LP_centered = log_softmax(Z_centered @ W_centered.T + b_centered).
       - Unmerges LoRA weights cleanly.
    4. Averages plain and centered log-probabilities across all fold adapters.
    5. Fuses evidence:
       LP_total = LP_plain + twin_w * LP_centered (for passages with cluster_size >= 2)
       LP_total = LP_plain (for singletons / unclustered passages)
    """
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    from safetensors.torch import load_file

    log(f"Initializing Tokenizer from {base_path}...")
    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    log(f"Loading Base Qwen3-8B Model truncated to {KEEP_LAYERS} layers ({dtype})...")
    cfg = AutoConfig.from_pretrained(base_path)
    tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    tc.num_hidden_layers = KEEP_LAYERS
    tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None:
        tc.layer_types = tc.layer_types[:KEEP_LAYERS]

    model = AutoModel.from_pretrained(
        base_path,
        config=cfg,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    ).eval()

    log(f"Tokenizing {len(texts)} texts (max_length={max_len})...")
    encodings = tok(list(texts), truncation=True, max_length=max_len)['input_ids']
    order = np.argsort([len(x) for x in encodings])
    n_samples = len(texts)

    mods = dict(model.named_modules())
    first_param_device = next(model.parameters()).device

    P_plain_acc = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    P_centered_acc = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)

    # Determine which passages belong to valid clusters (size >= 2)
    cluster_sizes = pd.Series(groups).value_counts().to_dict()
    has_cluster = np.array([cluster_sizes[g] >= 2 for g in groups], dtype=bool)
    n_clustered = int(has_cluster.sum())
    log(f"Clustering Mask: {n_clustered} clustered passages (twin applied), {n_samples - n_clustered} singletons (plain fallback).")

    def featurize_passages(norms, hid_dim):
        Z = np.zeros((n_samples, hid_dim * len(LORA24_LAYERS)), dtype=np.float32)
        for i in range(0, len(order), bs):
            batch_indices = order[i:i + bs]
            max_batch_len = max(len(encodings[j]) for j in batch_indices)

            input_ids = torch.full((len(batch_indices), max_batch_len), tok.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch_indices), max_batch_len), dtype=torch.long)

            for r, j in enumerate(batch_indices):
                l_j = len(encodings[j])
                input_ids[r, :l_j] = torch.tensor(encodings[j])
                attention_mask[r, :l_j] = 1

            input_ids = input_ids.to(first_param_device)
            attention_mask = attention_mask.to(first_param_device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            hidden_states = out.hidden_states

            mask = attention_mask.unsqueeze(-1).float()
            mask[:, 0] = 0.0
            mask_sum = mask.sum(1) + 1e-9

            pooled_layers = []
            for l_idx, nm in zip(LORA24_LAYERS, norms):
                hs = hidden_states[l_idx].float()
                hs_device = hs.device
                pooled = (hs * mask.to(hs_device)).sum(1) / mask_sum.to(hs_device)
                normed = nm(pooled.cpu())
                pooled_layers.append(normed)

            Z[batch_indices] = torch.cat(pooled_layers, dim=-1).numpy()

        return Z

    n_adapters = len(adapter_dirs)
    log(f"Evaluating across {n_adapters} LoRA adapter checkpoints with twin head...")

    for a_idx, ad in enumerate(adapter_dirs):
        log(f"  [{a_idx + 1}/{n_adapters}] Loading LoRA adapter from {ad}...")
        ad_path = os.path.join(ad, "adapter") if os.path.exists(os.path.join(ad, "adapter", "adapter_config.json")) else ad
        acfg = json.load(open(os.path.join(ad_path, "adapter_config.json")))
        scale = acfg['lora_alpha'] / acfg['r']
        
        weights_file = os.path.join(ad_path, "adapter_model.safetensors")
        if not os.path.exists(weights_file):
            weights_file = os.path.join(ad_path, "adapter_model.bin")
        
        if weights_file.endswith(".safetensors"):
            sd = load_file(weights_file)
        else:
            sd = torch.load(weights_file, map_location="cpu")

        # Merge LoRA weights
        deltas = []
        n_merged = 0
        for k, v in sd.items():
            if '.lora_A.' not in k:
                continue
            name = k.replace('base_model.model.', '', 1).split('.lora_A.')[0]
            if name not in mods and ('model.' + name) in mods:
                name = 'model.' + name
            if name in mods:
                B = sd[k.replace('.lora_A.', '.lora_B.')]
                W = mods[name].weight
                delta_W = ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)
                W += delta_W
                deltas.append((W, B, v))
                n_merged += 1

        log(f"    Merged {n_merged} low-rank matrices (scale={scale:.2f}).")

        # Load trained readout heads
        heads_path = os.path.join(ad, "heads.pt")
        if os.path.exists(heads_path):
            H = torch.load(heads_path, map_location="cpu")
        else:
            H = dict(load_file(os.path.join(ad, "heads.safetensors")))

        hid_dim = H['norms.0.weight'].shape[0]
        norms = [torch.nn.LayerNorm(hid_dim) for _ in LORA24_LAYERS]
        for i, nm in enumerate(norms):
            nm.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})

        # Featurize passages with merged adapter
        t0 = time.time()
        Z = featurize_passages(norms, hid_dim)
        log(f"    Featurized {len(texts)} samples into shape {Z.shape} in {time.time() - t0:.1f}s.")

        # Unmerge LoRA before next iteration
        for W, B, A_ in deltas:
            W -= ((B.to(W.device, torch.float32) @ A_.to(W.device, torch.float32)) * scale).to(W.dtype)

        del deltas, sd
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 1. Plain Head Evaluation
        W_plain = H['plain.weight'].float().numpy()
        b_plain = H['plain.bias'].float().numpy()
        logits_plain = Z @ W_plain.T + b_plain
        lp_plain = logits_plain - scipy.special.logsumexp(logits_plain, axis=1, keepdims=True)
        P_plain_acc += lp_plain / n_adapters

        # 2. Centred Head Evaluation (only on clustered rows)
        if 'centered.weight' in H:
            Z_cent = Z.copy()
            for g in np.unique(groups):
                idx = (groups == g)
                if cluster_sizes[g] >= 2:
                    Z_cent[idx] -= Z_cent[idx].mean(axis=0)
                else:
                    Z_cent[idx] = 0.0

            W_cent = H['centered.weight'].float().numpy()
            b_cent = H['centered.bias'].float().numpy()
            logits_cent = Z_cent @ W_cent.T + b_cent
            lp_cent = logits_cent - scipy.special.logsumexp(logits_cent, axis=1, keepdims=True)
            P_centered_acc += lp_cent / n_adapters
        else:
            log("    Warning: centered.weight not found in heads.pt; twin evaluation skipped.")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. Fuse Plain Head + Centred Twin Head Evidence
    LP_total = P_plain_acc.copy()
    if 'centered.weight' in H and n_clustered > 0:
        log(f"Fusing Centred Twin evidence (twin_weight={twin_w:.2f}) on {n_clustered} clustered rows...")
        LP_total[has_cluster] += twin_w * P_centered_acc[has_cluster]
    else:
        log("Notice: Plain head only active (zero twin head fusion).")

    # Normalized probabilities
    probs = np.exp(LP_total - LP_total.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, LP_total, P_plain_acc, P_centered_acc


# ==============================================================================
# 5. 4-View Stylometric LinearSVC Engine
# ==============================================================================
FW_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())


def shape_transform(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m.group(0)[0].isupper() else "x", text)


def func_transform(text):
    tokens = re.findall(r"\w+|[^\w\s]", text)
    transformed = [w.lower() if w.lower() in FW_WORDS or not w.isalnum() else "CONTENT" for w in tokens]
    return " ".join(transformed)


def fit_and_predict_svc(train_df, test_df, pool_path=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    
    log("Fitting 4-View Stylometric LinearSVC...")
    t_start = time.time()
    
    train_texts = train_df['text'].fillna("").astype(str).tolist()
    train_y = train_df['model'].map(L2I).values.tolist()
    train_weights = [1.0] * len(train_texts)
    
    fit_texts = train_texts
    fit_y = np.array(train_y, dtype=np.int64)
    fit_weights = np.array(train_weights, dtype=np.float32)
    
    if pool_path and os.path.exists(pool_path):
        log(f"Attempting to load Open Pool passages from {pool_path}...")
        try:
            if pool_path.endswith(".parquet"):
                pool_df = pd.read_parquet(pool_path)
            else:
                pool_df = pd.read_csv(pool_path)
            
            if 'text' not in pool_df.columns and 'response' in pool_df.columns:
                pool_df['text'] = pool_df['response']
            if 'model' not in pool_df.columns and 'label' in pool_df.columns:
                pool_df['model'] = pool_df['label']

            pool_df = pool_df[pool_df['model'].isin(LABELS)].copy()

            # Balanced sample up to 2,000 rows per class
            sampled_chunks = []
            for _, grp in pool_df.groupby('model'):
                n_sample = min(len(grp), 2000)
                sampled_chunks.append(grp.sample(n=n_sample, random_state=42))
            
            if sampled_chunks:
                sampled_pool = pd.concat(sampled_chunks, ignore_index=True)
                pool_texts = sampled_pool['text'].fillna("").astype(str).tolist()
                pool_y = sampled_pool['model'].map(L2I).values.tolist()
                pool_weights = [0.35] * len(pool_texts)
                
                fit_texts = train_texts + pool_texts
                fit_y = np.array(train_y + pool_y, dtype=np.int64)
                fit_weights = np.array(train_weights + pool_weights, dtype=np.float32)
                log(f"Open Pool loaded: {len(pool_texts):,} passages appended at sample weight 0.35.")
        except Exception as e:
            log(f"Warning: Open pool load failed ({e}); fitting on official train data only.")
            fit_texts = train_texts
            fit_y = np.array(train_y, dtype=np.int64)
            fit_weights = np.array(train_weights, dtype=np.float32)
    else:
        log("No external open pool found; fitting on official train data.")
        
    test_texts = test_df['text'].fillna("").astype(str).tolist()
    
    views_config = [
        ("char", 3, 5, 80000, 1.0),
        ("word", 1, 3, 60000, 0.7),
        ("shape", 2, 6, 30000, 0.5),
        ("function", 1, 3, 30000, 0.5),
    ]
    
    A_tr, B_te = [], []
    for view, lo, hi, mf, wv in views_config:
        prep = shape_transform if view == "shape" else func_transform if view == "function" else (lambda s: s)
        an = "char" if view == "shape" else "word" if view == "function" else view
        o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=3, max_features=mf, dtype=np.float32)
        if an == "word":
            o["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**o)
        tr_prep = [prep(t) for t in fit_texts]
        te_prep = [prep(t) for t in test_texts]
        X_tr_v = vec.fit_transform(tr_prep) * wv
        X_te_v = vec.transform(te_prep) * wv
        A_tr.append(X_tr_v)
        B_te.append(X_te_v)
        
    X_tr = sp.hstack(A_tr, format='csr')
    X_te = sp.hstack(B_te, format='csr')
    log(f"Combined 4-View Feature Matrix: {X_tr.shape[1]:,} features. Fitting LinearSVC...")
    
    clf = LinearSVC(C=0.5, class_weight='balanced', random_state=42, max_iter=3000)
    clf.fit(X_tr, fit_y, sample_weight=fit_weights)
    
    dec = clf.decision_function(X_te)
    lp_svc = dec - scipy.special.logsumexp(dec, axis=1, keepdims=True)
    log(f"LinearSVC fitting & scoring complete in {time.time() - t_start:.1f}s")
    return lp_svc


# ==============================================================================
# 6. Damped Sinkhorn-Knopp Matrix Calibration
# ==============================================================================
def apply_sinkhorn_knopp(P, target_marginal, alpha=0.30, max_iter=100, eps=1e-12):
    if alpha <= 0.0 or len(P) < 18:
        return P.copy()
    N = len(P)
    Q = P.copy()
    target_counts = N * target_marginal[None, :]
    for _ in range(max_iter):
        col_sums = np.clip(Q.sum(axis=0, keepdims=True), eps, None)
        Q *= (target_counts / col_sums)
        row_sums = np.clip(Q.sum(axis=1, keepdims=True), eps, None)
        Q /= row_sums
    log_P = np.log(np.clip(P, eps, 1.0))
    log_Q = np.log(np.clip(Q, eps, 1.0))
    log_blended = (1.0 - alpha) * log_P + alpha * log_Q
    Q_out = np.exp(log_blended - log_blended.max(axis=1, keepdims=True))
    Q_out /= Q_out.sum(axis=1, keepdims=True)
    return Q_out


# ==============================================================================
# 7. Main Standalone Execution Pipeline
# ==============================================================================
def main():
    print("=" * 80)
    print("EXP49: QWEN3-8B CENTRED TWIN + 4-VIEW LINEARSVC FUSION KERNEL")
    print("=" * 80)
    start_total = time.time()

    work_dir = "/kaggle/working" if os.path.exists("/kaggle") else "submissions"
    os.makedirs(work_dir, exist_ok=True)

    # 1. Resolve Data Paths
    data_dir = resolve_data_dir()
    test_path = os.path.join(data_dir, "test.csv")
    train_path = os.path.join(data_dir, "train.csv")
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    
    n_test = len(test_df)
    log(f"Competition Data: {data_dir} ({n_test} test rows)")
    log(f"Candidate Class Vocabulary ({NUM_CLASSES} classes): {LABELS}")

    # 2. Recover Prompt Clusters
    test_texts = test_df["text"].fillna("").astype(str).values
    recovered_groups = recover_prompt_clusters(test_texts, thr=CLUSTER_THRESHOLD, cap=CLUSTER_CAP)

    # 3. Predict with Qwen3-8B LoRA24 Centred Twin Head (EXP47 Foundation)
    lp_qwen = None
    try:
        base_model_path = resolve_base_model_path()
        adapter_dirs = resolve_adapters_dirs()
        log(f"Base Model Checkpoint: {base_model_path if base_model_path else 'NOT FOUND'}")
        log(f"LoRA Adapters Found:   {len(adapter_dirs) if adapter_dirs else 0}")
        if base_model_path is not None and adapter_dirs is not None and len(adapter_dirs) > 0:
            probs_qwen, lp_qwen, lp_plain, lp_cent = predict_qwen_lora24_with_twin(
                base_path=base_model_path,
                adapter_dirs=adapter_dirs,
                texts=test_texts,
                groups=recovered_groups,
                twin_w=TWIN_WEIGHT,
                max_len=MAX_SEQ_LEN,
                bs=BATCH_SIZE
            )
            log("Qwen3-8B LoRA24 inference completed successfully.")
        else:
            log("Warning: Base model or adapters not resolved.")
    except Exception as e:
        log(f"Warning: Qwen3-8B LoRA24 inference encountered exception: {e}")
        lp_qwen = None

    # 4. Predict with 4-View Stylometric LinearSVC
    lp_svc = None
    try:
        if train_df is not None:
            pool_path = resolve_open_pool_path()
            log(f"Open Pool Dataset: {pool_path if pool_path else 'Not attached (using train.csv only)'}")
            lp_svc = fit_and_predict_svc(train_df, test_df, pool_path=pool_path)
            log("4-View LinearSVC inference completed successfully.")
        else:
            log("Warning: train.csv not found; skipping LinearSVC.")
    except Exception as e:
        log(f"Warning: LinearSVC inference encountered exception: {e}")
        lp_svc = None

    # 5. Evidence Fusion & Graceful Degradation
    if lp_qwen is not None and lp_svc is not None:
        log(f"Fusing representations: w_svc={WEIGHT_SVC:.3f}, w_qwen={WEIGHT_QWEN:.3f}...")
        lp_fused = WEIGHT_QWEN * lp_qwen + WEIGHT_SVC * lp_svc
    elif lp_qwen is not None:
        log("NOTICE: Falling back to 100% Qwen3-8B Centred Twin Head (EXP47 verified baseline: 0.620 LB).")
        lp_fused = lp_qwen
    elif lp_svc is not None:
        log("NOTICE: Falling back to 100% 4-View Stylometric LinearSVC.")
        lp_fused = lp_svc
    else:
        log("CRITICAL FALLBACK: Neither model succeeded. Generating deterministic baseline predictions.")
        rng = np.random.RandomState(42)
        lp_fused = rng.randn(n_test, NUM_CLASSES).astype(np.float32)

    probs = np.exp(lp_fused - lp_fused.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    # 6. Damped Sinkhorn Regularisation
    try:
        if n_test >= 18:
            log(f"Applying Sinkhorn Regularisation: alpha={SINKHORN_ALPHA}, rho={SINKHORN_RHO}...")
            pred_marg = np.clip(probs.mean(axis=0), 1e-6, None)
            pred_marg /= pred_marg.sum()
            train_prior = np.ones(NUM_CLASSES) / float(NUM_CLASSES)
            if train_df is not None:
                train_prior = np.bincount(train_df['model'].map(L2I).values, minlength=NUM_CLASSES).astype(np.float64) + 1.0
                train_prior /= train_prior.sum()
            target = (1.0 - SINKHORN_RHO) * pred_marg + SINKHORN_RHO * train_prior
            target /= target.sum()
            P_final = apply_sinkhorn_knopp(probs, target, alpha=SINKHORN_ALPHA)
        else:
            log("Test set < 18 rows; skipping Sinkhorn regularisation.")
            P_final = probs
    except Exception as e:
        log(f"Warning: Sinkhorn regularisation encountered exception ({e}); using unregularised probabilities.")
        P_final = probs

    # 7. Generate Predictions & Guaranteed File Writing
    pred_indices = np.argmax(P_final, axis=1)
    pred_labels = [I2L[idx] for idx in pred_indices]

    sub_df = pd.DataFrame({
        "id": test_df["id"].values,
        "model": pred_labels
    })

    assert len(sub_df) == n_test, f"Row count mismatch: expected {n_test}, got {len(sub_df)}"
    assert set(sub_df.columns) == {"id", "model"}, "Columns mismatch!"
    assert sub_df["model"].isin(LABELS).all(), "Invalid model class!"
    assert sub_df["id"].is_unique, "Duplicate IDs in submission!"

    # Save to standard Kaggle output paths
    out_parquet = os.path.join(work_dir, "submission.parquet")
    out_csv = os.path.join(work_dir, "submission.csv")
    sub_df.to_csv(out_csv, index=False)
    log(f"Saved: {out_csv}")

    try:
        sub_df.to_parquet(out_parquet, index=False)
        log(f"Saved: {out_parquet}")
    except Exception as e:
        log(f"Parquet export skipped ({e}); CSV generated cleanly.")

    # Mirror to local directory if needed
    if not os.path.exists("submission.csv"):
        sub_df.to_csv("submission.csv", index=False)
    if not os.path.exists("submission.parquet"):
        try:
            sub_df.to_parquet("submission.parquet", index=False)
        except Exception:
            pass

    print("\nSubmission Preview:")
    print(sub_df.head(10))
    print(f"\nTotal Pipeline Runtime: {time.time() - start_total:.1f}s")
    print("=" * 80)
    print("EXP49 SUBMISSION READY FOR KAGGLE EVALUATOR")
    print("=" * 80)


if __name__ == "__main__":
    main()
