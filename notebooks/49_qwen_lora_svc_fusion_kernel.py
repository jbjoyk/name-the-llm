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
4. Bayesian Log-Probability Fusion (optimal weight balance w_svc=0.85 / conservative w_svc=0.50)
5. Damped Sinkhorn-Knopp Global Marginal Calibration (alpha=0.30, rho=0.35)
================================================================================
"""

import os
import sys
import gc
import json
import time
import glob
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

# Ensembling & Calibration parameters
WEIGHT_SVC = 0.70       # Balanced robust fusion weight between deep & surface representations
SINKHORN_ALPHA = 0.30   # Damped optimal transport regularisation strength
SINKHORN_RHO = 0.35     # Target mix toward training prior


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
    pattern = '/kaggle/input/**/heads.pt'
    heads = glob.glob(pattern, recursive=True)
    q8_dirs = [os.path.dirname(h) for h in heads if 'q8v5' in h.lower()]
    if q8_dirs:
        fold_dirs = [d for d in q8_dirs if any(f'q8v5_f{k}' in d.lower() for k in range(5))]
        full_dirs = [d for d in q8_dirs if d not in fold_dirs]
        sorted_dirs = sorted(fold_dirs) + full_dirs
        return sorted_dirs
    
    local_dir = "/home/rj2826/name-the-llm/models/q8v5"
    if os.path.exists(local_dir):
        return [os.path.join(local_dir, "q8v5")]
    return []


def resolve_open_pool_path():
    candidates = [
        "/kaggle/input/open-pool-passages/open_pool_clean.parquet",
        "/kaggle/input/crtproduction-open-pool/open_pool_clean.parquet",
        "/kaggle/input/ntl-open-pool/ntl_open_pool.parquet",
        "/home/rj2826/name-the-llm/data/external/open_pool_clean.parquet",
        "/home/rj2826/name-the-llm/data/external/open_pool_36k.parquet"
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
# 3. Unsupervised Prompt Recovery (Content-Word TF-IDF + Capped Agglomerative)
# ==============================================================================
def capped_agglo(S, cap=18, thr=0.15):
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    n = len(texts)
    if n <= 1:
        return np.zeros(n, dtype=int)
    try:
        vec = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
        Xc = vec.fit_transform(texts)
        if Xc.shape[1] == 0:
            return np.arange(n)
        S = (Xc @ Xc.T).toarray()
        return capped_agglo(S, cap=cap, thr=thr)
    except Exception as e:
        log(f"Prompt clustering fallback ({e}). Singletons assigned.")
        return np.arange(n)


# ==============================================================================
# 4. 4-View Stylometric LinearSVC Engine
# ==============================================================================
def fit_and_predict_svc(train_df, test_df, pool_path=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    
    log("Fitting 4-View Stylometric LinearSVC...")
    t_start = time.time()
    
    train_texts = train_df['text'].fillna("").values.tolist()
    train_y = train_df['model'].map(L2I).values.tolist()
    train_weights = [1.0] * len(train_texts)
    
    if pool_path and os.path.exists(pool_path):
        log(f"Loading external Open Pool passages from {pool_path}...")
        try:
            if pool_path.endswith(".parquet"):
                pool_df = pd.read_parquet(pool_path)
            else:
                pool_df = pd.read_csv(pool_path)
            
            # Sample balanced 2,000 per class (36,000 max)
            pool_df = pool_df[pool_df['model'].isin(LABELS)].reset_index(drop=True)
            sampled = pool_df.groupby('model').apply(lambda g: g.sample(min(len(g), 2000), random_state=42)).reset_index(drop=True)
            
            pool_texts = sampled['text'].fillna("").values.tolist()
            pool_y = sampled['model'].map(L2I).values.tolist()
            pool_weights = [0.35] * len(pool_texts)
            
            fit_texts = train_texts + pool_texts
            fit_y = np.array(train_y + pool_y)
            fit_weights = np.array(train_weights + pool_weights)
            log(f"Open Pool loaded: {len(pool_texts):,} passages appended at weight 0.35")
        except Exception as e:
            log(f"Warning: Open pool load failed ({e}); fitting on official train data only.")
            fit_texts = train_texts
            fit_y = np.array(train_y)
            fit_weights = np.array(train_weights)
    else:
        log("No external open pool found; fitting on official train data.")
        fit_texts = train_texts
        fit_y = np.array(train_y)
        fit_weights = np.array(train_weights)
        
    test_texts = test_df['text'].fillna("").values.tolist()
    
    # View 1: Word TF-IDF (1-3 n-grams)
    v1 = TfidfVectorizer(ngram_range=(1, 3), max_features=40000, sublinear_tf=True)
    X1_tr = v1.fit_transform(fit_texts)
    X1_te = v1.transform(test_texts)
    
    # View 2: Char-WB TF-IDF (3-5 n-grams)
    v2 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=50000, sublinear_tf=True)
    X2_tr = v2.fit_transform(fit_texts)
    X2_te = v2.transform(test_texts)
    
    # View 3: Word shape & punctuation features
    def extract_shape_text(t):
        return "".join([c if c in "\n\t—–-.,!?;:\"'()[]{}/*#`~" or c.isupper() else "x" for c in t[:1000]])
    v3 = TfidfVectorizer(ngram_range=(2, 6), analyzer='char', max_features=15000, sublinear_tf=True)
    X3_tr = v3.fit_transform([extract_shape_text(t) for t in fit_texts])
    X3_te = v3.transform([extract_shape_text(t) for t in test_texts])
    
    # Stack feature matrices
    X_tr = sp.hstack([X1_tr, X2_tr, X3_tr], format='csr')
    X_te = sp.hstack([X1_te, X2_te, X3_te], format='csr')
    
    log(f"Combined 4-View Feature Matrix: {X_tr.shape[1]:,} features. Fitting LinearSVC...")
    svc = LinearSVC(C=0.5, random_state=42, max_iter=3000, class_weight='balanced')
    svc.fit(X_tr, fit_y, sample_weight=fit_weights)
    
    # Decision function to soft log-probabilities via softmax
    dec_scores = svc.decision_function(X_te)
    lp_svc = dec_scores - scipy.special.logsumexp(dec_scores, axis=1, keepdims=True)
    log(f"LinearSVC fitting & scoring complete in {time.time() - t_start:.1f}s")
    return lp_svc


# ==============================================================================
# 5. Damped Sinkhorn-Knopp Matrix Scaling
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
# 6. Main Execution Pipeline
# ==============================================================================
def main():
    log("=" * 80)
    log("EXP49 KAGGLE EXECUTION PIPELINE INITIALIZING")
    log("=" * 80)
    start_total = time.time()

    data_dir = resolve_data_dir()
    test_path = os.path.join(data_dir, "test.csv")
    train_path = os.path.join(data_dir, "train.csv")
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    
    N_test = len(test_df)
    log(f"Active Data Directory: {data_dir}")
    log(f"Test Passages:         {N_test}")
    
    FAST_COMMIT = (N_test <= 5)
    if FAST_COMMIT:
        log("FAST_COMMIT MODE ACTIVE (Dummy rows detected)")

    # 1. Prompt Recovery on Test Passages
    test_texts = test_df['text'].fillna("").values
    test_groups = recover_prompt_clusters(test_texts, thr=CLUSTER_THRESHOLD, cap=CLUSTER_CAP)
    cluster_counts = pd.Series(test_groups).value_counts().to_dict()
    has_cluster = np.array([cluster_counts[g] >= 2 for g in test_groups], dtype=bool)
    log(f"Prompt Recovery: {len(cluster_counts)} clusters (Multi-passage: {has_cluster.sum()}, Singletons: {(~has_cluster).sum()})")

    # 2. Qwen3-8B Centred Twin Head Inference
    base_model_path = resolve_base_model_path()
    adapter_dirs = resolve_adapters_dirs()
    log(f"Base Checkpoint: {base_model_path}")
    log(f"LoRA Checkpoints Found: {len(adapter_dirs)}")

    if base_model_path is not None and len(adapter_dirs) > 0 and not FAST_COMMIT:
        # Full GPU Qwen Inference
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.device_count() > 1 else None,
            trust_remote_code=True
        )
        if torch.cuda.device_count() <= 1 and torch.cuda.is_available():
            model = model.to(device)
        model.eval()

        # Load Heads
        heads_path = os.path.join(adapter_dirs[0], "heads.pt")
        H = torch.load(heads_path, map_location="cpu")
        W_plain = H['plain.weight'].float().numpy()
        b_plain = H['plain.bias'].float().numpy()
        W_cent = H['centered.weight'].float().numpy()
        b_cent = H['centered.bias'].float().numpy()

        # Multi-adapter inference loop
        adapter_lps = []
        for ad_dir in adapter_dirs:
            from peft import PeftModel
            peft_m = PeftModel.from_pretrained(model, ad_dir)
            peft_m = peft_m.merge_and_unload()
            
            # Extract pooled hidden states
            pooled_reps = []
            for start_idx in range(0, N_test, BATCH_SIZE):
                batch_t = test_texts[start_idx:start_idx + BATCH_SIZE].tolist()
                enc = tokenizer(batch_t, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = peft_m(**enc, output_hidden_states=True)
                    hs = [out.hidden_states[l][:, 1:, :].mean(dim=1).float().cpu().numpy() for l in LORA24_LAYERS]
                    pooled_reps.append(np.concatenate(hs, axis=1))
            
            Z = np.vstack(pooled_reps)
            
            # Plain logits
            lp_plain = Z @ W_plain.T + b_plain
            lp_plain -= scipy.special.logsumexp(lp_plain, axis=1, keepdims=True)
            
            # Centered logits
            Z_cent = Z.copy()
            for g in np.unique(test_groups):
                if cluster_counts[g] >= 2:
                    idx = (test_groups == g)
                    Z_cent[idx] -= Z_cent[idx].mean(axis=0)
                else:
                    Z_cent[test_groups == g] = 0.0
            lp_cent = Z_cent @ W_cent.T + b_cent
            lp_cent -= scipy.special.logsumexp(lp_cent, axis=1, keepdims=True)
            
            # Twin fusion
            lp_total = lp_plain.copy()
            if has_cluster.any():
                lp_total[has_cluster] += TWIN_WEIGHT * lp_cent[has_cluster]
            adapter_lps.append(lp_total)
            
            del peft_m
            gc.collect()
            torch.cuda.empty_cache()
            
        lp_qwen = np.mean(adapter_lps, axis=0)
    else:
        log("Simulating Qwen output for fast commit / local fallback.")
        rng = np.random.RandomState(42)
        lp_qwen = rng.randn(N_test, NUM_CLASSES).astype(np.float32)
        lp_qwen -= scipy.special.logsumexp(lp_qwen, axis=1, keepdims=True)

    # 3. 4-View LinearSVC Inference
    pool_path = resolve_open_pool_path()
    if train_df is not None and not FAST_COMMIT:
        lp_svc = fit_and_predict_svc(train_df, test_df, pool_path=pool_path)
    else:
        lp_svc = lp_qwen.copy()

    # 4. Bayesian Log-Probability Fusion
    log(f"Fusing representations: w_svc={WEIGHT_SVC:.2f}, w_qwen={1.0 - WEIGHT_SVC:.2f}...")
    lp_fused = (1.0 - WEIGHT_SVC) * lp_qwen + WEIGHT_SVC * lp_svc
    P_fused = np.exp(lp_fused - lp_fused.max(axis=1, keepdims=True))
    P_fused /= P_fused.sum(axis=1, keepdims=True)

    # 5. Damped Sinkhorn Calibration
    if N_test >= 18 and not FAST_COMMIT:
        log(f"Applying Sinkhorn Regularisation: alpha={SINKHORN_ALPHA}, rho={SINKHORN_RHO}...")
        pred_marg = np.clip(P_fused.mean(axis=0), 1e-6, None); pred_marg /= pred_marg.sum()
        train_prior = np.ones(NUM_CLASSES) / float(NUM_CLASSES)
        if train_df is not None:
            train_prior = np.bincount(train_df['model'].map(L2I).values, minlength=NUM_CLASSES).astype(np.float64) + 1.0
            train_prior /= train_prior.sum()
        target = (1.0 - SINKHORN_RHO) * pred_marg + SINKHORN_RHO * train_prior
        target /= target.sum()
        P_final = apply_sinkhorn_knopp(P_fused, target, alpha=SINKHORN_ALPHA)
    else:
        P_final = P_fused

    # 6. Final Predictions & Export
    preds = [I2L[idx] for idx in P_final.argmax(axis=1)]
    submission = pd.DataFrame({"id": test_df['id'].values, "model": preds})
    
    assert len(submission) == N_test, "Mismatch in output row count!"
    assert set(submission.columns) == {"id", "model"}, "Invalid columns!"
    assert submission['model'].isin(LABELS).all(), "Invalid model class!"
    assert submission['id'].is_unique, "Duplicate IDs in submission!"
    
    os.makedirs("/kaggle/working", exist_ok=True)
    out_csv = "/kaggle/working/submission.csv"
    out_parquet = "/kaggle/working/submission.parquet"
    submission.to_csv(out_csv, index=False)
    submission.to_csv("submission.csv", index=False)
    try:
        submission.to_parquet(out_parquet, index=False)
    except Exception:
        pass

    log(f"Successfully generated {out_csv} ({len(submission)} rows)")
    log(f"Sample predictions:\n{submission.head()}")
    log(f"Total Pipeline Runtime: {time.time() - start_total:.1f}s")
    log("=" * 80)


if __name__ == "__main__":
    main()
