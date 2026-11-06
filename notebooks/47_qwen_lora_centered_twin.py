"""
================================================================================
KAGGLE "NAME THAT LLM" COMPETITION: EXP47 QWEN3-8B CENTRED TWIN HEAD KERNEL
================================================================================
Experiment: EXP47_qwen3_8b_centered_twin
Lineage: The Usual Suspects / Dhruv Tibarewal / Sovereign Gold Apex
Baseline: EXP46A (Public LB: 0.606)
Addition: Unsupervised Content-Word Prompt Recovery + Centred Twin Head (twin_weight=0.35)
Hardware: 2x T4 GPU or 1x P100 / A100
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
import torch
import scipy.special

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

    local_candidates = [
        "models/q8v5/q8v5",
        "models/q8v5",
        "/tmp/q8_test/q8v5"
    ]
    for lc in local_candidates:
        if os.path.exists(os.path.join(lc, "heads.pt")):
            return [lc]

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
        # Filter content words: ignore English stopwords, min_df=2, max_df=0.05
        v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
        Xc = v.fit_transform(texts)
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

    log(f"Loading Base Qwen3-8B Model truncated to {KEEP_LAYERS} layers in float16...")
    cfg = AutoConfig.from_pretrained(base_path)
    tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    tc.num_hidden_layers = KEEP_LAYERS
    tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None:
        tc.layer_types = tc.layer_types[:KEEP_LAYERS]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        base_path,
        config=cfg,
        torch_dtype=torch.float16,
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
                idx = groups == g
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
# 5. Main Standalone Execution
# ==============================================================================
def main():
    print("=" * 80)
    print("EXP47: QWEN3-8B CENTRED TWIN HEAD ATTRIBUTION KERNEL")
    print("=" * 80)

    work_dir = "/kaggle/working" if os.path.exists("/kaggle") else "submissions"
    os.makedirs(work_dir, exist_ok=True)

    # 1. Resolve Data Paths
    data_dir = resolve_data_dir()
    test_path = os.path.join(data_dir, "test.csv")
    test_df = pd.read_csv(test_path)
    n_test = len(test_df)

    log(f"Competition Data: {data_dir} ({n_test} test rows)")
    log(f"Candidate Class Vocabulary ({NUM_CLASSES} classes): {LABELS}")

    # 2. Resolve Base Model and Adapters
    base_model_path = resolve_base_model_path()
    adapter_dirs = resolve_adapters_dirs()

    print("\n" + "=" * 80)
    print("PRE-FLIGHT COMPONENT VERIFICATION")
    print("=" * 80)
    print(f"[Base Model Checkpoint]:  {base_model_path if base_model_path else 'MISSING'}")
    print(f"[LoRA Adapters Dataset]:  {len(adapter_dirs) if adapter_dirs else 0} adapters found")
    if adapter_dirs:
        for ad in adapter_dirs:
            print(f"  - {ad}")
    print("=" * 80 + "\n")

    if base_model_path is None:
        raise FileNotFoundError(
            "CRITICAL: Qwen3-8B-Base model was not found!\n"
            "Please attach the Kaggle Model: 'qwen-lm/qwen-3' (Variation: 8b-base / Transformers).\n"
            "Execution halted without fallback to prevent silent failure."
        )

    if adapter_dirs is None or len(adapter_dirs) == 0:
        raise FileNotFoundError(
            "CRITICAL: LoRA24 Adapters dataset was not found!\n"
            "Please attach the Kaggle Dataset: 'dhruvtibarewal/nlm-l24-adapters-v5'.\n"
            "Execution halted without fallback to prevent silent failure."
        )

    # 3. Recover Prompt Clusters
    test_texts = test_df["text"].fillna("").astype(str).values
    recovered_groups = recover_prompt_clusters(test_texts, thr=CLUSTER_THRESHOLD, cap=CLUSTER_CAP)

    # 4. Predict with LoRA24 + Centred Twin Head
    t0 = time.time()
    probs, lp_total, lp_plain, lp_cent = predict_qwen_lora24_with_twin(
        base_path=base_model_path,
        adapter_dirs=adapter_dirs,
        texts=test_texts,
        groups=recovered_groups,
        twin_w=TWIN_WEIGHT,
        max_len=MAX_SEQ_LEN,
        bs=BATCH_SIZE
    )
    inference_time = time.time() - t0
    log(f"Inference completed in {inference_time:.2f}s ({n_test / max(inference_time, 0.001):.1f} rows/s).")

    # 5. Generate Predictions & Output Formats
    pred_indices = np.argmax(probs, axis=1)
    pred_labels = [I2L[idx] for idx in pred_indices]

    sub_df = pd.DataFrame({
        "id": test_df["id"].values,
        "model": pred_labels
    })

    # Save CSV and Parquet
    out_parquet = os.path.join(work_dir, "submission.parquet")
    out_csv = os.path.join(work_dir, "submission.csv")
    sub_df.to_csv(out_csv, index=False)
    log(f"Saved: {out_csv}")

    try:
        sub_df.to_parquet(out_parquet, index=False)
        log(f"Saved: {out_parquet}")
    except Exception as e:
        log(f"Parquet export skipped ({e}); CSV generated cleanly.")

    if not os.path.exists("submission.csv"):
        sub_df.to_csv("submission.csv", index=False)
    if not os.path.exists("submission.parquet"):
        try:
            sub_df.to_parquet("submission.parquet", index=False)
        except Exception:
            pass

    # Report diagnostic preview
    plain_indices = np.argmax(lp_plain, axis=1)
    n_changed = int((pred_indices != plain_indices).sum())
    log(f"Twin Head Effect: {n_changed} / {n_test} predictions updated from plain baseline.")

    print("\nSubmission Preview:")
    print(sub_df.head(10))
    print("=" * 80)
    print("EXP47 SUBMISSION READY FOR KAGGLE EVALUATOR")
    print("=" * 80)


if __name__ == "__main__":
    main()
