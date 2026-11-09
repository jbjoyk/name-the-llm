"""
================================================================================
KAGGLE "NAME THAT LLM" COMPETITION: EXP52 DUAL-LORA INFERENCE KERNEL
================================================================================
Experiment: EXP52_dual_lora_inference
Architecture: Pure Two-Backbone Semantic Ensemble (No SVC / No Sinkhorn)
- Model 1: Qwen3-8B-Base (Truncated 24 layers) + q8v5 LoRA24 + Centred Twin (w = 0.60)
- Model 2: Qwen3-14B-Base (Truncated 27 layers) + q14v5 LoRA24 + Centred Twin (w = 0.40)
Fusion: logP_final = 0.60 * logP_q8 + 0.40 * logP_q14
Lineage: Dhruv Tibarewal / Anto Pil / Sovereign Gold Apex
Hardware Target: Kaggle 2x T4 GPU (Sequential model evaluation for zero OOM risk)
================================================================================
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import gc
import json
import time
import glob
import numpy as np
import pandas as pd
import scipy.special
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ==============================================================================
# 1. Candidate LLM Class Definitions (18 Classes - Alphabetical)
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

# Architecture specifications
Q8_SPEC = {
    "name": "q8v5",
    "size_str": "8b",
    "keep": 24,
    "layers": [2, 4, 6, 8, 12, 16, 20, 24],
    "hidden_size": 4096,
    "total_dim": 32768,
    "eos_id": 151643,
    "max_len": 384,
    "bs": 16,
    "twin_weight": 0.35,
    "blend_weight": 0.60
}

Q14_SPEC = {
    "name": "q14v5",
    "size_str": "14b",
    "keep": 27,
    "layers": [2, 4, 6, 9, 13, 17, 22, 27],
    "hidden_size": 5120,
    "total_dim": 40960,
    "eos_id": 151643,
    "max_len": 447,
    "bs": 8,
    "twin_weight": 0.35,
    "blend_weight": 0.40
}


def log(*args):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)


# ==============================================================================
# 2. Kaggle Path Discovery
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


def verify_local_checkpoint(base_path, model_name="model"):
    """Validates that base_path is a valid local directory containing required files."""
    if not isinstance(base_path, str) or not os.path.isabs(base_path):
        raise ValueError(f"[{model_name}] Base path must be an absolute local filesystem path, got: {base_path}")

    if os.path.exists("/kaggle") and not base_path.startswith("/kaggle/input/"):
        raise RuntimeError(f"[{model_name}] On Kaggle, base path MUST start with '/kaggle/input/', got: {base_path}")

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"[{model_name}] Base directory does not exist: {base_path}")

    config_file = os.path.join(base_path, "config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"[{model_name}] Missing required config.json in: {base_path}")

    tok_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
    if not any(os.path.isfile(os.path.join(base_path, f)) for f in tok_files):
        raise FileNotFoundError(f"[{model_name}] Missing required tokenizer files in: {base_path}")

    log(f"[{model_name}] Local offline checkpoint verified: {base_path}")
    return base_path


def resolve_base_model(size_str="8b"):
    """
    Deterministic local offline base model resolver for Qwen3-8B or Qwen3-14B.
    Strictly offline: NEVER falls back to a Hugging Face repository ID.
    """
    size_key = size_str.lower().strip()  # '8b' or '14b'

    # 1. Primary canonical Kaggle model path
    primary_kaggle = f"/kaggle/input/models/qwen-lm/qwen-3/transformers/{size_key}-base/1"
    if os.path.isdir(primary_kaggle) and os.path.isfile(os.path.join(primary_kaggle, "config.json")):
        return verify_local_checkpoint(primary_kaggle, f"qwen3_{size_key}")

    # 2. Known Kaggle alternative model mount locations
    alt_kaggle_paths = [
        f"/kaggle/input/qwen-3/transformers/{size_key}-base/1",
        f"/kaggle/input/qwen-3/{size_key}-base",
        f"/kaggle/input/qwen3-{size_key}-base",
        f"/kaggle/input/qwen-3-base-{size_key}",
        f"/kaggle/input/models--qwen--qwen3-{size_key}-base"
    ]
    for p in alt_kaggle_paths:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
            return verify_local_checkpoint(p, f"qwen3_{size_key}")

    # 3. Dynamic search across /kaggle/input (robust against symlinks / nesting)
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files:
                low_root = root.lower()
                if "qwen" in low_root and size_key in low_root:
                    if "base" in low_root or "transformers" in low_root:
                        return verify_local_checkpoint(root, f"qwen3_{size_key}")

    # 4. Local machine cache fallback (strictly existing local filesystem directories only)
    hf_hub_name = f"models--Qwen--Qwen3-{size_key.upper()}-Base"
    local_cache_base = os.path.expanduser(f"~/.cache/huggingface/hub/{hf_hub_name}/snapshots")
    if os.path.isdir(local_cache_base):
        snaps = sorted(os.listdir(local_cache_base))
        for s in snaps:
            snap_path = os.path.join(local_cache_base, s)
            if os.path.isdir(snap_path) and os.path.isfile(os.path.join(snap_path, "config.json")):
                return verify_local_checkpoint(snap_path, f"qwen3_{size_key}")

    # 5. Local repo fallback
    local_repo_path = os.path.abspath(f"models/base/qwen3_{size_key}")
    if os.path.isdir(local_repo_path) and os.path.isfile(os.path.join(local_repo_path, "config.json")):
        return verify_local_checkpoint(local_repo_path, f"qwen3_{size_key}")

    # FAIL FAST: NEVER return a Hugging Face repo ID
    err_msg = (
        f"CRITICAL ERROR: Offline base model for Qwen3-{size_key.upper()} was not found on local filesystem!\n"
        f"Checked primary Kaggle path: {primary_kaggle}\n"
        f"When running on Kaggle, please ensure the Kaggle Model 'qwen-lm/qwen-3' with variation '{size_key}-base' is attached."
    )
    raise FileNotFoundError(err_msg)


def resolve_adapter_dir(adapter_name="q8v5"):
    """Finds adapter directory with adapter_model.safetensors and heads.pt."""
    # 1. Primary canonical Kaggle paths
    kaggle_paths = [
        f"/kaggle/input/nlm-l24-adapters-v5/{adapter_name}/{adapter_name}",
        f"/kaggle/input/nlm-l24-adapters-v5/{adapter_name}",
        f"/kaggle/input/dhruvtibarewal-nlm-l24-adapters-v5/{adapter_name}/{adapter_name}",
        f"/kaggle/input/dhruvtibarewal-nlm-l24-adapters-v5/{adapter_name}"
    ]
    for p in kaggle_paths:
        if os.path.isfile(os.path.join(p, "heads.pt")):
            return os.path.abspath(p)

    # 2. Dynamic search in /kaggle/input
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "heads.pt" in files and adapter_name.lower() in root.lower():
                return os.path.abspath(root)

    # 3. Local relative paths
    local_candidates = [
        f"models/adapters_v5/{adapter_name}/{adapter_name}",
        f"models/adapters_v5/{adapter_name}",
        f"models/{adapter_name}/{adapter_name}",
        f"models/{adapter_name}"
    ]
    for lc in local_candidates:
        if os.path.isfile(os.path.join(lc, "heads.pt")):
            return os.path.abspath(lc)

    raise FileNotFoundError(
        f"CRITICAL ERROR: Adapter directory for '{adapter_name}' not found on local filesystem!\n"
        "Please ensure dataset 'dhruvtibarewal/nlm-l24-adapters-v5' is attached."
    )


# ==============================================================================
# 3. Unsupervised Prompt Recovery (Clustering)
# ==============================================================================
def capped_agglo(S, cap=18, thr=0.15):
    """Lance-Williams Average Linkage Agglomerative Clustering."""
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


def recover_clusters(texts, thr=0.15, cap=18):
    from sklearn.feature_extraction.text import TfidfVectorizer
    n = len(texts)
    if n <= 1:
        return np.arange(n)
    try:
        max_df = 0.05 if n >= 20 else 1.0
        min_df = 2 if n >= 20 else 1
        v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=min_df, max_df=max_df)
        Xc = v.fit_transform(texts)
        if Xc.shape[1] == 0:
            return np.arange(n)
        S = (Xc @ Xc.T).toarray()
        return capped_agglo(S, cap=cap, thr=thr)
    except Exception as e:
        log(f"  [Notice] Clustering fallback: {e}. Singletons assigned.")
        return np.arange(n)


# ==============================================================================
# 4. LoRA24 Inference Engine with Centred Twin Head
# ==============================================================================
@torch.no_grad()
def run_backbone_inference(spec, base_path, adapter_path, texts, prompt_groups=None):
    """
    Executes LoRA24 inference:
    1. Loads truncated base model in fp16.
    2. Injects and merges LoRA delta weights in-memory.
    3. Runs forward pass extracting target hidden layers.
    4. Applies masked mean pooling and LayerNorms.
    5. Evaluates plain and centered twin heads.
    6. Returns log-probabilities: (n_samples, 18).
    """
    name = spec["name"]
    keep = spec["keep"]
    layers = spec["layers"]
    hid = spec["hidden_size"]
    total_dim = spec["total_dim"]
    eos_id = spec.get("eos_id", 151643)
    max_len = spec.get("max_len", 384)
    bs = spec.get("bs", 8)
    twin_weight = spec.get("twin_weight", 0.35)

    log(f"=== Starting {name} Inference (Keep={keep}, Hidden={hid}, Dim={total_dim}) ===")
    t0 = time.time()

    # Pre-flight strict offline verification
    verify_local_checkpoint(base_path, name)

    # 1. Tokenizer (offline only)
    tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Tokenize with explicit EOS append
    tokens_list = tok(list(texts), truncation=True, max_length=max_len - 1)["input_ids"]
    enc = [x + [eos_id] for x in tokens_list]
    n_samples = len(enc)

    # 2. Config & Truncation (offline only)
    cfg = AutoConfig.from_pretrained(base_path, local_files_only=True)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tc.num_hidden_layers = keep
    tc.use_cache = False
    if getattr(tc, "layer_types", None) is not None:
        tc.layer_types = tc.layer_types[:keep]

    # 3. Multi-GPU Device Allocation
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        device_map = "auto"
    elif n_gpus == 1:
        device_map = {"": 0}
    else:
        device_map = "cpu"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 4. Load Base Model (Truncated, offline only)
    try:
        model = AutoModel.from_pretrained(
            base_path,
            config=cfg,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True
        ).eval()
    except Exception as e:
        log(f"  [Fallback] from_pretrained failed ({e}); instantiating from config with auto-dispatch...")
        from accelerate import infer_auto_device_map, dispatch_model
        with torch.device("meta"):
            meta_model = AutoModel.from_config(cfg, torch_dtype=dtype)
        no_split = getattr(meta_model, "_no_split_modules", ["Qwen3DecoderLayer", "Qwen2DecoderLayer"])
        if os.path.exists("/kaggle"):
            max_mem = {0: "14GiB", 1: "14GiB", "cpu": "25GiB"} if n_gpus > 1 else {0: "14GiB", "cpu": "25GiB"}
        else:
            max_mem = {0: "9.5GiB", "cpu": "25GiB"}
        dmap = infer_auto_device_map(meta_model, max_memory=max_mem, no_split_module_classes=no_split)
        torch.manual_seed(42)
        model = AutoModel.from_config(cfg, torch_dtype=dtype)
        model = dispatch_model(model, device_map=dmap).eval()

    # Disable final norm to match Dhruv Tibarewal / Lucas Boesen readout head specification
    if hasattr(model, "norm"):
        model.norm = torch.nn.Identity()

    # 5. In-Memory LoRA Delta Injection
    adapter_weights_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(adapter_weights_file):
        lora_weights = load_file(adapter_weights_file)
        with open(os.path.join(adapter_path, "adapter_config.json")) as f:
            lora_cfg = json.load(f)
        scale = lora_cfg.get("lora_alpha", 32) / lora_cfg.get("r", 16)

        lora_pairs = {}
        for k, v in lora_weights.items():
            if "lora_A" in k:
                base_k = k.replace(".lora_A.weight", "").replace("base_model.model.", "")
                lora_pairs.setdefault(base_k, {})["A"] = v
            elif "lora_B" in k:
                base_k = k.replace(".lora_B.weight", "").replace("base_model.model.", "")
                lora_pairs.setdefault(base_k, {})["B"] = v

        for base_k, ab in lora_pairs.items():
            if "A" in ab and "B" in ab:
                delta = (ab["B"].float() @ ab["A"].float()) * scale
                curr_module = model
                for part in base_k.split("."):
                    if hasattr(curr_module, part):
                        curr_module = getattr(curr_module, part)
                    elif part.isdigit() and int(part) < len(curr_module):
                        curr_module = curr_module[int(part)]
                    else:
                        curr_module = None
                        break
                if curr_module is not None and hasattr(curr_module, "weight"):
                    orig_dev = curr_module.weight.device
                    orig_dt = curr_module.weight.dtype
                    curr_module.weight.data.add_(delta.to(orig_dev, dtype=orig_dt))
        del lora_weights, lora_pairs
        gc.collect()

    # 6. Load Classification Heads & LayerNorms
    heads_file = os.path.join(adapter_path, "heads.pt")
    heads_data = torch.load(heads_file, map_location="cpu")

    head_plain_w = heads_data["plain.weight"].float()
    head_plain_b = heads_data["plain.bias"].float()
    head_cent_w = heads_data["centered.weight"].float()
    head_cent_b = heads_data["centered.bias"].float()

    norms = []
    for l_idx in range(len(layers)):
        norm = torch.nn.LayerNorm(hid, elementwise_affine=True)
        norm.weight.data.copy_(heads_data[f"norms.{l_idx}.weight"].float())
        norm.bias.data.copy_(heads_data[f"norms.{l_idx}.bias"].float())
        norms.append(norm)

    # 7. Dynamic Batching & Memory Sizing
    if torch.cuda.is_available():
        free_mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if free_mem_gib < 14.0:
            bs = min(bs, 4 if hid <= 4096 else 2)

    primary_device = next(model.parameters()).device
    sort_idx = np.argsort([len(x) for x in enc])
    unsort_idx = np.argsort(sort_idx)

    pooled_features = []

    for i in range(0, n_samples, bs):
        batch_ids = sort_idx[i:i + bs]
        b_tokens = [enc[idx] for idx in batch_ids]
        b_lens = [len(x) for x in b_tokens]
        max_b_len = max(b_lens)

        input_ids = torch.full((len(b_tokens), max_b_len), tok.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(b_tokens), max_b_len), dtype=torch.long)
        for row_i, tok_ids in enumerate(b_tokens):
            input_ids[row_i, :len(tok_ids)] = torch.tensor(tok_ids, dtype=torch.long)
            attention_mask[row_i, :len(tok_ids)] = 1

        input_ids = input_ids.to(primary_device)
        attention_mask = attention_mask.to(primary_device)

        try:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hs = out.hidden_states
        except torch.OutOfMemoryError:
            log(f"  [OOM Warning] Fallback to single sample execution for batch {i}..{i+len(b_tokens)}")
            torch.cuda.empty_cache()
            hs_accum = [[] for _ in range(len(layers))]
            for single_id in batch_ids:
                s_toks = enc[single_id]
                s_in = torch.tensor([s_toks], dtype=torch.long, device=primary_device)
                s_mask = torch.ones_like(s_in)
                s_out = model(input_ids=s_in, attention_mask=s_mask, output_hidden_states=True, return_dict=True)
                for l_k, tgt_layer in enumerate(layers):
                    h_l = s_out.hidden_states[tgt_layer][0, 1:].float()
                    hs_accum[l_k].append(h_l.mean(dim=0, keepdim=True).cpu())
            b_pooled_normed = []
            for l_k, norm_k in enumerate(norms):
                cat_l = torch.cat(hs_accum[l_k], dim=0)
                b_pooled_normed.append(norm_k(cat_l))
            b_feat = torch.cat(b_pooled_normed, dim=1)
            pooled_features.append(b_feat)
            continue

        # Extract target layers, mask out token 0, and mean-pool
        b_pooled_normed = []
        mask_content = attention_mask[:, 1:].unsqueeze(-1).float()
        for l_k, tgt_layer in enumerate(layers):
            h_l = hs[tgt_layer][:, 1:, :].float()
            sum_h = (h_l * mask_content).sum(dim=1)
            cnt_h = mask_content.sum(dim=1).clamp(min=1e-6)
            pool_l = (sum_h / cnt_h).cpu()
            b_pooled_normed.append(norms[l_k](pool_l))

        b_feat = torch.cat(b_pooled_normed, dim=1)
        pooled_features.append(b_feat)

    all_features = torch.cat(pooled_features, dim=0)[unsort_idx]

    # Free Model from VRAM completely before head scoring
    del model
    if 'out' in locals():
        del out, hs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 8. Classification Head Scoring (Plain + Centred Twin)
    logits_plain = (all_features @ head_plain_w.T + head_plain_b).numpy()
    lp_plain = logits_plain - scipy.special.logsumexp(logits_plain, axis=1, keepdims=True)

    lp_total = lp_plain.copy()
    if prompt_groups is not None and head_cent_w is not None:
        cluster_counts = pd.Series(prompt_groups).value_counts().to_dict()
        has_cluster = np.array([cluster_counts.get(g, 1) >= 2 for g in prompt_groups], dtype=bool)

        if has_cluster.any():
            feat_np = all_features.numpy().copy()
            for g_id in np.unique(prompt_groups):
                idx = prompt_groups == g_id
                if cluster_counts.get(g_id, 1) >= 2:
                    feat_np[idx] -= feat_np[idx].mean(axis=0)
                else:
                    feat_np[idx] = 0.0

            logits_cent = (torch.tensor(feat_np).float() @ head_cent_w.T + head_cent_b).numpy()
            lp_cent = logits_cent - scipy.special.logsumexp(logits_cent, axis=1, keepdims=True)
            lp_total[has_cluster] += twin_weight * lp_cent[has_cluster]

    probs = np.exp(lp_total - scipy.special.logsumexp(lp_total, axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    elapsed = time.time() - t0
    log(f"  {name} inference finished in {elapsed:.2f}s ({n_samples / max(elapsed, 1e-6):.2f} rows/s)")
    return {
        "probs": probs,
        "log_probs": lp_total,
        "elapsed": elapsed
    }


# ==============================================================================
# 5. Main Kaggle Pipeline
# ==============================================================================
def main():
    print("=" * 80)
    print("EXP52: DUAL-LORA QWEN3-8B (0.60) + QWEN3-14B (0.40) EXECUTION")
    print("=" * 80)

    work_dir = "/kaggle/working" if os.path.exists("/kaggle") else "submissions"
    os.makedirs(work_dir, exist_ok=True)

    # 1. Discover Data
    data_dir = resolve_data_dir()
    test_path = os.path.join(data_dir, "test.csv")
    test_df = pd.read_csv(test_path)
    n_test = len(test_df)
    log(f"Dataset resolved: {test_path} ({n_test} rows)")

    # 2. Discover Checkpoints & Adapters
    base_8b = resolve_base_model("8b")
    base_14b = resolve_base_model("14b")
    ad_8b = resolve_adapter_dir("q8v5")
    ad_14b = resolve_adapter_dir("q14v5")

    print("\nPre-Flight Model & Adapter Check:")
    print(f"  Qwen3-8B Base:  {base_8b}")
    print(f"  q8v5 Adapter:   {ad_8b}")
    print(f"  Qwen3-14B Base: {base_14b}")
    print(f"  q14v5 Adapter:  {ad_14b}\n")

    # 3. Prompt Recovery
    test_texts = test_df["text"].fillna("").astype(str).values
    groups = recover_clusters(test_texts, thr=0.15, cap=18)
    n_groups = len(np.unique(groups))
    log(f"Recovered {n_groups} clusters across {n_test} test rows.")

    # 4. Sequential Model 1: Qwen3-8B LoRA24 Centred Twin
    t_start = time.time()
    res_8b = run_backbone_inference(Q8_SPEC, base_8b, ad_8b, test_texts, groups)

    # Garbage collect before Model 2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Sequential Model 2: Qwen3-14B LoRA24 Centred Twin
    res_14b = run_backbone_inference(Q14_SPEC, base_14b, ad_14b, test_texts, groups)

    # 6. Exact Log-Probability Fusion
    w_8b = Q8_SPEC["blend_weight"]
    w_14b = Q14_SPEC["blend_weight"]
    log(f"Fusing Log-Probabilities: {w_8b:.2f} * logP(q8v5) + {w_14b:.2f} * logP(q14v5)")

    log_probs_fused = w_8b * res_8b["log_probs"] + w_14b * res_14b["log_probs"]
    probs_fused = np.exp(log_probs_fused - scipy.special.logsumexp(log_probs_fused, axis=1, keepdims=True))
    probs_fused /= probs_fused.sum(axis=1, keepdims=True)

    # 7. Predictions & Formatting
    pred_indices = probs_fused.argmax(axis=1)
    pred_labels = [I2L[idx] for idx in pred_indices]

    sub_df = pd.DataFrame({
        "id": test_df["id"].values,
        "model": pred_labels
    })

    out_csv = os.path.join(work_dir, "submission.csv")
    out_parquet = os.path.join(work_dir, "submission.parquet")
    sub_df.to_csv(out_csv, index=False)
    log(f"Saved: {out_csv}")

    try:
        sub_df.to_parquet(out_parquet, index=False)
        log(f"Saved: {out_parquet}")
    except Exception as e:
        log(f"Parquet export skipped ({e}); CSV generated cleanly.")

    # Mirror to current working directory if different
    if not os.path.exists("submission.csv"):
        sub_df.to_csv("submission.csv", index=False)
    if not os.path.exists("submission.parquet"):
        try:
            sub_df.to_parquet("submission.parquet", index=False)
        except Exception:
            pass

    total_time = time.time() - t_start
    log(f"Total pipeline completed in {total_time:.2f}s ({n_test / max(total_time, 1e-6):.2f} rows/s)")

    print("\nSubmission Preview:")
    print(sub_df.head(10))
    print("\nPredicted Class Counts:")
    print(sub_df["model"].value_counts())
    print("=" * 80)
    print("EXP52 SUBMISSION READY FOR KAGGLE EVALUATOR")
    print("=" * 80)


if __name__ == "__main__":
    main()
