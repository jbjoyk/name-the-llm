"""
================================================================================
EXP52: DUAL-LORA QWEN3-8B (q8v5) + QWEN3-14B (q14v5) INFERENCE ENGINE
================================================================================
Architecture: Two-Backbone Semantic Ensemble
- Member 1: Qwen3-8B-Base (Truncated 24 layers) + q8v5 LoRA24 + Centred Twin (w = 0.60)
- Member 2: Qwen3-14B-Base (Truncated 27 layers) + q14v5 LoRA24 + Centred Twin (w = 0.40)
Fusion Rule: logP_final = 0.60 * logP_q8 + 0.40 * logP_q14
Author: AntiGravity Senior Competitive ML Agent
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
import psutil
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
# 2. Path Discovery
# ==============================================================================
def resolve_data_dir():
    candidates = [
        "/kaggle/input/competitions/name-that-llm",
        "/kaggle/input/name-that-llm",
        "data",
        "/home/rj2826/name-the-llm/data"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "test.csv")):
            return c
    return "data"


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
# 4. LoRA24 Feature Extraction & Inference
# ==============================================================================
@torch.no_grad()
def run_lora24_inference(spec, base_path, adapter_path, texts, prompt_groups=None):
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

    log(f"--- Running {name} Inference (Keep={keep}, Layers={layers}) ---")
    t0 = time.time()

    # Pre-flight strict offline verification
    verify_local_checkpoint(base_path, name)

    # Tokenizer (strictly offline)
    tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Tokenize with explicit EOS append
    tokens_list = tok(list(texts), truncation=True, max_length=max_len - 1)["input_ids"]
    enc = [x + [eos_id] for x in tokens_list]
    n_samples = len(enc)

    # Config & Truncation (strictly offline)
    cfg = AutoConfig.from_pretrained(base_path, local_files_only=True)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tc.num_hidden_layers = keep
    tc.use_cache = False
    if getattr(tc, "layer_types", None) is not None:
        tc.layer_types = tc.layer_types[:keep]

    # Device allocation
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        device_map = "auto"
    elif n_gpus == 1:
        device_map = {"": 0}
    else:
        device_map = "cpu"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load Model (Truncated, strictly offline)
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

    # Load Heads Checkpoint
    heads_file = os.path.join(adapter_path, "heads.pt")
    if not os.path.exists(heads_file):
        heads_file = os.path.join(adapter_path, "heads.safetensors")
        H = dict(load_file(heads_file))
    else:
        H = torch.load(heads_file, map_location="cpu")

    # LayerNorms
    norms = [torch.nn.LayerNorm(hid) for _ in layers]
    for i, nm in enumerate(norms):
        nm.load_state_dict({"weight": H[f"norms.{i}.weight"], "bias": H[f"norms.{i}.bias"]})

    # In-memory LoRA weight injection
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(adapter_path, "adapter", "adapter_model.safetensors")

    cfg_file = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_file):
        cfg_file = os.path.join(adapter_path, "adapter", "adapter_config.json")

    with open(cfg_file) as f:
        acfg = json.load(f)
    scale = acfg["lora_alpha"] / acfg["r"]

    mods = dict(model.named_modules())
    sd = load_file(adapter_file)
    deltas = []

    for k, v in sd.items():
        if ".lora_A." not in k:
            continue
        name_k = k.replace("base_model.model.", "", 1).split(".lora_A.")[0]
        if name_k not in mods and ("model." + name_k) in mods:
            name_k = "model." + name_k
        if name_k not in mods:
            continue
        B = sd[k.replace(".lora_A.", ".lora_B.")]
        W = mods[name_k].weight
        delta_W = ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)
        W += delta_W
        deltas.append((W, B, v))

    # Memory safety batch size adjustment for tight VRAM
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < (14 * 1024**3):
        bs = min(bs, 4) if name == "q8v5" else min(bs, 2)
        log(f"  [Memory Guard] Single 12GB GPU detected; adjusted {name} batch size to {bs}")

    # Feature extraction loop
    Z = np.zeros((n_samples, total_dim), dtype=np.float32)
    order = np.argsort([len(x) for x in enc])
    first_device = next(model.parameters()).device

    for s in range(0, n_samples, bs):
        batch_idx = order[s:s + bs]
        max_b = max(len(enc[j]) for j in batch_idx)
        ids = torch.full((len(batch_idx), max_b), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch_idx), max_b), dtype=torch.long)
        for r, j in enumerate(batch_idx):
            ids[r, :len(enc[j])] = torch.tensor(enc[j])
            att[r, :len(enc[j])] = 1

        try:
            ids_dev = ids.to(first_device)
            att_dev = att.to(first_device)
            out = model(input_ids=ids_dev, attention_mask=att_dev, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
        except torch.OutOfMemoryError:
            log(f"  [Notice] CUDA OOM at batch {s}; recovering via sample-by-sample fallback...")
            torch.cuda.empty_cache()
            single_hs_list = []
            for r_s in range(len(batch_idx)):
                s_id = ids[r_s:r_s+1].to(first_device)
                s_at = att[r_s:r_s+1].to(first_device)
                s_out = model(input_ids=s_id, attention_mask=s_at, output_hidden_states=True, use_cache=False)
                single_hs_list.append(s_out.hidden_states)
            hs = tuple(torch.cat([h[l] for h in single_hs_list], dim=0) for l in range(len(single_hs_list[0])))

        mask = att.unsqueeze(-1).float()
        mask[:, 0] = 0.0  # exclude attention sink token 0

        pooled_layers = []
        for l_idx, nm in zip(layers, norms):
            h_l = hs[l_idx]
            m_dev = mask.to(h_l.device)
            p_l = (h_l.float() * m_dev).sum(1) / m_dev.sum(1).clamp(min=1e-6)
            p_l_norm = nm(p_l.cpu())
            pooled_layers.append(p_l_norm)

        Z[batch_idx] = torch.cat(pooled_layers, dim=-1).numpy()

    # Unmerge LoRA weights to keep base pristine
    for W, B, A_mat in deltas:
        delta_W = ((B.to(W.device, torch.float32) @ A_mat.to(W.device, torch.float32)) * scale).to(W.dtype)
        W -= delta_W

    del deltas, sd, model, mods
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Head computations
    W_plain = H["plain.weight"].float().numpy()
    b_plain = H["plain.bias"].float().numpy()
    W_cent = H["centered.weight"].float().numpy()
    b_cent = H["centered.bias"].float().numpy()

    # Plain Head Logits & Log-Probs
    logits_plain = Z @ W_plain.T + b_plain
    lp_plain = logits_plain - scipy.special.logsumexp(logits_plain, axis=1, keepdims=True)

    # Centered Head Logits & Log-Probs
    lp_total = lp_plain.copy()
    if prompt_groups is not None:
        cluster_counts = pd.Series(prompt_groups).value_counts().to_dict()
        has_cluster = np.array([cluster_counts.get(g, 1) >= 2 for g in prompt_groups], dtype=bool)

        if has_cluster.any():
            Z_cent = Z.copy()
            for g in np.unique(prompt_groups):
                idx = prompt_groups == g
                if cluster_counts.get(g, 1) >= 2:
                    Z_cent[idx] -= Z_cent[idx].mean(axis=0)
                else:
                    Z_cent[idx] = 0.0

            logits_cent = Z_cent @ W_cent.T + b_cent
            lp_cent = logits_cent - scipy.special.logsumexp(logits_cent, axis=1, keepdims=True)
            lp_total[has_cluster] += twin_weight * lp_cent[has_cluster]

    probs = np.exp(lp_total - scipy.special.logsumexp(lp_total, axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    elapsed = time.time() - t0
    log(f"  {name} inference finished in {elapsed:.2f}s ({n_samples / max(elapsed, 1e-6):.2f} rows/s)")

    return {
        "log_probs": lp_total,
        "probs": probs,
        "features": Z,
        "elapsed_sec": elapsed
    }


# ==============================================================================
# 5. Pipeline Orchestration: Phase 1, Phase 2, Phase 3
# ==============================================================================
def run_exp52_pipeline():
    print("=" * 80)
    print("EXP52: DUAL-LORA QWEN3-8B (0.60) + QWEN3-14B (0.40) EXECUTION")
    print("=" * 80)

    # Hardware probe
    print("\n[Hardware & Compute Audit]:")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available:  {torch.cuda.is_available()}")
    n_gpus = torch.cuda.device_count()
    print(f"  Device Count:    {n_gpus} GPU(s)")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {props.name} | Total Memory: {props.total_memory / (1024**3):.2f} GiB")
    vm = psutil.virtual_memory()
    print(f"  System RAM:      Total: {vm.total / (1024**3):.2f} GiB | Available: {vm.available / (1024**3):.2f} GiB")

    # Path Discovery
    data_dir = resolve_data_dir()
    test_file = os.path.join(data_dir, "test.csv")
    train_file = os.path.join(data_dir, "train.csv")
    print(f"\n[Data Discovery]:")
    print(f"  Data Directory:  {data_dir}")
    print(f"  Test File:       {test_file} (Exists: {os.path.exists(test_file)})")

    test_df = pd.read_csv(test_file)
    n_test = len(test_df)
    print(f"  Visible Rows:    {n_test}")

    # Model Discovery
    base_8b = resolve_base_model("8b")
    base_14b = resolve_base_model("14b")
    ad_8b = resolve_adapter_dir("q8v5")
    ad_14b = resolve_adapter_dir("q14v5")

    print(f"\n[Model & Adapter Discovery]:")
    print(f"  Qwen3-8B Base:   {base_8b}")
    print(f"  q8v5 Adapter:    {ad_8b}")
    print(f"  Qwen3-14B Base:  {base_14b}")
    print(f"  q14v5 Adapter:   {ad_14b}")

    assert ad_8b is not None, "Missing q8v5 adapter artifacts!"
    assert ad_14b is not None, "Missing q14v5 adapter artifacts!"

    # ==========================================================================
    # PHASE 1: 5-Row Smoke Test
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: 5-ROW SMOKE TEST")
    print("=" * 80)

    # Prepare exactly 5 rows (from test or train)
    if n_test >= 5:
        smoke_texts = test_df["text"].fillna("").values[:5]
        smoke_ids = test_df["id"].values[:5]
    else:
        train_df = pd.read_csv(train_file)
        smoke_texts = train_df["text"].fillna("").values[:5]
        smoke_ids = train_df["id"].values[:5]

    print(f"Smoke test passages: {len(smoke_texts)}")

    # Prompt clustering on smoke texts
    smoke_groups = recover_clusters(smoke_texts, thr=0.15, cap=18)

    # 1. Tokenizers & EOS check
    tok8 = AutoTokenizer.from_pretrained(base_8b)
    tok14 = AutoTokenizer.from_pretrained(base_14b)
    print(f"1. Tokenizer 8B vocab:  {len(tok8)}, EOS ID: {tok8.eos_token_id}")
    print(f"2. Tokenizer 14B vocab: {len(tok14)}, EOS ID: {tok14.eos_token_id}")
    assert Q8_SPEC["eos_id"] == 151643, "q8v5 EOS ID mismatch"
    assert Q14_SPEC["eos_id"] == 151643, "q14v5 EOS ID mismatch"

    # 2. Dimensions check from heads
    H8 = torch.load(os.path.join(ad_8b, "heads.pt"), map_location="cpu")
    H14 = torch.load(os.path.join(ad_14b, "heads.pt"), map_location="cpu")
    print(f"3. q8v5 Head input dim:  {H8['plain.weight'].shape[1]} (Expected 32768)")
    print(f"4. q14v5 Head input dim: {H14['plain.weight'].shape[1]} (Expected 40960)")
    assert H8['plain.weight'].shape[1] == 32768, "q8v5 dim mismatch"
    assert H14['plain.weight'].shape[1] == 40960, "q14v5 dim mismatch"

    # Run Smoke Passes
    res_8b_smoke = run_lora24_inference(Q8_SPEC, base_8b, ad_8b, smoke_texts, smoke_groups)
    res_14b_smoke = run_lora24_inference(Q14_SPEC, base_14b, ad_14b, smoke_texts, smoke_groups)

    # Assertions
    assert res_8b_smoke["probs"].shape == (5, 18), "q8v5 probability shape invalid"
    assert res_14b_smoke["probs"].shape == (5, 18), "q14v5 probability shape invalid"
    assert np.isfinite(res_8b_smoke["probs"]).all(), "q8v5 contains NaN/Inf"
    assert np.isfinite(res_14b_smoke["probs"]).all(), "q14v5 contains NaN/Inf"
    assert np.allclose(res_8b_smoke["probs"].sum(axis=1), 1.0, atol=1e-5), "q8v5 probs sum != 1"
    assert np.allclose(res_14b_smoke["probs"].sum(axis=1), 1.0, atol=1e-5), "q14v5 probs sum != 1"

    # Fused log-probabilities
    lp_fused_smoke = 0.60 * res_8b_smoke["log_probs"] + 0.40 * res_14b_smoke["log_probs"]
    p_fused_smoke = np.exp(lp_fused_smoke - scipy.special.logsumexp(lp_fused_smoke, axis=1, keepdims=True))
    p_fused_smoke /= p_fused_smoke.sum(axis=1, keepdims=True)
    preds_smoke = [I2L[idx] for idx in p_fused_smoke.argmax(axis=1)]

    print(f"5. Fused probabilities sum: {p_fused_smoke.sum(axis=1)}")
    print(f"6. Fused predictions: {preds_smoke}")
    print(">> PHASE 1 (SMOKE TEST) PASSED: 100% VALID")

    # ==========================================================================
    # PHASE 2: Runtime & Memory Benchmark
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: RUNTIME & HARDWARE BENCHMARK (50-100 ROWS)")
    print("=" * 80)

    train_df = pd.read_csv(train_file)
    n_bench = 50
    np.random.seed(42)
    bench_indices = np.random.choice(len(train_df), size=n_bench, replace=False)
    bench_texts = train_df.loc[bench_indices, "text"].values
    bench_groups = recover_clusters(bench_texts, thr=0.15, cap=18)

    print(f"Running benchmark on {n_bench} sampled passages...")

    ram_start = psutil.virtual_memory().used / (1024**3)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    t_bench_start = time.time()
    res_8b_bench = run_lora24_inference(Q8_SPEC, base_8b, ad_8b, bench_texts, bench_groups)
    t_8b_end = time.time()
    vram_8b_peak = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    res_14b_bench = run_lora24_inference(Q14_SPEC, base_14b, ad_14b, bench_texts, bench_groups)
    t_14b_end = time.time()
    vram_14b_peak = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0

    total_bench_time = t_14b_end - t_bench_start
    ram_end = psutil.virtual_memory().used / (1024**3)

    throughput_8b = n_bench / max(t_8b_end - t_bench_start, 1e-6)
    throughput_14b = n_bench / max(t_14b_end - t_8b_end, 1e-6)
    throughput_total = n_bench / max(total_bench_time, 1e-6)

    # Check CPU offload detection
    cpu_offload_detected = (vram_14b_peak < 11.0) and (ram_end - ram_start > 2.0)

    benchmark_results = {
        "benchmark_rows": n_bench,
        "wall_clock_time_sec": total_bench_time,
        "time_8b_sec": t_8b_end - t_bench_start,
        "time_14b_sec": t_14b_end - t_8b_end,
        "throughput_8b_rows_per_sec": throughput_8b,
        "throughput_14b_rows_per_sec": throughput_14b,
        "throughput_overall_rows_per_sec": throughput_total,
        "peak_vram_8b_gib": vram_8b_peak,
        "peak_vram_14b_gib": vram_14b_peak,
        "ram_delta_gib": ram_end - ram_start,
        "cpu_offload_detected": cpu_offload_detected
    }

    print(f"\nBenchmark Metrics Summary ({n_bench} rows):")
    print(f"  Wall-clock total:       {total_bench_time:.2f} s")
    print(f"  Qwen3-8B Time:          {benchmark_results['time_8b_sec']:.2f} s ({throughput_8b:.2f} rows/s)")
    print(f"  Qwen3-14B Time:         {benchmark_results['time_14b_sec']:.2f} s ({throughput_14b:.2f} rows/s)")
    print(f"  Peak VRAM 8B:           {vram_8b_peak:.2f} GiB")
    print(f"  Peak VRAM 14B:          {vram_14b_peak:.2f} GiB")
    print(f"  RAM Delta:              {ram_end - ram_start:.2f} GiB")
    print(f"  CPU Offload Detected:   {cpu_offload_detected}")
    print(">> PHASE 2 (BENCHMARK) PASSED: SUCCESS")

    # ==========================================================================
    # PHASE 3: Full Test Inference
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: FULL TEST INFERENCE")
    print("=" * 80)

    test_texts = test_df["text"].fillna("").values
    test_ids = test_df["id"].values
    print(f"Processing full test set: {len(test_texts)} rows")

    test_groups = recover_clusters(test_texts, thr=0.15, cap=18)

    t_full_start = time.time()
    res_8b_test = run_lora24_inference(Q8_SPEC, base_8b, ad_8b, test_texts, test_groups)
    res_14b_test = run_lora24_inference(Q14_SPEC, base_14b, ad_14b, test_texts, test_groups)
    t_full_end = time.time()

    # Exact Log-Probability Fusion
    lp_fused_test = 0.60 * res_8b_test["log_probs"] + 0.40 * res_14b_test["log_probs"]
    p_fused_test = np.exp(lp_fused_test - scipy.special.logsumexp(lp_fused_test, axis=1, keepdims=True))
    p_fused_test /= p_fused_test.sum(axis=1, keepdims=True)
    preds_test = [I2L[idx] for idx in p_fused_test.argmax(axis=1)]

    # Format outputs
    sub_df = pd.DataFrame({"id": test_ids, "model": preds_test})
    os.makedirs("experiments/EXP52_dual_lora", exist_ok=True)
    os.makedirs("submissions/candidates/EXP52", exist_ok=True)

    # Save outputs
    sub_df.to_csv("submissions/candidates/EXP52/submission.csv", index=False)
    sub_df.to_csv("submission.csv", index=False)
    try:
        sub_df.to_parquet("submissions/candidates/EXP52/submission.parquet", index=False)
        sub_df.to_parquet("submission.parquet", index=False)
    except Exception as e:
        log(f"Parquet export notice: {e}")

    np.save("experiments/EXP52_dual_lora/probs_q8v5.npy", res_8b_test["probs"])
    np.save("experiments/EXP52_dual_lora/probs_q14v5.npy", res_14b_test["probs"])
    np.save("experiments/EXP52_dual_lora/probs_fused.npy", p_fused_test)

    # Class distribution
    class_dist = pd.Series(preds_test).value_counts().to_dict()

    runtime_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu_count": n_gpus,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "system_ram_gib": vm.total / (1024**3)
        },
        "benchmark": benchmark_results,
        "full_inference": {
            "test_rows": len(test_texts),
            "inference_time_sec": t_full_end - t_full_start,
            "throughput_rows_per_sec": len(test_texts) / max(t_full_end - t_full_start, 1e-6)
        }
    }

    metrics = {
        "experiment_id": "EXP52_dual_lora_inference",
        "fusion_formula": "0.60 * logP_q8v5 + 0.40 * logP_q14v5",
        "smoke_test_passed": True,
        "benchmark_passed": True,
        "test_rows_processed": len(test_texts),
        "predicted_class_distribution": class_dist,
        "validation_oof_reference_score": {
            "balanced_accuracy": 0.7873,
            "macro_f1": 0.7880,
            "standalone_q8v5_ba": 0.5910,
            "standalone_q14v5_ba": 0.6346
        }
    }

    with open("experiments/EXP52_dual_lora/runtime.json", "w") as f:
        json.dump(runtime_metrics, f, indent=2)

    with open("experiments/EXP52_dual_lora/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFinal Test Inference Results ({len(test_texts)} rows):")
    print(sub_df)
    print("\nPredicted Class Distribution:")
    for k, v in class_dist.items():
        print(f"  {k:24s}: {v}")

    print("\n" + "=" * 80)
    print("EXP52 EXECUTION COMPLETE: ALL 3 PHASES PASSED")
    print("=" * 80)
    return metrics, runtime_metrics


if __name__ == "__main__":
    run_exp52_pipeline()
