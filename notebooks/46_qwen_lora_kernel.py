"""
================================================================================
KAGGLE "NAME THAT LLM" COMPETITION: STANDALONE QWEN3-8B LoRA24 ATTRIBUTION KERNEL
================================================================================
Component: CMP_09 (Qwen3-8B LoRA24 Attribution Model)
Lineage: The Usual Suspects / Dhruv Tibarewal / Sovereign Gold Apex (0.672 LB)
Task: Multiclass Sequence Attribution (18 Candidate LLMs)
Author: Antigravity Team
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

# ==============================================================================
# 1. Candidate LLM Class Definitions
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

# Target layers for LoRA24 multi-layer hidden state pooling
LORA24_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]
KEEP_LAYERS = 24
MAX_SEQ_LEN = 384
BATCH_SIZE = 16


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
        # Fallback to any qwen 8b
        candidates = [os.path.dirname(x) for x in cfgs if 'qwen' in x.lower() and '8b' in x.lower()]
    if candidates:
        return candidates[0]
    
    # Check local HuggingFace cache
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
    
    # Check for q8v5 adapters
    q8_dirs = [os.path.dirname(h) for h in heads if 'q8v5' in h.lower()]
    if q8_dirs:
        # Sort fold adapters 0-4, then full fit
        fold_dirs = [d for d in q8_dirs if any(f'q8v5_f{k}' in d.lower() for k in range(5))]
        full_dirs = [d for d in q8_dirs if d not in fold_dirs]
        sorted_dirs = sorted(fold_dirs) + full_dirs
        return sorted_dirs

    # Local fallback directory
    local_dir = "/tmp/q8_test"
    if os.path.exists(local_dir):
        local_heads = glob.glob(os.path.join(local_dir, "**/heads.pt"), recursive=True)
        if local_heads:
            return [os.path.dirname(h) for h in local_heads]

    return None


# ==============================================================================
# 3. LoRA24 Fast In-Memory Prediction Engine
# ==============================================================================
@torch.no_grad()
def predict_qwen_lora24(base_path, adapter_dirs, texts, max_len=384, bs=16):
    """
    Executes multi-adapter LoRA24 inference:
    1. Loads truncated base model (keep=24 layers) ONCE into GPU VRAM in fp16.
    2. For each fold adapter:
       - Merges LoRA weights directly into base linear layers: W += scale * (B @ A).
       - Extracts hidden states at layers [2, 4, 6, 8, 12, 16, 20, 24].
       - Mean-pools over tokens (excluding token 0).
       - Passes pooled states through trained LayerNorm modules.
       - Evaluates plain linear classification head.
       - Unmerges LoRA weights cleanly: W -= scale * (B @ A).
    3. Averages log-probabilities across all adapters.
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

    # Pre-tokenize all texts
    log(f"Tokenizing {len(texts)} texts (max_length={max_len})...")
    encodings = tok(list(texts), truncation=True, max_length=max_len)['input_ids']
    order = np.argsort([len(x) for x in encodings])
    n_samples = len(texts)

    mods = dict(model.named_modules())
    first_param_device = next(model.parameters()).device
    P_accumulated = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)

    def featurize_passages(norms, hid_dim):
        """Extracts and normalizes multi-layer hidden states."""
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

            # Masked mean pooling excluding token 0
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

    # Iterate through all available fold adapters
    n_adapters = len(adapter_dirs)
    log(f"Evaluating across {n_adapters} LoRA adapter checkpoints...")

    for a_idx, ad in enumerate(adapter_dirs):
        log(f"  [{a_idx + 1}/{n_adapters}] Loading LoRA adapter from {ad}...")
        ad_path = os.path.join(ad, "adapter") if os.path.exists(os.path.join(ad, "adapter", "adapter_config.json")) else ad
        acfg = json.load(open(os.path.join(ad_path, "adapter_config.json")))
        scale = acfg['lora_alpha'] / acfg['r']
        
        # Load LoRA safetensors
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

        # Load trained readout head
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

        # Compute log-probabilities via plain linear head
        W_head = H['plain.weight'].float().numpy()
        b_head = H['plain.bias'].float().numpy()
        logits = Z @ W_head.T + b_head
        log_probs = logits - logits.max(axis=1, keepdims=True) - np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))

        P_accumulated += log_probs / n_adapters

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Final normalized probabilities
    probs = np.exp(P_accumulated - P_accumulated.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, P_accumulated


# ==============================================================================
# 4. Main Standalone Execution
# ==============================================================================
def main():
    print("=" * 80)
    print("PHASE 46A: STANDALONE QWEN3-8B LoRA24 ATTRIBUTION KERNEL")
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

    # 3. FAST_COMMIT Handler for Kaggle Submissions
    # On commit run (3 dummy test rows), execute quick verification to pass build.
    # On hidden scoring rerun (approx 4,000 rows), execute full inference.
    is_commit_run = n_test < 50
    if is_commit_run:
        log("Notice: Small test set detected (< 50 rows). Running verification test pass...")

    t0 = time.time()
    test_texts = test_df["text"].fillna("").astype(str).values
    probs, log_probs = predict_qwen_lora24(
        base_path=base_model_path,
        adapter_dirs=adapter_dirs,
        texts=test_texts,
        max_len=MAX_SEQ_LEN,
        bs=BATCH_SIZE
    )
    inference_time = time.time() - t0
    log(f"Inference completed in {inference_time:.2f}s ({n_test / max(inference_time, 0.001):.1f} rows/s).")

    # 4. Generate Predictions & Output Formats
    pred_indices = np.argmax(probs, axis=1)
    pred_labels = [I2L[idx] for idx in pred_indices]

    sub_df = pd.DataFrame({
        "id": test_df["id"].values,
        "model": pred_labels
    })

    # Save CSV and Parquet
    out_parquet = os.path.join(work_dir, "submission.parquet")
    out_csv = os.path.join(work_dir, "submission.csv")
    sub_df.to_parquet(out_parquet, index=False)
    sub_df.to_csv(out_csv, index=False)

    log(f"Saved: {out_parquet}")
    log(f"Saved: {out_csv}")

    # Ensure present in root working directory for Kaggle evaluator
    if not os.path.exists("submission.parquet"):
        sub_df.to_parquet("submission.parquet", index=False)
    if not os.path.exists("submission.csv"):
        sub_df.to_csv("submission.csv", index=False)

    print("\nSubmission Preview:")
    print(sub_df.head(10))
    print("=" * 80)
    print("SUBMISSION READY FOR KAGGLE EVALUATOR")
    print("=" * 80)


if __name__ == "__main__":
    main()
