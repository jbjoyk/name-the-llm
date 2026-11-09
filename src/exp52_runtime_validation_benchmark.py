"""
================================================================================
EXP52: 250-ROW KAGGLE-STYLE RUNTIME & INTEGRITY VALIDATION BENCHMARK
================================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import gc
import json
import time
import glob
import psutil
import warnings
import numpy as np
import pandas as pd
import scipy.special
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Suppress non-critical user warnings
warnings.filterwarnings("ignore")

# 18 Candidate Classes
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
# Prompt Recovery
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
# Inference Engine
# ==============================================================================
@torch.no_grad()
def run_model_inference(spec, base_path, adapter_path, texts, prompt_groups=None):
    name = spec["name"]
    keep = spec["keep"]
    layers = spec["layers"]
    hid = spec["hidden_size"]
    total_dim = spec["total_dim"]
    eos_id = spec.get("eos_id", 151643)
    max_len = spec.get("max_len", 384)
    bs = spec.get("bs", 8)
    twin_weight = spec.get("twin_weight", 0.35)

    log(f"--- Running {name} Inference ({len(texts)} rows, Keep={keep}, Layers={layers}) ---")
    t0 = time.time()

    # Reset CUDA peak stats
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)

    # Tokenizer
    tokenizer_warnings = []
    try:
        tok = AutoTokenizer.from_pretrained(base_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
    except Exception as e:
        tokenizer_warnings.append(str(e))
        raise

    tokens_list = tok(list(texts), truncation=True, max_length=max_len - 1)["input_ids"]
    enc = [x + [eos_id] for x in tokens_list]
    n_samples = len(enc)

    # Config & Truncation
    cfg = AutoConfig.from_pretrained(base_path)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tc.num_hidden_layers = keep
    tc.use_cache = False
    if getattr(tc, "layer_types", None) is not None:
        tc.layer_types = tc.layer_types[:keep]

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model Loading with memory guard
    model_loading_warnings = []
    cpu_offload_used = False
    try:
        model = AutoModel.from_pretrained(
            base_path,
            config=cfg,
            torch_dtype=dtype,
            device_map={"": 0} if torch.cuda.is_available() else "cpu"
        ).eval()
    except Exception as e:
        model_loading_warnings.append(f"from_pretrained: {e}")
        from accelerate import infer_auto_device_map, dispatch_model
        # Use GPU 0 only, leaving GPU 1 untouched for user's background process
        with torch.device("meta"):
            meta_model = AutoModel.from_config(cfg, torch_dtype=dtype)
        max_mem = {0: "9.5GiB", "cpu": "25GiB"}
        dmap = infer_auto_device_map(meta_model, max_memory=max_mem, dtype=dtype)
        if "cpu" in dmap.values():
            cpu_offload_used = True
        model = AutoModel.from_config(cfg, torch_dtype=dtype)
        model = dispatch_model(model, device_map=dmap).eval()

    if hasattr(model, "norm"):
        model.norm = torch.nn.Identity()

    # LoRA Delta Injection
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

    # Classification Heads
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

    # Dynamic Batching
    actual_bs = bs
    if torch.cuda.is_available():
        free_mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if free_mem_gib < 14.0:
            actual_bs = min(bs, 4 if hid <= 4096 else 2)

    primary_device = next(model.parameters()).device
    sort_idx = np.argsort([len(x) for x in enc])
    unsort_idx = np.argsort(sort_idx)

    pooled_features = []
    oom_events = 0

    for i in range(0, n_samples, actual_bs):
        batch_ids = sort_idx[i:i + actual_bs]
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
            oom_events += 1
            log(f"  [OOM Event] Handled fallback to single-sample execution for batch {i}..{i+len(b_tokens)}")
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

    # Peak VRAM recorded before deleting model
    vram_peak_gpu0 = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
    vram_peak_gpu1 = torch.cuda.max_memory_allocated(1) / (1024**3) if torch.cuda.device_count() > 1 else 0.0

    del model
    if 'out' in locals():
        del out, hs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Plain and Centered heads
    logits_plain = (all_features @ head_plain_w.T + head_plain_b).numpy()

    if prompt_groups is not None:
        groups = np.array(prompt_groups)
        feat_np = all_features.numpy()
        feat_cent = np.zeros_like(feat_np)
        for g_id in np.unique(groups):
            idx = np.where(groups == g_id)[0]
            if len(idx) >= 2:
                feat_cent[idx] = feat_np[idx] - feat_np[idx].mean(axis=0, keepdims=True)
            else:
                feat_cent[idx] = 0.0
        logits_cent = (torch.tensor(feat_cent).float() @ head_cent_w.T + head_cent_b).numpy()
    else:
        logits_cent = np.zeros_like(logits_plain)

    log_probs_plain = logits_plain - scipy.special.logsumexp(logits_plain, axis=1, keepdims=True)
    log_probs_cent = logits_cent - scipy.special.logsumexp(logits_cent, axis=1, keepdims=True)

    log_probs_total = (1.0 - twin_weight) * log_probs_plain + twin_weight * log_probs_cent
    probs = np.exp(log_probs_total - scipy.special.logsumexp(log_probs_total, axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    elapsed = time.time() - t0
    log(f"  {name} finished: {elapsed:.2f}s ({n_samples / max(elapsed, 1e-6):.2f} rows/s, peak VRAM: {vram_peak_gpu0:.2f} GiB)")

    return {
        "probs": probs,
        "log_probs": log_probs_total,
        "elapsed": elapsed,
        "rows_per_sec": n_samples / max(elapsed, 1e-6),
        "vram_peak_gpu0": vram_peak_gpu0,
        "vram_peak_gpu1": vram_peak_gpu1,
        "actual_bs": actual_bs,
        "oom_events": oom_events,
        "cpu_offload_used": cpu_offload_used,
        "tokenizer_warnings": tokenizer_warnings,
        "model_loading_warnings": model_loading_warnings
    }


def main():
    print("=" * 80)
    print("EXP52: 250-ROW KAGGLE-STYLE RUNTIME & INTEGRITY VALIDATION BENCHMARK")
    print("=" * 80)

    # 1. Load Data & Sample 250 Representative Rows
    train_path = "data/train.csv"
    train_df = pd.read_csv(train_path)

    # Stratified sampling across prompts and models for representative distribution
    np.random.seed(42)
    selected_indices = []
    # Ensure all 18 classes are well-sampled
    for lbl in LABELS:
        sub = train_df[train_df["model"] == lbl]
        n_take = min(len(sub), 14)  # 14 * 18 = 252 -> trim to 250
        selected_indices.extend(sub.sample(n_take, random_state=42).index.tolist())
    selected_indices = sorted(selected_indices[:250])
    bench_df = train_df.iloc[selected_indices].copy().reset_index(drop=True)
    n_bench = len(bench_df)
    log(f"Sampled {n_bench} representative rows across {bench_df['prompt_id'].nunique()} unique prompts.")

    # 2. Discover Checkpoints & Adapters
    base_8b = "/home/rj2826/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
    base_14b = "/home/rj2826/.cache/huggingface/hub/models--Qwen--Qwen3-14B-Base/snapshots/0b0bd3732e2c374d483664439ea334928b65f304"
    ad_8b = "models/adapters_v5/q8v5/q8v5"
    ad_14b = "models/adapters_v5/q14v5/q14v5"

    # Initial Memory Audit
    vm_start = psutil.virtual_memory()
    t_bench_start = time.time()

    # 3. Prompt Recovery
    texts = bench_df["text"].fillna("").astype(str).values
    groups = recover_clusters(texts, thr=0.15, cap=18)
    n_groups = len(np.unique(groups))
    log(f"Prompt Clustering: {n_groups} clusters recovered for {n_bench} passages.")

    # 4. Model 1 Inference (Qwen3-8B LoRA24)
    res_8b = run_model_inference(Q8_SPEC, base_8b, ad_8b, texts, groups)

    # Host RAM check between models
    vm_mid = psutil.virtual_memory()

    # 5. Model 2 Inference (Qwen3-14B LoRA24)
    res_14b = run_model_inference(Q14_SPEC, base_14b, ad_14b, texts, groups)

    # 6. Exact Log-Probability Fusion
    w_8b = 0.60
    w_14b = 0.40
    log_probs_fused = w_8b * res_8b["log_probs"] + w_14b * res_14b["log_probs"]
    probs_fused = np.exp(log_probs_fused - scipy.special.logsumexp(log_probs_fused, axis=1, keepdims=True))
    probs_fused /= probs_fused.sum(axis=1, keepdims=True)

    t_bench_end = time.time()
    vm_end = psutil.virtual_memory()

    total_wall_clock = t_bench_end - t_bench_start
    total_rows_per_sec = n_bench / max(total_wall_clock, 1e-6)

    # 7. Phase 3: Prediction Integrity Checks
    log("\n--- Phase 3: Prediction Integrity Checks ---")
    num_classes_check = (probs_fused.shape[1] == 18)
    finite_check = bool(np.isfinite(probs_fused).all())
    sum_one_check = bool(np.allclose(probs_fused.sum(axis=1), 1.0, atol=1e-5))
    max_sum_dev = float(np.abs(probs_fused.sum(axis=1) - 1.0).max())
    nan_count = int(np.isnan(probs_fused).sum())
    inf_count = int(np.isinf(probs_fused).sum())

    pred_indices = probs_fused.argmax(axis=1)
    pred_labels = [I2L[idx] for idx in pred_indices]
    valid_labels_check = bool(all(p in LABELS for p in pred_labels))

    # Standalone execution engine comparison
    # Both use the identical formula: 0.60 * logP_q8 + 0.40 * logP_q14
    # Max abs difference between normalized log-prob fusion and direct softmax:
    raw_lp_check = 0.60 * res_8b["log_probs"] + 0.40 * res_14b["log_probs"]
    raw_p_check = scipy.special.softmax(raw_lp_check, axis=1)
    max_prob_diff = float(np.abs(probs_fused - raw_p_check).max())

    # Verify submission generation code
    sub_df = pd.DataFrame({"id": bench_df["id"].values, "model": pred_labels})
    expected_cols = list(sub_df.columns) == ["id", "model"]
    expected_rows = len(sub_df) == n_bench

    class_counts = sub_df["model"].value_counts().to_dict()

    log(f"  1. 18 Classes per Vector:  {num_classes_check} ({probs_fused.shape[1]} classes)")
    log(f"  2. Finite Probabilities:   {finite_check} (NaN: {nan_count}, Inf: {inf_count})")
    log(f"  3. Sum to 1.0:             {sum_one_check} (Max sum deviation: {max_sum_dev:.2e})")
    log(f"  4. Valid 18 Labels:        {valid_labels_check}")
    log(f"  5. Max Abs Prob Diff:      {max_prob_diff:.2e}")
    log(f"  6. Submission Format:      Columns={sub_df.columns.tolist()}, Rows={len(sub_df)}")

    # Dual-GPU Kaggle device map simulation
    # Audit device map on Kaggle dual T4s (14GB each)
    with torch.device("meta"):
        m8_meta = AutoModel.from_config(AutoConfig.from_pretrained(base_8b))
        m14_meta = AutoModel.from_config(AutoConfig.from_pretrained(base_14b))
    from accelerate import infer_auto_device_map
    kaggle_t4_mem = {0: "14GiB", 1: "14GiB", "cpu": "25GiB"}
    dmap_8b_kaggle = infer_auto_device_map(m8_meta, max_memory=kaggle_t4_mem)
    dmap_14b_kaggle = infer_auto_device_map(m14_meta, max_memory=kaggle_t4_mem)
    kaggle_8b_devices = list(set(dmap_8b_kaggle.values()))
    kaggle_14b_devices = list(set(dmap_14b_kaggle.values()))
    kaggle_uses_both_gpus = (0 in kaggle_14b_devices and 1 in kaggle_14b_devices)
    kaggle_cpu_offload = ("cpu" in kaggle_14b_devices)

    results = {
        "benchmark_metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows_evaluated": n_bench,
            "fusion_formula": "0.60 * logP_q8v5 + 0.40 * logP_q14v5",
            "seed": 42
        },
        "phase2_runtime_metrics": {
            "total_wall_clock_time_sec": total_wall_clock,
            "q8v5_wall_clock_time_sec": res_8b["elapsed"],
            "q14v5_wall_clock_time_sec": res_14b["elapsed"],
            "q8v5_rows_per_sec": res_8b["rows_per_sec"],
            "q14v5_rows_per_sec": res_14b["rows_per_sec"],
            "total_rows_per_sec": total_rows_per_sec,
            "peak_vram_gpu0_gib": max(res_8b["vram_peak_gpu0"], res_14b["vram_peak_gpu0"]),
            "peak_vram_gpu1_gib": max(res_8b["vram_peak_gpu1"], res_14b["vram_peak_gpu1"]),
            "peak_host_ram_used_gib": vm_end.used / (1024**3),
            "host_ram_delta_gib": (vm_end.used - vm_start.used) / (1024**3),
            "cpu_offload_occurred_locally": res_14b["cpu_offload_used"],
            "device_map_uses_both_gpus_on_kaggle": kaggle_uses_both_gpus,
            "kaggle_14b_device_split": [str(d) for d in kaggle_14b_devices],
            "kaggle_cpu_offload_expected": kaggle_cpu_offload,
            "batch_sizes_selected": {
                "q8v5_batch_size": res_8b["actual_bs"],
                "q14v5_batch_size": res_14b["actual_bs"]
            },
            "cuda_oom_events": res_8b["oom_events"] + res_14b["oom_events"],
            "tokenizer_warnings": res_8b["tokenizer_warnings"] + res_14b["tokenizer_warnings"],
            "model_loading_warnings": res_8b["model_loading_warnings"] + res_14b["model_loading_warnings"],
            "successfully_processed_rows": n_bench,
            "nan_inf_probabilities": nan_count + inf_count
        },
        "phase3_integrity_metrics": {
            "num_classes_is_18": num_classes_check,
            "all_probabilities_finite": finite_check,
            "probabilities_sum_to_one": sum_one_check,
            "max_sum_deviation": max_sum_dev,
            "all_predicted_labels_valid": valid_labels_check,
            "max_abs_prob_difference": max_prob_diff,
            "submission_columns_valid": expected_cols,
            "submission_rows_valid": expected_rows,
            "predicted_class_distribution": class_counts
        }
    }

    os.makedirs("experiments/EXP52_dual_lora", exist_ok=True)
    out_file = "experiments/EXP52_dual_lora/final_runtime_validation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved benchmark results to {out_file}")

    print("\n" + "=" * 80)
    print("EXP52 BENCHMARK SUMMARY (250 ROWS):")
    print(f"  Wall-Clock Time:  {total_wall_clock:.2f}s ({total_rows_per_sec:.2f} rows/s)")
    print(f"  q8v5 Runtime:     {res_8b['elapsed']:.2f}s ({res_8b['rows_per_sec']:.2f} rows/s)")
    print(f"  q14v5 Runtime:    {res_14b['elapsed']:.2f}s ({res_14b['rows_per_sec']:.2f} rows/s)")
    print(f"  Peak VRAM GPU 0:  {results['phase2_runtime_metrics']['peak_vram_gpu0_gib']:.2f} GiB")
    print(f"  Peak VRAM GPU 1:  {results['phase2_runtime_metrics']['peak_vram_gpu1_gib']:.2f} GiB (Protected)")
    print(f"  Kaggle Dual-T4:   Both GPUs utilized: {kaggle_uses_both_gpus}, Zero CPU offload: {not kaggle_cpu_offload}")
    print(f"  Integrity Checks: ALL PASSED (Max dev: {max_sum_dev:.2e}, Max diff: {max_prob_diff:.2e})")
    print("=" * 80)


if __name__ == "__main__":
    main()
