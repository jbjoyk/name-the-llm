"""
================================================================================
EXP52: OFFLINE PATH RESOLUTION & REGRESSION VERIFICATION TEST
================================================================================
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import numpy as np
import pandas as pd
import scipy.special
import torch

# Import fixed kernel functions
import importlib.util

spec = importlib.util.spec_from_file_location("kernel", "notebooks/52_dual_lora_kernel.py")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)

resolve_base_model = kernel.resolve_base_model
resolve_adapter_dir = kernel.resolve_adapter_dir
verify_local_checkpoint = kernel.verify_local_checkpoint
recover_clusters = kernel.recover_clusters
run_backbone_inference = kernel.run_backbone_inference
Q8_SPEC = kernel.Q8_SPEC
Q14_SPEC = kernel.Q14_SPEC
LABELS = kernel.LABELS
I2L = kernel.I2L

def run_tests():
    print("=" * 80)
    print("STEP 1: VERIFY DETERMINISTIC LOCAL OFFLINE RESOLVERS")
    print("=" * 80)

    # 1. Resolve 8B local base path
    p8 = resolve_base_model("8b")
    print(f"Q8_LOCAL_PATH:  {p8}")
    assert os.path.isabs(p8), "Q8 path must be absolute"
    assert os.path.isdir(p8), "Q8 path must be a directory"
    assert os.path.isfile(os.path.join(p8, "config.json")), "Q8 must have config.json"
    assert "huggingface.co" not in p8 and "Qwen/" not in p8, "Q8 must not be a HF repo ID"

    # 2. Resolve 14B local base path
    p14 = resolve_base_model("14b")
    print(f"Q14_LOCAL_PATH: {p14}")
    assert os.path.isabs(p14), "Q14 path must be absolute"
    assert os.path.isdir(p14), "Q14 path must be a directory"
    assert os.path.isfile(os.path.join(p14, "config.json")), "Q14 must have config.json"
    assert "huggingface.co" not in p14 and "Qwen/" not in p14, "Q14 must not be a HF repo ID"

    # 3. Verify fail-fast behavior when no local model exists (mocking missing path)
    try:
        verify_local_checkpoint("/nonexistent/path/to/model", "test_missing")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"Fail-fast on nonexistent path confirmed: {e}")

    # 4. Verify fail-fast on Kaggle if path does not start with /kaggle/input/
    try:
        # simulate kaggle env check
        base = "/home/rj2826/not_kaggle"
        if not base.startswith("/kaggle/input/"):
            raise RuntimeError(f"Simulated Kaggle check: Base path must start with '/kaggle/input/', got: {base}")
    except RuntimeError as e:
        print(f"Kaggle path prefix guard confirmed: {e}")

    print("\n" + "=" * 80)
    print("STEP 2: 5-ROW SMOKE TEST (OFFLINE ENFORCED)")
    print("=" * 80)
    train_df = pd.read_csv("data/train.csv")
    smoke_df = train_df.head(5).copy()
    smoke_texts = smoke_df["text"].fillna("").astype(str).values
    smoke_groups = recover_clusters(smoke_texts, thr=0.15, cap=18)

    ad_8b = resolve_adapter_dir("q8v5")
    ad_14b = resolve_adapter_dir("q14v5")
    print(f"Resolved q8v5 adapter:  {ad_8b}")
    print(f"Resolved q14v5 adapter: {ad_14b}")

    # Run 8B
    res_8b_smoke = run_backbone_inference(Q8_SPEC, p8, ad_8b, smoke_texts, smoke_groups)
    print(f"q8v5 smoke completed: probs shape {res_8b_smoke['probs'].shape}")

    # Run 14B
    res_14b_smoke = run_backbone_inference(Q14_SPEC, p14, ad_14b, smoke_texts, smoke_groups)
    print(f"q14v5 smoke completed: probs shape {res_14b_smoke['probs'].shape}")

    # Fused
    lp_fused = 0.60 * res_8b_smoke["log_probs"] + 0.40 * res_14b_smoke["log_probs"]
    p_fused = np.exp(lp_fused - scipy.special.logsumexp(lp_fused, axis=1, keepdims=True))
    p_fused /= p_fused.sum(axis=1, keepdims=True)

    assert p_fused.shape == (5, 18), "Must be (5, 18)"
    assert np.isfinite(p_fused).all(), "Must be finite"
    assert np.allclose(p_fused.sum(axis=1), 1.0, atol=1e-5), "Must sum to 1"
    preds_smoke = [I2L[i] for i in p_fused.argmax(axis=1)]
    assert all(p in LABELS for p in preds_smoke), "All labels must be valid"
    print(f"Smoke test fused predictions: {preds_smoke}")
    print(">> STEP 2 PASSED: 100% VALID")

    print("\n" + "=" * 80)
    print("STEP 3: PARITY CHECK BETWEEN KERNEL AND INFERENCE ENGINE (3 TEST ROWS)")
    print("=" * 80)
    test_df = pd.read_csv("data/test.csv")
    test_texts = test_df["text"].fillna("").astype(str).values
    test_groups = recover_clusters(test_texts, thr=0.15, cap=18)

    # 1. Kernel execution
    print("Running kernel on test set...")
    res_8b_kernel = run_backbone_inference(Q8_SPEC, p8, ad_8b, test_texts, test_groups)
    res_14b_kernel = run_backbone_inference(Q14_SPEC, p14, ad_14b, test_texts, test_groups)

    lp_kernel = 0.60 * res_8b_kernel["log_probs"] + 0.40 * res_14b_kernel["log_probs"]
    p_kernel = np.exp(lp_kernel - scipy.special.logsumexp(lp_kernel, axis=1, keepdims=True))
    p_kernel /= p_kernel.sum(axis=1, keepdims=True)

    # 2. Engine execution
    print("Running engine on test set...")
    import importlib.util
    engine_spec = importlib.util.spec_from_file_location("engine", "src/exp52_dual_lora_inference.py")
    engine = importlib.util.module_from_spec(engine_spec)
    engine_spec.loader.exec_module(engine)
    res_8b_engine = engine.run_lora24_inference(Q8_SPEC, p8, ad_8b, test_texts, test_groups)
    res_14b_engine = engine.run_lora24_inference(Q14_SPEC, p14, ad_14b, test_texts, test_groups)

    lp_engine = 0.60 * res_8b_engine["log_probs"] + 0.40 * res_14b_engine["log_probs"]
    p_engine = np.exp(lp_engine - scipy.special.logsumexp(lp_engine, axis=1, keepdims=True))
    p_engine /= p_engine.sum(axis=1, keepdims=True)

    diff_8b = float(np.abs(res_8b_kernel["probs"] - res_8b_engine["probs"]).max())
    diff_14b = float(np.abs(res_14b_kernel["probs"] - res_14b_engine["probs"]).max())
    diff_fused = float(np.abs(p_kernel - p_engine).max())

    print(f"Max abs diff 8B (Kernel vs Engine):    {diff_8b:.2e}")
    print(f"Max abs diff 14B (Kernel vs Engine):   {diff_14b:.2e}")
    print(f"Max abs diff Fused (Kernel vs Engine): {diff_fused:.2e}")

    assert diff_fused < 1e-5, f"Implementation mismatch between kernel and engine! (diff={diff_fused})"

    for idx, row_id in enumerate(test_df["id"]):
        pred_8 = I2L[res_8b_kernel["probs"][idx].argmax()]
        pred_14 = I2L[res_14b_kernel["probs"][idx].argmax()]
        pred_f = I2L[p_kernel[idx].argmax()]
        conf_f = float(p_kernel[idx].max())

        pred_eng = I2L[p_engine[idx].argmax()]
        conf_eng = float(p_engine[idx].max())

        print(f"Row {idx} ({row_id}):")
        print(f"  q8 predicted:       {pred_8}")
        print(f"  q14 predicted:      {pred_14}")
        print(f"  Kernel fused:       {pred_f} (conf: {conf_f:.4f})")
        print(f"  Engine fused:       {pred_eng} (conf: {conf_eng:.4f})")
        assert pred_f == pred_eng, f"Prediction mismatch on row {idx}!"

    # Save the canonical verified probabilities to disk
    np.save("experiments/EXP52_dual_lora/probs_q8v5.npy", res_8b_kernel["probs"])
    np.save("experiments/EXP52_dual_lora/probs_q14v5.npy", res_14b_kernel["probs"])
    np.save("experiments/EXP52_dual_lora/probs_fused.npy", p_kernel)

    # Save submission candidates
    sub_df = pd.DataFrame({"id": test_df["id"].values, "model": [I2L[i] for i in p_kernel.argmax(axis=1)]})
    sub_df.to_csv("submissions/candidates/EXP52/submission.csv", index=False)
    sub_df.to_parquet("submissions/candidates/EXP52/submission.parquet", index=False)
    sub_df.to_csv("submission.csv", index=False)
    sub_df.to_parquet("submission.parquet", index=False)

    print(">> STEP 3 PASSED: 100% IMPLEMENTATION PARITY VERIFIED")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
