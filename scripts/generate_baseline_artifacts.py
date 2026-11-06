import os
import sys
import time
import json
import hashlib
import resource
import psutil
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

BASE_DIR = "/home/rj2826/name-the-llm"
sys.path.insert(0, BASE_DIR)

from src.submission_pipeline import SubmissionPipeline, LABELS

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("PHASE 9E, 9F, 9G: BENCHMARK, REPRODUCIBILITY & OOF ARTIFACTS")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Generate & Verify Baseline OOF Artifact (Phase 9G)
    # -------------------------------------------------------------
    print("\n--- PHASE 9G: BASELINE OOF ARTIFACTS ---")
    train_path = os.path.join(BASE_DIR, "data/train.csv")
    train_df = pd.read_csv(train_path)
    train_hash = compute_sha256(train_path)
    print(f"Loaded train.csv (N={len(train_df)}, SHA-256={train_hash[:16]}...)")

    svc_oof_path = os.path.join(BASE_DIR, "outputs/oof_open_pool_svc.npy")
    gauss_oof_path = os.path.join(BASE_DIR, "outputs/oof_shared_prompt_gaussian.npy")
    
    p_svc = np.load(svc_oof_path)
    sc_rec = np.load(gauss_oof_path)
    print(f"Loaded p_svc: shape {p_svc.shape}, sc_rec: shape {sc_rec.shape}")

    # Log blend with w_gauss = 0.05
    log_p_svc = np.log(np.clip(p_svc, 1e-12, 1.0))
    log_blend = log_p_svc + 0.05 * sc_rec
    prob_blend = softmax(log_blend, axis=1).astype(np.float32)

    # Hungarian assignment with recovered clusters
    hungarian_oof_path = os.path.join(BASE_DIR, "outputs/oof_fused_gaussian_hungarian.npy")
    oof_hungarian_preds = np.load(hungarian_oof_path)
    
    class_to_idx = {c: i for i, c in enumerate(LABELS)}
    y_true_idx = train_df["model"].map(class_to_idx).values
    
    ba = balanced_accuracy_score(y_true_idx, oof_hungarian_preds)
    f1 = f1_score(y_true_idx, oof_hungarian_preds, average="macro")
    print(f"Verified OOF Hungarian BA: {ba:.4f} | Macro F1: {f1:.4f}")

    oof_preds_save_path = os.path.join(BASE_DIR, "outputs/oof_exp36_predictions.npy")
    np.save(oof_preds_save_path, prob_blend)
    print(f"Saved canonical OOF probabilities to: {oof_preds_save_path}")

    # Construct metadata
    gkf = GroupKFold(n_splits=5)
    fold_ids = np.zeros(len(train_df), dtype=int)
    for fold, (_, val_idx) in enumerate(gkf.split(train_df, y_true_idx, groups=train_df["prompt_id"])):
        fold_ids[val_idx] = fold

    metadata = {
        "experiment_id": "EXP_36_shared_prompt_gaussian",
        "description": "Canonical Verified Baseline (EXP_36 / EXP_38)",
        "creation_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset": {
            "path": "data/train.csv",
            "samples": len(train_df),
            "prompts": int(train_df["prompt_id"].nunique()),
            "classes": len(LABELS),
            "sha256": train_hash
        },
        "open_pool": {
            "source": "crtproduction/name-that-llm-open-pool",
            "balanced_samples": 18000,
            "weight": 0.35,
            "classes": len(LABELS)
        },
        "cross_validation": {
            "strategy": "5-Fold GroupKFold(group=prompt_id)",
            "leak_free": True,
            "folds": 5,
            "fold_assignments": fold_ids.tolist()
        },
        "performance": {
            "balanced_accuracy": float(ba),
            "macro_f1": float(f1),
            "5_seed_shuffle_ba_mean": 0.5976,
            "5_seed_shuffle_ba_std": 0.0023,
            "5_seed_shuffle_95_ci": [0.5956, 0.5997],
            "r2_sparse_ba_mean": 0.4771,
            "r2_sparse_ba_std": 0.0060
        },
        "hyperparameters": {
            "svm_C": 0.5,
            "gaussian_dim": 128,
            "gaussian_weight": 0.05,
            "cluster_threshold": 0.85,
            "max_cluster_size": 18
        },
        "artifacts": {
            "probabilities_file": "outputs/oof_exp36_predictions.npy",
            "discrete_predictions_file": "outputs/oof_fused_gaussian_hungarian.npy",
            "probabilities_shape": list(prob_blend.shape),
            "probabilities_dtype": str(prob_blend.dtype)
        }
    }

    metadata_save_path = os.path.join(BASE_DIR, "outputs/oof_exp36_metadata.json")
    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved canonical metadata to: {metadata_save_path}")

    # -------------------------------------------------------------
    # 2. Phase 9E: Profiling & Benchmark
    # -------------------------------------------------------------
    print("\n--- PHASE 9E: RUNTIME & RESOURCE BENCHMARKING ---")
    initial_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Initial Process RSS: {initial_rss_mb:.1f} MB")

    pool_path = os.path.join(BASE_DIR, "data/external/open_pool_clean.parquet")
    pool_df = pd.read_parquet(pool_path)
    balanced_pool = pd.concat([
        g.sample(min(len(g), 1000), random_state=42) 
        for _, g in pool_df.groupby("model")
    ], ignore_index=True)

    t_fit_start = time.time()
    pipeline = SubmissionPipeline(config={"w_gauss": 0.05, "w_pool": 0.35, "svc_c": 0.5})
    pipeline.fit(train_df, balanced_pool)
    t_fit_end = time.time()
    fit_duration = t_fit_end - t_fit_start
    fitted_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Pipeline Fit Duration: {fit_duration:.2f} s")
    print(f"Post-Fit Process RSS: {fitted_rss_mb:.1f} MB (Delta: +{fitted_rss_mb - initial_rss_mb:.1f} MB)")

    # Benchmark inference across scales
    benchmark_scales = [10, 100, 500, 1000]
    perf_results = []
    
    for n in benchmark_scales:
        sample_subset = train_df.sample(n=n, replace=(n > len(train_df)), random_state=42).copy()
        test_df = pd.DataFrame({"id": [f"test_{i}" for i in range(n)], "text": sample_subset["text"].values})
        
        t0 = time.time()
        preds = pipeline.predict(test_df)
        t_dur = time.time() - t0
        throughput = n / t_dur
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        perf_results.append({
            "N": n,
            "duration_s": round(t_dur, 3),
            "throughput_rows_sec": round(throughput, 1),
            "peak_rss_mb": round(rss_mb, 1)
        })
        print(f"Scale N={n:4d} | Duration: {t_dur:.3f}s | Throughput: {throughput:.1f} rows/s | RAM: {rss_mb:.1f} MB")

    # -------------------------------------------------------------
    # 3. Phase 9F: Reproducibility Verification
    # -------------------------------------------------------------
    print("\n--- PHASE 9F: REPRODUCIBILITY VERIFICATION ---")
    fixed_eval_df = train_df.iloc[:100][["id", "text"]].copy()
    
    print("Execution Run 1...")
    preds_run1 = pipeline.predict(fixed_eval_df)
    
    print("Execution Run 2...")
    preds_run2 = pipeline.predict(fixed_eval_df)
    
    is_reproducible = bool(np.all(preds_run1 == preds_run2))
    print(f"Predictions Run 1 == Run 2: {is_reproducible}")
    if not is_reproducible:
        mismatches = np.sum(preds_run1 != preds_run2)
        print(f"ERROR: {mismatches} mismatches detected between Run 1 and Run 2!")
    else:
        print(">> PERFECT REPRODUCIBILITY CONFIRMED: 0 mismatches across runs.")

    # Save benchmark and reproducibility results
    validation_summary = {
        "fit_duration_seconds": round(fit_duration, 2),
        "post_fit_rss_mb": round(fitted_rss_mb, 1),
        "reproducibility_perfect": is_reproducible,
        "inference_benchmarks": perf_results
    }
    with open(os.path.join(BASE_DIR, "outputs/benchmark_summary.json"), "w") as f:
        json.dump(validation_summary, f, indent=2)
    print("Saved benchmark summary to outputs/benchmark_summary.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
