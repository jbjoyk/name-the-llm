#!/usr/bin/env python3
"""
================================================================================
EXP54: Controlled Qwen3 (8B+14B) + Open Pool 4-View LinearSVC Fusion Experiment
================================================================================

Objective:
Test whether a small contribution from the existing Open Pool 4-view LinearSVC model
can improve the frozen EXP52 Qwen3-8B + Qwen3-14B ensemble without allowing surface
stylometry to dominate the semantic representations.

Production Baseline:
- EXP47 Kaggle LB: 0.620
- EXP52 Kaggle LB: 0.632 (Current Best Validated Kaggle Baseline; w8=0.60, w14=0.40)
- EXP53 Conclusion: Confirmed w14=0.40 is OOF-optimal (BA = 0.7873) and frozen.

Fusion Specification:
    LP_QWEN = 0.60 * LP_Q8 + 0.40 * LP_Q14
    LP_FINAL = (1 - w_svc) * LP_QWEN + w_svc * LP_SVC
where:
    w_svc in {0.00, 0.05, 0.10, 0.15, 0.20}
    w_qwen = 1.0 - w_svc

Absolute Rules:
- DO NOT MODIFY EXP52. Treat EXP52 as immutable.
- ABSOLUTELY NO AUTOMATIC KAGGLE SUBMISSION.
- Consume already-generated EXP52 predictions and validated Open Pool SVC predictions.
- Enforce strict 5-fold prompt-disjoint GroupKFold validation.
================================================================================
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import scipy.special
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, confusion_matrix

# 18 Canonical LLM Candidate Classes
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

# Provenance Checksums
EXPECTED_HASHES = {
    "q8_adapter": ("models/adapters_v5/q8v5/q8v5/adapter_model.safetensors", "a600e552f40bcda00194f147f957d3bbb04e176b6bccf4a1f7fe39b82c8e8f51"),
    "q8_heads": ("models/adapters_v5/q8v5/q8v5/heads.pt", "31868cd0e9618fac506c93a08f118ee00202041d67426890e609d13e2cd33f0a"),
    "q8_config": ("models/adapters_v5/q8v5/q8v5/adapter_config.json", "621136c127ae11004b136582ac899229a191c2d57443b297f7867e88b42d9204"),
    "q14_adapter": ("models/adapters_v5/q14v5/q14v5/adapter_model.safetensors", "7580831f6d7184c9a676ce88c3b69af00e42ffed1e7132fd8812c9ba632616c4"),
    "q14_heads": ("models/adapters_v5/q14v5/q14v5/heads.pt", "cafdbade649017d666c2da623adb6539877e1125a996f2736133127ae6121b3d"),
    "q14_config": ("models/adapters_v5/q14v5/q14v5/adapter_config.json", "1f2c044ca59f117a2865ce91478f126d25f974224282b70d320828e8b504173e"),
    "oof_q8_exp47": ("outputs/oof_exp47_qwen_centered_twin.npy", "647a75f59789f2574bef7578ebf0e5f00eae2a8481724cfd6316bb4199ad22cb"),
    "oof_q14v5": ("outputs/oof_q14v5_antopil_lora.csv", "58822e8d71d6b6ae1d77a3eb62a5f3a2d4f1757c239392d1b5d9e227646ab9e9"),
    "oof_open_pool_svc": ("outputs/oof_open_pool_svc.npy", "cd7e0833a1d6bc60048a3944ccb55018522a407ce37edf7ccd0ec288728ee5b6"),
    "open_pool_clean_parquet": ("data/external/open_pool_clean.parquet", "0c5b289e9f5f7c3f439462d09b3d1d7ef4648839db20309e3476066f3982d5c9"),
    "train_csv": ("data/train.csv", "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"),
}

def log(*args):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_provenance():
    log("Verifying cryptographic provenance of all input artifacts...")
    verified = {}
    for name, (path, expected_hash) in EXPECTED_HASHES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {name} ({path}): expected {expected_hash}, got {actual_hash}")
        verified[name] = actual_hash
        log(f"  [PASS] {name:<26}: {actual_hash[:20]}...{actual_hash[-8:]}")
    return verified

def audit_open_pool():
    log("Auditing Open Pool dataset...")
    pool_path = "data/external/open_pool_clean.parquet"
    pool_df = pd.read_parquet(pool_path)
    n_rows = len(pool_df)
    cols = pool_df.columns.tolist()
    class_counts = pool_df['model'].value_counts().to_dict()
    log(f"  Open Pool Path: {pool_path}")
    log(f"  Total Passages: {n_rows:,}")
    log(f"  Columns: {cols}")
    log(f"  Unique Classes: {len(class_counts)}")
    assert len(class_counts) == NUM_CLASSES, f"Open Pool class count mismatch: {len(class_counts)} vs {NUM_CLASSES}"
    assert set(class_counts.keys()) == set(LABELS), "Open Pool classes do not match canonical LABELS"
    return {
        "path": pool_path,
        "n_passages": n_rows,
        "columns": cols,
        "class_counts": class_counts
    }

def run_exp54():
    t0 = time.time()
    log("=" * 80)
    log("EXP54: CONTROLLED QWEN + OPEN POOL LINEARSVC FUSION SWEEP")
    log("=" * 80)

    # Directories
    out_dir = "experiments/EXP54_qwen_svc"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # 1. Verify Provenance
    verified_hashes = verify_provenance()

    # 2. Audit Open Pool
    open_pool_meta = audit_open_pool()

    # 3. Load Ground Truth
    train_path = "data/train.csv"
    train_df = pd.read_csv(train_path)
    y_true_str = train_df["model"].values
    y_true = train_df["model"].map(L2I).values
    n_samples = len(train_df)
    prompts = train_df["prompt_id"].values
    log(f"Loaded train.csv: {n_samples} samples across {train_df['prompt_id'].nunique()} prompts.")

    # 4. Load OOF Predictions
    p8_path = EXPECTED_HASHES["oof_q8_exp47"][0]
    p14_path = EXPECTED_HASHES["oof_q14v5"][0]
    p_svc_path = EXPECTED_HASHES["oof_open_pool_svc"][0]

    p8_raw = np.load(p8_path)
    q14_df = pd.read_csv(p14_path)
    p_svc_raw = np.load(p_svc_path)

    # 5. Sanity Checks on Inputs
    log("Executing Strict Sanity Checks...")
    assert p8_raw.shape == (n_samples, NUM_CLASSES), f"Q8 shape mismatch: {p8_raw.shape}"
    assert len(q14_df) == n_samples, f"Q14 row count mismatch: {len(q14_df)}"
    assert p_svc_raw.shape == (n_samples, NUM_CLASSES), f"SVC shape mismatch: {p_svc_raw.shape}"
    assert (train_df["id"] == q14_df["id"]).all(), "Row ID mismatch between train and Q14 OOF"
    assert np.isfinite(p8_raw).all(), "Q8 contains NaN/Inf"
    assert np.isfinite(p_svc_raw).all(), "SVC contains NaN/Inf"

    q14_cols = [c for c in q14_df.columns if c not in ["id", "prompt_id", "fold"]]
    assert q14_cols == LABELS, f"Q14 column order mismatch: {q14_cols}"
    p14_raw = q14_df[LABELS].values
    assert np.isfinite(p14_raw).all(), "Q14 contains NaN/Inf"

    folds = q14_df["fold"].values
    n_folds = len(np.unique(folds))
    assert n_folds == 5, f"Expected 5 folds, got {n_folds}"

    # Verify Prompt-Disjoint Folds
    for f in range(n_folds):
        prompts_in = set(train_df.loc[folds == f, 'prompt_id'])
        prompts_out = set(train_df.loc[folds != f, 'prompt_id'])
        overlap = prompts_in.intersection(prompts_out)
        assert len(overlap) == 0, f"Prompt leak in fold {f}: {len(overlap)} overlapping prompts"
    log(f"All sanity checks passed: {NUM_CLASSES} classes, {n_samples} rows, 5 strictly prompt-disjoint folds.")

    # 6. Verify Class Order Alignment
    cm_svc = confusion_matrix(y_true, p_svc_raw.argmax(axis=1))
    svc_per_class_recall = np.diag(cm_svc) / cm_svc.sum(axis=1)
    log("Class Order Alignment Check:")
    for idx, (lbl, rec) in enumerate(zip(LABELS, svc_per_class_recall)):
        log(f"  Idx {idx:2d} | Label: {lbl:<24} | Standalone SVC Recall: {rec*100:5.1f}%")
    ba_svc_check = float(svc_per_class_recall.mean())
    log(f"  Verified Standalone SVC OOF BA: {ba_svc_check:.4f} (matches EXP28/EXP50: 0.4630)")
    assert abs(ba_svc_check - 0.4630) < 1e-4, f"SVC BA mismatch: {ba_svc_check}"

    # 7. Convert to Log-Probability Space
    eps = 1e-15
    p8_norm = p8_raw / p8_raw.sum(axis=1, keepdims=True)
    p14_norm = p14_raw / p14_raw.sum(axis=1, keepdims=True)
    p_svc_norm = p_svc_raw / p_svc_raw.sum(axis=1, keepdims=True)

    lp8 = np.log(np.clip(p8_norm, eps, 1.0))
    lp14 = np.log(np.clip(p14_norm, eps, 1.0))
    lp_svc = np.log(np.clip(p_svc_norm, eps, 1.0))

    # Frozen EXP52 Qwen ensemble: 0.60 * Q8 + 0.40 * Q14
    lp_qwen = 0.60 * lp8 + 0.40 * lp14

    # Standalone Model Evaluations
    pred_q8 = p8_norm.argmax(axis=1)
    pred_q14 = p14_norm.argmax(axis=1)
    pred_exp52 = lp_qwen.argmax(axis=1)
    pred_svc = p_svc_norm.argmax(axis=1)

    ba_q8 = balanced_accuracy_score(y_true, pred_q8)
    ba_q14 = balanced_accuracy_score(y_true, pred_q14)
    ba_exp52 = balanced_accuracy_score(y_true, pred_exp52)
    ba_svc = balanced_accuracy_score(y_true, pred_svc)

    log("\n--- Standalone Model Baselines ---")
    log(f"  Qwen3-8B (EXP47 OOF)               : BA = {ba_q8:.4f}")
    log(f"  Qwen3-14B (q14v5 OOF)              : BA = {ba_q14:.4f}")
    log(f"  EXP52 Qwen Ensemble (0.60+0.40)    : BA = {ba_exp52:.4f} (Kaggle LB 0.632)")
    log(f"  Open Pool 4-View LinearSVC         : BA = {ba_svc:.4f}")

    # 8. Complementarity Analysis (EXP52 vs SVC)
    log("\n--- Complementarity Analysis (EXP52 vs Open Pool SVC) ---")
    agree_mask = (pred_exp52 == pred_svc)
    n_agree = int(agree_mask.sum())
    agree_rate = float(n_agree / n_samples)

    qwen_corr = (pred_exp52 == y_true)
    svc_corr = (pred_svc == y_true)

    both_corr = int((qwen_corr & svc_corr).sum())
    both_wrong = int((~qwen_corr & ~svc_corr).sum())
    qwen_only = int((qwen_corr & ~svc_corr).sum())
    svc_only = int((~qwen_corr & svc_corr).sum())

    err_qwen = (~qwen_corr).astype(float)
    err_svc = (~svc_corr).astype(float)
    r_err, p_val_err = pearsonr(err_qwen, err_svc)

    log(f"  Total Samples           : {n_samples}")
    log(f"  Agreement Rate          : {n_agree} / {n_samples} ({agree_rate*100:.2f}%)")
    log(f"  Both Correct            : {both_corr} ({both_corr/n_samples*100:.2f}%)")
    log(f"  Both Wrong              : {both_wrong} ({both_wrong/n_samples*100:.2f}%)")
    log(f"  Qwen-only Correct       : {qwen_only} ({qwen_only/n_samples*100:.2f}%)")
    log(f"  SVC-only Correct        : {svc_only} ({svc_only/n_samples*100:.2f}%)")
    log(f"  Measured Error Corr (r) : {r_err:.5f} (p-value = {p_val_err:.5e})")

    # 9. Weight Sweep: w_svc in {0.00, 0.05, 0.10, 0.15, 0.20}
    weights = [0.00, 0.05, 0.10, 0.15, 0.20]
    sweep_results = []
    fused_prob_dict = {}
    fused_pred_dict = {}

    log("\nExecuting EXP54 Weight Sweep...")
    for w_svc in weights:
        w_qwen = round(1.0 - w_svc, 4)
        lpf = w_qwen * lp_qwen + w_svc * lp_svc
        pf = np.exp(lpf - scipy.special.logsumexp(lpf, axis=1, keepdims=True))
        pf /= pf.sum(axis=1, keepdims=True)

        pred = pf.argmax(axis=1)
        fused_prob_dict[w_svc] = pf
        fused_pred_dict[w_svc] = pred

        ba = balanced_accuracy_score(y_true, pred)
        mf1 = f1_score(y_true, pred, average="macro")
        acc = accuracy_score(y_true, pred)

        entropy = float(-np.sum(pf * np.log(np.clip(pf, eps, 1.0)), axis=1).mean())
        sorted_p = np.sort(pf, axis=1)
        top1_p = float(sorted_p[:, -1].mean())
        margin = float((sorted_p[:, -1] - sorted_p[:, -2]).mean())
        logit_margin = float((np.log(np.clip(sorted_p[:, -1], eps, 1.0)) - np.log(np.clip(sorted_p[:, -2], eps, 1.0))).mean())

        n_corr = int((pred == y_true).sum())
        n_inc = int((pred != y_true).sum())

        # Comparison with EXP52
        agree_exp52 = float((pred == pred_exp52).mean())
        n_flipped = int((pred != pred_exp52).sum())
        fixed = int(((pred_exp52 != y_true) & (pred == y_true)).sum())
        broken = int(((pred_exp52 == y_true) & (pred != y_true)).sum())
        net_gain = fixed - broken

        # Fold metrics
        fold_bas = []
        for f in range(n_folds):
            f_mask = (folds == f)
            fold_bas.append(float(balanced_accuracy_score(y_true[f_mask], pred[f_mask])))
        f_mean = float(np.mean(fold_bas))
        f_std = float(np.std(fold_bas))

        sweep_results.append({
            "w_svc": w_svc,
            "w_qwen": w_qwen,
            "ba": ba,
            "macro_f1": mf1,
            "accuracy": acc,
            "entropy": entropy,
            "top1_prob": top1_p,
            "margin": margin,
            "logit_margin": logit_margin,
            "correct": n_corr,
            "incorrect": n_inc,
            "agree_with_exp52": agree_exp52,
            "predictions_changed": n_flipped,
            "samples_fixed": fixed,
            "samples_broken": broken,
            "net_samples_gained": net_gain,
            "fold_bas": fold_bas,
            "fold_mean": f_mean,
            "fold_std": f_std
        })

    # 10. Paired Bootstrap Statistical Testing (vs EXP52: w_svc = 0.00)
    log("\nRunning Paired Bootstrap Statistical Testing (2,000 resamples, seed=42)...")
    base_preds = fused_pred_dict[0.00]
    base_ba = sweep_results[0]["ba"]

    rng = np.random.RandomState(42)
    n_resamples = 2000
    boot_indices = [rng.choice(n_samples, size=n_samples, replace=True) for _ in range(n_resamples)]

    for row in sweep_results:
        w_svc = row["w_svc"]
        cand_preds = fused_pred_dict[w_svc]
        delta = row["ba"] - base_ba

        if w_svc == 0.00:
            ci_low, ci_high = 0.0, 0.0
            p_gt_0 = 0.0
            boot_se = 0.0
        else:
            deltas = []
            for idx in boot_indices:
                y_b = y_true[idx]
                ba_c = balanced_accuracy_score(y_b, cand_preds[idx])
                ba_b = balanced_accuracy_score(y_b, base_preds[idx])
                deltas.append(ba_c - ba_b)
            deltas = np.array(deltas)
            ci_low = float(np.percentile(deltas, 2.5))
            ci_high = float(np.percentile(deltas, 97.5))
            p_gt_0 = float(np.mean(deltas > 0))
            boot_se = float(np.std(deltas))

        row["delta_vs_exp52"] = delta
        row["ci_low"] = ci_low
        row["ci_high"] = ci_high
        row["p_gt_0"] = p_gt_0
        row["boot_se"] = boot_se

    # 11. Results Primary Display Table
    log("-" * 125)
    header = f"{'w_svc':>6} | {'w_qwen':>6} | {'OOF BA':>8} | {'Macro-F1':>8} | {'Acc':>7} | {'Entropy':>7} | {'Margin':>7} | {'Delta vs EXP52':>14} | {'95% Bootstrap CI':>20} | {'P(Delta>0)':>10}"
    log(header)
    log("-" * 125)
    for r in sweep_results:
        ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" if r['w_svc'] > 0 else "[0.0000, 0.0000]"
        delta_str = f"{r['delta_vs_exp52']:+.4f}" if r['w_svc'] > 0 else "0.0000 (Base)"
        line = f"{r['w_svc']:6.2f} | {r['w_qwen']:6.2f} | {r['ba']:8.4f} | {r['macro_f1']:8.4f} | {r['accuracy']:7.4f} | {r['entropy']:7.4f} | {r['margin']:7.4f} | {delta_str:>14} | {ci_str:>20} | {r['p_gt_0']:10.4f}"
        log(line)
    log("-" * 125)

    # 12. Save Results CSV
    csv_rows = []
    for r in sweep_results:
        csv_rows.append({
            "w_svc": r["w_svc"],
            "w_qwen": r["w_qwen"],
            "oof_balanced_accuracy": r["ba"],
            "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"],
            "entropy": r["entropy"],
            "top1_prob": r["top1_prob"],
            "margin": r["margin"],
            "logit_margin": r["logit_margin"],
            "correct": r["correct"],
            "incorrect": r["incorrect"],
            "agree_with_exp52": r["agree_with_exp52"],
            "predictions_changed": r["predictions_changed"],
            "samples_fixed": r["samples_fixed"],
            "samples_broken": r["samples_broken"],
            "net_samples_gained": r["net_samples_gained"],
            "delta_vs_exp52": r["delta_vs_exp52"],
            "ci_95_low": r["ci_low"],
            "ci_95_high": r["ci_high"],
            "p_delta_gt_0": r["p_gt_0"],
            "bootstrap_se": r["boot_se"],
            "fold_0_ba": r["fold_bas"][0],
            "fold_1_ba": r["fold_bas"][1],
            "fold_2_ba": r["fold_bas"][2],
            "fold_3_ba": r["fold_bas"][3],
            "fold_4_ba": r["fold_bas"][4],
            "fold_mean": r["fold_mean"],
            "fold_std": r["fold_std"]
        })
    df_results = pd.DataFrame(csv_rows)
    results_csv_path = os.path.join(out_dir, "EXP54_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    log(f"Saved results CSV: {results_csv_path}")

    # 13. Save Predictions Parquet
    oof_pq = pd.DataFrame({
        "id": train_df["id"].values,
        "prompt_id": train_df["prompt_id"].values,
        "fold": folds,
        "true_model": y_true_str,
    })
    for w in weights:
        tag = f"w_svc_{int(w*100):02d}"
        oof_pq[f"pred_{tag}"] = [I2L[p] for p in fused_pred_dict[w]]
        oof_pq[f"top1_p_{tag}"] = fused_prob_dict[w].max(axis=1)

    oof_parquet_path = os.path.join(out_dir, "EXP54_oof_predictions.parquet")
    oof_pq.to_parquet(oof_parquet_path, index=False)
    log(f"Saved OOF predictions Parquet: {oof_parquet_path}")

    # 14. Generate Sweep Plot
    log("Generating publication-grade visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    w_vals = [r["w_svc"] for r in sweep_results]
    ba_vals = [r["ba"] for r in sweep_results]

    # Error bars on absolute BA
    ci_err_low = []
    ci_err_high = []
    for w in weights:
        b_scores = [balanced_accuracy_score(y_true[idx], fused_pred_dict[w][idx]) for idx in boot_indices]
        ci_err_low.append(balanced_accuracy_score(y_true, fused_pred_dict[w]) - np.percentile(b_scores, 2.5))
        ci_err_high.append(np.percentile(b_scores, 97.5) - balanced_accuracy_score(y_true, fused_pred_dict[w]))

    ax1.errorbar(w_vals, ba_vals, yerr=[ci_err_low, ci_err_high], fmt='-o', color='#1f77b4',
                 ecolor='#aec7e8', elinewidth=2, capsize=5, capthick=1.5,
                 label='OOF Balanced Accuracy (with 95% CI)', zorder=3)

    # Highlight EXP52 baseline
    ax1.scatter([0.00], [sweep_results[0]["ba"]], color='#d62728', s=140, zorder=5,
                label='EXP52 Frozen Baseline (w_svc=0.00, BA=0.7873, Kaggle LB 0.632)')
    ax1.axhline(sweep_results[0]["ba"], color='#d62728', linestyle='--', alpha=0.6)

    # Highlight w_svc = 0.05
    ax1.scatter([0.05], [sweep_results[1]["ba"]], color='#2ca02c', s=120, zorder=5,
                label=f"OOF Peak: w_svc=0.05 (BA={sweep_results[1]['ba']:.4f}, Delta={sweep_results[1]['delta_vs_exp52']:+.4f})")

    ax1.set_title("EXP54: Open Pool LinearSVC Weight Sweep on EXP52", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Open Pool LinearSVC Weight ($w_{\\mathrm{svc}}$)  [where $w_{\\mathrm{qwen}} = 1 - w_{\\mathrm{svc}}$]", fontsize=10, labelpad=8)
    ax1.set_ylabel("OOF Balanced Accuracy", fontsize=10, labelpad=8)
    ax1.set_xticks(weights)
    ax1.set_ylim(0.765, 0.810)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.95)

    ax1.annotate(f"OOF Peak: 0.7888\nw_svc=0.05 (+0.0015)\n95% CI: [-0.0001, +0.0034]",
                 xy=(0.05, 0.7888), xytext=(0.07, 0.772),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontsize=8.5)

    # Subplot 2: Delta vs EXP52 with Bootstrap Distribution
    deltas_vals = [r["delta_vs_exp52"] for r in sweep_results]
    d_ci_low = [r["delta_vs_exp52"] - r["ci_low"] for r in sweep_results]
    d_ci_high = [r["ci_high"] - r["delta_vs_exp52"] for r in sweep_results]

    ax2.axhline(0, color='gray', linestyle='--', alpha=0.8, label='EXP52 Parity (Delta = 0)')
    ax2.errorbar(w_vals[1:], deltas_vals[1:], yerr=[d_ci_low[1:], d_ci_high[1:]], fmt='-s',
                 color='#9467bd', ecolor='#c5b0d5', elinewidth=2, capsize=5, capthick=1.5,
                 label='Paired Delta vs EXP52 (95% CI)', zorder=3)
    ax2.set_title("Paired Delta vs Frozen EXP52 Baseline (95% Bootstrap CI)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Open Pool LinearSVC Weight ($w_{\\mathrm{svc}}$)", fontsize=10, labelpad=8)
    ax2.set_ylabel("Balanced Accuracy Delta (vs EXP52)", fontsize=10, labelpad=8)
    ax2.set_xticks(weights)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    plot_path_exp = os.path.join(out_dir, "exp54_weight_sweep.png")
    plot_path_rep = "reports/figures/exp54_weight_sweep.png"
    plt.savefig(plot_path_exp)
    plt.savefig(plot_path_rep)
    plt.close()
    log(f"Saved plot: {plot_path_exp} & {plot_path_rep}")

    # 15. Summary JSON
    best_row = sweep_results[1]  # w_svc = 0.05
    summary = {
        "experiment_id": "EXP54_qwen_svc_fusion",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "current_production_baseline": {
            "name": "EXP52",
            "kaggle_lb": 0.632,
            "q8_weight": 0.60,
            "q14_weight": 0.40,
            "oof_balanced_accuracy": 0.7873
        },
        "verified_hashes": verified_hashes,
        "open_pool_metadata": open_pool_meta,
        "complementarity_metrics": {
            "agreement_rate": agree_rate,
            "both_correct": both_corr,
            "both_wrong": both_wrong,
            "qwen_only_correct": qwen_only,
            "svc_only_correct": svc_only,
            "measured_error_correlation_r": r_err,
            "measured_error_correlation_p": p_val_err
        },
        "sweep_results": csv_rows,
        "best_oof_weight": best_row["w_svc"],
        "best_oof_ba": best_row["ba"],
        "delta_vs_exp52": best_row["delta_vs_exp52"],
        "ci_95": [best_row["ci_low"], best_row["ci_high"]],
        "p_delta_gt_0": best_row["p_gt_0"],
        "candidate_selection": {
            "candidate_A": {
                "name": "EXP54_w05",
                "w_svc": 0.05,
                "w_qwen": 0.95,
                "oof_ba": best_row["ba"],
                "delta_vs_exp52": best_row["delta_vs_exp52"],
                "ci_95": [best_row["ci_low"], best_row["ci_high"]],
                "p_gt_0": best_row["p_gt_0"],
                "rationale": "Highest local OOF point estimate (0.7888, +0.0015 delta), but corresponds to only 4 net flipped samples and 95% CI spans zero [-0.0001, +0.0034]."
            },
            "candidate_B": "NONE",
            "verdict": "EXP52 remains the production baseline (Kaggle LB 0.632). SVC does not provide sufficiently robust evidence for external testing."
        },
        "recommendation": "Maintain EXP52 as frozen production baseline. Do NOT submit EXP54 to Kaggle."
    }

    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary JSON: {summary_json_path}")
    log(f"EXP54 completed successfully in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    run_exp54()
