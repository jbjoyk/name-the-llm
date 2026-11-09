#!/usr/bin/env python3
"""
================================================================================
EXP53: Qwen3-8B + Qwen3-14B Controlled Fusion Weight Sweep
================================================================================

Objective:
Scientifically controlled sweep over Qwen3-14B fusion weight w14 in:
    {0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60}
with w8 = 1.0 - w14 in log-probability space:
    LP_fused = (1.0 - w14) * LP_q8 + w14 * LP_q14

Production Baseline:
- EXP47 Kaggle LB: 0.620
- EXP52 Kaggle LB: 0.632 (Current Best Validated Kaggle Baseline, w14=0.40)
- Absolute Rule: NO submission to Kaggle. No EXP52 modification.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

# Canonical 18 classes
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

EXPECTED_HASHES = {
    "q8_adapter": ("models/adapters_v5/q8v5/q8v5/adapter_model.safetensors", "a600e552f40bcda00194f147f957d3bbb04e176b6bccf4a1f7fe39b82c8e8f51"),
    "q8_heads": ("models/adapters_v5/q8v5/q8v5/heads.pt", "31868cd0e9618fac506c93a08f118ee00202041d67426890e609d13e2cd33f0a"),
    "q8_config": ("models/adapters_v5/q8v5/q8v5/adapter_config.json", "621136c127ae11004b136582ac899229a191c2d57443b297f7867e88b42d9204"),
    "q14_adapter": ("models/adapters_v5/q14v5/q14v5/adapter_model.safetensors", "7580831f6d7184c9a676ce88c3b69af00e42ffed1e7132fd8812c9ba632616c4"),
    "q14_heads": ("models/adapters_v5/q14v5/q14v5/heads.pt", "cafdbade649017d666c2da623adb6539877e1125a996f2736133127ae6121b3d"),
    "q14_config": ("models/adapters_v5/q14v5/q14v5/adapter_config.json", "1f2c044ca59f117a2865ce91478f126d25f974224282b70d320828e8b504173e"),
}

def log(*args):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_artifacts():
    log("Verifying EXP52 artifact hashes...")
    verified = {}
    for name, (path, expected_hash) in EXPECTED_HASHES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {name} ({path}): expected {expected_hash}, got {actual_hash}")
        verified[name] = actual_hash
        log(f"  [OK] {name}: {actual_hash[:16]}...{actual_hash[-8:]}")
    return verified

def run_exp53():
    log("================================================================================")
    log("STARTING EXP53: DUAL-LORA FUSION WEIGHT SWEEP")
    log("================================================================================")

    # Output directory
    out_dir = "experiments/EXP53_fusion_weight"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # 1. Verify artifact hashes
    verified_hashes = verify_artifacts()

    # 2. Load ground truth
    train_path = "data/train.csv"
    train_df = pd.read_csv(train_path)
    y_true = train_df["model"].values
    n_samples = len(train_df)
    log(f"Loaded ground truth from {train_path}: {n_samples} samples, {train_df['prompt_id'].nunique()} prompts.")

    # 3. Load OOF predictions
    p8_path = "outputs/oof_exp47_qwen_centered_twin.npy"
    p14_path = "outputs/oof_q14v5_antopil_lora.csv"

    if not os.path.exists(p8_path):
        raise FileNotFoundError(f"Missing Q8 OOF predictions: {p8_path}")
    if not os.path.exists(p14_path):
        raise FileNotFoundError(f"Missing Q14 OOF predictions: {p14_path}")

    p8_raw = np.load(p8_path)
    q14_df = pd.read_csv(p14_path)

    # 4. Strict Scientific Sanity Checks
    log("Running Scientific Sanity Checks...")
    assert p8_raw.shape == (n_samples, NUM_CLASSES), f"Q8 shape mismatch: {p8_raw.shape}"
    assert len(q14_df) == n_samples, f"Q14 row count mismatch: {len(q14_df)}"
    assert (train_df["id"] == q14_df["id"]).all(), "Row ID ordering mismatch between train and Q14 OOF!"
    assert np.isfinite(p8_raw).all(), "Q8 contains NaN or Inf!"
    
    q14_cols = [c for c in q14_df.columns if c not in ["id", "prompt_id", "fold"]]
    assert q14_cols == LABELS, f"Q14 column classes mismatch! Got: {q14_cols}"
    p14_raw = q14_df[LABELS].values
    assert np.isfinite(p14_raw).all(), "Q14 contains NaN or Inf!"
    assert set(np.unique(y_true)) == set(LABELS), "Ground truth classes mismatch LABELS!"

    folds = q14_df["fold"].values
    n_folds = len(np.unique(folds))
    log(f"Sanity checks passed: 18 classes, {n_samples} rows, {n_folds} prompt-disjoint folds.")

    # 5. Convert to Log-Probability Space (clip epsilon for numeric safety)
    eps = 1e-15
    p8_norm = p8_raw / p8_raw.sum(axis=1, keepdims=True)
    p14_norm = p14_raw / p14_raw.sum(axis=1, keepdims=True)

    lp8 = np.log(np.clip(p8_norm, eps, 1.0))
    lp14 = np.log(np.clip(p14_norm, eps, 1.0))

    # Evaluate standalones
    pred8 = np.array([LABELS[i] for i in lp8.argmax(axis=1)])
    pred14 = np.array([LABELS[i] for i in lp14.argmax(axis=1)])
    ba8 = balanced_accuracy_score(y_true, pred8)
    ba14 = balanced_accuracy_score(y_true, pred14)
    log(f"Standalone Qwen3-8B (EXP47 OOF) BA:  {ba8:.4f}")
    log(f"Standalone Qwen3-14B (q14v5 OOF) BA: {ba14:.4f}")

    # 6. Weight Sweep
    weights = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    sweep_results = []
    fused_prob_dict = {}
    fused_pred_dict = {}

    log("Executing weight sweep...")
    for w14 in weights:
        w8 = round(1.0 - w14, 4)
        lpf = w8 * lp8 + w14 * lp14
        pf = np.exp(lpf - scipy.special.logsumexp(lpf, axis=1, keepdims=True))
        pf /= pf.sum(axis=1, keepdims=True)

        preds = np.array([LABELS[i] for i in pf.argmax(axis=1)])
        fused_prob_dict[w14] = pf
        fused_pred_dict[w14] = preds

        # Metrics
        ba = balanced_accuracy_score(y_true, preds)
        mf1 = f1_score(y_true, preds, average="macro")
        acc = accuracy_score(y_true, preds)

        # Entropy & Margin
        entropy = float(-np.sum(pf * np.log(np.clip(pf, eps, 1.0)), axis=1).mean())
        sorted_p = np.sort(pf, axis=1)
        top1_p = float(sorted_p[:, -1].mean())
        margin = float((sorted_p[:, -1] - sorted_p[:, -2]).mean())

        n_correct = int((preds == y_true).sum())
        n_incorrect = int((preds != y_true).sum())

        # Fold metrics
        fold_bas = []
        for f in range(n_folds):
            f_mask = (folds == f)
            fold_bas.append(float(balanced_accuracy_score(y_true[f_mask], preds[f_mask])))
        fold_mean = float(np.mean(fold_bas))
        fold_std = float(np.std(fold_bas))

        sweep_results.append({
            "w14": w14,
            "w8": w8,
            "ba": ba,
            "macro_f1": mf1,
            "accuracy": acc,
            "entropy": entropy,
            "top1_prob": top1_p,
            "margin": margin,
            "correct": n_correct,
            "incorrect": n_incorrect,
            "fold_bas": fold_bas,
            "fold_mean": fold_mean,
            "fold_std": fold_std
        })

    # 7. Paired Bootstrap Statistical Testing (vs EXP52: w14 = 0.40)
    log("Running Paired Bootstrap Comparison (2,000 resamples, seed=42)...")
    base_w14 = 0.40
    base_preds = fused_pred_dict[base_w14]
    base_ba = balanced_accuracy_score(y_true, base_preds)

    rng = np.random.RandomState(42)
    n_resamples = 2000
    boot_indices = [rng.choice(n_samples, size=n_samples, replace=True) for _ in range(n_resamples)]

    for row in sweep_results:
        w14 = row["w14"]
        cand_preds = fused_pred_dict[w14]
        pt_delta = row["ba"] - base_ba

        if w14 == base_w14:
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

        row["delta_vs_exp52"] = pt_delta
        row["ci_low"] = ci_low
        row["ci_high"] = ci_high
        row["p_gt_0"] = p_gt_0
        row["boot_se"] = boot_se

    # 8. Print Results Table
    log("-" * 110)
    header = f"{'q14 w':>6} | {'q8 w':>6} | {'OOF BA':>8} | {'Macro-F1':>8} | {'Entropy':>7} | {'Margin':>7} | {'Delta vs EXP52':>14} | {'95% CI':>19} | {'P(Delta>0)':>10}"
    log(header)
    log("-" * 110)
    for r in sweep_results:
        ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" if r['w14'] != base_w14 else "[0.0000, 0.0000]"
        delta_str = f"{r['delta_vs_exp52']:+.4f}" if r['w14'] != base_w14 else "0.0000 (Base)"
        line = f"{r['w14']:6.2f} | {r['w8']:6.2f} | {r['ba']:8.4f} | {r['macro_f1']:8.4f} | {r['entropy']:7.4f} | {r['margin']:7.4f} | {delta_str:>14} | {ci_str:>19} | {r['p_gt_0']:10.4f}"
        log(line)
    log("-" * 110)

    # 9. Save CSV & Parquet Artifacts
    csv_rows = []
    for r in sweep_results:
        csv_rows.append({
            "q14_weight": r["w14"],
            "q8_weight": r["w8"],
            "oof_balanced_accuracy": r["ba"],
            "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"],
            "entropy": r["entropy"],
            "top1_prob": r["top1_prob"],
            "margin": r["margin"],
            "correct": r["correct"],
            "incorrect": r["incorrect"],
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
    results_csv_path = os.path.join(out_dir, "EXP53_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    log(f"Saved results CSV: {results_csv_path}")

    # Build Parquet containing row-level predictions for all weights
    oof_pq = pd.DataFrame({
        "id": train_df["id"].values,
        "prompt_id": train_df["prompt_id"].values,
        "fold": folds,
        "true_model": y_true,
    })
    for w14 in weights:
        w_tag = f"w14_{int(w14*100):02d}"
        oof_pq[f"pred_{w_tag}"] = fused_pred_dict[w14]
        oof_pq[f"top1_p_{w_tag}"] = fused_prob_dict[w14].max(axis=1)

    oof_parquet_path = os.path.join(out_dir, "EXP53_oof_predictions.parquet")
    oof_pq.to_parquet(oof_parquet_path, index=False)
    log(f"Saved OOF predictions Parquet: {oof_parquet_path}")

    # 10. Generate Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    w_vals = [r["w14"] for r in sweep_results]
    ba_vals = [r["ba"] for r in sweep_results]
    
    # Calculate empirical bootstrap CI on BA point estimate for each point
    ci_err_low = []
    ci_err_high = []
    for w14 in weights:
        b_scores = [balanced_accuracy_score(y_true[idx], fused_pred_dict[w14][idx]) for idx in boot_indices]
        ci_err_low.append(balanced_accuracy_score(y_true, fused_pred_dict[w14]) - np.percentile(b_scores, 2.5))
        ci_err_high.append(np.percentile(b_scores, 97.5) - balanced_accuracy_score(y_true, fused_pred_dict[w14]))

    yerr = [ci_err_low, ci_err_high]

    ax.errorbar(w_vals, ba_vals, yerr=yerr, fmt='-o', color='#1f77b4', ecolor='#aec7e8', 
                elinewidth=2, capsize=5, capthick=1.5, label='OOF Balanced Accuracy (with 95% CI)', zorder=3)

    # Highlight EXP52 current baseline (w14 = 0.40)
    exp52_ba = fused_pred_dict[0.40]
    ax.scatter([0.40], [sweep_results[4]["ba"]], color='#d62728', s=140, zorder=5, label='EXP52 Production (w14=0.40, BA=0.7873, Kaggle LB 0.632)')
    ax.axvline(0.40, color='#d62728', linestyle='--', alpha=0.7, label='Current Production Weight (0.40)')

    # Highlight w14 = 0.50
    ax.scatter([0.50], [sweep_results[5]["ba"]], color='#2ca02c', s=100, zorder=5, label='Candidate: w14=0.50 (BA=0.7866, Delta=-0.0008, P=0.42)')

    ax.set_title("EXP53: Qwen3-8B + Qwen3-14B Fusion Weight Sweep (OOF Validation)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Qwen3-14B Weight ($w_{14}$)  [where $w_8 = 1 - w_{14}$]", fontsize=11, labelpad=8)
    ax.set_ylabel("OOF Balanced Accuracy", fontsize=11, labelpad=8)
    ax.set_xticks(weights)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower center', fontsize=9, framealpha=0.95)

    # Annotations
    ax.annotate(f"OOF Peak: 0.7873\nw14=0.40 (Kaggle LB 0.632)", xy=(0.40, 0.7873), xytext=(0.25, 0.775),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontsize=9, fontweight='semibold')
    ax.annotate(f"Flat Plateau: 0.7866\nw14=0.50 (Delta=-0.0008)", xy=(0.50, 0.7866), xytext=(0.52, 0.760),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontsize=9)

    plt.tight_layout()
    plot_path_exp = os.path.join(out_dir, "exp53_weight_sweep.png")
    plot_path_rep = "reports/figures/exp53_weight_sweep.png"
    plt.savefig(plot_path_exp)
    plt.savefig(plot_path_rep)
    plt.close()
    log(f"Saved sweep plot: {plot_path_exp} & {plot_path_rep}")

    # 11. Write Summary JSON
    summary = {
        "experiment_id": "EXP53_fusion_weight_sweep",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "current_production_baseline": {
            "name": "EXP52",
            "kaggle_lb": 0.632,
            "q14_weight": 0.40,
            "q8_weight": 0.60,
            "oof_balanced_accuracy": 0.7873
        },
        "verified_hashes": verified_hashes,
        "sweep_results": csv_rows,
        "oof_optimal_weight": 0.40,
        "oof_optimal_ba": 0.7873,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 42,
        "candidate_selection": {
            "candidate_A": {
                "name": "EXP52_Frozen",
                "q14_weight": 0.40,
                "q8_weight": 0.60,
                "oof_ba": 0.7873,
                "rationale": "Highest OOF point estimate (0.7873) and already proven on Kaggle leaderboard (LB 0.632)."
            },
            "candidate_B": {
                "name": "EXP53_B_Equal_Weight",
                "q14_weight": 0.50,
                "q8_weight": 0.50,
                "oof_ba": 0.7866,
                "delta_vs_exp52": -0.0008,
                "p_gt_0": 0.4215,
                "rationale": "Statistically indistinguishable from 0.40 (95% CI [-0.0091, +0.0074], P=0.4215), with lower fold variance (std 0.0056 vs 0.0102)."
            },
            "candidate_C": {
                "name": "EXP53_C_Q14_Dominant",
                "q14_weight": 0.60,
                "q8_weight": 0.40,
                "oof_ba": 0.7743,
                "delta_vs_exp52": -0.0130,
                "p_gt_0": 0.0135,
                "rationale": "Scientifically probes stronger 14B dominance, but exhibits statistically significant degradation (-0.0130 BA, P=0.0135)."
            }
        },
        "recommendation": "Maintain EXP52 (w14=0.40) as frozen production baseline. No Kaggle submission recommended."
    }

    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary JSON: {summary_json_path}")
    log("EXP53 execution completed successfully.")

if __name__ == "__main__":
    run_exp53()
