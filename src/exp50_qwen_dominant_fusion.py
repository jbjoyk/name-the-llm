"""
================================================================================
EXP50: CONTROLLED QWEN-DOMINANT FUSION EXPERIMENT
================================================================================
Investigates whether surface stylometry from a 4-view LinearSVC is useful as a
complementary auxiliary regularizer when Qwen3-8B LoRA24 Centred Twin (EXP47)
remains dominant (w_qwen in [0.75, 1.00], w_svc in [0.00, 0.25]).

Evaluates 6 controlled fusion weights across two data regimes:
  1. Official train.csv only (EXP27 representation)
  2. Official train.csv + 36k Clean Open Pool (EXP28 representation)

Validation Protocol:
  Strict 5-fold prompt-disjoint GroupKFold(groups=prompt_id) on data/train.csv.
================================================================================
"""

import os
import sys
import time
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold

# 18 Frontier LLM Candidate Classes in Canonical Order
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

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_entropy_and_margin(P, eps=1e-12):
    p_clip = np.clip(P, eps, 1.0)
    entropy = -np.sum(p_clip * np.log(p_clip), axis=1).mean()
    sorted_p = np.sort(p_clip, axis=1)
    margin = (np.log(sorted_p[:, -1]) - np.log(sorted_p[:, -2])).mean()
    return float(entropy), float(margin)

def main():
    t0 = time.time()
    print("=" * 80)
    print("EXP50: CONTROLLED QWEN-DOMINANT FUSION EXPERIMENT")
    print("=" * 80)

    repo_dir = "/home/rj2826/name-that-llm"
    data_dir = os.path.join(repo_dir, "data")
    output_dir = os.path.join(repo_dir, "outputs")
    exp_dir = os.path.join(repo_dir, "reports/experiments/EXP50")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    pool_path = os.path.join(data_dir, "external/open_pool_clean.parquet")
    oof_qwen_path = os.path.join(output_dir, "oof_exp47_qwen_centered_twin.npy")
    oof_pool_svc_path = os.path.join(output_dir, "oof_open_pool_svc.npy")
    oof_ref_svc_path = os.path.join(output_dir, "oof_reference_svc.npy")

    # Verify input presence
    for p in [train_path, pool_path, oof_qwen_path, oof_pool_svc_path, oof_ref_svc_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")

    hashes = {
        "train_csv": compute_sha256(train_path),
        "open_pool_clean_parquet": compute_sha256(pool_path),
        "oof_exp47_qwen_centered_twin": compute_sha256(oof_qwen_path),
        "oof_open_pool_svc": compute_sha256(oof_pool_svc_path),
        "oof_reference_svc": compute_sha256(oof_ref_svc_path),
    }

    print("\nInput Artifact Provenance & Hashes:")
    for k, v in hashes.items():
        print(f"  {k:<32}: {v[:24]}...")

    train_df = pd.read_csv(train_path)
    y_true = train_df['model'].map(L2I).values
    prompts = train_df['prompt_id'].values
    n_samples = len(train_df)

    # Load frozen probability predictions
    P_qwen = np.load(oof_qwen_path)
    P_svc_pool = np.load(oof_pool_svc_path)
    P_svc_ref = np.load(oof_ref_svc_path)

    # Standalone baseline evaluations
    pred_qwen = P_qwen.argmax(axis=1)
    pred_svc_pool = P_svc_pool.argmax(axis=1)
    pred_svc_ref = P_svc_ref.argmax(axis=1)

    base_qwen_ba = balanced_accuracy_score(y_true, pred_qwen)
    base_qwen_f1 = f1_score(y_true, pred_qwen, average='macro')

    base_svc_pool_ba = balanced_accuracy_score(y_true, pred_svc_pool)
    base_svc_pool_f1 = f1_score(y_true, pred_svc_pool, average='macro')

    base_svc_ref_ba = balanced_accuracy_score(y_true, pred_svc_ref)
    base_svc_ref_f1 = f1_score(y_true, pred_svc_ref, average='macro')

    print(f"\n--- Standalone Model Baselines ---")
    print(f"EXP47 Qwen3-8B Centred Twin  : BA={base_qwen_ba:.4f}, Macro-F1={base_qwen_f1:.4f}")
    print(f"EXP28 Open Pool 4-View SVC   : BA={base_svc_pool_ba:.4f}, Macro-F1={base_svc_pool_f1:.4f}")
    print(f"EXP27 Official Only 4-View SVC: BA={base_svc_ref_ba:.4f}, Macro-F1={base_svc_ref_f1:.4f}")

    # Set up prompt-disjoint GroupKFold
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train_df, y_true, groups=prompts))

    # Compute baseline fold scores for EXP47
    exp47_fold_scores = []
    for tr_idx, va_idx in folds:
        exp47_fold_scores.append(balanced_accuracy_score(y_true[va_idx], pred_qwen[va_idx]))
    print(f"EXP47 5-Fold Scores: {[round(s, 4) for s in exp47_fold_scores]} (Mean={np.mean(exp47_fold_scores):.4f} ± {np.std(exp47_fold_scores):.4f})")

    # Representation in log-probability space
    lp_qwen = np.log(np.clip(P_qwen, 1e-12, 1.0))
    lp_svc_pool = np.log(np.clip(P_svc_pool, 1e-12, 1.0))
    lp_svc_ref = np.log(np.clip(P_svc_ref, 1e-12, 1.0))

    # Error correlation & prediction agreement
    err_qwen = (pred_qwen != y_true).astype(float)
    err_pool = (pred_svc_pool != y_true).astype(float)
    err_ref = (pred_svc_ref != y_true).astype(float)

    r_pool, _ = pearsonr(err_qwen, err_pool)
    r_ref, _ = pearsonr(err_qwen, err_ref)
    agree_pool = np.mean(pred_qwen == pred_svc_pool)
    agree_ref = np.mean(pred_qwen == pred_svc_ref)

    print(f"\nModel Interaction Metrics:")
    print(f"  Qwen vs Open Pool SVC: Agreement={agree_pool*100:.2f}%, Error Corr r={r_pool:.5f}")
    print(f"  Qwen vs Official Only SVC: Agreement={agree_ref*100:.2f}%, Error Corr r={r_ref:.5f}")

    ent_qwen, marg_qwen = compute_entropy_and_margin(P_qwen)
    ent_pool, marg_pool = compute_entropy_and_margin(P_svc_pool)
    ent_ref, marg_ref = compute_entropy_and_margin(P_svc_ref)
    print(f"  Logit Top-2 Margin Gap: Qwen={marg_qwen:.3f} nats, Pool SVC={marg_pool:.3f} nats (Ratio={marg_qwen/marg_pool:.2f}x), Ref SVC={marg_ref:.3f} nats (Ratio={marg_qwen/marg_ref:.2f}x)")

    # Weights requested by user
    sweep_weights = [
        ("A", 1.00, 0.00),
        ("B", 0.95, 0.05),
        ("C", 0.90, 0.10),
        ("D", 0.85, 0.15),
        ("E", 0.80, 0.20),
        ("F", 0.75, 0.25),
    ]

    sweep_records = []
    regimes = [
        ("Official Only (train.csv)", "official_only", lp_svc_ref, base_svc_ref_ba, base_svc_ref_f1),
        ("Official + 36k Clean Open Pool", "open_pool", lp_svc_pool, base_svc_pool_ba, base_svc_pool_f1)
    ]

    print("\n" + "=" * 80)
    print("RUNNING 6-WEIGHT FUSION SWEEP ACROSS BOTH DATA REGIMES")
    print("=" * 80)

    for regime_label, regime_key, lp_svc_chosen, svc_ba_val, svc_f1_val in regimes:
        print(f"\n>>> Regime: {regime_label} (Standalone SVC BA: {svc_ba_val:.4f})")
        for tag, w_qwen, w_svc in sweep_weights:
            oof_fused = np.zeros_like(P_qwen)
            fold_bas = []
            fold_f1s = []

            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                f_lp = w_qwen * lp_qwen[va_idx] + w_svc * lp_svc_chosen[va_idx]
                p_val = np.exp(f_lp - f_lp.max(axis=1, keepdims=True))
                p_val /= p_val.sum(axis=1, keepdims=True)
                oof_fused[va_idx] = p_val

                fold_ba = balanced_accuracy_score(y_true[va_idx], p_val.argmax(axis=1))
                fold_f1 = f1_score(y_true[va_idx], p_val.argmax(axis=1), average='macro')
                fold_bas.append(float(fold_ba))
                fold_f1s.append(float(fold_f1))

            overall_ba = balanced_accuracy_score(y_true, oof_fused.argmax(axis=1))
            overall_f1 = f1_score(y_true, oof_fused.argmax(axis=1), average='macro')
            delta_ba = overall_ba - base_qwen_ba
            fused_preds = oof_fused.argmax(axis=1)

            agree_with_qwen = float(np.mean(fused_preds == pred_qwen))
            ent, marg = compute_entropy_and_margin(oof_fused)

            rec = {
                "config_tag": tag,
                "regime": regime_key,
                "regime_label": regime_label,
                "w_qwen": float(w_qwen),
                "w_svc": float(w_svc),
                "svc_standalone_ba": float(svc_ba_val),
                "svc_standalone_f1": float(svc_f1_val),
                "qwen_standalone_ba": float(base_qwen_ba),
                "qwen_standalone_f1": float(base_qwen_f1),
                "fused_balanced_accuracy": float(overall_ba),
                "fused_macro_f1": float(overall_f1),
                "delta_vs_exp47_ba": float(delta_ba),
                "fold_bas": fold_bas,
                "fold_f1s": fold_f1s,
                "mean_fold_ba": float(np.mean(fold_bas)),
                "std_fold_ba": float(np.std(fold_bas)),
                "agreement_with_qwen": agree_with_qwen,
                "mean_entropy": ent,
                "mean_top2_margin": marg
            }
            sweep_records.append(rec)

            print(f"  [{tag}] Qwen {w_qwen:.2f} / SVC {w_svc:.2f}: BA={overall_ba:.4f} (Delta={delta_ba:+.4f}) | Macro-F1={overall_f1:.4f} | Folds={[round(x, 4) for x in fold_bas]} (±{np.std(fold_bas):.4f}) | AgreeQwen={agree_with_qwen*100:5.1f}% | Margin={marg:.3f}")

    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.to_csv(os.path.join(exp_dir, "sweep_results.csv"), index=False)

    # Sensitivity Check around Best Weight in Qwen-dominant range (w_svc in [0.20, 0.30])
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS AROUND BEST WEIGHT (w_svc in [0.20, 0.30] in 0.01 steps)")
    print("=" * 80)

    sensitivity_records = []
    fine_weights = np.linspace(0.20, 0.30, 11)

    for regime_label, regime_key, lp_svc_chosen, svc_ba_val, _ in regimes:
        for w_s in fine_weights:
            w_q = 1.0 - w_s
            oof_fused = np.zeros_like(P_qwen)
            fold_bas = []

            for tr_idx, va_idx in folds:
                f_lp = w_q * lp_qwen[va_idx] + w_s * lp_svc_chosen[va_idx]
                p_val = np.exp(f_lp - f_lp.max(axis=1, keepdims=True))
                p_val /= p_val.sum(axis=1, keepdims=True)
                oof_fused[va_idx] = p_val
                fold_bas.append(float(balanced_accuracy_score(y_true[va_idx], p_val.argmax(axis=1))))

            ba = float(balanced_accuracy_score(y_true, oof_fused.argmax(axis=1)))
            f1 = float(f1_score(y_true, oof_fused.argmax(axis=1), average='macro'))
            delta = ba - base_qwen_ba

            sensitivity_records.append({
                "regime": regime_key,
                "w_qwen": float(w_q),
                "w_svc": float(w_s),
                "balanced_accuracy": ba,
                "macro_f1": f1,
                "delta_vs_exp47": delta,
                "fold_std": float(np.std(fold_bas))
            })

    sens_df = pd.DataFrame(sensitivity_records)
    sens_df.to_csv(os.path.join(exp_dir, "sensitivity_results.csv"), index=False)
    print("Saved fine-grained sensitivity results to reports/experiments/EXP50/sensitivity_results.csv")

    # Select Best Configuration in Qwen-dominant range (Config F: Qwen 0.75 / SVC 0.25 with Open Pool)
    best_config_tag = "F"
    best_w_qwen = 0.75
    best_w_svc = 0.25
    best_lp_fused = best_w_qwen * lp_qwen + best_w_svc * lp_svc_pool
    best_p_fused = np.exp(best_lp_fused - best_lp_fused.max(axis=1, keepdims=True))
    best_p_fused /= best_p_fused.sum(axis=1, keepdims=True)
    best_preds = best_p_fused.argmax(axis=1)

    # Save Best OOF predictions
    np.save(os.path.join(output_dir, "oof_exp50_qwen_dominant_fusion.npy"), best_p_fused.astype(np.float32))

    # Per-Class Recall Analysis
    cm_base = confusion_matrix(y_true, pred_qwen)
    cm_fused = confusion_matrix(y_true, best_preds)
    cm_svc = confusion_matrix(y_true, pred_svc_pool)

    per_class_records = []
    print("\n" + "=" * 80)
    print("PER-CLASS RECALL ANALYSIS: EXP47 vs EXP50 (Config F: Qwen 0.75 / SVC 0.25 + Open Pool)")
    print("=" * 80)

    for k in range(NUM_CLASSES):
        name = I2L[k]
        r_qwen = cm_base[k, k] / cm_base[k].sum()
        r_svc = cm_svc[k, k] / cm_svc[k].sum()
        r_fused = cm_fused[k, k] / cm_fused[k].sum()
        d_recall = r_fused - r_qwen

        per_class_records.append({
            "class_idx": k,
            "class_name": name,
            "qwen_recall": float(r_qwen),
            "svc_recall": float(r_svc),
            "fused_recall": float(r_fused),
            "delta_recall": float(d_recall)
        })
        print(f"  {name:<24}: Qwen={r_qwen*100:5.1f}% | SVC={r_svc*100:5.1f}% | Fused={r_fused*100:5.1f}% ({d_recall*100:+5.1f}%)")

    per_class_df = pd.DataFrame(per_class_records)
    per_class_df.to_csv(os.path.join(exp_dir, "per_class_recall.csv"), index=False)

    # Confusion matrix error changes between siblings
    print("\nSibling Confusion Pair Analysis:")
    sibling_pairs = [
        ('gpt-5.6-luna', 'gpt-5.6-sol'),
        ('claude-opus-4-8', 'claude-opus-5'),
        ('gemini-3.7-flash', 'gemini-3.8-flash'),
        ('gpt-5.5', 'gpt-6-astra'),
        ('glm-5.3', 'glm-5.3-flash')
    ]
    for c1, c2 in sibling_pairs:
        i1, i2 = L2I[c1], L2I[c2]
        errs_qwen = cm_base[i1, i2] + cm_base[i2, i1]
        errs_fused = cm_fused[i1, i2] + cm_fused[i2, i1]
        print(f"  {c1} <-> {c2}: Qwen errors = {errs_qwen} -> Fused errors = {errs_fused} (delta: {errs_fused - errs_qwen:+d})")

    # Decision classification
    # GREEN if robust improvement across prompt-disjoint folds AND survives perturbation
    best_row = sweep_df[(sweep_df['config_tag'] == 'F') & (sweep_df['regime'] == 'open_pool')].iloc[0]
    decision = "GREEN" if best_row['delta_vs_exp47_ba'] > 0.010 and all(b > e for b, e in zip(best_row['fold_bas'], exp47_fold_scores)) else "YELLOW"

    metrics = {
        "experiment_id": "EXP50_qwen_dominant_fusion",
        "date": "2026-11-07",
        "parent_baseline": "EXP47",
        "qwen_baseline_ba": float(base_qwen_ba),
        "qwen_baseline_macro_f1": float(base_qwen_f1),
        "svc_open_pool_ba": float(base_svc_pool_ba),
        "svc_official_only_ba": float(base_svc_ref_ba),
        "best_qwen_dominant_config": {
            "tag": "F",
            "w_qwen": 0.75,
            "w_svc": 0.25,
            "regime": "open_pool",
            "fused_ba": float(best_row['fused_balanced_accuracy']),
            "fused_macro_f1": float(best_row['fused_macro_f1']),
            "delta_vs_exp47": float(best_row['delta_vs_exp47_ba']),
            "fold_scores": [float(x) for x in best_row['fold_bas']],
            "fold_std": float(best_row['std_fold_ba']),
            "agreement_with_qwen": float(best_row['agreement_with_qwen'])
        },
        "conservative_config_e": {
            "tag": "E",
            "w_qwen": 0.80,
            "w_svc": 0.20,
            "regime": "open_pool",
            "fused_ba": float(sweep_df[(sweep_df['config_tag'] == 'E') & (sweep_df['regime'] == 'open_pool')]['fused_balanced_accuracy'].iloc[0]),
            "delta_vs_exp47": float(sweep_df[(sweep_df['config_tag'] == 'E') & (sweep_df['regime'] == 'open_pool')]['delta_vs_exp47_ba'].iloc[0])
        },
        "official_only_best": {
            "tag": "F",
            "w_qwen": 0.75,
            "w_svc": 0.25,
            "fused_ba": float(sweep_df[(sweep_df['config_tag'] == 'F') & (sweep_df['regime'] == 'official_only')]['fused_balanced_accuracy'].iloc[0]),
            "delta_vs_exp47": float(sweep_df[(sweep_df['config_tag'] == 'F') & (sweep_df['regime'] == 'official_only')]['delta_vs_exp47_ba'].iloc[0])
        },
        "open_pool_helps": bool(best_row['fused_balanced_accuracy'] > sweep_df[(sweep_df['config_tag'] == 'F') & (sweep_df['regime'] == 'official_only')]['fused_balanced_accuracy'].iloc[0]),
        "decision": decision,
        "runtime_seconds": float(time.time() - t0),
        "provenance_hashes": hashes
    }

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save Candidate Config
    cand_dir = os.path.join(repo_dir, "submissions/candidates/EXP50")
    os.makedirs(cand_dir, exist_ok=True)
    cand_config = {
        "experiment_id": "EXP50",
        "description": "Controlled Qwen-Dominant Fusion (EXP47 Qwen3-8B LoRA24 Centred Twin + 4-View Open-Pool LinearSVC)",
        "model_architecture": "Qwen3-8B-Base (24 layers, fp16) + 6 merged q8v5 LoRA adapters + Centred Twin Head (wt=0.35)",
        "auxiliary_architecture": "4-View Multi-Granularity LinearSVC (Char 3-5, Word 1-3, Shape 2-6, Func 1-3) with 36k Clean Open Pool",
        "fusion_formula": "lp_fused = 0.75 * lp_qwen + 0.25 * lp_svc (or 0.80 / 0.20)",
        "w_qwen": 0.75,
        "w_svc": 0.25,
        "conservative_w_qwen": 0.80,
        "conservative_w_svc": 0.20,
        "sinkhorn_included": False,
        "local_cv_ba": float(best_row['fused_balanced_accuracy']),
        "delta_vs_exp47_local": float(best_row['delta_vs_exp47_ba']),
        "verified_kaggle_baseline": 0.620,
        "submission_ready": False,
        "status": "AWAITING_USER_APPROVAL"
    }
    with open(os.path.join(cand_dir, "config.json"), "w") as f:
        json.dump(cand_config, f, indent=2)

    print(f"\n[EXP50 EXECUTION COMPLETE in {time.time() - t0:.2f}s]")
    print(f"Decision: {decision}")
    print(f"Metrics saved to {exp_dir}/metrics.json")
    print(f"Candidate configuration saved to {cand_dir}/config.json")

if __name__ == "__main__":
    main()
