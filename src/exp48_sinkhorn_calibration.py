"""
================================================================================
EXP48: SINKHORN / GLOBAL MARGINAL CALIBRATION
================================================================================
Investigates the impact of damped Sinkhorn-Knopp optimal transport balancing
on EXP47 (Qwen3-8B LoRA24 Centred Twin Head) predictions under strict
prompt-disjoint GroupKFold cross-validation.
================================================================================
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold

# 1. Constants & Label Taxonomy
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

def sinkhorn_knopp_balance(P, target_marginal, alpha, max_iter=100, eps=1e-12):
    """
    Applies damped Sinkhorn-Knopp optimal transport to pull class column totals
    towards target_marginal.
    
    Args:
        P: (N, K) input probability matrix where rows sum to 1.
        target_marginal: (K,) target class distribution where elements sum to 1.
        alpha: balancing strength in [0.0, 1.0]. 0.0 = identity (no change).
        max_iter: maximum matrix scaling iterations.
        eps: numerical stability floor.
    Returns:
        Q_out: (N, K) calibrated probability matrix.
    """
    if alpha <= 0.0:
        return P.copy()
    
    N = len(P)
    Q = P.copy()
    target_counts = N * target_marginal[None, :]
    
    for _ in range(max_iter):
        # 1. Scale columns to match target marginal
        col_sums = np.clip(Q.sum(axis=0, keepdims=True), eps, None)
        Q *= (target_counts / col_sums)
        # 2. Normalize rows to sum to 1
        row_sums = np.clip(Q.sum(axis=1, keepdims=True), eps, None)
        Q /= row_sums
        
    # Interpolate in log-probability space
    log_P = np.log(np.clip(P, eps, 1.0))
    log_Q = np.log(np.clip(Q, eps, 1.0))
    log_blended = (1.0 - alpha) * log_P + alpha * log_Q
    
    # Softmax
    Q_out = np.exp(log_blended - log_blended.max(axis=1, keepdims=True))
    Q_out /= Q_out.sum(axis=1, keepdims=True)
    return Q_out

def main():
    t0 = time.time()
    print("=" * 80)
    print("EXP48: SINKHORN / GLOBAL MARGINAL CALIBRATION EXPERIMENT")
    print("=" * 80)

    # 1. Load Data & OOF Predictions
    data_dir = "/home/rj2826/name-the-llm/data"
    output_dir = "/home/rj2826/name-the-llm/outputs"
    exp_dir = "/home/rj2826/name-the-llm/reports/experiments/EXP48"
    os.makedirs(exp_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    y_true = train_df['model'].map(L2I).values
    prompts = train_df['prompt_id'].values
    N = len(train_df)

    oof_path = os.path.join(output_dir, "oof_exp47_qwen_centered_twin.npy")
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"Missing {oof_path}")
    P_exp47 = np.load(oof_path)

    # Baseline evaluation
    base_preds = P_exp47.argmax(axis=1)
    base_ba = balanced_accuracy_score(y_true, base_preds)
    base_f1 = f1_score(y_true, base_preds, average='macro')
    print(f"EXP47 Baseline Local BA:       {base_ba:.4f}")
    print(f"EXP47 Baseline Local Macro-F1: {base_f1:.4f}")
    print(f"EXP47 Kaggle Public LB:        0.620")

    # 2. Strict Prompt-Disjoint GroupKFold Validation Protocol
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train_df, y_true, groups=prompts))

    # Grid parameters
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.50, 0.60]
    rhos = [0.0, 0.15, 0.25, 0.35, 0.50, 0.70, 1.0]

    ablation_records = []
    best_ba = -1.0
    best_config = None
    best_oof_probs = None

    print("\n--- Running 2D Hyperparameter Grid Sweep (alpha x rho) ---")
    for rho in rhos:
        for alpha in alphas:
            oof_calibrated = np.zeros_like(P_exp47)
            fold_bas = []
            
            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                P_val = P_exp47[va_idx]
                y_tr = y_true[tr_idx]
                
                # Training prior from training folds only (strict no-leakage)
                train_prior = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64) + 1.0
                train_prior /= train_prior.sum()
                
                # Model-implied test marginal from validation fold
                pred_marg = np.clip(P_val.mean(axis=0), 1e-6, None)
                pred_marg /= pred_marg.sum()
                
                # Target marginal mix
                target = (1.0 - rho) * pred_marg + rho * train_prior
                target /= target.sum()
                
                # Apply Sinkhorn
                calib_val = sinkhorn_knopp_balance(P_val, target, alpha=alpha)
                oof_calibrated[va_idx] = calib_val
                
                f_ba = balanced_accuracy_score(y_true[va_idx], calib_val.argmax(axis=1))
                fold_bas.append(f_ba)
                
            overall_ba = balanced_accuracy_score(y_true, oof_calibrated.argmax(axis=1))
            overall_f1 = f1_score(y_true, oof_calibrated.argmax(axis=1), average='macro')
            delta = overall_ba - base_ba
            
            ablation_records.append({
                "alpha": alpha,
                "rho": rho,
                "balanced_accuracy": float(overall_ba),
                "macro_f1": float(overall_f1),
                "delta": float(delta),
                "fold_std": float(np.std(fold_bas))
            })
            
            if overall_ba > best_ba:
                best_ba = overall_ba
                best_config = {"alpha": alpha, "rho": rho}
                best_oof_probs = oof_calibrated.copy()

    ablation_df = pd.DataFrame(ablation_records)
    ablation_df.to_csv(os.path.join(exp_dir, "ablation_results.csv"), index=False)
    print(f"Saved ablation results to {os.path.join(exp_dir, 'ablation_results.csv')}")

    print(f"\nGrid Sweep Complete:")
    print(f"  Best Configuration: alpha = {best_config['alpha']}, rho = {best_config['rho']}")
    print(f"  Best Balanced Accuracy: {best_ba:.4f} (Delta vs EXP47: {best_ba - base_ba:+.4f})")
    print(f"  Reference Notebook Settings (alpha=0.30, rho=0.35):")
    ref_row = ablation_df[(ablation_df['alpha'] == 0.30) & (ablation_df['rho'] == 0.35)].iloc[0]
    print(f"    BA = {ref_row['balanced_accuracy']:.4f}, Delta = {ref_row['delta']:+.4f}")

    # 3. Per-Class Analysis
    best_preds = best_oof_probs.argmax(axis=1)
    cm_base = confusion_matrix(y_true, base_preds)
    cm_best = confusion_matrix(y_true, best_preds)

    per_class_stats = []
    for k in range(NUM_CLASSES):
        name = I2L[k]
        base_recall = cm_base[k, k] / cm_base[k].sum()
        best_recall = cm_best[k, k] / cm_best[k].sum()
        per_class_stats.append({
            "class_idx": k,
            "class_name": name,
            "baseline_recall": float(base_recall),
            "calibrated_recall": float(best_recall),
            "recall_delta": float(best_recall - base_recall)
        })

    # Save metrics & predictions
    np.save(os.path.join(exp_dir, "predictions.npy"), best_oof_probs.astype(np.float32))

    metrics = {
        "experiment_id": "EXP48_sinkhorn_calibration",
        "baseline_ba": float(base_ba),
        "calibrated_ba": float(best_ba),
        "delta": float(best_ba - base_ba),
        "baseline_macro_f1": float(base_f1),
        "calibrated_macro_f1": float(f1_score(y_true, best_preds, average='macro')),
        "best_alpha": float(best_config['alpha']),
        "best_rho": float(best_config['rho']),
        "reference_alpha_0_30_ba": float(ref_row['balanced_accuracy']),
        "reference_alpha_0_30_delta": float(ref_row['delta']),
        "runtime_seconds": float(time.time() - t0),
        "decision": "YELLOW"
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "experiment_id": "EXP48",
        "parent_baseline": "EXP47",
        "component": "Damped Sinkhorn-Knopp Global Marginal Calibration",
        "alphas_tested": alphas,
        "rhos_tested": rhos,
        "selected_alpha": best_config['alpha'],
        "selected_rho": best_config['rho'],
        "max_iter": 100,
        "validation_strategy": "5-Fold GroupKFold(group=prompt_id)"
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    comparison = {
        "EXP46A": {"local_ba": 0.5862, "kaggle_lb": 0.606, "delta": 0.0},
        "EXP47": {"local_ba": 0.5910, "kaggle_lb": 0.620, "delta": 0.0048},
        "EXP48": {"local_ba": float(best_ba), "kaggle_lb": "NOT_SUBMITTED", "delta": float(best_ba - 0.5910)}
    }
    with open(os.path.join(exp_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n[Done] All EXP48 metrics, configs, and predictions written successfully.")

if __name__ == "__main__":
    main()
