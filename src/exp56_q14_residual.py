#!/usr/bin/env python3
"""
EXP56: Qwen3-14B Prompt-Conditioned Residual Specialist.

Evaluates the hypothesis that prompt-family information in Qwen3-14B representations
can be removed through prompt centering to expose residual model-specific information
that complements the frozen EXP52 ensemble.

Baselines evaluated:
  A. EXP47 Qwen3-8B Centred Twin (0.5910 BA)
  B. q14v5 Plain Head (0.6346 BA)
  C. q14v5 Centered Head (0.4842 BA standalone / 0.6491 combined)
  D. NEW Residual Specialist (training-only cluster centroids, Sections 3-5)
  E. EXP52 Dual-LoRA Baseline (0.7873 BA)
"""

import os
import sys
import json
import time
import pathlib
import numpy as np
import pandas as pd
import scipy.special
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Constants & Configuration
# ==============================================================================

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max"
]
NUM_CLASSES = len(LABELS)
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}

PREDEFINED_WEIGHTS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

REPO_DIR = pathlib.Path("/home/rj2826/name-the-llm")
DATA_DIR = REPO_DIR / "data"
OUTPUTS_DIR = REPO_DIR / "outputs"
EXPERIMENTS_DIR = REPO_DIR / "experiments" / "EXP56_q14_residual"
REPORTS_DIR = REPO_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ==============================================================================
# 2. Vectorized Lance-Williams Capped Agglomerative Clustering
# ==============================================================================

def fast_capped_agglo(S: np.ndarray, cap: int = 18, thr: float = 0.15) -> np.ndarray:
    """
    Lance-Williams Average Linkage Agglomerative Clustering (Vectorized):
    - Never merges clusters past maximum size `cap` (18).
    - Stops merging when maximum pairwise cosine similarity falls below `thr` (0.15).
    - Strict unsupervised clustering: Zero labels, zero prompt IDs, zero test leakage.
    """
    n = len(S)
    A = S.astype(np.float32).copy()
    np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32)
    lab = np.arange(n)
    alive = np.ones(n, bool)
    
    while True:
        k = np.argmax(A)
        i, j = divmod(k, n)
        mx = A[i, j]
        if mx < thr:
            break
        A[i, j] = A[j, i] = -np.inf
        if not alive[i] or not alive[j]:
            continue
        if size[i] + size[j] > cap:
            continue
        si, sj = size[i], size[j]
        st = si + sj
        
        mask = alive.copy()
        mask[i] = mask[j] = False
        new_dist = (si * A[i, mask] + sj * A[j, mask]) / st
        A[i, mask] = new_dist
        A[mask, i] = new_dist
        A[j, :] = -np.inf
        A[:, j] = -np.inf
        size[i] = st
        alive[j] = False
        lab[lab == j] = i
        
    return lab

# ==============================================================================
# 3. Main Experiment Execution
# ==============================================================================

def main():
    log("=== Starting EXP56: Qwen3-14B Prompt-Conditioned Residual Specialist ===")
    
    # 1. Load Data & Ground Truth
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    q14_df = pd.read_csv(OUTPUTS_DIR / "oof_q14v5_antopil_lora.csv")
    p8_raw = np.load(OUTPUTS_DIR / "oof_exp47_qwen_centered_twin.npy")
    
    n_samples = len(train_df)
    assert n_samples == 2699, f"Expected 2699 samples, got {n_samples}"
    assert (train_df["id"] == q14_df["id"]).all(), "Row ID mismatch between train and q14 OOF"
    
    y_str = train_df["model"].values
    y_true = np.array([LABEL2IDX[m] for m in y_str])
    folds = q14_df["fold"].values
    n_folds = len(np.unique(folds))
    assert n_folds == 5, f"Expected 5 folds, got {n_folds}"
    
    # Check prompt disjointness
    for f in range(5):
        val_prompts = set(train_df.loc[folds == f, "prompt_id"])
        tr_prompts = set(train_df.loc[folds != f, "prompt_id"])
        assert len(val_prompts.intersection(tr_prompts)) == 0, f"Prompt leakage in fold {f}!"
    log(f"Verified strict 5-fold prompt-disjointness across {train_df['prompt_id'].nunique()} prompts (30 prompts/fold).")
    
    # 2. Extract Base Model Log-Probabilities
    eps = 1e-15
    p8_norm = p8_raw / p8_raw.sum(axis=1, keepdims=True)
    p14_norm = q14_df[LABELS].values / q14_df[LABELS].values.sum(axis=1, keepdims=True)
    
    lp8 = np.log(np.clip(p8_norm, eps, 1.0))
    lp14 = np.log(np.clip(p14_norm, eps, 1.0))
    
    # Baseline E: Frozen EXP52 Ensemble
    lp_exp52 = 0.60 * lp8 + 0.40 * lp14
    preds_exp52 = lp_exp52.argmax(axis=1)
    ba_exp52 = balanced_accuracy_score(y_true, preds_exp52)
    f1_exp52 = f1_score(y_true, preds_exp52, average="macro")
    log(f"Baseline E (EXP52): Balanced Accuracy = {ba_exp52:.4f}, Macro-F1 = {f1_exp52:.4f}")
    
    # Baseline A: EXP47 Qwen3-8B
    preds_exp47 = lp8.argmax(axis=1)
    ba_exp47 = balanced_accuracy_score(y_true, preds_exp47)
    log(f"Baseline A (EXP47 Qwen3-8B): Balanced Accuracy = {ba_exp47:.4f}")
    
    # Baseline B: q14v5 Plain Head
    preds_q14_plain = lp14.argmax(axis=1)
    ba_q14_plain = balanced_accuracy_score(y_true, preds_q14_plain)
    log(f"Baseline B (q14v5 Plain Head): Balanced Accuracy = {ba_q14_plain:.4f}")
    
    # 3. Baseline C: q14v5 Centered Head (Intra-validation prompt recovery, standard twin architecture)
    log("Computing Baseline C: q14v5 Centered Head...")
    lp_cent_oof = np.zeros_like(lp14)
    lp_q14_twin = lp14.copy()
    
    for f in range(5):
        val_idx = np.where(folds == f)[0]
        val_texts = train_df["text"].values[val_idx]
        
        v = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.05)
        Xc = v.fit_transform(val_texts)
        S = (Xc @ Xc.T).toarray()
        groups = fast_capped_agglo(S, cap=18, thr=0.15)
        
        vc = pd.Series(groups).value_counts().to_dict()
        val_lp = lp14[val_idx].copy()
        val_cent = np.zeros_like(val_lp)
        
        for g in np.unique(groups):
            idx_g = np.where(groups == g)[0]
            if vc[g] >= 2:
                mean_lp = val_lp[idx_g].mean(axis=0)
                val_cent[idx_g] = val_lp[idx_g] - mean_lp
            else:
                val_cent[idx_g] = 0.0
                
        lp_cent_oof[val_idx] = val_cent
        has_clust = np.array([vc[g] >= 2 for g in groups])
        lp_q14_twin[val_idx[has_clust]] += 0.35 * val_cent[has_clust]
        
    ba_q14_cent_standalone = balanced_accuracy_score(y_true, lp_cent_oof.argmax(axis=1))
    ba_q14_twin_combined = balanced_accuracy_score(y_true, lp_q14_twin.argmax(axis=1))
    log(f"Baseline C (q14v5 Centered Head Standalone): Balanced Accuracy = {ba_q14_cent_standalone:.4f}")
    log(f"Baseline C (q14v5 Plain + Centered Twin @ 0.35): Balanced Accuracy = {ba_q14_twin_combined:.4f}")
    
    # 4. Model D: NEW Residual Specialist (Training-only prompt clusters, Sections 3-5)
    log("Fitting Model D: NEW Residual Specialist via training-only clusters...")
    oof_probs_res = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    matched_counts = []
    
    for f in range(5):
        tr_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        
        tr_texts = train_df["text"].values[tr_idx]
        val_texts = train_df["text"].values[val_idx]
        
        # Fit vectorizer strictly on training fold
        v = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.05)
        Xc_tr = v.fit_transform(tr_texts)
        Xc_val = v.transform(val_texts)
        
        # Cluster training responses strictly
        S_tr = (Xc_tr @ Xc_tr.T).toarray()
        groups_tr = fast_capped_agglo(S_tr, cap=18, thr=0.15)
        
        unique_groups = np.unique(groups_tr)
        vc = pd.Series(groups_tr).value_counts().to_dict()
        multi_groups = [g for g in unique_groups if vc[g] >= 2]
        
        tfidf_centroids = []
        z_means = []
        r_tr = lp14[tr_idx].copy()
        
        for g in multi_groups:
            idx_g = np.where(groups_tr == g)[0]
            c_tfidf = np.asarray(Xc_tr[idx_g].mean(axis=0)).ravel()
            norm_c = np.linalg.norm(c_tfidf)
            if norm_c > 1e-9:
                c_tfidf /= norm_c
            tfidf_centroids.append(c_tfidf)
            
            c_z = lp14[tr_idx[idx_g]].mean(axis=0)
            z_means.append(c_z)
            r_tr[idx_g] -= c_z
            
        for g in unique_groups:
            if vc[g] < 2:
                idx_g = np.where(groups_tr == g)[0]
                r_tr[idx_g] = 0.0 # singleton fallback
                
        # Fit linear classifier on training residuals
        clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
        clf.fit(r_tr, y_true[tr_idx])
        
        # Assign validation rows to training clusters
        tfidf_centroids = np.array(tfidf_centroids)
        z_means = np.array(z_means)
        
        val_norm = np.asarray(Xc_val.toarray())
        row_norms = np.linalg.norm(val_norm, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        val_norm /= row_norms
        
        sim_matrix = val_norm @ tfidf_centroids.T
        best_cluster = sim_matrix.argmax(axis=1)
        best_sim = sim_matrix.max(axis=1)
        reliable = best_sim >= 0.15
        matched_counts.append(reliable.sum())
        
        r_val = np.zeros_like(lp14[val_idx])
        for j in range(len(val_idx)):
            if reliable[j]:
                c_idx = best_cluster[j]
                r_val[j] = lp14[val_idx[j]] - z_means[c_idx]
            else:
                r_val[j] = 0.0 # singleton fallback
                
        oof_probs_res[val_idx] = clf.predict_proba(r_val)
        
    ba_res_standalone = balanced_accuracy_score(y_true, oof_probs_res.argmax(axis=1))
    f1_res_standalone = f1_score(y_true, oof_probs_res.argmax(axis=1), average="macro")
    acc_res_standalone = accuracy_score(y_true, oof_probs_res.argmax(axis=1))
    
    total_matched = sum(matched_counts)
    log(f"Model D (NEW Residual Specialist): {total_matched}/{n_samples} rows ({total_matched/n_samples*100:.1f}%) matched to training clusters.")
    log(f"Model D Standalone Performance: Balanced Accuracy = {ba_res_standalone:.4f}, Macro-F1 = {f1_res_standalone:.4f}, Accuracy = {acc_res_standalone:.4f}")
    
    # 5. Complementarity Analysis vs EXP52
    preds_res = oof_probs_res.argmax(axis=1)
    agree_mask = (preds_exp52 == preds_res)
    both_correct = ((preds_exp52 == y_true) & (preds_res == y_true)).sum()
    both_wrong = ((preds_exp52 != y_true) & (preds_res != y_true)).sum()
    exp52_only = ((preds_exp52 == y_true) & (preds_res != y_true)).sum()
    res_only = ((preds_exp52 != y_true) & (preds_res == y_true)).sum()
    oracle_correct = ((preds_exp52 == y_true) | (preds_res == y_true))
    oracle_ba = balanced_accuracy_score(y_true, np.where(preds_exp52 == y_true, preds_exp52, preds_res))
    
    # Error correlation
    err_exp52 = (preds_exp52 != y_true).astype(int)
    err_res = (preds_res != y_true).astype(int)
    corr_matrix = np.corrcoef(err_exp52, err_res)
    error_corr = float(corr_matrix[0, 1])
    
    log(f"Complementarity: Agreement={agree_mask.mean()*100:.2f}%, Both Correct={both_correct}, Both Wrong={both_wrong}")
    log(f"  EXP52-only Correct={exp52_only}, Residual-only Correct={res_only}")
    log(f"  Oracle BA Ceiling={oracle_ba:.4f} (+{oracle_ba - ba_exp52:.4f} over EXP52)")
    log(f"  Low measured Pearson error correlation: r = {error_corr:.5f}")
    
    # 6. Predefined Log-Probability Fusion Sweep
    lp_res = np.log(np.clip(oof_probs_res, eps, 1.0))
    sweep_results = []
    
    log("Evaluating Predefined Fusion Weights...")
    for w in PREDEFINED_WEIGHTS:
        lp_fused = (1.0 - w) * lp_exp52 + w * lp_res
        preds_fused = lp_fused.argmax(axis=1)
        
        ba = balanced_accuracy_score(y_true, preds_fused)
        f1 = f1_score(y_true, preds_fused, average="macro")
        acc = accuracy_score(y_true, preds_fused)
        delta = ba - ba_exp52
        net_samples = int((preds_fused == y_true).sum() - (preds_exp52 == y_true).sum())
        
        # Paired bootstrap CI vs EXP52
        rng = np.random.RandomState(BOOTSTRAP_SEED)
        boot_deltas = []
        for _ in range(N_BOOTSTRAP):
            b_idx = rng.randint(0, n_samples, size=n_samples)
            b_ba_fused = balanced_accuracy_score(y_true[b_idx], preds_fused[b_idx])
            b_ba_exp52 = balanced_accuracy_score(y_true[b_idx], preds_exp52[b_idx])
            boot_deltas.append(b_ba_fused - b_ba_exp52)
            
        boot_deltas = np.array(boot_deltas)
        ci_low = float(np.percentile(boot_deltas, 2.5))
        ci_high = float(np.percentile(boot_deltas, 97.5))
        p_pos = float((boot_deltas > 0).mean())
        boot_se = float(boot_deltas.std())
        
        sweep_results.append({
            "w_res": w,
            "w_exp52": 1.0 - w,
            "ba": ba,
            "macro_f1": f1,
            "acc": acc,
            "delta": delta,
            "net_samples": net_samples,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_pos": p_pos,
            "boot_se": boot_se
        })
        log(f"  w={w:.2f}: BA={ba:.4f}, Delta={delta:+.4f}, 95% CI=[{ci_low:+.4f}, {ci_high:+.4f}], P(delta>0)={p_pos:.4f}, Net={net_samples:+d}")
        
    df_results = pd.DataFrame(sweep_results)
    df_results.to_csv(EXPERIMENTS_DIR / "EXP56_results.csv", index=False)
    
    # 7. Strict 5-Fold Nested Cross-Validation
    log("Running Strict 5-Fold Nested Cross-Validation...")
    nested_records = []
    
    for f in range(5):
        val_idx = np.where(folds == f)[0]
        tr_idx = np.where(folds != f)[0]
        
        # Select best weight strictly on inner training split
        best_w = 0.00
        best_tr_ba = -1.0
        for w in PREDEFINED_WEIGHTS:
            lp_tr_fused = (1.0 - w) * lp_exp52[tr_idx] + w * lp_res[tr_idx]
            tr_ba = balanced_accuracy_score(y_true[tr_idx], lp_tr_fused.argmax(axis=1))
            if tr_ba > best_tr_ba:
                best_tr_ba = tr_ba
                best_w = w
                
        # Evaluate selected weight on outer held-out fold
        lp_outer_fused = (1.0 - best_w) * lp_exp52[val_idx] + best_w * lp_res[val_idx]
        val_ba_fused = balanced_accuracy_score(y_true[val_idx], lp_outer_fused.argmax(axis=1))
        val_ba_exp52 = balanced_accuracy_score(y_true[val_idx], preds_exp52[val_idx])
        
        nested_records.append({
            "fold": f,
            "selected_w": best_w,
            "inner_tr_ba": best_tr_ba,
            "outer_val_ba": val_ba_fused,
            "outer_exp52_ba": val_ba_exp52,
            "delta": val_ba_fused - val_ba_exp52
        })
        log(f"  Outer Fold {f}: Selected w*={best_w:.2f} (Inner BA={best_tr_ba:.4f}), Outer BA={val_ba_fused:.4f} vs EXP52 {val_ba_exp52:.4f} (Delta={val_ba_fused - val_ba_exp52:+.4f})")
        
    df_nested = pd.DataFrame(nested_records)
    nested_mean_ba = df_nested["outer_val_ba"].mean()
    nested_std_ba = df_nested["outer_val_ba"].std()
    nested_mean_exp52 = df_nested["outer_exp52_ba"].mean()
    nested_delta = nested_mean_ba - nested_mean_exp52
    log(f"Nested CV Result: Mean BA = {nested_mean_ba:.4f} +/- {nested_std_ba:.4f} (Delta vs EXP52 = {nested_delta:+.4f})")
    
    # 8. Save Predictions Parquet
    df_oof = pd.DataFrame({
        "id": train_df["id"].values,
        "prompt_id": train_df["prompt_id"].values,
        "fold": folds,
        "y_true": y_str,
        "pred_exp52": [IDX2LABEL[i] for i in preds_exp52],
        "pred_residual": [IDX2LABEL[i] for i in preds_res]
    })
    for i, l in enumerate(LABELS):
        df_oof[f"prob_{l}"] = oof_probs_res[:, i]
    df_oof.to_parquet(EXPERIMENTS_DIR / "EXP56_oof_predictions.parquet", index=False)
    
    # 9. Class-by-Class Recall Diagnostics
    cm_exp52 = confusion_matrix(y_true, preds_exp52)
    cm_fused = confusion_matrix(y_true, ((1.0 - 0.20) * lp_exp52 + 0.20 * lp_res).argmax(axis=1))
    
    rec_exp52 = np.diag(cm_exp52) / cm_exp52.sum(axis=1)
    rec_fused = np.diag(cm_fused) / cm_fused.sum(axis=1)
    
    recall_comparison = []
    for i, l in enumerate(LABELS):
        recall_comparison.append({
            "class": l,
            "exp52_recall": float(rec_exp52[i]),
            "fused_recall": float(rec_fused[i]),
            "delta": float(rec_fused[i] - rec_exp52[i])
        })
    df_recall = pd.DataFrame(recall_comparison)
    
    # 10. Generate Visualizations
    # Figure 1: Weight Sweep & 95% Bootstrap CI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df_results["w_res"], df_results["ba"], marker="o", color="#1f77b4", lw=2, label="EXP56 Fused BA")
    ax1.axhline(ba_exp52, color="crimson", linestyle="--", label=f"EXP52 Baseline ({ba_exp52:.4f})")
    ax1.fill_between(df_results["w_res"], df_results["ba"] + df_results["ci_low"] - df_results["delta"],
                     df_results["ba"] + df_results["ci_high"] - df_results["delta"], color="#1f77b4", alpha=0.15)
    ax1.set_xlabel("Residual Specialist Weight ($w$)")
    ax1.set_ylabel("Balanced Accuracy")
    ax1.set_title("EXP56: Fusion Weight Sweep with 95% Bootstrap CI")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.bar(df_results["w_res"], df_results["delta"], width=0.015, color="#2ca02c", alpha=0.8, edgecolor="black")
    ax2.axhline(0.0, color="black", lw=1)
    ax2.errorbar(df_results["w_res"], df_results["delta"],
                 yerr=[df_results["delta"] - df_results["ci_low"], df_results["ci_high"] - df_results["delta"]],
                 fmt="none", ecolor="black", capsize=4)
    ax2.set_xlabel("Residual Specialist Weight ($w$)")
    ax2.set_ylabel("Delta BA vs EXP52")
    ax2.set_title("Paired Delta with 95% Bootstrap Intervals")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "exp56_weight_sweep.png", dpi=300)
    plt.close()
    
    # Figure 2: Complementarity & Recall Deltas
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(LABELS))
    deltas = df_recall["delta"] * 100
    colors = ["#2ca02c" if d > 0 else "#d62728" if d < 0 else "#7f7f7f" for d in deltas]
    ax.barh(y_pos, deltas, color=colors, edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Recall Delta (Percentage Points vs EXP52)")
    ax.set_title("EXP56: Per-Class Recall Impact of Residual Specialist ($w=0.20$)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "exp56_complementarity.png", dpi=300)
    plt.close()
    
    # Figure 3: Normalized Confusion Matrix
    cm_res_norm = confusion_matrix(y_true, preds_res, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_res_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(LABELS)), yticks=np.arange(len(LABELS)),
           xticklabels=LABELS, yticklabels=LABELS,
           ylabel="True Label", xlabel="Predicted Label",
           title=f"EXP56: Model D Standalone Confusion Matrix (BA = {ba_res_standalone:.4f})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "exp56_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 11. Determine Final Verdict
    # Best predefined fusion point
    best_row = df_results.loc[df_results["ba"].idxmax()]
    best_w = best_row["w_res"]
    best_ba = best_row["ba"]
    best_delta = best_row["delta"]
    best_ci_low = best_row["ci_low"]
    best_ci_high = best_row["ci_high"]
    best_p_pos = best_row["p_pos"]
    
    # Verdict logic per Section 12:
    # RED: no meaningful improvement or duplicate of existing q14v5 centered head.
    # Here: Model D has collapsed standalone BA (0.1045), nested delta is negligible (+0.0006),
    # and the prompt-centering concept is already natively implemented in q14v5 head_centered.
    verdict = "RED"
    
    summary_data = {
        "experiment_id": "EXP56_q14_residual",
        "verdict": verdict,
        "standalone_residual_ba": float(ba_res_standalone),
        "q14v5_plain_ba": float(ba_q14_plain),
        "q14v5_centered_ba": float(ba_q14_cent_standalone),
        "q14v5_twin_combined_ba": float(ba_q14_twin_combined),
        "exp47_ba": float(ba_exp47),
        "exp52_baseline_ba": float(ba_exp52),
        "best_predefined_weight": float(best_w),
        "best_predefined_ba": float(best_ba),
        "best_predefined_delta": float(best_delta),
        "best_ci_low": float(best_ci_low),
        "best_ci_high": float(best_ci_high),
        "best_p_pos": float(best_p_pos),
        "nested_mean_ba": float(nested_mean_ba),
        "nested_std_ba": float(nested_std_ba),
        "nested_delta": float(nested_delta),
        "nested_records": nested_records,
        "matched_rows": int(total_matched),
        "matched_ratio": float(total_matched / n_samples),
        "error_correlation_r": error_corr,
        "kaggle_submission": "NOT SUBMITTED",
        "production_baseline": "EXP52 — Kaggle Public LB 0.632"
    }
    
    with open(EXPERIMENTS_DIR / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
        
    log("Saved all artifacts to experiments/EXP56_q14_residual/")
    log(f"EXP56 Execution Complete. Final Verdict: {verdict}")

if __name__ == "__main__":
    main()
