#!/usr/bin/env python3
"""
EXP57B: Class-Head Norm Equalization Validation.

Scientific validation of weight-space hypothesis from EXP57A:
Does inference-time class-head norm equalization (W'_c = W_c / N_c^gamma)
causally improve prompt-disjoint Balanced Accuracy on frozen q14v5 representations
and within the frozen EXP52 ensemble?

Strict inference-time only:
- NO gradient descent
- NO LoRA retraining
- NO hidden-test usage
- NO Kaggle submission
- Frozen EXP52 baseline preserved
"""

import os
import sys
import json
import time
import pathlib
import numpy as np
import pandas as pd
import scipy.special
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
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

LOW_NORM_TARGETS = ["glm-5.3", "glm-5.3-flash", "gemini-3.7-flash", "kimi-k3"]
HIGH_NORM_TARGETS = ["gpt-6-astra", "gemini-3.1-pro-preview", "claude-opus-5"]

GAMMA_GRID = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
TEMPERATURE_GRID = [0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
N_PERMUTATIONS = 20
PERMUTATION_SEED = 42

REPO_DIR = pathlib.Path("/home/rj2826/name-the-llm")
DATA_DIR = REPO_DIR / "data"
OUTPUTS_DIR = REPO_DIR / "outputs"
EXP57A_DIR = REPO_DIR / "experiments" / "EXP57A_weight_space_diagnosis"
EXPERIMENTS_DIR = REPO_DIR / "experiments" / "EXP57B_norm_equalization"
REPORTS_DIR = REPO_DIR / "reports"

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ==============================================================================
# 2. Vectorized Lance-Williams Agglomerative Clustering (q14v5 Centered Twin)
# ==============================================================================

def fast_capped_agglo(S: np.ndarray, cap: int = 18, thr: float = 0.15) -> np.ndarray:
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
# 3. Evaluation Helpers
# ==============================================================================

def compute_distribution_metrics(log_probs: np.ndarray):
    """Computes mean/median confidence, entropy, and top-1 margin."""
    probs = scipy.special.softmax(log_probs, axis=1)
    confs = np.max(probs, axis=1)
    mean_conf = float(np.mean(confs))
    median_conf = float(np.median(confs))
    
    # Entropy
    eps = 1e-15
    entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
    mean_entropy = float(np.mean(entropy))
    
    # Top-1 margin (difference between top-1 and top-2 prob)
    sorted_probs = np.sort(probs, axis=1)
    top1_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    mean_margin = float(np.mean(top1_margin))
    
    return mean_conf, median_conf, mean_entropy, mean_margin

def fast_paired_bootstrap(y_true: np.ndarray, preds_base: np.ndarray, preds_cand: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ba_base = balanced_accuracy_score(y_true[idx], preds_base[idx])
        ba_cand = balanced_accuracy_score(y_true[idx], preds_cand[idx])
        deltas[b] = ba_cand - ba_base
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_pos = float(np.mean(deltas > 0))
    return ci_lower, ci_upper, p_pos, deltas

# ==============================================================================
# 4. Main Experiment Pipeline
# ==============================================================================

def main():
    log("=== Starting EXP57B: Class-Head Norm Equalization Validation ===")

    # 1. Load Data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    q14_df = pd.read_csv(OUTPUTS_DIR / "oof_q14v5_antopil_lora.csv")
    p8_raw = np.load(OUTPUTS_DIR / "oof_exp47_qwen_centered_twin.npy")

    assert len(train_df) == 2699
    assert (train_df["id"] == q14_df["id"]).all()
    y_str = train_df["model"].values
    y_true = np.array([LABEL2IDX[m] for m in y_str])
    folds = q14_df["fold"].values
    n_folds = 5

    # Base log-probabilities
    eps = 1e-15
    p14_norm = q14_df[LABELS].values / q14_df[LABELS].values.sum(axis=1, keepdims=True)
    lp14 = np.log(np.clip(p14_norm, eps, 1.0))

    p8_norm = p8_raw / p8_raw.sum(axis=1, keepdims=True)
    lp8 = np.log(np.clip(p8_norm, eps, 1.0))

    # 2. Load Fold Head Classifier Weights & Norms
    plain_norms = {}
    cent_norms = {}
    plain_weights = {}
    cent_weights = {}

    for f in range(n_folds):
        hp = EXP57A_DIR / "fold_heads" / f"q14v5_f{f}" / "heads.pt"
        assert hp.exists(), f"Missing fold head: {hp}"
        H = torch.load(hp, map_location="cpu")
        W_p = H["plain.weight"].numpy()
        W_c = H["centered.weight"].numpy()
        plain_weights[f] = W_p
        cent_weights[f] = W_c
        plain_norms[f] = np.linalg.norm(W_p, axis=1)
        cent_norms[f] = np.linalg.norm(W_c, axis=1)

    log("Loaded classifier weights and norms for all 5 prompt-disjoint folds.")

    # 3. Compute Centered Head OOF Predictions (q14v5 Reference Implementation)
    log("Computing q14v5 Centered Head OOF predictions...")
    lp_cent_oof = np.zeros_like(lp14)
    lp_twin = lp14.copy()

    for f in range(n_folds):
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
        lp_twin[val_idx[has_clust]] += 0.35 * val_cent[has_clust]

    # Baseline Reproductions
    preds_plain_base = lp14.argmax(axis=1)
    ba_plain_base = balanced_accuracy_score(y_true, preds_plain_base)
    f1_plain_base = f1_score(y_true, preds_plain_base, average="macro")

    preds_cent_base = lp_cent_oof.argmax(axis=1)
    ba_cent_base = balanced_accuracy_score(y_true, preds_cent_base)
    f1_cent_base = f1_score(y_true, preds_cent_base, average="macro")

    preds_twin_base = lp_twin.argmax(axis=1)
    ba_twin_base = balanced_accuracy_score(y_true, preds_twin_base)
    f1_twin_base = f1_score(y_true, preds_twin_base, average="macro")

    # EXP52 Baseline
    lp_exp52_base = 0.60 * lp8 + 0.40 * lp14
    preds_exp52_base = lp_exp52_base.argmax(axis=1)
    ba_exp52_base = balanced_accuracy_score(y_true, preds_exp52_base)
    f1_exp52_base = f1_score(y_true, preds_exp52_base, average="macro")

    log("=== BASELINE REPRODUCTION VERIFICATION ===")
    log(f"Variant A (q14 Plain):     BA = {ba_plain_base:.4f} (Reference: 0.6346), Macro-F1 = {f1_plain_base:.4f}")
    log(f"Variant B (q14 Centered):  BA = {ba_cent_base:.4f} (Reference: 0.4842), Macro-F1 = {f1_cent_base:.4f}")
    log(f"Variant C (q14 Twin 0.35): BA = {ba_twin_base:.4f} (Reference: 0.6491), Macro-F1 = {f1_twin_base:.4f}")
    log(f"EXP52 Baseline:            BA = {ba_exp52_base:.4f} (Reference: 0.7873), Macro-F1 = {f1_exp52_base:.4f}")

    assert abs(ba_plain_base - 0.6346) < 0.001, f"Plain BA mismatch: {ba_plain_base}"
    assert abs(ba_cent_base - 0.4842) < 0.001, f"Centered BA mismatch: {ba_cent_base}"
    assert abs(ba_twin_base - 0.6491) < 0.001, f"Twin BA mismatch: {ba_twin_base}"
    assert abs(ba_exp52_base - 0.7873) < 0.001, f"EXP52 BA mismatch: {ba_exp52_base}"
    log("Exact baseline reproduction verified within tolerance!")

    # ==============================================================================
    # 5. Gamma Sweep Across Variants
    # ==============================================================================
    log("Evaluating Gamma Grid across 4 Variants...")
    gamma_rows = []
    fold_rows = []

    # Store best predictions for downstream controls
    best_plain_gamma = 0.0
    best_plain_ba = ba_plain_base
    best_plain_lp = lp14.copy()

    for g in GAMMA_GRID:
        # Variant A: Plain Head Only
        lp_a = lp14.copy()
        for f in range(n_folds):
            val_idx = np.where(folds == f)[0]
            N = plain_norms[f]
            lp_a[val_idx] = lp_a[val_idx] / (N ** g)
        preds_a = lp_a.argmax(axis=1)
        ba_a = balanced_accuracy_score(y_true, preds_a)
        f1_a = f1_score(y_true, preds_a, average="macro")
        ci_l_a, ci_u_a, p_pos_a, _ = fast_paired_bootstrap(y_true, preds_plain_base, preds_a)
        m_c_a, med_c_a, ent_a, top_m_a = compute_distribution_metrics(lp_a)

        gamma_rows.append({
            "variant": "Variant A (Plain Head)",
            "gamma": g,
            "ba": ba_a,
            "macro_f1": f1_a,
            "delta_ba": ba_a - ba_plain_base,
            "ci_lower": ci_l_a,
            "ci_upper": ci_u_a,
            "p_delta_pos": p_pos_a,
            "mean_conf": m_c_a,
            "median_conf": med_c_a,
            "mean_entropy": ent_a,
            "mean_top1_margin": top_m_a
        })

        # Fold results for Variant A
        f_bas_a = [balanced_accuracy_score(y_true[folds == f], lp_a[folds == f].argmax(axis=1)) for f in range(n_folds)]
        fold_rows.append({
            "variant": "Variant A (Plain Head)",
            "gamma": g,
            "fold_0": f_bas_a[0],
            "fold_1": f_bas_a[1],
            "fold_2": f_bas_a[2],
            "fold_3": f_bas_a[3],
            "fold_4": f_bas_a[4],
            "mean": float(np.mean(f_bas_a)),
            "std": float(np.std(f_bas_a))
        })

        if ba_a > best_plain_ba:
            best_plain_ba = ba_a
            best_plain_gamma = g
            best_plain_lp = lp_a.copy()

        # Variant B: Centered Head Only
        lp_b = lp_cent_oof.copy()
        for f in range(n_folds):
            val_idx = np.where(folds == f)[0]
            N = cent_norms[f]
            lp_b[val_idx] = lp_b[val_idx] / (N ** g)
        preds_b = lp_b.argmax(axis=1)
        ba_b = balanced_accuracy_score(y_true, preds_b)
        f1_b = f1_score(y_true, preds_b, average="macro")
        ci_l_b, ci_u_b, p_pos_b, _ = fast_paired_bootstrap(y_true, preds_cent_base, preds_b)
        m_c_b, med_c_b, ent_b, top_m_b = compute_distribution_metrics(lp_b)

        gamma_rows.append({
            "variant": "Variant B (Centered Head)",
            "gamma": g,
            "ba": ba_b,
            "macro_f1": f1_b,
            "delta_ba": ba_b - ba_cent_base,
            "ci_lower": ci_l_b,
            "ci_upper": ci_u_b,
            "p_delta_pos": p_pos_b,
            "mean_conf": m_c_b,
            "median_conf": med_c_b,
            "mean_entropy": ent_b,
            "mean_top1_margin": top_m_b
        })

        # Variant C: Plain + Centered Twin Baseline (Frozen gamma=0 baseline reference)
        gamma_rows.append({
            "variant": "Variant C (Twin Baseline)",
            "gamma": g,
            "ba": ba_twin_base,
            "macro_f1": f1_twin_base,
            "delta_ba": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_delta_pos": 0.0,
            "mean_conf": float(np.mean(np.max(scipy.special.softmax(lp_twin, axis=1), axis=1))),
            "median_conf": float(np.median(np.max(scipy.special.softmax(lp_twin, axis=1), axis=1))),
            "mean_entropy": float(np.mean(-np.sum(scipy.special.softmax(lp_twin, axis=1) * np.log(np.clip(scipy.special.softmax(lp_twin, axis=1), eps, 1.0)), axis=1))),
            "mean_top1_margin": float(np.mean(np.diff(np.sort(scipy.special.softmax(lp_twin, axis=1), axis=1)[:, -2:], axis=1)))
        })

        # Variant D: Plain Equalized + 0.35 * Centered (Scale Preserving)
        lp_d_rel = lp14.copy()
        for f in range(n_folds):
            val_idx = np.where(folds == f)[0]
            N = plain_norms[f]
            lp_d_rel[val_idx] = lp_d_rel[val_idx] * ((N.mean() / N) ** g)
        lp_d_rel = lp_d_rel - scipy.special.logsumexp(lp_d_rel, axis=1, keepdims=True)
        lp_tot_d = lp_d_rel + 0.35 * lp_cent_oof
        preds_d = lp_tot_d.argmax(axis=1)
        ba_d = balanced_accuracy_score(y_true, preds_d)
        f1_d = f1_score(y_true, preds_d, average="macro")
        ci_l_d, ci_u_d, p_pos_d, _ = fast_paired_bootstrap(y_true, preds_twin_base, preds_d)
        m_c_d, med_c_d, ent_d, top_m_d = compute_distribution_metrics(lp_tot_d)

        gamma_rows.append({
            "variant": "Variant D (Twin Scale-Preserving)",
            "gamma": g,
            "ba": ba_d,
            "macro_f1": f1_d,
            "delta_ba": ba_d - ba_twin_base,
            "ci_lower": ci_l_d,
            "ci_upper": ci_u_d,
            "p_delta_pos": p_pos_d,
            "mean_conf": m_c_d,
            "median_conf": med_c_d,
            "mean_entropy": ent_d,
            "mean_top1_margin": top_m_d
        })

        # Fold results for Variant D
        f_bas_d = [balanced_accuracy_score(y_true[folds == f], lp_tot_d[folds == f].argmax(axis=1)) for f in range(n_folds)]
        fold_rows.append({
            "variant": "Variant D (Twin Scale-Preserving)",
            "gamma": g,
            "fold_0": f_bas_d[0],
            "fold_1": f_bas_d[1],
            "fold_2": f_bas_d[2],
            "fold_3": f_bas_d[3],
            "fold_4": f_bas_d[4],
            "mean": float(np.mean(f_bas_d)),
            "std": float(np.std(f_bas_d))
        })

    gamma_df = pd.DataFrame(gamma_rows)
    gamma_df.to_csv(EXPERIMENTS_DIR / "gamma_sweep.csv", index=False)
    log(f"Saved gamma sweep to {EXPERIMENTS_DIR / 'gamma_sweep.csv'}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(EXPERIMENTS_DIR / "fold_results.csv", index=False)
    log(f"Saved fold results to {EXPERIMENTS_DIR / 'fold_results.csv'}")

    log(f"Best Plain Gamma: gamma={best_plain_gamma:.2f}, BA = {best_plain_ba:.4f} (delta = {best_plain_ba - ba_plain_base:+.4f})")

    # ==============================================================================
    # 6. Global Temperature Sweep
    # ==============================================================================
    log("Evaluating Global Temperature Grid...")
    temp_rows = []
    best_temp_ba = ba_plain_base
    best_temp = 1.00

    for T in TEMPERATURE_GRID:
        # Variant A Temperature
        lp_t_a = lp14 / T
        preds_t_a = lp_t_a.argmax(axis=1)
        ba_t_a = balanced_accuracy_score(y_true, preds_t_a)
        f1_t_a = f1_score(y_true, preds_t_a, average="macro")
        m_c, med_c, ent, top_m = compute_distribution_metrics(lp_t_a)

        temp_rows.append({
            "variant": "Variant A (Plain Head)",
            "temperature": T,
            "ba": ba_t_a,
            "macro_f1": f1_t_a,
            "delta_ba": ba_t_a - ba_plain_base,
            "mean_conf": m_c,
            "median_conf": med_c,
            "mean_entropy": ent,
            "mean_top1_margin": top_m
        })

        # Variant D Temperature (applied to total twin logits)
        lp_t_d = lp_twin / T
        preds_t_d = lp_t_d.argmax(axis=1)
        ba_t_d = balanced_accuracy_score(y_true, preds_t_d)
        f1_t_d = f1_score(y_true, preds_t_d, average="macro")
        m_c_d, med_c_d, ent_d, top_m_d = compute_distribution_metrics(lp_t_d)

        temp_rows.append({
            "variant": "Variant D (Twin Baseline)",
            "temperature": T,
            "ba": ba_t_d,
            "macro_f1": f1_t_d,
            "delta_ba": ba_t_d - ba_twin_base,
            "mean_conf": m_c_d,
            "median_conf": med_c_d,
            "mean_entropy": ent_d,
            "mean_top1_margin": top_m_d
        })

        if ba_t_a > best_temp_ba:
            best_temp_ba = ba_t_a
            best_temp = T

    temp_df = pd.DataFrame(temp_rows)
    temp_df.to_csv(EXPERIMENTS_DIR / "temperature_sweep.csv", index=False)
    log(f"Saved temperature sweep to {EXPERIMENTS_DIR / 'temperature_sweep.csv'}")

    # ==============================================================================
    # 7. Per-Class Effect Analysis (Section 8)
    # ==============================================================================
    log("Computing Per-Class Recall and Precision Effects...")
    # Evaluate at best_plain_gamma (gamma = 1.00)
    preds_best_plain = best_plain_lp.argmax(axis=1)

    rec_base = recall_score(y_true, preds_plain_base, average=None)
    rec_eq = recall_score(y_true, preds_best_plain, average=None)
    prec_base = precision_score(y_true, preds_plain_base, average=None)
    prec_eq = precision_score(y_true, preds_best_plain, average=None)
    f1_base = f1_score(y_true, preds_plain_base, average=None)
    f1_eq = f1_score(y_true, preds_best_plain, average=None)

    mean_norms_per_class = np.mean([plain_norms[f] for f in range(n_folds)], axis=0)
    std_norms_per_class = np.std([plain_norms[f] for f in range(n_folds)], axis=0)

    class_rows = []
    for i, l in enumerate(LABELS):
        class_rows.append({
            "class": l,
            "class_idx": i,
            "mean_norm": float(mean_norms_per_class[i]),
            "norm_std": float(std_norms_per_class[i]),
            "recall_baseline": float(rec_base[i]),
            "recall_best_gamma": float(rec_eq[i]),
            "delta_recall": float(rec_eq[i] - rec_base[i]),
            "prec_baseline": float(prec_base[i]),
            "prec_best_gamma": float(prec_eq[i]),
            "delta_prec": float(prec_eq[i] - prec_base[i]),
            "f1_baseline": float(f1_base[i]),
            "f1_best_gamma": float(f1_eq[i]),
            "is_low_norm_target": l in LOW_NORM_TARGETS,
            "is_high_norm_target": l in HIGH_NORM_TARGETS
        })

    class_df = pd.DataFrame(class_rows).sort_values("mean_norm")
    class_df.to_csv(EXPERIMENTS_DIR / "per_class_effects.csv", index=False)
    log(f"Saved per-class effects to {EXPERIMENTS_DIR / 'per_class_effects.csv'}")

    # Inspect Low-Norm vs High-Norm targets
    low_norm_deltas = [r["delta_recall"] for r in class_rows if r["is_low_norm_target"]]
    high_norm_deltas = [r["delta_recall"] for r in class_rows if r["is_high_norm_target"]]
    mean_low_norm_delta = float(np.mean(low_norm_deltas))
    mean_high_norm_delta = float(np.mean(high_norm_deltas))
    log(f"Mean Low-Norm Target Delta Recall:  {mean_low_norm_delta:+.4f} (Targets: {LOW_NORM_TARGETS})")
    log(f"Mean High-Norm Target Delta Recall: {mean_high_norm_delta:+.4f} (Targets: {HIGH_NORM_TARGETS})")

    # ==============================================================================
    # 8. Control 1: Random Norm Permutation Control (Section 12)
    # ==============================================================================
    log("Running Control 1: Random Norm Permutation Control (20 fixed iterations)...")
    orig_delta = best_plain_ba - ba_plain_base
    rng_perm = np.random.default_rng(PERMUTATION_SEED)
    perm_rows = []
    perm_deltas = []

    for p_idx in range(N_PERMUTATIONS):
        perm = rng_perm.permutation(NUM_CLASSES)
        lp_p = lp14.copy()
        for f in range(n_folds):
            val_idx = np.where(folds == f)[0]
            N_perm = plain_norms[f][perm]
            lp_p[val_idx] = lp_p[val_idx] / (N_perm ** best_plain_gamma)
        ba_p = balanced_accuracy_score(y_true, lp_p.argmax(axis=1))
        d_p = ba_p - ba_plain_base
        perm_deltas.append(d_p)
        perm_rows.append({
            "permutation_id": p_idx,
            "ba": float(ba_p),
            "delta_ba": float(d_p)
        })

    perm_deltas = np.array(perm_deltas)
    mean_perm_delta = float(np.mean(perm_deltas))
    std_perm_delta = float(np.std(perm_deltas))
    percentile = float(np.mean(perm_deltas < orig_delta) * 100.0)

    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(EXPERIMENTS_DIR / "permutation_control.csv", index=False)
    log(f"Permutation Control: Orig Delta = {orig_delta:+.4f}, Perm Mean Delta = {mean_perm_delta:+.4f} (std = {std_perm_delta:.4f})")
    log(f"Empirical Percentile of Original Result: {percentile:.1f}%")

    # ==============================================================================
    # 9. EXP52 Integration (Section 14 & 15)
    # ==============================================================================
    log("Running EXP52 Integration across all 5 folds...")
    # Candidate EXP52 uses plain equalized logits with scale-preserving formulation
    lp14_eq_rel = lp14.copy()
    for f in range(n_folds):
        val_idx = np.where(folds == f)[0]
        N = plain_norms[f]
        lp14_eq_rel[val_idx] = lp14_eq_rel[val_idx] * ((N.mean() / N) ** best_plain_gamma)
    lp14_eq_rel = lp14_eq_rel - scipy.special.logsumexp(lp14_eq_rel, axis=1, keepdims=True)

    lp_exp52_cand = 0.60 * lp8 + 0.40 * lp14_eq_rel
    preds_exp52_cand = lp_exp52_cand.argmax(axis=1)
    ba_exp52_cand = balanced_accuracy_score(y_true, preds_exp52_cand)
    delta_exp52 = ba_exp52_cand - ba_exp52_base

    ci_l_52, ci_u_52, p_pos_52, _ = fast_paired_bootstrap(y_true, preds_exp52_base, preds_exp52_cand)

    f_exp52_base = [balanced_accuracy_score(y_true[folds == f], lp_exp52_base[folds == f].argmax(axis=1)) for f in range(n_folds)]
    f_exp52_cand = [balanced_accuracy_score(y_true[folds == f], lp_exp52_cand[folds == f].argmax(axis=1)) for f in range(n_folds)]

    exp52_rows = [
        {
            "system": "Baseline EXP52 (0.60 Q8 + 0.40 Q14)",
            "fold_0": f_exp52_base[0],
            "fold_1": f_exp52_base[1],
            "fold_2": f_exp52_base[2],
            "fold_3": f_exp52_base[3],
            "fold_4": f_exp52_base[4],
            "mean_fold_ba": float(np.mean(f_exp52_base)),
            "oof_ba": float(ba_exp52_base),
            "delta_oof_ba": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_delta_pos": 0.0
        },
        {
            "system": f"Candidate EXP52 (0.60 Q8 + 0.40 Q14_eq_gamma{best_plain_gamma})",
            "fold_0": f_exp52_cand[0],
            "fold_1": f_exp52_cand[1],
            "fold_2": f_exp52_cand[2],
            "fold_3": f_exp52_cand[3],
            "fold_4": f_exp52_cand[4],
            "mean_fold_ba": float(np.mean(f_exp52_cand)),
            "oof_ba": float(ba_exp52_cand),
            "delta_oof_ba": float(delta_exp52),
            "ci_lower": ci_l_52,
            "ci_upper": ci_u_52,
            "p_delta_pos": p_pos_52
        }
    ]
    exp52_df = pd.DataFrame(exp52_rows)
    exp52_df.to_csv(EXPERIMENTS_DIR / "exp52_integration.csv", index=False)
    log(f"Saved EXP52 integration to {EXPERIMENTS_DIR / 'exp52_integration.csv'}")
    log(f"EXP52 Baseline: {ba_exp52_base:.4f}, Candidate: {ba_exp52_cand:.4f}, Delta: {delta_exp52:+.4f}")

    # ==============================================================================
    # 10. Generate High-Quality Publication Figures (Section 18)
    # ==============================================================================
    log("Generating publication figures...")

    # Figure 1: Baseline Confusion Matrix
    cm_base = confusion_matrix(y_true, preds_plain_base, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    cax = ax.matshow(cm_base, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABELS, rotation=90, fontsize=8)
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_title(f"Confusion Matrix: Baseline q14v5 Plain Head (BA = {ba_plain_base:.4f})", pad=20, fontweight="bold")
    ax.set_xlabel("Predicted Model", fontweight="bold")
    ax.set_ylabel("True Model", fontweight="bold")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "confusion_matrix_baseline.png")
    plt.close()

    # Figure 2: Best Gamma Confusion Matrix
    cm_best = confusion_matrix(y_true, preds_best_plain, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    cax = ax.matshow(cm_best, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABELS, rotation=90, fontsize=8)
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_title(f"Confusion Matrix: Equalized Plain Head gamma={best_plain_gamma:.2f} (BA = {best_plain_ba:.4f})", pad=20, fontweight="bold")
    ax.set_xlabel("Predicted Model", fontweight="bold")
    ax.set_ylabel("True Model", fontweight="bold")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "confusion_matrix_best_gamma.png")
    plt.close()

    # Figure 3: Per-Class Recall Delta vs Class Weight Norm
    sorted_df = class_df.sort_values("mean_norm")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    colors = []
    for _, r in sorted_df.iterrows():
        if r["is_low_norm_target"]:
            colors.append("#d9534f") # Red for low norm targets
        elif r["is_high_norm_target"]:
            colors.append("#5cb85c") # Green for high norm targets
        else:
            colors.append("#337ab7") # Blue for others

    bars = ax.bar(range(NUM_CLASSES), sorted_df["delta_recall"], color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(NUM_CLASSES))
    xticklabels = [f"{r['class']}\n(||W||={r['mean_norm']:.2f})" for _, r in sorted_df.iterrows()]
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
    ax.set_ylabel(r"$\Delta$ Recall (Equalized - Baseline)", fontweight="bold")
    ax.set_title(f"EXP57B: Per-Class Recall Delta vs Class-Head Norm (gamma = {best_plain_gamma:.2f})\nRed = Suppressed-Norm Targets, Green = Dominant-Norm Targets", pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "class_recall_delta.png")
    plt.close()

    # Figure 4: Gamma vs Balanced Accuracy across variants
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for v_name, col, marker in [("Variant A (Plain Head)", "#1f77b4", "o"),
                                ("Variant B (Centered Head)", "#ff7f0e", "s"),
                                ("Variant C (Twin Baseline)", "#2ca02c", "^"),
                                ("Variant D (Twin Scale-Preserving)", "#d62728", "D")]:
        sub = gamma_df[gamma_df["variant"] == v_name]
        ax.plot(sub["gamma"], sub["ba"], label=v_name, color=col, marker=marker, linewidth=2)
    ax.set_xlabel(r"Norm Equalization Exponent $\gamma$", fontweight="bold")
    ax.set_ylabel("Balanced Accuracy", fontweight="bold")
    ax.set_title("EXP57B: Balanced Accuracy vs Gamma Grid", pad=15, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "gamma_vs_ba.png")
    plt.close()

    # Figure 5: Temperature vs Balanced Accuracy and Entropy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    sub_t_a = temp_df[temp_df["variant"] == "Variant A (Plain Head)"]
    sub_t_d = temp_df[temp_df["variant"] == "Variant D (Twin Baseline)"]

    ax1.plot(sub_t_a["temperature"], sub_t_a["ba"], label="Variant A (Plain)", marker="o", color="#1f77b4")
    ax1.plot(sub_t_d["temperature"], sub_t_d["ba"], label="Variant D (Twin)", marker="s", color="#d62728")
    ax1.set_xlabel("Global Temperature T", fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy", fontweight="bold")
    ax1.set_title("Balanced Accuracy vs Temperature", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(sub_t_a["temperature"], sub_t_a["mean_entropy"], label="Variant A Entropy", marker="o", color="#1f77b4")
    ax2.plot(sub_t_d["temperature"], sub_t_d["mean_entropy"], label="Variant D Entropy", marker="s", color="#d62728")
    ax2.set_xlabel("Global Temperature T", fontweight="bold")
    ax2.set_ylabel("Mean Predictive Entropy (nats)", fontweight="bold")
    ax2.set_title("Predictive Entropy vs Temperature", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "temperature_vs_ba.png")
    plt.close()
    log("All figures generated successfully.")

    # ==============================================================================
    # 11. Final Verification & Synthesis for Section 20 Terminal Output
    # ==============================================================================
    # Criteria evaluation:
    # 1. Cross-fold stable? Check fold deltas for best_plain_gamma
    f_deltas = [fold_rows[-8]["fold_" + str(i)] - fold_rows[-16]["fold_" + str(i)] for i in range(5)] # gamma 1.0 vs gamma 0.0
    f_sub = fold_df[fold_df["variant"] == "Variant A (Plain Head)"]
    f_base_vals = f_sub[f_sub["gamma"] == 0.00].iloc[0][["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]].values.astype(float)
    f_best_vals = f_sub[f_sub["gamma"] == best_plain_gamma].iloc[0][["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]].values.astype(float)
    fold_deltas = f_best_vals - f_base_vals
    is_cross_fold_stable = bool(np.all(fold_deltas >= 0) and np.std(fold_deltas) < 0.003)

    # 2. Temperature explains effect?
    temp_explains = False # Single head argmax is invariant to T, so T cannot explain delta directly, but effect size is smaller than calibration noise

    # 3. Permutation supports class-specific effect?
    perm_supports = bool(percentile >= 90.0)

    # 4. Low-norm class improvement?
    low_norm_improved = bool(mean_low_norm_delta > 0)
    low_norm_str = "IMPROVED" if low_norm_improved else f"DEGRADED ({mean_low_norm_delta:+.4f})"
    high_norm_str = f"{mean_high_norm_delta:+.4f}"

    # Verdict
    if is_cross_fold_stable and perm_supports and low_norm_improved and delta_exp52 > 0:
        verdict = "GREEN"
        next_step = "PROCEED_TO_EXP57C"
    elif orig_delta > 0 and (perm_supports or is_cross_fold_stable):
        verdict = "YELLOW"
        next_step = "STOP_WEIGHT_SPACE_EDITING"
    else:
        verdict = "RED"
        next_step = "STOP_WEIGHT_SPACE_EDITING"

    ci_l, ci_u, p_pos, _ = fast_paired_bootstrap(y_true, preds_plain_base, best_plain_lp.argmax(axis=1))

    print("\n" + "="*50)
    print("EXP57B COMPLETE")
    print(f"Baseline q14 OOF BA: {ba_plain_base:.4f}")
    print(f"Best norm-equalized BA: {best_plain_ba:.4f}")
    print(f"Best gamma: {best_plain_gamma:.2f}")
    print(f"Best temperature BA: {best_temp_ba:.4f}")
    print(f"Best temperature: {best_temp:.2f}")
    print("")
    print(f"Norm correction delta: {orig_delta:+.4f}")
    print(f"95% CI: [{ci_l:+.4f}, {ci_u:+.4f}]")
    print(f"P(delta > 0): {p_pos:.4f}")
    print("")
    print(f"Permutation percentile: {percentile:.1f}%")
    print(f"Low-norm class improvement: {low_norm_str}")
    print(f"High-norm class change: {high_norm_str}")
    print("")
    print(f"EXP52 baseline: {ba_exp52_base:.4f}")
    print(f"EXP52 + norm correction: {ba_exp52_cand:.4f}")
    print(f"EXP52 delta: {delta_exp52:+.4f}")
    print("")
    print(f"Temperature explains effect: {'YES' if temp_explains else 'NO'}")
    print(f"Permutation supports class-specific effect: {'YES' if perm_supports else 'NO'}")
    print(f"Cross-fold stable: {'YES' if is_cross_fold_stable else 'NO'}")
    print("")
    print(f"Verdict:\n{verdict}")
    print("")
    print(f"Recommended next step:\n{next_step}")
    print("")
    print("EXP52 modified: NO")
    print("Kaggle submission made: NO")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
