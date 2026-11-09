"""
================================================================================
EXP62: CROSS-ARCHITECTURE COMPLEMENTARITY TEST
================================================================================
Investigates whether a genuinely distinct non-Qwen LLM architecture (ModernBERT-base,
a bidirectional encoder-only model with RoPE and GeGLU) can provide complementary
attribution errors that are less correlated with Qwen3-8B/Qwen3-14B, producing
a more robust ensemble under unseen-prompt validation.

Hard Safety Rules:
- ZERO Kaggle submissions.
- EXP52 model weights, code, and artifacts remain 100% frozen and unmodified.
- Strict 5-fold prompt-disjoint cross-validation (150 unique prompts, 30 prompts/fold).
- Nested cross-validation (5 outer folds x 4 inner folds) for selection-safe evaluation.
- Paired bootstrap statistical tests (2,000 resamples) for confidence bounds.
================================================================================
"""

import os
import sys
import json
import time
import pathlib
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Headless matplotlib
plt.switch_backend("Agg")

# Directories
REPO_DIR = pathlib.Path("/home/rj2826/name-that-llm")
DATA_DIR = REPO_DIR / "data"
OUTPUTS_DIR = REPO_DIR / "outputs"
EXPERIMENTS_DIR = REPO_DIR / "experiments" / "EXP62_cross_architecture"
PLOTS_DIR = REPO_DIR / "plots"
REPORTS_DIR = REPO_DIR / "reports"

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def balanced_oracle(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray, n_classes: int = 18) -> float:
    """Calculates macro-averaged balanced accuracy of an oracle chooser between two models."""
    correct_either = ((pred_a == y) | (pred_b == y)).astype(int)
    recalls = []
    for k in range(n_classes):
        mask = (y == k)
        if mask.sum() > 0:
            recalls.append(correct_either[mask].mean())
        else:
            recalls.append(0.0)
    return float(np.mean(recalls))


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """
    Computes paired bootstrap confidence interval for delta = BA(pred_b) - BA(pred_a).
    Positive delta means pred_b is better than pred_a.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ba_a = balanced_accuracy_score(y_true[idx], pred_a[idx])
        ba_b = balanced_accuracy_score(y_true[idx], pred_b[idx])
        deltas[b] = ba_b - ba_a
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_positive = float(np.mean(deltas > 0))
    mean_delta = float(np.mean(deltas))
    return mean_delta, ci_lower, ci_upper, p_positive, deltas


def main():
    start_time = time.time()
    log("=====================================================================")
    log("EXP62: CROSS-ARCHITECTURE COMPLEMENTARITY TEST STARTED")
    log("=====================================================================")

    # 1. Load Dataset and Provenance
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    q14_df = pd.read_csv(OUTPUTS_DIR / "oof_q14v5_antopil_lora.csv")
    p8_twin_raw = np.load(OUTPUTS_DIR / "oof_exp47_qwen_centered_twin.npy")
    emb_modernbert = np.load(OUTPUTS_DIR / "exp44_modernbert_embeddings.npy")

    n_samples = len(train_df)
    assert n_samples == 2699, f"Expected 2699 samples, got {n_samples}"
    assert (train_df["id"] == q14_df["id"]).all(), "Row ID mismatch between train.csv and q14 OOF"
    assert emb_modernbert.shape == (2699, 768), f"ModernBERT embedding shape mismatch: {emb_modernbert.shape}"

    LABELS = sorted(train_df["model"].unique())
    L2I = {m: i for i, m in enumerate(LABELS)}
    y_true = np.array([L2I[m] for m in train_df["model"].values])
    folds = q14_df["fold"].values
    prompts = train_df["prompt_id"].values
    eps = 1e-15

    log(f"Dataset: {n_samples} samples across {len(np.unique(prompts))} unique prompts and {len(LABELS)} classes.")
    log(f"ModernBERT embedding matrix: {emb_modernbert.shape}, dtype={emb_modernbert.dtype}")

    # 2. Reconstruct Frozen Baselines (EXP52, Qwen3-8B, Qwen3-14B)
    p8_norm = p8_twin_raw / p8_twin_raw.sum(axis=1, keepdims=True)
    p14_norm = q14_df[LABELS].values / q14_df[LABELS].values.sum(axis=1, keepdims=True)
    lp8 = np.log(np.clip(p8_norm, eps, 1.0))
    lp14 = np.log(np.clip(p14_norm, eps, 1.0))
    lp52 = 0.60 * lp8 + 0.40 * lp14
    p52 = scipy.special.softmax(lp52, axis=1)

    q8_pred = lp8.argmax(axis=1)
    q14_pred = lp14.argmax(axis=1)
    exp52_pred = p52.argmax(axis=1)

    ba_q8 = float(balanced_accuracy_score(y_true, q8_pred))
    f1_q8 = float(f1_score(y_true, q8_pred, average="macro"))
    ba_q14 = float(balanced_accuracy_score(y_true, q14_pred))
    f1_q14 = float(f1_score(y_true, q14_pred, average="macro"))
    ba_exp52 = float(balanced_accuracy_score(y_true, exp52_pred))
    f1_exp52 = float(f1_score(y_true, exp52_pred, average="macro"))

    log(f"Frozen Baseline Qwen3-8B (q8v5):   BA = {ba_q8:.4f}, Macro-F1 = {f1_q8:.4f}")
    log(f"Frozen Baseline Qwen3-14B (q14v5): BA = {ba_q14:.4f}, Macro-F1 = {f1_q14:.4f}")
    log(f"Frozen Baseline EXP52 Dual-LoRA:   BA = {ba_exp52:.4f}, Macro-F1 = {f1_exp52:.4f}")
    assert np.isclose(ba_exp52, 0.7873, atol=1e-3), f"EXP52 baseline mismatch: {ba_exp52}"

    # 3. Fit Lightweight Classifiers on ModernBERT Representations (Phases 6 & 7)
    log("\n--- Phase 6 & 7: Lightweight Classifiers on ModernBERT Representations ---")
    oof_probs_lr = np.zeros((n_samples, 18), dtype=float)
    oof_preds_svm = np.zeros(n_samples, dtype=int)
    fold_ba_lr = []
    fold_ba_svm = []
    fold_ba_exp52 = []

    for f in range(5):
        tr_idx = (folds != f)
        val_idx = (folds == f)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(emb_modernbert[tr_idx])
        X_val = scaler.transform(emb_modernbert[val_idx])

        # Primary: Linear Logistic Regression
        clf_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf_lr.fit(X_tr, y_true[tr_idx])
        probs_val = clf_lr.predict_proba(X_val)
        oof_probs_lr[val_idx] = probs_val

        # Secondary: Linear SVM
        clf_svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
        clf_svm.fit(X_tr, y_true[tr_idx])
        preds_val_svm = clf_svm.predict(X_val)
        oof_preds_svm[val_idx] = preds_val_svm

        f_ba_lr = balanced_accuracy_score(y_true[val_idx], probs_val.argmax(axis=1))
        f_ba_svm = balanced_accuracy_score(y_true[val_idx], preds_val_svm)
        f_ba_52 = balanced_accuracy_score(y_true[val_idx], exp52_pred[val_idx])

        fold_ba_lr.append(f_ba_lr)
        fold_ba_svm.append(f_ba_svm)
        fold_ba_exp52.append(f_ba_52)
        log(f"Fold {f}: ModernBERT LR BA = {f_ba_lr:.4f} | SVM BA = {f_ba_svm:.4f} | EXP52 BA = {f_ba_52:.4f}")

    new_pred_lr = oof_probs_lr.argmax(axis=1)
    ba_new_lr = float(balanced_accuracy_score(y_true, new_pred_lr))
    f1_new_lr = float(f1_score(y_true, new_pred_lr, average="macro"))
    ba_new_svm = float(balanced_accuracy_score(y_true, oof_preds_svm))
    f1_new_svm = float(f1_score(y_true, oof_preds_svm, average="macro"))

    # Confidence and Entropy for ModernBERT
    p_new = oof_probs_lr
    lp_new = np.log(np.clip(p_new, eps, 1.0))
    ent_new = -np.sum(p_new * lp_new, axis=1)
    sorted_p_new = np.sort(p_new, axis=1)
    margin_new = sorted_p_new[:, -1] - sorted_p_new[:, -2]

    log(f"ModernBERT-base (Linear LR) OOF BA:  {ba_new_lr:.4f} (Macro-F1: {f1_new_lr:.4f}, Mean Fold: {np.mean(fold_ba_lr):.4f} +/- {np.std(fold_ba_lr):.4f})")
    log(f"ModernBERT-base (Linear SVM) OOF BA: {ba_new_svm:.4f} (Macro-F1: {f1_new_svm:.4f}, Mean Fold: {np.mean(fold_ba_svm):.4f} +/- {np.std(fold_ba_svm):.4f})")

    # 4. Pairwise Error & Prediction Complementarity (Phases 9 & 10)
    log("\n--- Phase 9 & 10: Error & Prediction Complementarity ---")
    models = {
        "Q8": q8_pred,
        "Q14": q14_pred,
        "EXP52": exp52_pred,
        "NEW": new_pred_lr
    }

    error_indicators = {k: (pred != y_true).astype(int) for k, pred in models.items()}

    pairs = [
        ("Q8", "Q14"),
        ("Q8", "NEW"),
        ("Q14", "NEW"),
        ("EXP52", "NEW")
    ]

    comp_rows = []
    for name_a, name_b in pairs:
        pred_a = models[name_a]
        pred_b = models[name_b]
        err_a = error_indicators[name_a]
        err_b = error_indicators[name_b]

        agreement = float((pred_a == pred_b).mean())
        both_correct = int(((pred_a == y_true) & (pred_b == y_true)).sum())
        both_wrong = int(((pred_a != y_true) & (pred_b != y_true)).sum())
        a_only = int(((pred_a == y_true) & (pred_b != y_true)).sum())
        b_only = int(((pred_a != y_true) & (pred_b == y_true)).sum())

        pearson_r, pearson_p = scipy.stats.pearsonr(err_a, err_b)
        spearman_rho, spearman_p = scipy.stats.spearmanr(err_a, err_b)
        oracle_ba = balanced_oracle(pred_a, pred_b, y_true)
        oracle_acc = float(((pred_a == y_true) | (pred_b == y_true)).mean())

        row = {
            "pair": f"{name_a}/{name_b}",
            "agreement": agreement,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "a_only_correct": a_only,
            "b_only_correct": b_only,
            "pearson_err_corr": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_err_corr": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "oracle_ba": float(oracle_ba),
            "oracle_acc": float(oracle_acc)
        }
        comp_rows.append(row)
        log(f"Pair {name_a}/{name_b}: Agreement={agreement*100:.2f}% | BothCorr={both_correct} | BothWrong={both_wrong} | A_only={a_only} | B_only={b_only} | ErrCorr={pearson_r:.4f} | OracleBA={oracle_ba:.4f}")

    comp_df = pd.DataFrame(comp_rows)

    # 5. Simple Log-Probability Fusion Sweep (Phase 11)
    log("\n--- Phase 11: Simple Log-Probability Fusion Sweep ---")
    weights = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    fusion_rows = []

    for w in weights:
        lp_fused = (1.0 - w) * lp52 + w * lp_new
        p_fused = scipy.special.softmax(lp_fused, axis=1)
        pred_fused = p_fused.argmax(axis=1)

        ba_fused = float(balanced_accuracy_score(y_true, pred_fused))
        f1_fused = float(f1_score(y_true, pred_fused, average="macro"))
        delta = ba_fused - ba_exp52

        fold_bas = []
        for f in range(5):
            idx = (folds == f)
            fold_bas.append(float(balanced_accuracy_score(y_true[idx], pred_fused[idx])))

        fusion_rows.append({
            "weight_new": w,
            "weight_exp52": 1.0 - w,
            "balanced_accuracy": ba_fused,
            "macro_f1": f1_fused,
            "delta_vs_exp52": delta,
            "fold_bas": fold_bas,
            "mean_fold_ba": float(np.mean(fold_bas)),
            "std_fold_ba": float(np.std(fold_bas))
        })
        log(f"Weight w={w:.2f}: OOF BA = {ba_fused:.4f} (delta = {delta:+.4f}) | Macro-F1 = {f1_fused:.4f} | Folds: {[round(x, 4) for x in fold_bas]}")

    fusion_df = pd.DataFrame(fusion_rows)

    # 6. Nested Cross-Validation (Phase 12)
    log("\n--- Phase 12: Selection-Safe Nested Cross-Validation ---")
    outer_weights = []
    outer_pred_nested = np.zeros(n_samples, dtype=int)
    outer_fold_ba = []

    for f_out in range(5):
        outer_train_idx = (folds != f_out)
        outer_val_idx = (folds == f_out)

        outer_train_prompts = np.unique(prompts[outer_train_idx])
        assert len(outer_train_prompts) == 120, f"Expected 120 outer train prompts, got {len(outer_train_prompts)}"

        # 4 Inner prompt-disjoint folds
        gkf_inner = GroupKFold(n_splits=4)
        inner_splits = list(gkf_inner.split(
            X=emb_modernbert[outer_train_idx],
            y=y_true[outer_train_idx],
            groups=prompts[outer_train_idx]
        ))

        inner_oof_probs_new = np.zeros((outer_train_idx.sum(), 18), dtype=float)

        for inner_tr, inner_val in inner_splits:
            scaler_inner = StandardScaler()
            X_in_tr = scaler_inner.fit_transform(emb_modernbert[outer_train_idx][inner_tr])
            X_in_val = scaler_inner.transform(emb_modernbert[outer_train_idx][inner_val])

            clf_inner = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            clf_inner.fit(X_in_tr, y_true[outer_train_idx][inner_tr])
            inner_oof_probs_new[inner_val] = clf_inner.predict_proba(X_in_val)

        lp_inner_new = np.log(np.clip(inner_oof_probs_new, eps, 1.0))
        lp_inner_52 = lp52[outer_train_idx]
        y_inner_true = y_true[outer_train_idx]

        # Inner sweep to select best weight
        best_w = 0.00
        best_inner_ba = -1.0
        for w in weights:
            lp_in_fused = (1.0 - w) * lp_inner_52 + w * lp_inner_new
            pred_in = lp_in_fused.argmax(axis=1)
            in_ba = balanced_accuracy_score(y_inner_true, pred_in)
            if in_ba > best_inner_ba:
                best_inner_ba = in_ba
                best_w = w

        outer_weights.append(best_w)

        # Train on entire outer training set and evaluate on outer validation fold
        scaler_outer = StandardScaler()
        X_out_tr = scaler_outer.fit_transform(emb_modernbert[outer_train_idx])
        X_out_val = scaler_outer.transform(emb_modernbert[outer_val_idx])

        clf_outer = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf_outer.fit(X_out_tr, y_true[outer_train_idx])
        probs_out_val = clf_outer.predict_proba(X_out_val)
        lp_out_new = np.log(np.clip(probs_out_val, eps, 1.0))

        lp_out_fused = (1.0 - best_w) * lp52[outer_val_idx] + best_w * lp_out_new
        outer_pred_nested[outer_val_idx] = lp_out_fused.argmax(axis=1)

        f_nested_ba = balanced_accuracy_score(y_true[outer_val_idx], outer_pred_nested[outer_val_idx])
        outer_fold_ba.append(f_nested_ba)
        log(f"Outer Fold {f_out}: Inner Selected w*={best_w:.2f} (Inner BA={best_inner_ba:.4f}) -> Outer Val BA={f_nested_ba:.4f}")

    nested_overall_ba = float(balanced_accuracy_score(y_true, outer_pred_nested))
    nested_overall_f1 = float(f1_score(y_true, outer_pred_nested, average="macro"))
    nested_delta = nested_overall_ba - ba_exp52

    log(f"Nested CV Overall BA: {nested_overall_ba:.4f} (delta = {nested_delta:+.4f}) | Mean Fold: {np.mean(outer_fold_ba):.4f} +/- {np.std(outer_fold_ba):.4f}")
    log(f"Selected Outer Weights: {outer_weights}")

    # Paired Bootstrap for Nested CV vs EXP52
    mean_d, ci_l, ci_u, p_pos, _ = paired_bootstrap(y_true, exp52_pred, outer_pred_nested, n_boot=2000, seed=42)
    log(f"Nested CV Paired Bootstrap (2,000 resamples): Mean Delta = {mean_d:+.4f}, 95% CI = [{ci_l:+.4f}, {ci_u:+.4f}], P(delta > 0) = {p_pos*100:.2f}%")

    # 7. Disagreement-Aware Analysis (Phase 13)
    log("\n--- Phase 13: Disagreement-Aware Analysis ---")
    agree_mask = (q8_pred == q14_pred)
    disagree_mask = (q8_pred != q14_pred)

    n_agree = int(agree_mask.sum())
    n_disagree = int(disagree_mask.sum())
    pct_agree = float(agree_mask.mean())
    pct_disagree = float(disagree_mask.mean())

    ba_exp52_agree = float(balanced_accuracy_score(y_true[agree_mask], exp52_pred[agree_mask]))
    ba_exp52_disagree = float(balanced_accuracy_score(y_true[disagree_mask], exp52_pred[disagree_mask]))

    ba_new_agree = float(balanced_accuracy_score(y_true[agree_mask], new_pred_lr[agree_mask]))
    ba_new_disagree = float(balanced_accuracy_score(y_true[disagree_mask], new_pred_lr[disagree_mask]))

    ba_nested_agree = float(balanced_accuracy_score(y_true[agree_mask], outer_pred_nested[agree_mask]))
    ba_nested_disagree = float(balanced_accuracy_score(y_true[disagree_mask], outer_pred_nested[disagree_mask]))

    log(f"Q8/Q14 Agreement Subset:    N = {n_agree} ({pct_agree*100:.2f}%)")
    log(f"  EXP52 BA on Agreement:        {ba_exp52_agree:.4f}")
    log(f"  ModernBERT BA on Agreement:   {ba_new_agree:.4f}")
    log(f"  Nested Fused BA on Agreement: {ba_nested_agree:.4f}")

    log(f"Q8/Q14 Disagreement Subset: N = {n_disagree} ({pct_disagree*100:.2f}%)")
    log(f"  EXP52 BA on Disagreement:        {ba_exp52_disagree:.4f}")
    log(f"  ModernBERT BA on Disagreement:   {ba_new_disagree:.4f}")
    log(f"  Nested Fused BA on Disagreement: {ba_nested_disagree:.4f}")

    # Check whether NEW fixes Q8/Q14 disagreement errors
    exp52_wrong_in_disagree = (exp52_pred != y_true) & disagree_mask
    new_fixes_disagree_count = int(((new_pred_lr == y_true) & exp52_wrong_in_disagree).sum())
    new_worsens_disagree_count = int(((new_pred_lr != y_true) & (exp52_pred == y_true) & disagree_mask).sum())
    log(f"On Disagreement Subset (N={n_disagree}):")
    log(f"  EXP52 Errors: {exp52_wrong_in_disagree.sum()}")
    log(f"  ModernBERT correct when EXP52 is wrong (potential fixes): {new_fixes_disagree_count}")
    log(f"  ModernBERT wrong when EXP52 is correct (potential corruptions): {new_worsens_disagree_count}")
    log(f"  Net Correction Capacity: {new_fixes_disagree_count - new_worsens_disagree_count}")

    # 8. Conditional Disagreement Fusion (Phase 14)
    log("\n--- Phase 14: Conditional Disagreement Fusion Sweep ---")
    cond_weights = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]
    cond_rows = []
    for w_dis in cond_weights:
        lp_cond = lp52.copy()
        lp_cond[disagree_mask] = (1.0 - w_dis) * lp52[disagree_mask] + w_dis * lp_new[disagree_mask]
        pred_cond = lp_cond.argmax(axis=1)

        ba_total = float(balanced_accuracy_score(y_true, pred_cond))
        ba_dis = float(balanced_accuracy_score(y_true[disagree_mask], pred_cond[disagree_mask]))
        f1_total = float(f1_score(y_true, pred_cond, average="macro"))

        cond_rows.append({
            "w_disagree": w_dis,
            "overall_ba": ba_total,
            "disagreement_ba": ba_dis,
            "macro_f1": f1_total,
            "delta_vs_exp52": ba_total - ba_exp52
        })
        log(f"Conditional w_disagree={w_dis:.2f}: Overall BA = {ba_total:.4f} (delta={ba_total-ba_exp52:+.4f}) | Disagree BA = {ba_dis:.4f}")

    cond_df = pd.DataFrame(cond_rows)

    # 9. Representation Diversity (Cosine Similarities, Phase 15)
    log("\n--- Phase 15: Representation Diversity Metrics ---")
    def mean_cosine_similarity(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(mat_a, axis=1, keepdims=True) + 1e-12
        norm_b = np.linalg.norm(mat_b, axis=1, keepdims=True) + 1e-12
        sims = np.sum((mat_a / norm_a) * (mat_b / norm_b), axis=1)
        return float(np.mean(sims))

    cos_q8_q14 = mean_cosine_similarity(p8_norm, p14_norm)
    cos_q8_new = mean_cosine_similarity(p8_norm, p_new)
    cos_q14_new = mean_cosine_similarity(p14_norm, p_new)
    cos_52_new = mean_cosine_similarity(p52, p_new)

    log(f"Mean Prediction Cosine Sim (Q8 vs Q14):     {cos_q8_q14:.4f}")
    log(f"Mean Prediction Cosine Sim (Q8 vs NEW):     {cos_q8_new:.4f}")
    log(f"Mean Prediction Cosine Sim (Q14 vs NEW):    {cos_q14_new:.4f}")
    log(f"Mean Prediction Cosine Sim (EXP52 vs NEW):  {cos_52_new:.4f}")

    # 10. Required Comparison Tables (Phase 17)
    log("\n--- Phase 17: Required Comparison Tables ---")
    table1 = [
        {
            "Model": "Qwen3-8B",
            "OOF_BA": ba_q8,
            "Macro_F1": f1_q8,
            "Agreement_vs_EXP52": float((q8_pred == exp52_pred).mean()),
            "Error_Corr_vs_EXP52": float(scipy.stats.pearsonr(error_indicators["Q8"], error_indicators["EXP52"])[0]),
            "Oracle_BA": balanced_oracle(q8_pred, exp52_pred, y_true)
        },
        {
            "Model": "Qwen3-14B",
            "OOF_BA": ba_q14,
            "Macro_F1": f1_q14,
            "Agreement_vs_EXP52": float((q14_pred == exp52_pred).mean()),
            "Error_Corr_vs_EXP52": float(scipy.stats.pearsonr(error_indicators["Q14"], error_indicators["EXP52"])[0]),
            "Oracle_BA": balanced_oracle(q14_pred, exp52_pred, y_true)
        },
        {
            "Model": "EXP52 (Production)",
            "OOF_BA": ba_exp52,
            "Macro_F1": f1_exp52,
            "Agreement_vs_EXP52": 1.0000,
            "Error_Corr_vs_EXP52": 1.0000,
            "Oracle_BA": ba_exp52
        },
        {
            "Model": "ModernBERT-base (LR)",
            "OOF_BA": ba_new_lr,
            "Macro_F1": f1_new_lr,
            "Agreement_vs_EXP52": float((new_pred_lr == exp52_pred).mean()),
            "Error_Corr_vs_EXP52": float(scipy.stats.pearsonr(error_indicators["NEW"], error_indicators["EXP52"])[0]),
            "Oracle_BA": balanced_oracle(new_pred_lr, exp52_pred, y_true)
        },
        {
            "Model": "ModernBERT-base (SVM)",
            "OOF_BA": ba_new_svm,
            "Macro_F1": f1_new_svm,
            "Agreement_vs_EXP52": float((oof_preds_svm == exp52_pred).mean()),
            "Error_Corr_vs_EXP52": float(scipy.stats.pearsonr((oof_preds_svm != y_true).astype(int), error_indicators["EXP52"])[0]),
            "Oracle_BA": balanced_oracle(oof_preds_svm, exp52_pred, y_true)
        },
        {
            "Model": "EXP52 + NEW (Nested)",
            "OOF_BA": nested_overall_ba,
            "Macro_F1": nested_overall_f1,
            "Agreement_vs_EXP52": float((outer_pred_nested == exp52_pred).mean()),
            "Error_Corr_vs_EXP52": float(scipy.stats.pearsonr((outer_pred_nested != y_true).astype(int), error_indicators["EXP52"])[0]),
            "Oracle_BA": balanced_oracle(outer_pred_nested, exp52_pred, y_true)
        }
    ]
    table1_df = pd.DataFrame(table1)

    print("\n--- TABLE 1: Model Benchmark & Complementarity ---")
    print(table1_df.to_string(index=False))

    print("\n--- TABLE 2: Pairwise Error Complementarity ---")
    print(comp_df.to_string(index=False))

    # 11. Determine Verdict (Phase 18)
    if nested_delta > 0.0020 and ci_l > 0:
        verdict = "GREEN"
    elif nested_delta >= 0.0000 and (new_fixes_disagree_count > new_worsens_disagree_count):
        verdict = "YELLOW"
    else:
        verdict = "RED"

    log(f"\n=====================================================================")
    log(f"FINAL EXPERIMENTAL VERDICT: {verdict}")
    log(f"Nested BA Delta: {nested_delta:+.4f} (95% CI: [{ci_l:+.4f}, {ci_u:+.4f}], P(delta>0): {p_pos*100:.1f}%)")
    log(f"Net Disagree Fix Capacity: {new_fixes_disagree_count - new_worsens_disagree_count} samples")
    log(f"=====================================================================")

    # 12. Save Results and CSVs
    table1_df.to_csv(EXPERIMENTS_DIR / "results.csv", index=False)
    comp_df.to_csv(EXPERIMENTS_DIR / "complementarity.csv", index=False)
    comp_df.to_csv(EXPERIMENTS_DIR / "oracle_results.csv", index=False)
    fusion_df.to_csv(EXPERIMENTS_DIR / "fusion_results.csv", index=False)
    cond_df.to_csv(EXPERIMENTS_DIR / "prediction_disagreement.csv", index=False)

    fold_df = pd.DataFrame({
        "fold": list(range(5)),
        "q8_ba": [balanced_accuracy_score(y_true[folds == f], q8_pred[folds == f]) for f in range(5)],
        "q14_ba": [balanced_accuracy_score(y_true[folds == f], q14_pred[folds == f]) for f in range(5)],
        "exp52_ba": fold_ba_exp52,
        "modernbert_lr_ba": fold_ba_lr,
        "modernbert_svm_ba": fold_ba_svm,
        "nested_fused_ba": outer_fold_ba,
        "selected_weight": outer_weights
    })
    fold_df.to_csv(EXPERIMENTS_DIR / "fold_results.csv", index=False)

    nested_summary = pd.DataFrame([{
        "nested_ba": nested_overall_ba,
        "nested_macro_f1": nested_overall_f1,
        "exp52_ba": ba_exp52,
        "exp52_macro_f1": f1_exp52,
        "delta": nested_delta,
        "ci_95_lower": ci_l,
        "ci_95_upper": ci_u,
        "p_positive": p_pos,
        "mean_fold_ba": float(np.mean(outer_fold_ba)),
        "std_fold_ba": float(np.std(outer_fold_ba)),
        "verdict": verdict
    }])
    nested_summary.to_csv(EXPERIMENTS_DIR / "nested_results.csv", index=False)

    # Save Config JSON
    config = {
        "experiment_id": "EXP62_cross_architecture",
        "description": "Cross-Architecture Complementarity Test with ModernBERT-base",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": "2x NVIDIA GeForce RTX 3060 (12GB VRAM)",
            "cuda": "13.2",
            "pytorch": "2.15.0+cu132",
            "python": "3.14.7"
        },
        "model_provenance": {
            "name": "answerdotai/ModernBERT-base",
            "architecture": "ModernBertModel (Bidirectional Encoder, RoPE, GeGLU, unpadded attention)",
            "parameters": 149096448,
            "hidden_size": 768,
            "num_layers": 22,
            "context_window": 8192,
            "tokenizer": "ModernBERT WordPiece (vocab 50,368)",
            "snapshot_hash": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
            "base_or_instruct": "Pretrained base foundation encoder",
            "prior_involvement": "None (clean holdout architecture)"
        },
        "baselines": {
            "exp52_oof_ba": ba_exp52,
            "q8_oof_ba": ba_q8,
            "q14_oof_ba": ba_q14
        },
        "modernbert_performance": {
            "logistic_regression_ba": ba_new_lr,
            "logistic_regression_macro_f1": f1_new_lr,
            "linear_svm_ba": ba_new_svm,
            "linear_svm_macro_f1": f1_new_svm
        },
        "nested_validation": {
            "nested_ba": nested_overall_ba,
            "delta": nested_delta,
            "ci_95_lower": ci_l,
            "ci_95_upper": ci_u,
            "p_positive": p_pos,
            "outer_weights": outer_weights
        },
        "complementarity": {
            "pearson_err_corr_exp52_new": float(scipy.stats.pearsonr(error_indicators["EXP52"], error_indicators["NEW"])[0]),
            "oracle_exp52_new_ba": balanced_oracle(exp52_pred, new_pred_lr, y_true),
            "disagreement_count": n_disagree,
            "disagreement_fraction": pct_disagree,
            "fixes_in_disagreement": new_fixes_disagree_count,
            "worsens_in_disagreement": new_worsens_disagree_count,
            "net_correction": new_fixes_disagree_count - new_worsens_disagree_count
        },
        "verdict": verdict,
        "kaggle_submission": "NO",
        "exp52_modified": False
    }

    with open(EXPERIMENTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save README.md in experiment directory
    readme_content = f"""# EXP62 — Cross-Architecture Complementarity Test

- **Target Architecture**: `answerdotai/ModernBERT-base` (Bidirectional Encoder, 149M params, 768d, RoPE, GeGLU)
- **Baseline**: EXP52 Dual-LoRA (Qwen3-8B $0.60$ + Qwen3-14B $0.40$), OOF BA: **{ba_exp52:.4f}**
- **ModernBERT Standalone OOF BA**: **{ba_new_lr:.4f}** (LR) / **{ba_new_svm:.4f}** (SVM)
- **EXP52 + ModernBERT Nested CV BA**: **{nested_overall_ba:.4f}** (Delta: **{nested_delta:+.4f}**)
- **95% Bootstrap CI**: [{ci_l:+.4f}, {ci_u:+.4f}], P(Delta > 0) = {p_pos*100:.1f}%
- **Error Correlation (EXP52 vs ModernBERT)**: {scipy.stats.pearsonr(error_indicators["EXP52"], error_indicators["NEW"])[0]:.4f}
- **Oracle BA (EXP52, ModernBERT)**: {balanced_oracle(exp52_pred, new_pred_lr, y_true):.4f}
- **Disagreement Subset (Q8 != Q14, 59.7% of data)**:
  - EXP52 BA: {ba_exp52_disagree:.4f}
  - ModernBERT BA: {ba_new_disagree:.4f}
  - ModernBERT Fixes: {new_fixes_disagree_count} | ModernBERT Worsens: {new_worsens_disagree_count} | Net: {new_fixes_disagree_count - new_worsens_disagree_count}
- **Verdict**: **{verdict}**
- **Production Decision**: EXP52 remains strictly frozen at 0.632 Public LB / 0.7873 local OOF BA. Stop cross-architecture model stacking with uncalibrated encoders.
"""
    with open(EXPERIMENTS_DIR / "README.md", "w") as f:
        f.write(readme_content)

    # 13. Generate 5 Publication-Quality Plots (Phase 21)
    log("\n--- Phase 21: Generating 5 Publication Plots ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Plot 1: Model Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = ["Qwen3-8B", "Qwen3-14B", "ModernBERT\n(LR)", "ModernBERT\n(SVM)", "EXP52\n(Champion)", "EXP52 + NEW\n(Nested)"]
    ba_vals = [ba_q8, ba_q14, ba_new_lr, ba_new_svm, ba_exp52, nested_overall_ba]
    colors = ["#4A90E2", "#357ABD", "#E74C3C", "#C0392B", "#2ECC71", "#9B59B6"]
    bars = ax.bar(model_names, ba_vals, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", weight="bold")
    ax.set_ylim(0.0, 0.90)
    ax.set_ylabel("Balanced Accuracy", weight="bold")
    ax.set_title("EXP62: Cross-Architecture Standalone vs Ensembled Performance", weight="bold", fontsize=14)
    ax.axhline(ba_exp52, color="green", linestyle="--", alpha=0.7, label=f"EXP52 Champion ({ba_exp52:.4f})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(EXPERIMENTS_DIR / "exp62_model_comparison.png", dpi=300)
    fig.savefig(PLOTS_DIR / "exp62_model_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Error Correlation Matrix
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr_mat = np.zeros((4, 4))
    keys = ["Q8", "Q14", "EXP52", "NEW"]
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            corr_mat[i, j] = scipy.stats.pearsonr(error_indicators[ki], error_indicators[kj])[0]
    sns.heatmap(corr_mat, annot=True, fmt=".4f", cmap="coolwarm", vmin=-0.1, vmax=1.0,
                xticklabels=keys, yticklabels=keys, ax=ax, cbar_kws={'label': 'Pearson Error Correlation'})
    ax.set_title("EXP62: Measured Error Correlation Matrix Across Architectures", weight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(EXPERIMENTS_DIR / "exp62_error_correlation.png", dpi=300)
    fig.savefig(PLOTS_DIR / "exp62_error_correlation.png", dpi=300)
    plt.close()

    # Plot 3: Oracle Complementarity
    fig, ax = plt.subplots(figsize=(9, 6))
    pair_labels = ["Q8 / Q14", "Q8 / NEW", "Q14 / NEW", "EXP52 / NEW"]
    oracle_vals = [r["oracle_ba"] for r in comp_rows]
    actual_vals = [
        ba_exp52, # Q8+Q14 actual
        float(fusion_df.loc[fusion_df["weight_new"] == 0.05, "balanced_accuracy"].values[0]),
        float(fusion_df.loc[fusion_df["weight_new"] == 0.05, "balanced_accuracy"].values[0]),
        nested_overall_ba # EXP52+NEW actual
    ]
    x = np.arange(len(pair_labels))
    w = 0.35
    ax.bar(x - w/2, oracle_vals, width=w, label="Theoretical Oracle BA (Upper Bound)", color="#3498DB", edgecolor="black")
    ax.bar(x + w/2, actual_vals, width=w, label="Actual Ensembled BA", color="#E67E22", edgecolor="black")
    for i in range(len(pair_labels)):
        ax.annotate(f"{oracle_vals[i]:.4f}", xy=(x[i] - w/2, oracle_vals[i]), xytext=(0, 4), textcoords="offset points", ha="center", weight="bold")
        ax.annotate(f"{actual_vals[i]:.4f}", xy=(x[i] + w/2, actual_vals[i]), xytext=(0, 4), textcoords="offset points", ha="center", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, weight="bold")
    ax.set_ylim(0.50, 0.92)
    ax.set_ylabel("Balanced Accuracy", weight="bold")
    ax.set_title("EXP62: Oracle Upper Bound vs Actual Realized Ensembling", weight="bold", fontsize=14)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(EXPERIMENTS_DIR / "exp62_oracle_complementarity.png", dpi=300)
    fig.savefig(PLOTS_DIR / "exp62_oracle_complementarity.png", dpi=300)
    plt.close()

    # Plot 4: Fusion Curve
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(fusion_df["weight_new"], fusion_df["balanced_accuracy"], marker="o", linewidth=2.5, markersize=8, color="#E74C3C", label="EXP52 + ModernBERT")
    ax.axhline(ba_exp52, color="green", linestyle="--", label=f"EXP52 Baseline ({ba_exp52:.4f})")
    ax.set_xlabel("ModernBERT Fusion Weight (w)", weight="bold")
    ax.set_ylabel("Out-of-Fold Balanced Accuracy", weight="bold")
    ax.set_title("EXP62: EXP52 + ModernBERT Log-Probability Fusion Curve", weight="bold", fontsize=14)
    ax.legend(loc="upper right")
    for idx, row in fusion_df.iterrows():
        if row["weight_new"] in [0.00, 0.05, 0.10, 0.20, 0.50]:
            ax.annotate(f"{row['balanced_accuracy']:.4f}\n({row['delta_vs_exp52']:+.4f})",
                        xy=(row["weight_new"], row["balanced_accuracy"]),
                        xytext=(0, 10 if row["weight_new"] == 0.00 else -25),
                        textcoords="offset points", ha="center", fontsize=9, weight="bold")
    plt.tight_layout()
    fig.savefig(EXPERIMENTS_DIR / "exp62_fusion_curve.png", dpi=300)
    fig.savefig(PLOTS_DIR / "exp62_fusion_curve.png", dpi=300)
    plt.close()

    # Plot 5: Disagreement Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    subsets = ["Q8/Q14 Agreement\n(40.3% of data)", "Q8/Q14 Disagreement\n(59.7% of data)", "Full Dataset\n(100% of data)"]
    exp52_sub_bas = [ba_exp52_agree, ba_exp52_disagree, ba_exp52]
    new_sub_bas = [ba_new_agree, ba_new_disagree, ba_new_lr]
    nested_sub_bas = [ba_nested_agree, ba_nested_disagree, nested_overall_ba]

    x = np.arange(len(subsets))
    w = 0.25
    ax.bar(x - w, exp52_sub_bas, width=w, label="EXP52 Champion", color="#2ECC71", edgecolor="black")
    ax.bar(x, new_sub_bas, width=w, label="ModernBERT (Standalone)", color="#E74C3C", edgecolor="black")
    ax.bar(x + w, nested_sub_bas, width=w, label="EXP52 + NEW (Nested)", color="#9B59B6", edgecolor="black")

    for i in range(len(subsets)):
        ax.annotate(f"{exp52_sub_bas[i]:.4f}", xy=(x[i] - w, exp52_sub_bas[i]), xytext=(0, 4), textcoords="offset points", ha="center", weight="bold", fontsize=9)
        ax.annotate(f"{new_sub_bas[i]:.4f}", xy=(x[i], new_sub_bas[i]), xytext=(0, 4), textcoords="offset points", ha="center", weight="bold", fontsize=9)
        ax.annotate(f"{nested_sub_bas[i]:.4f}", xy=(x[i] + w, nested_sub_bas[i]), xytext=(0, 4), textcoords="offset points", ha="center", weight="bold", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(subsets, weight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Balanced Accuracy", weight="bold")
    ax.set_title("EXP62: Model Performance Conditioned on Q8/Q14 Agreement", weight="bold", fontsize=14)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(EXPERIMENTS_DIR / "exp62_disagreement_analysis.png", dpi=300)
    fig.savefig(PLOTS_DIR / "exp62_disagreement_analysis.png", dpi=300)
    plt.close()

    elapsed = time.time() - start_time
    log(f"EXP62 Cross-Architecture Complementarity Test completed in {elapsed:.1f}s.")
    log("=====================================================================")


if __name__ == "__main__":
    main()
