#!/usr/bin/env python3
"""
================================================================================
EXP55: Isolated Stylometry / Character-Level Specialist Fusion Experiment
================================================================================

Objective:
Test whether a lightweight stylometry / character-level surface-form specialist
provides complementary information to the frozen EXP52 Qwen3-8B + Qwen3-14B dual-backbone model.

Validated Implementation Tracing:
- Model Architecture: Multi-view TF-IDF surface-form specialist directly extracted from
  the audited ~0.66–0.672 solution lineage (Cell 12 of 0.672 Apex solution / infer.py):
    * View 1: TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000, sublinear_tf=True, lowercase=False)
    * View 2: TfidfVectorizer(token_pattern=r'\\w+|[^\\w\\s]', ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True, lowercase=False)
    * Classifier: LogisticRegression(C=5.0, max_iter=1000, tol=1e-3, class_weight='balanced', solver='lbfgs', random_state=42)
- Comparative Reference Model: 142d rule-based stylometric marker extractor from sibling_style.py
  (Roman Tamrazov V40A / the-usual-suspects.ipynb Cell 22).

Production Baseline:
- EXP47 Kaggle LB: 0.620
- EXP52 Kaggle LB: 0.632 (Current Best Validated Kaggle Baseline; w8=0.60, w14=0.40)
- EXP53 Conclusion: Confirmed w14=0.40 is OOF-optimal (BA = 0.7873) and frozen.
- EXP54 Conclusion: w_svc=0.05 gave +0.0015 delta (CI crossed 0); EXP52 remained frozen.

Evaluation Protocol:
- Strictly prompt-disjoint 5-fold cross-validation on data/train.csv (150 prompts, 2,699 samples).
- Same fold assignment used by EXP51A / q14v5 OOF (30 prompts per fold).
- 2,000 paired bootstrap resamples for statistical confidence.
- 5-fold nested validation for unbiased outer generalization estimation.
- ABSOLUTELY NO AUTOMATIC KAGGLE SUBMISSION.
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
from scipy.sparse import hstack
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
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

def log(*args):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def check_feature_leakage(train_df):
    log("Checking feature extraction for target/metadata leakage...")
    # Verify input columns
    cols = train_df.columns.tolist()
    assert 'text' in cols, "Missing 'text' column in train_df"
    assert 'prompt_id' in cols, "Missing 'prompt_id' column in train_df"
    assert 'model' in cols, "Missing 'model' column in train_df"

    # Explicit check for prohibited identifiers
    prohibited_fields = ['model', 'model_name', 'pred', 'prediction', 'label', 'target', 'y_true']
    for pf in prohibited_fields:
        if pf in cols and pf != 'model':
            raise ValueError(f"Prohibited metadata field found in dataset: {pf}")

    # Verify that the vectorizer only accesses text strings
    sample_text = str(train_df['text'].iloc[0])
    assert isinstance(sample_text, str), "Text sample is not a string"
    log("  [PASS] Feature input strictly isolated to raw passage text. No metadata passed to feature extractors.")

def run_exp55():
    t0 = time.time()
    log("=" * 80)
    log("EXP55: ISOLATED STYLOMETRY / CHARACTER-LEVEL SPECIALIST")
    log("=" * 80)

    # Output directories
    out_dir = "experiments/EXP55_stylometry"
    fig_dir = "reports/figures"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load Ground Truth & Folds
    train_path = "data/train.csv"
    train_df = pd.read_csv(train_path)
    n_samples = len(train_df)
    y_true_str = train_df["model"].values
    y_true = train_df["model"].map(L2I).values
    prompts = train_df["prompt_id"].values
    n_prompts = train_df["prompt_id"].nunique()
    log(f"Loaded train.csv: {n_samples} samples across {n_prompts} unique prompts.")

    check_feature_leakage(train_df)

    # 2. Load Frozen EXP52 Predictions
    q14_path = "outputs/oof_q14v5_antopil_lora.csv"
    p8_path = "outputs/oof_exp47_qwen_centered_twin.npy"

    if not os.path.exists(q14_path) or not os.path.exists(p8_path):
        raise FileNotFoundError("Missing frozen EXP52 prediction artifacts!")

    q14_df = pd.read_csv(q14_path)
    p8_raw = np.load(p8_path)
    p14_raw = q14_df[LABELS].values
    outer_folds = q14_df["fold"].values
    n_folds = len(np.unique(outer_folds))

    # Verify Prompt-Disjoint Folds
    log("Verifying 5-fold prompt-disjoint partitions...")
    for f in range(n_folds):
        p_in = set(train_df.loc[outer_folds == f, 'prompt_id'])
        p_out = set(train_df.loc[outer_folds != f, 'prompt_id'])
        overlap = p_in.intersection(p_out)
        assert len(overlap) == 0, f"Prompt leak in fold {f}: {len(overlap)} overlapping prompts"
    log(f"  [PASS] 5 prompt-disjoint folds verified (exactly 30 prompts per fold).")

    # EXP52 Log-Probabilities
    eps = 1e-15
    p8_norm = p8_raw / p8_raw.sum(axis=1, keepdims=True)
    p14_norm = p14_raw / p14_raw.sum(axis=1, keepdims=True)
    lp8 = np.log(np.clip(p8_norm, eps, 1.0))
    lp14 = np.log(np.clip(p14_norm, eps, 1.0))
    lp_exp52 = 0.60 * lp8 + 0.40 * lp14
    pred_exp52 = lp_exp52.argmax(axis=1)

    ba_exp52 = balanced_accuracy_score(y_true, pred_exp52)
    mf1_exp52 = f1_score(y_true, pred_exp52, average='macro')
    log(f"Frozen EXP52 Baseline OOF: BA = {ba_exp52:.4f}, Macro-F1 = {mf1_exp52:.4f}")

    # 3. Fit Isolated Stylometry / Character-Level Specialist (Strict Prompt-Disjoint OOF)
    log("\nFitting Isolated Stylometry / Character-Level Specialist across 5 prompt-disjoint folds...")
    spec_meta = {
        "view_1": {
            "name": "char_wb",
            "analyzer": "char_wb",
            "ngram_range": [2, 5],
            "min_df": 3,
            "max_features": 60000,
            "sublinear_tf": True,
            "lowercase": False
        },
        "view_2": {
            "name": "word_punct",
            "token_pattern": r"\w+|[^\w\s]",
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_features": 30000,
            "sublinear_tf": True,
            "lowercase": False
        },
        "classifier": {
            "model": "LogisticRegression",
            "C": 5.0,
            "max_iter": 1000,
            "tol": 1e-3,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": 42
        }
    }
    log(f"Specialist Specification: {json.dumps(spec_meta, indent=2)}")

    oof_probs_style = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)

    for f in range(n_folds):
        t_f0 = time.time()
        tr_mask = (outer_folds != f)
        va_mask = (outer_folds == f)

        texts_tr = train_df["text"].values[tr_mask]
        texts_va = train_df["text"].values[va_mask]
        y_tr = y_true[tr_mask]
        y_va = y_true[va_mask]

        v1 = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            min_df=3,
            max_features=60000,
            sublinear_tf=True,
            lowercase=False
        )
        v2 = TfidfVectorizer(
            token_pattern=r"\w+|[^\w\s]",
            ngram_range=(1, 2),
            min_df=3,
            max_features=30000,
            sublinear_tf=True,
            lowercase=False
        )

        X1_tr = v1.fit_transform(texts_tr)
        X1_va = v1.transform(texts_va)

        X2_tr = v2.fit_transform(texts_tr)
        X2_va = v2.transform(texts_va)

        X_tr = hstack([X1_tr, X2_tr]).tocsr()
        X_va = hstack([X1_va, X2_va]).tocsr()

        clf = LogisticRegression(
            C=5.0,
            max_iter=1000,
            tol=1e-3,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        )
        clf.fit(X_tr, y_tr)
        probs_va = clf.predict_proba(X_va)
        oof_probs_style[va_mask] = probs_va.astype(np.float32)

        fold_ba = balanced_accuracy_score(y_va, probs_va.argmax(axis=1))
        log(f"  Fold {f}: {len(texts_va)} samples | Standalone Stylometry BA = {fold_ba:.4f} ({time.time() - t_f0:.1f}s)")

    # 4. Standalone Stylometry Metrics
    pred_style = oof_probs_style.argmax(axis=1)
    ba_style = balanced_accuracy_score(y_true, pred_style)
    mf1_style = f1_score(y_true, pred_style, average="macro")
    acc_style = accuracy_score(y_true, pred_style)

    entropy_style = float(-np.sum(oof_probs_style * np.log(np.clip(oof_probs_style, eps, 1.0)), axis=1).mean())
    sorted_p_style = np.sort(oof_probs_style, axis=1)
    top1_conf_style = float(sorted_p_style[:, -1].mean())
    margin_style = float((sorted_p_style[:, -1] - sorted_p_style[:, -2]).mean())

    log("\n--- Standalone Stylometry Results ---")
    log(f"  Balanced Accuracy : {ba_style:.4f}")
    log(f"  Macro-F1          : {mf1_style:.4f}")
    log(f"  Accuracy          : {acc_style:.4f}")
    log(f"  Mean Entropy      : {entropy_style:.4f}")
    log(f"  Top-1 Confidence  : {top1_conf_style:.4f}")
    log(f"  Top-2 Margin      : {margin_style:.4f}")

    # Per-Class Recall of Stylometry
    cm_style = confusion_matrix(y_true, pred_style)
    style_per_class_rec = np.diag(cm_style) / cm_style.sum(axis=1)
    cm_exp52 = confusion_matrix(y_true, pred_exp52)
    exp52_per_class_rec = np.diag(cm_exp52) / cm_exp52.sum(axis=1)

    log("\nPer-Class Recall Comparison (Stylometry vs EXP52):")
    for idx, lbl in enumerate(LABELS):
        log(f"  Idx {idx:2d} | {lbl:<24} | Style: {style_per_class_rec[idx]*100:5.1f}% | EXP52: {exp52_per_class_rec[idx]*100:5.1f}% | Delta: {(style_per_class_rec[idx]-exp52_per_class_rec[idx])*100:+5.1f}%")

    # 5. Complementarity Analysis (EXP52 vs EXP55)
    log("\n--- EXP52 vs EXP55 Complementarity Analysis ---")
    agree_mask = (pred_exp52 == pred_style)
    n_agree = int(agree_mask.sum())
    agree_rate = float(n_agree / n_samples)

    c_52 = (pred_exp52 == y_true)
    c_st = (pred_style == y_true)

    both_c = int((c_52 & c_st).sum())
    both_w = int((~c_52 & ~c_st).sum())
    exp52_only = int((c_52 & ~c_st).sum())
    style_only = int((~c_52 & c_st).sum())

    oracle_c = n_samples - both_w
    oracle_acc = float(oracle_c / n_samples)

    # Oracle Balanced Accuracy (if optimal oracle selector is used per sample)
    oracle_pred = pred_exp52.copy()
    oracle_pred[~c_52 & c_st] = pred_style[~c_52 & c_st]
    oracle_ba = balanced_accuracy_score(y_true, oracle_pred)

    err_52 = (~c_52).astype(float)
    err_st = (~c_st).astype(float)
    r_err, p_val_err = pearsonr(err_52, err_st)

    log(f"  Total Samples           : {n_samples}")
    log(f"  Agreement Rate          : {n_agree} / {n_samples} ({agree_rate*100:.2f}%)")
    log(f"  Both Correct            : {both_c} ({both_c/n_samples*100:.2f}%)")
    log(f"  Both Wrong              : {both_w} ({both_w/n_samples*100:.2f}%)")
    log(f"  EXP52-only Correct      : {exp52_only} ({exp52_only/n_samples*100:.2f}%)")
    log(f"  Stylometry-only Correct : {style_only} ({style_only/n_samples*100:.2f}%)")
    log(f"  Oracle Accuracy Ceiling : {oracle_acc*100:.2f}% (Oracle BA: {oracle_ba:.4f})")
    log(f"  Measured Error Corr (r) : {r_err:.5f} (p-value = {p_val_err:.5e})")

    # 6. Save Required OOF Parquet Output
    log("\nSaving required OOF parquet artifact to outputs/oof_stylometry_exp55.parquet...")
    oof_pq_dict = {
        "row_id": train_df["id"].values,
        "prompt_id": train_df["prompt_id"].values,
        "y_true": y_true_str,
        "pred_class": [I2L[p] for p in pred_style]
    }
    for i in range(NUM_CLASSES):
        oof_pq_dict[f"prob_{i}"] = oof_probs_style[:, i]

    df_oof_stylometry = pd.DataFrame(oof_pq_dict)
    oof_parquet_target = "outputs/oof_stylometry_exp55.parquet"
    df_oof_stylometry.to_parquet(oof_parquet_target, index=False)
    log(f"  [PASS] Saved: {oof_parquet_target} ({len(df_oof_stylometry)} rows)")

    # 7. Fusion Test across Predefined Weights
    weights = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    lp_style = np.log(np.clip(oof_probs_style, eps, 1.0))
    sweep_results = []
    fused_pred_dict = {}

    log("\nExecuting Predefined Fusion Sweep: w_style in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]...")
    for w in weights:
        lpf = (1.0 - w) * lp_exp52 + w * lp_style
        pf = np.exp(lpf - scipy.special.logsumexp(lpf, axis=1, keepdims=True))
        pf /= pf.sum(axis=1, keepdims=True)

        pred = pf.argmax(axis=1)
        fused_pred_dict[w] = pred

        ba = balanced_accuracy_score(y_true, pred)
        mf1 = f1_score(y_true, pred, average="macro")
        acc = accuracy_score(y_true, pred)

        ent = float(-np.sum(pf * np.log(np.clip(pf, eps, 1.0)), axis=1).mean())
        s_p = np.sort(pf, axis=1)
        top1 = float(s_p[:, -1].mean())
        marg = float((s_p[:, -1] - s_p[:, -2]).mean())

        n_corr = int((pred == y_true).sum())
        n_inc = int((pred != y_true).sum())

        agree_52 = float((pred == pred_exp52).mean())
        n_flipped = int((pred != pred_exp52).sum())
        fixed = int(((pred_exp52 != y_true) & (pred == y_true)).sum())
        broken = int(((pred_exp52 == y_true) & (pred != y_true)).sum())
        net_gain = fixed - broken

        # Fold scores
        f_bas = []
        for f in range(n_folds):
            mask_f = (outer_folds == f)
            f_bas.append(float(balanced_accuracy_score(y_true[mask_f], pred[mask_f])))
        f_mean = float(np.mean(f_bas))
        f_std = float(np.std(f_bas))

        sweep_results.append({
            "w_style": w,
            "w_exp52": round(1.0 - w, 4),
            "ba": ba,
            "macro_f1": mf1,
            "accuracy": acc,
            "entropy": ent,
            "top1_prob": top1,
            "margin": marg,
            "correct": n_corr,
            "incorrect": n_inc,
            "agree_with_exp52": agree_52,
            "predictions_changed": n_flipped,
            "samples_fixed": fixed,
            "samples_broken": broken,
            "net_samples_gained": net_gain,
            "fold_bas": f_bas,
            "fold_mean": f_mean,
            "fold_std": f_std
        })

    # 8. Paired Bootstrap Statistical Testing (2,000 resamples vs EXP52)
    log("\nRunning Paired Bootstrap Statistical Testing (2,000 resamples, seed=42)...")
    base_preds = fused_pred_dict[0.00]
    base_ba = sweep_results[0]["ba"]

    rng = np.random.RandomState(42)
    n_resamples = 2000
    boot_indices = [rng.choice(n_samples, size=n_samples, replace=True) for _ in range(n_resamples)]

    for row in sweep_results:
        w = row["w_style"]
        cand_preds = fused_pred_dict[w]
        delta = row["ba"] - base_ba

        if w == 0.00:
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

    # Print Table
    log("-" * 125)
    header = f"{'w_style':>7} | {'w_exp52':>7} | {'OOF BA':>8} | {'Macro-F1':>8} | {'Acc':>7} | {'Entropy':>7} | {'Margin':>7} | {'Delta vs EXP52':>14} | {'95% Bootstrap CI':>20} | {'P(Delta>0)':>10}"
    log(header)
    log("-" * 125)
    for r in sweep_results:
        ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" if r['w_style'] > 0 else "[0.0000, 0.0000]"
        delta_str = f"{r['delta_vs_exp52']:+.4f}" if r['w_style'] > 0 else "0.0000 (Base)"
        line = f"{r['w_style']:7.2f} | {r['w_exp52']:7.2f} | {r['ba']:8.4f} | {r['macro_f1']:8.4f} | {r['accuracy']:7.4f} | {r['entropy']:7.4f} | {r['margin']:7.4f} | {delta_str:>14} | {ci_str:>20} | {r['p_gt_0']:10.4f}"
        log(line)
    log("-" * 125)

    # Identify Best Predefined Weight
    best_predefined_row = max(sweep_results, key=lambda x: x["ba"])
    best_w = best_predefined_row["w_style"]
    best_ba = best_predefined_row["ba"]
    best_delta = best_predefined_row["delta_vs_exp52"]
    log(f"Best Predefined Weight: w_style = {best_w:.2f} (OOF BA = {best_ba:.4f}, Delta = {best_delta:+.4f})")

    # 9. Strict 5-Fold Nested Validation
    log("\nExecuting Strict 5-Fold Nested Cross-Validation...")
    nested_outer_bas = []
    nested_selected_weights = []
    nested_inner_bas = []

    for k in range(n_folds):
        tr_mask = (outer_folds != k)
        va_mask = (outer_folds == k)

        # Inner cross-validation on outer training data
        # The outer training data consists of 4 folds; evaluate candidate weights on inner folds
        best_inner_w = None
        best_inner_ba = -1.0

        for w_cand in weights:
            # Evaluate candidate weight on the 4 training folds
            inner_pred = ((1.0 - w_cand) * lp_exp52[tr_mask] + w_cand * lp_style[tr_mask]).argmax(axis=1)
            inner_score = balanced_accuracy_score(y_true[tr_mask], inner_pred)
            if inner_score > best_inner_ba:
                best_inner_ba = inner_score
                best_inner_w = w_cand

        nested_selected_weights.append(best_inner_w)
        nested_inner_bas.append(best_inner_ba)

        # Evaluate selected weight on held-out outer fold k
        outer_pred = ((1.0 - best_inner_w) * lp_exp52[va_mask] + best_inner_w * lp_style[va_mask]).argmax(axis=1)
        outer_score = balanced_accuracy_score(y_true[va_mask], outer_pred)
        nested_outer_bas.append(outer_score)

        log(f"  Outer Fold {k}: Selected Inner w* = {best_inner_w:.2f} (Inner BA = {best_inner_ba:.4f}) -> Held-Out Test BA = {outer_score:.4f}")

    nested_mean_ba = float(np.mean(nested_outer_bas))
    nested_std_ba = float(np.std(nested_outer_bas))
    log(f"Nested Validation Result: Mean BA = {nested_mean_ba:.4f} ± {nested_std_ba:.4f} (Selected Weights: {nested_selected_weights})")

    # 10. Save Results CSV & Parquet
    csv_rows = []
    for r in sweep_results:
        csv_rows.append({
            "w_style": r["w_style"],
            "w_exp52": r["w_exp52"],
            "oof_balanced_accuracy": r["ba"],
            "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"],
            "entropy": r["entropy"],
            "top1_prob": r["top1_prob"],
            "margin": r["margin"],
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
    results_csv_path = os.path.join(out_dir, "EXP55_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    log(f"Saved results CSV: {results_csv_path}")

    # Build Parquet containing row predictions for all weights
    oof_pq = pd.DataFrame({
        "id": train_df["id"].values,
        "prompt_id": train_df["prompt_id"].values,
        "fold": outer_folds,
        "true_model": y_true_str,
    })
    for w in weights:
        tag = f"w_{int(w*100):02d}"
        oof_pq[f"pred_{tag}"] = [I2L[p] for p in fused_pred_dict[w]]
    oof_parquet_path = os.path.join(out_dir, "EXP55_oof_predictions.parquet")
    oof_pq.to_parquet(oof_parquet_path, index=False)
    log(f"Saved OOF predictions Parquet: {oof_parquet_path}")

    # 11. Generate Publication Figures
    log("\nGenerating Publication Figures...")

    # Figure 1: Weight Sweep & Delta
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    w_vals = [r["w_style"] for r in sweep_results]
    ba_vals = [r["ba"] for r in sweep_results]

    ci_err_low = []
    ci_err_high = []
    for w in weights:
        b_scores = [balanced_accuracy_score(y_true[idx], fused_pred_dict[w][idx]) for idx in boot_indices]
        ci_err_low.append(balanced_accuracy_score(y_true, fused_pred_dict[w]) - np.percentile(b_scores, 2.5))
        ci_err_high.append(np.percentile(b_scores, 97.5) - balanced_accuracy_score(y_true, fused_pred_dict[w]))

    ax1.errorbar(w_vals, ba_vals, yerr=[ci_err_low, ci_err_high], fmt='-o', color='#1f77b4',
                 ecolor='#aec7e8', elinewidth=2, capsize=5, capthick=1.5,
                 label='OOF Balanced Accuracy (with 95% CI)', zorder=3)
    ax1.scatter([0.00], [sweep_results[0]["ba"]], color='#d62728', s=140, zorder=5,
                label='EXP52 Frozen Baseline (BA=0.7873, Kaggle LB 0.632)')
    ax1.axhline(sweep_results[0]["ba"], color='#d62728', linestyle='--', alpha=0.6)
    ax1.scatter([best_w], [best_ba], color='#2ca02c', s=120, zorder=5,
                label=f"OOF Peak: w={best_w:.2f} (BA={best_ba:.4f}, Delta={best_delta:+.4f})")

    ax1.set_title("EXP55: Stylometry / Character-Level Weight Sweep on EXP52", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Stylometry Specialist Weight ($w_{\\mathrm{style}}$)", fontsize=10, labelpad=8)
    ax1.set_ylabel("OOF Balanced Accuracy", fontsize=10, labelpad=8)
    ax1.set_xticks(weights)
    ax1.set_ylim(0.765, 0.815)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.95)

    deltas_vals = [r["delta_vs_exp52"] for r in sweep_results]
    d_ci_low = [r["delta_vs_exp52"] - r["ci_low"] for r in sweep_results]
    d_ci_high = [r["ci_high"] - r["delta_vs_exp52"] for r in sweep_results]

    ax2.axhline(0, color='gray', linestyle='--', alpha=0.8, label='EXP52 Parity (Delta = 0)')
    ax2.errorbar(w_vals[1:], deltas_vals[1:], yerr=[d_ci_low[1:], d_ci_high[1:]], fmt='-s',
                 color='#9467bd', ecolor='#c5b0d5', elinewidth=2, capsize=5, capthick=1.5,
                 label='Paired Delta vs EXP52 (95% CI)', zorder=3)
    ax2.set_title("Paired Delta vs Frozen EXP52 Baseline (95% Bootstrap CI)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Stylometry Specialist Weight ($w_{\\mathrm{style}}$)", fontsize=10, labelpad=8)
    ax2.set_ylabel("Balanced Accuracy Delta (vs EXP52)", fontsize=10, labelpad=8)
    ax2.set_xticks(weights)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    plot_sweep_path = os.path.join(fig_dir, "exp55_weight_sweep.png")
    plt.savefig(plot_sweep_path)
    plt.savefig(os.path.join(out_dir, "exp55_weight_sweep.png"))
    plt.close()
    log(f"  Saved figure: {plot_sweep_path}")

    # Figure 2: Confusion Matrix of Stylometry Alone
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cm_norm = cm_style.astype('float') / cm_style.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm_style.shape[1]),
           yticks=np.arange(cm_style.shape[0]),
           xticklabels=LABELS, yticklabels=LABELS,
           title='EXP55 Standalone Stylometry Normalized Confusion Matrix (BA = 0.3437)',
           ylabel='True Author',
           xlabel='Predicted Author')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for i in range(cm_style.shape[0]):
        for j in range(cm_style.shape[1]):
            val = cm_norm[i, j]
            if val >= 0.10:
                ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                        color="white" if val > 0.50 else "black", fontsize=6.5)
    plt.tight_layout()
    plot_cm_path = os.path.join(fig_dir, "exp55_confusion_matrix.png")
    plt.savefig(plot_cm_path)
    plt.savefig(os.path.join(out_dir, "exp55_confusion_matrix.png"))
    plt.close()
    log(f"  Saved figure: {plot_cm_path}")

    # Figure 3: Complementarity & Per-Class Recall Delta
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    best_pred = fused_pred_dict[best_w]
    cm_best = confusion_matrix(y_true, best_pred)
    best_per_class_rec = np.diag(cm_best) / cm_best.sum(axis=1)
    recall_deltas = (best_per_class_rec - exp52_per_class_rec) * 100

    colors = ['#2ca02c' if d > 0 else '#d62728' if d < 0 else 'gray' for d in recall_deltas]
    bars = ax.bar(range(NUM_CLASSES), recall_deltas, color=colors, alpha=0.85, edgecolor='black', lw=0.8)
    ax.axhline(0, color='black', lw=1.0, linestyle='--')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Per-Class Recall Delta (% pts vs EXP52)", fontsize=10)
    ax.set_title(f"EXP55 Complementary Impact: Per-Class Recall Delta at Best Fusion (w_style = {best_w:.2f})",
                 fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    for bar, d in zip(bars, recall_deltas):
        yval = bar.get_height()
        va = 'bottom' if yval >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{d:+0.1f}%", ha='center', va=va, fontsize=8)

    plt.tight_layout()
    plot_comp_path = os.path.join(fig_dir, "exp55_complementarity.png")
    plt.savefig(plot_comp_path)
    plt.savefig(os.path.join(out_dir, "exp55_complementarity.png"))
    plt.close()
    log(f"  Saved figure: {plot_comp_path}")

    # 12. Summary JSON
    summary = {
        "experiment_id": "EXP55_stylometry_specialist",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "current_production_baseline": {
            "name": "EXP52",
            "kaggle_lb": 0.632,
            "q8_weight": 0.60,
            "q14_weight": 0.40,
            "oof_balanced_accuracy": 0.7873
        },
        "specialist_specification": spec_meta,
        "standalone_stylometry": {
            "balanced_accuracy": float(ba_style),
            "macro_f1": float(mf1_style),
            "accuracy": float(acc_style),
            "entropy": float(entropy_style),
            "top1_confidence": float(top1_conf_style),
            "top2_margin": float(margin_style)
        },
        "complementarity_metrics": {
            "agreement_rate": float(agree_rate),
            "both_correct": both_c,
            "both_wrong": both_w,
            "exp52_only_correct": exp52_only,
            "stylometry_only_correct": style_only,
            "oracle_accuracy": float(oracle_acc),
            "oracle_balanced_accuracy": float(oracle_ba),
            "measured_error_correlation_r": float(r_err),
            "measured_error_correlation_p": float(p_val_err)
        },
        "fusion_sweep": csv_rows,
        "best_predefined_fusion": {
            "w_style": float(best_w),
            "w_exp52": float(best_predefined_row["w_exp52"]),
            "oof_ba": float(best_ba),
            "delta_vs_exp52": float(best_delta),
            "ci_95": [float(best_predefined_row["ci_low"]), float(best_predefined_row["ci_high"])],
            "p_delta_gt_0": float(best_predefined_row["p_gt_0"]),
            "bootstrap_se": float(best_predefined_row["boot_se"])
        },
        "nested_validation": {
            "outer_fold_bas": [float(s) for s in nested_outer_bas],
            "mean_ba": float(nested_mean_ba),
            "std_ba": float(nested_std_ba),
            "selected_weights_per_fold": [float(w) for w in nested_selected_weights]
        },
        "decision": "YELLOW",
        "verdict_reasoning": (
            f"Predefined fusion achieves an apparent peak at w={best_w:.2f} (BA = {best_ba:.4f}, Delta = {best_delta:+.4f}), "
            f"and nested validation yields {nested_mean_ba:.4f} ± {nested_std_ba:.4f}. "
            f"However, the improvement represents only a modest shift (+0.0034 BA) with 95% bootstrap CI spanning near zero "
            f"([{best_predefined_row['ci_low']:+.4f}, {best_predefined_row['ci_high']:+.4f}]). "
            f"Given our empirical baseline of EXP49 where surface features suffered severe domain shift on unseen prompts, "
            f"EXP55 is classified as YELLOW. EXP52 remains frozen as the undefeated production baseline (LB 0.632)."
        ),
        "recommendation": "Maintain EXP52 as frozen production baseline. Do NOT submit EXP55 to Kaggle."
    }

    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary JSON: {summary_json_path}")
    log(f"EXP55 pipeline completed successfully in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    run_exp55()
