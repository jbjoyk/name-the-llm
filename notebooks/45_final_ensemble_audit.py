"""
# 45 — EXP_45: Final Ensemble Selection, Error Diagnostics & Bootstrap Audit
EXPERIMENT_ID: EXP_45_final_ensemble_audit
DATE: 2026-11-06
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, classification_report

BASE_DIR = "/home/rj2826/name-the-llm"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.submission_pipeline import LABELS, LABEL_TO_IDX, IDX_TO_LABEL, recover_prompt_clusters, hungarian_assignment

DATA_PATH = os.path.join(BASE_DIR, "data/train.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def main():
    print("=" * 80)
    print("EXP_45: FINAL ENSEMBLE SELECTION & ERROR AUDIT")
    print("=" * 80)

    train_df = pd.read_csv(DATA_PATH)
    texts = train_df["text"].values
    prompts = train_df["prompt_id"].values
    y_raw = train_df["model"].values
    y_idx = np.array([LABEL_TO_IDX[m] for m in y_raw])
    n_samples = len(train_df)
    unique_prompts = np.unique(prompts)
    n_prompts = len(unique_prompts)

    p_36 = np.load(os.path.join(OUTPUTS_DIR, "oof_exp36_predictions.npy"))
    p_40 = np.load(os.path.join(OUTPUTS_DIR, "oof_exp40_gat_predictions.npy"))
    p_44 = np.load(os.path.join(OUTPUTS_DIR, "oof_exp44_hybrid_gat_predictions.npy"))

    rec_clusters = recover_prompt_clusters(texts)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(train_df, y_idx, groups=prompts))

    # Baselines
    h_36 = hungarian_assignment(p_36, rec_clusters)
    p_42c = 0.95 * p_36 + 0.05 * p_40
    h_42c = hungarian_assignment(p_42c, rec_clusters)

    # Final Candidate: EXP_44-C with pre-specified alpha=0.05
    p_final_05 = 0.95 * p_36 + 0.05 * p_44
    h_final_05 = hungarian_assignment(p_final_05, rec_clusters)

    # Final Candidate: EXP_44-C with nested alpha
    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.08, 0.10, 0.15, 0.20]
    p_final_nested = np.zeros_like(p_36)
    h_final_nested = np.zeros(n_samples, dtype=int)
    chosen_alphas = []

    for tr, val in splits:
        inner_gkf = GroupKFold(n_splits=4)
        best_a, best_in_ba = 0.05, -1.0
        for a in grid:
            in_bas = []
            for in_tr, in_val in inner_gkf.split(tr, y_idx[tr], groups=prompts[tr]):
                v_idx = tr[in_val]
                p_in = (1.0 - a) * p_36[v_idx] + a * p_44[v_idx]
                h_in = hungarian_assignment(p_in, rec_clusters[v_idx])
                in_bas.append(balanced_accuracy_score(y_idx[v_idx], h_in))
            m_ba = np.mean(in_bas)
            if m_ba > best_in_ba:
                best_in_ba = m_ba
                best_a = a
        chosen_alphas.append(best_a)
        p_final_nested[val] = (1.0 - best_a) * p_36[val] + best_a * p_44[val]
        h_final_nested[val] = hungarian_assignment(p_final_nested[val], rec_clusters[val])

    # Save final OOF prediction array
    np.save(os.path.join(OUTPUTS_DIR, "exp45_final_oof_predictions.npy"), p_final_nested)
    print(">> Saved outputs/exp45_final_oof_predictions.npy")

    # ======================================================================
    # Error Analysis: Final Candidate vs EXP_42C Control
    # ======================================================================
    corr_42c = (h_42c == y_idx)
    corr_final = (h_final_nested == y_idx)

    repaired = int(np.sum((~corr_42c) & corr_final))
    damaged = int(np.sum(corr_42c & (~corr_final)))
    net_improvement = repaired - damaged

    both_correct = int(np.sum(corr_42c & corr_final))
    both_wrong = int(np.sum((~corr_42c) & (~corr_final)))
    exp42c_only = int(np.sum(corr_42c & (~corr_final)))
    final_only = int(np.sum((~corr_42c) & corr_final))

    print("\n--- Sample Transitions vs EXP_42C ---")
    print(f"Repaired Samples (42C wrong -> Final correct): {repaired}")
    print(f"Damaged Samples (42C correct -> Final wrong):   {damaged}")
    print(f"Net Sample Improvement:                        {net_improvement:+d}")
    print(f"Both Correct: {both_correct} | Both Wrong: {both_wrong}")

    # Sibling Pair Breakdown
    sibling_pairs = [
        ("Gemini Sibling", "gemini-3.7-flash", "gemini-3.8-flash"),
        ("GLM Sibling", "glm-5.3", "glm-5.3-flash"),
        ("GPT Sibling", "gpt-5.6-sol", "gpt-5.6-terra"),
        ("GPT Sibling", "gpt-5.6-luna", "gpt-5.6-sol"),
        ("GPT Sibling", "gpt-5.6-luna", "gpt-5.6-terra"),
        ("Claude Sibling", "claude-opus-4-6", "claude-opus-4-8"),
        ("Claude Sibling", "claude-opus-4-8", "claude-opus-5"),
        ("Claude Sibling", "claude-opus-5", "claude-sonnet-5"),
        ("Qwen/Kimi", "qwen3.8-max", "kimi-k3")
    ]

    sibling_records = []
    sibling_repaired_total = 0
    sibling_damaged_total = 0

    for cat_name, m1, m2 in sibling_pairs:
        i1, i2 = LABEL_TO_IDX[m1], LABEL_TO_IDX[m2]
        # Errors where true was m1/m2 and prediction was m2/m1
        e_42c = int(np.sum(((y_idx == i1) & (h_42c == i2)) | ((y_idx == i2) & (h_42c == i1))))
        e_final = int(np.sum(((y_idx == i1) & (h_final_nested == i2)) | ((y_idx == i2) & (h_final_nested == i1))))
        
        c_fixed = int(np.sum(((y_idx == i1) & (h_42c == i2) & (h_final_nested == i1)) |
                             ((y_idx == i2) & (h_42c == i1) & (h_final_nested == i2))))
        c_damaged = int(np.sum(((y_idx == i1) & (h_42c == i1) & (h_final_nested == i2)) |
                               ((y_idx == i2) & (h_42c == i2) & (h_final_nested == i1))))
        
        sibling_repaired_total += c_fixed
        sibling_damaged_total += c_damaged
        
        sibling_records.append({
            "category": cat_name,
            "pair": f"{m1} <-> {m2}",
            "exp42c_errors": e_42c,
            "final_errors": e_final,
            "corrected": c_fixed,
            "damaged": c_damaged,
            "net": c_fixed - c_damaged
        })

    cross_family_repaired = repaired - sibling_repaired_total
    cross_family_damaged = damaged - sibling_damaged_total
    cross_family_net = cross_family_repaired - cross_family_damaged

    print(f"\nSibling Confusions: Repaired={sibling_repaired_total}, Damaged={sibling_damaged_total}, Net={sibling_repaired_total - sibling_damaged_total:+d}")
    print(f"Cross-Family:       Repaired={cross_family_repaired}, Damaged={cross_family_damaged}, Net={cross_family_net:+d}")

    # ======================================================================
    # Uncertainty Analysis (Quintiles vs EXP_42C)
    # ======================================================================
    conf_36 = np.max(p_36, axis=1)
    q_labels = ["Q5_lowest", "Q4_low", "Q3_mid", "Q2_high", "Q1_highest"]
    q_bins = pd.qcut(conf_36, q=5, labels=q_labels)

    uncertainty_records = []
    for q in q_labels:
        mask = np.array(q_bins == q)
        n_q = int(np.sum(mask))
        ba_42c_q = float(balanced_accuracy_score(y_idx[mask], h_42c[mask]))
        ba_fin_q = float(balanced_accuracy_score(y_idx[mask], h_final_nested[mask]))
        gain_ba = ba_fin_q - ba_42c_q
        pct_diff = float(np.mean(h_42c[mask] != h_final_nested[mask]))

        rec = {
            "bin": q,
            "n_samples": n_q,
            "exp42c_ba": ba_42c_q,
            "final_candidate_ba": ba_fin_q,
            "ba_gain": gain_ba,
            "pct_diff": pct_diff
        }
        uncertainty_records.append(rec)
        print(f"Uncertainty {q:10s} (N={n_q}): 42C BA={ba_42c_q:.4f} | Final BA={ba_fin_q:.4f} | Gain={gain_ba:+.4f} | % Changed={pct_diff:.2%}")

    # ======================================================================
    # Prompt-Level Cluster Bootstrap (2,000 resamples)
    # ======================================================================
    prompt_to_indices = {pid: np.where(prompts == pid)[0] for pid in unique_prompts}
    deltas_prompt = []
    imp_p, wrs_p, tie_p = 0, 0, 0

    for pid in unique_prompts:
        mask = (prompts == pid)
        acc_42 = np.mean(h_42c[mask] == y_idx[mask])
        acc_fin = np.mean(h_final_nested[mask] == y_idx[mask])
        d = float(acc_fin - acc_42)
        deltas_prompt.append(d)
        if d > 1e-6: imp_p += 1
        elif d < -1e-6: wrs_p += 1
        else: tie_p += 1

    deltas_prompt = np.array(deltas_prompt)
    mean_prompt_d = float(np.mean(deltas_prompt))
    median_prompt_d = float(np.median(deltas_prompt))
    std_prompt_d = float(np.std(deltas_prompt, ddof=1))
    cohen_d = float(mean_prompt_d / std_prompt_d) if std_prompt_d > 0 else 0.0

    n_boot = 2000
    boot_deltas = []
    rng_boot = np.random.RandomState(42)

    for _ in range(n_boot):
        b_pids = rng_boot.choice(unique_prompts, size=n_prompts, replace=True)
        b_idx = np.concatenate([prompt_to_indices[pid] for pid in b_pids])
        boot_deltas.append(balanced_accuracy_score(y_idx[b_idx], h_final_nested[b_idx]) - balanced_accuracy_score(y_idx[b_idx], h_42c[b_idx]))

    boot_deltas = np.array(boot_deltas)
    ci_90_lower = float(np.percentile(boot_deltas, 5.0))
    ci_90_upper = float(np.percentile(boot_deltas, 95.0))
    ci_95_lower = float(np.percentile(boot_deltas, 2.5))
    ci_95_upper = float(np.percentile(boot_deltas, 97.5))
    p_pos = float(np.mean(boot_deltas > 0))

    print(f"\nPrompt-Level Cluster Bootstrap (B=2000):")
    print(f"  Improved: {imp_p} ({imp_p/n_prompts:.1%}) | Worsened: {wrs_p} ({wrs_p/n_prompts:.1%}) | Tied: {tie_p} ({tie_p/n_prompts:.1%})")
    print(f"  Mean Delta: {mean_prompt_d:+.4f} | Median Delta: {median_prompt_d:+.4f} | Cohen's d: {cohen_d:.4f}")
    print(f"  90% CI: [{ci_90_lower:+.4f}, {ci_90_upper:+.4f}]")
    print(f"  95% CI: [{ci_95_lower:+.4f}, {ci_95_upper:+.4f}]")
    print(f"  P(Delta > 0): {p_pos:.4f}")

    # ======================================================================
    # Save Metadata Artifact
    # ======================================================================
    metadata = {
        "experiment_id": "EXP_45_final_ensemble_audit",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_baseline_exp42c": {
            "ba": float(balanced_accuracy_score(y_idx, h_42c)),
            "macro_f1": float(f1_score(y_idx, h_42c, average="macro"))
        },
        "final_candidate_nested": {
            "ba": float(balanced_accuracy_score(y_idx, h_final_nested)),
            "macro_f1": float(f1_score(y_idx, h_final_nested, average="macro")),
            "chosen_alphas": chosen_alphas,
            "mean_alpha": float(np.mean(chosen_alphas))
        },
        "final_candidate_prespecified_05": {
            "ba": float(balanced_accuracy_score(y_idx, h_final_05)),
            "macro_f1": float(f1_score(y_idx, h_final_05, average="macro"))
        },
        "sample_transitions": {
            "repaired": repaired,
            "damaged": damaged,
            "net": net_improvement,
            "sibling_repaired": sibling_repaired_total,
            "sibling_damaged": sibling_damaged_total,
            "cross_family_repaired": cross_family_repaired,
            "cross_family_damaged": cross_family_damaged
        },
        "sibling_pair_breakdown": sibling_records,
        "uncertainty_analysis": uncertainty_records,
        "bootstrap_results": {
            "n_boot": n_boot,
            "mean_delta": mean_prompt_d,
            "median_delta": median_prompt_d,
            "cohen_d": cohen_d,
            "ci_90": [ci_90_lower, ci_90_upper],
            "ci_95": [ci_95_lower, ci_95_upper],
            "p_positive": p_pos,
            "improved_prompts": imp_p,
            "worsened_prompts": wrs_p,
            "tied_prompts": tie_p
        }
    }

    with open(os.path.join(OUTPUTS_DIR, "exp45_final_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(">> Saved metadata to outputs/exp45_final_metadata.json")
    print("=" * 80)
    print("EXP_45 AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
