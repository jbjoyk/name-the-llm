# %% [markdown]
"""
# 34 — EXP_29: Official + Open Pool + 4-View SVC + Prompt Residual Features
EXPERIMENT_ID: 29_open_pool_residual
DATE: 2026-11-06
RESEARCH_QUESTION: Does combining the broad external vocabulary of Open Pool 4-View SVC
                   with prompt-conditioned residual representations and blind prompt recovery
                   substantially outperform standalone Open Pool SVC?
HYPOTHESIS:
1. Open Pool 4-View SVC addresses sample sparsity, providing robust n-gram and casing coverage across the 18 models.
2. Prompt-conditioned residuals eliminate prompt-topic variance, isolating author stylistic deviation.
3. Blind prompt recovery via size-capped average linkage clustering reconstructs prompt groupings (ARI ~0.88),
   allowing Hungarian bipartite matching to enforce injective model assignment per prompt.
4. Fusing Open Pool SVC with prompt residuals under nested leak-free log-loss optimization will produce
   a breakthrough in prompt-disjoint balanced accuracy.
DATA_VERSION: Official v1 (2,699) + Clean Open Pool (36,000)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id) (Strict Prompt-Disjoint Nested CV)
OUTPUTS: outputs/oof_open_pool_residual.npy, reports/open_pool_residual_results.md
"""

# %% — Imports & Setup
import os, sys, time, json, datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linear_sum_assignment
from scipy.special import softmax, logsumexp
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, log_loss

BASE_DIR = "/home/rj2826/name-the-llm"
DATA_PATH = os.path.join(BASE_DIR, "data/train.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# %% — Load Data and Precomputed OOF Probabilities
print("Loading competition dataset and candidate OOF streams...")
train = pd.read_csv(DATA_PATH)
texts = train["text"].fillna("").astype(str).values
y_true = train["model"].values
groups = train["prompt_id"].values
classes = sorted(np.unique(y_true))
n_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
y_true_idx = np.array([class_to_idx[c] for c in y_true])

oof_svc_off = np.load(os.path.join(OUTPUTS_DIR, "oof_reference_svc.npy"))        # EXP_27: 0.3749
oof_svc_pool = np.load(os.path.join(OUTPUTS_DIR, "oof_open_pool_svc.npy"))       # EXP_28
oof_residuals = np.load(os.path.join(OUTPUTS_DIR, "oof_prompt_residuals.npy"))   # EXP_22: 0.4453
oof_sibling = np.load(os.path.join(OUTPUTS_DIR, "oof_sibling_specialists.npy")) # EXP_24: 0.4378

print("Candidate Streams Loaded:")
print(f"  EXP_27 Official 4-View SVC:     BalAcc = {balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_off, axis=1)):.4f}")
print(f"  EXP_28 Open Pool 4-View SVC:    BalAcc = {balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_pool, axis=1)):.4f}")
print(f"  EXP_22 Prompt Residual Model:   BalAcc = {balanced_accuracy_score(y_true_idx, np.argmax(oof_residuals, axis=1)):.4f}")
print(f"  EXP_24 Sibling Specialists:     BalAcc = {balanced_accuracy_score(y_true_idx, np.argmax(oof_sibling, axis=1)):.4f}")

# %% — Prompt Recovery & Hungarian Assignment Helper
def recover_clusters_and_assign(val_texts, probs, max_cluster_size=18, threshold=0.15):
    """Recovers prompt clusters via content-word TF-IDF and applies Hungarian assignment."""
    vec = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.08, sublinear_tf=True)
    X_content = vec.fit_transform(val_texts)
    sim = (X_content @ X_content.T).toarray()
    dist = np.clip(1.0 - sim, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    
    condensed_dist = squareform(dist, checks=False)
    Z = linkage(condensed_dist, method='average')
    cluster_labels = fcluster(Z, t=1.0 - threshold, criterion='distance')
    
    unique_c, counts = np.unique(cluster_labels, return_counts=True)
    final_labels = cluster_labels.copy()
    next_id = max(unique_c) + 1
    for c, cnt in zip(unique_c, counts):
        if cnt > max_cluster_size:
            members = np.where(cluster_labels == c)[0]
            sub_Z = linkage(squareform(dist[np.ix_(members, members)], checks=False), method='average')
            sub_labels = fcluster(sub_Z, t=max_cluster_size, criterion='maxclust')
            for sl in np.unique(sub_labels):
                sub_idx = members[sub_labels == sl]
                final_labels[sub_idx] = next_id
                next_id += 1
                
    out_preds = np.zeros(len(probs), dtype=int)
    for c in np.unique(final_labels):
        idx = np.where(final_labels == c)[0]
        if len(idx) <= 1:
            out_preds[idx] = np.argmax(probs[idx], axis=1)
        else:
            cost_matrix = -probs[idx]
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            out_preds[idx[row_ind]] = col_ind
    return out_preds, final_labels

# %% — 5-Fold Nested Leak-Free Optimization & Fusion
kf = GroupKFold(n_splits=5)
streams = [oof_svc_pool, oof_residuals, oof_sibling]
M = len(streams)

nested_oof_probs = np.zeros((len(train), n_classes), dtype=np.float32)
nested_oof_preds = np.zeros(len(train), dtype=int)
hungarian_oof_preds = np.zeros(len(train), dtype=int)
fold_scores_raw = []
fold_scores_hungarian = []
fold_weights = []

t0 = time.time()
print("\n" + "=" * 75)
print("EXECUTING EXP_29: NESTED LEAK-FREE OPEN POOL + PROMPT RESIDUAL ENSEMBLE")
print("=" * 75)

for fold, (tr_idx, val_idx) in enumerate(kf.split(train, y_true, groups=groups), 1):
    f_t0 = time.time()
    y_tr = y_true_idx[tr_idx]
    y_val = y_true_idx[val_idx]
    val_texts = texts[val_idx]
    
    # Strictly optimize weights only on training fold using cross-entropy
    def loss_func(logits):
        w = np.exp(logits - np.max(logits))
        w /= np.sum(w)
        blended = sum(w[k] * streams[k][tr_idx] for k in range(M))
        blended = np.clip(blended, 1e-12, 1.0 - 1e-12)
        blended /= np.sum(blended, axis=1, keepdims=True)
        return log_loss(y_tr, blended)
        
    res = minimize(loss_func, x0=np.zeros(M), method="L-BFGS-B")
    best_w = np.exp(res.x - np.max(res.x))
    best_w /= np.sum(best_w)
    fold_weights.append(best_w)
    
    # Evaluate blended probabilities on unseen validation fold
    val_blended = sum(best_w[k] * streams[k][val_idx] for k in range(M))
    val_blended /= np.sum(val_blended, axis=1, keepdims=True)
    nested_oof_probs[val_idx] = val_blended
    
    preds_raw = np.argmax(val_blended, axis=1)
    nested_oof_preds[val_idx] = preds_raw
    f_acc_raw = balanced_accuracy_score(y_val, preds_raw)
    fold_scores_raw.append(f_acc_raw)
    
    # Apply blind prompt recovery & Hungarian assignment
    preds_hung, _ = recover_clusters_and_assign(val_texts, val_blended)
    hungarian_oof_preds[val_idx] = preds_hung
    f_acc_hung = balanced_accuracy_score(y_val, preds_hung)
    fold_scores_hungarian.append(f_acc_hung)
    
    print(f"Fold {fold} in {time.time() - f_t0:.1f}s | Raw BalAcc={f_acc_raw:.4f} | Hungarian BalAcc={f_acc_hung:.4f} | Weights={np.round(best_w, 3)}")

total_elapsed = time.time() - t0
acc_raw = balanced_accuracy_score(y_true_idx, nested_oof_preds)
acc_hung = balanced_accuracy_score(y_true_idx, hungarian_oof_preds)
f1_hung = f1_score(y_true_idx, hungarian_oof_preds, average="macro")
mean_weights = np.mean(fold_weights, axis=0)

print("\n" + "=" * 75)
print("EXP_29 RESULTS: OPEN POOL + PROMPT RESIDUAL + SIBLING + HUNGARIAN")
print("=" * 75)
print(f"EXP_27 Official 4-View SVC BalAcc:            {balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_off, axis=1)):.4f}")
print(f"EXP_28 Open Pool 4-View SVC BalAcc:           {balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_pool, axis=1)):.4f}")
print(f"EXP_29 Raw Ensemble BalAcc:                   {acc_raw:.4f} (±{np.std(fold_scores_raw):.4f})")
print(f"EXP_29 Ensemble + Hungarian Assignment BalAcc:{acc_hung:.4f} (±{np.std(fold_scores_hungarian):.4f})")
print(f"Macro F1 Score (Hungarian):                   {f1_hung:.4f}")
print(f"Delta vs EXP_28 Open Pool SVC:                {acc_hung - balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_pool, axis=1)):+.4f}")
print(f"Delta vs EXP_27 Official SVC:                 {acc_hung - balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_off, axis=1)):+.4f}")
print(f"Optimal Stream Weights (Pool SVC, Res, Sibling): {np.round(mean_weights, 3)}")
print("=" * 75)

# %% — Save OOF Predictions
out_path_oof = os.path.join(OUTPUTS_DIR, "oof_open_pool_residual.npy")
np.save(out_path_oof, nested_oof_probs)
print(f"Saved OOF probabilities to {out_path_oof}")

# %% — Generate Report
report_path = os.path.join(REPORTS_DIR, "open_pool_residual_results.md")
cls_rep = classification_report(y_true_idx, hungarian_oof_preds, target_names=classes, output_dict=True)
with open(report_path, "w") as f:
    f.write(f"""# EXP_29: Official + Open Pool + 4-View SVC + Prompt Residual Report
**Experiment ID**: `EXP_29_open_pool_residual`  
**Date**: {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Validation Strategy**: 5-Fold `GroupKFold(group=prompt_id)` (Strict Prompt-Disjoint Nested CV)  
**Primary Metric**: Balanced Accuracy = **{acc_hung:.4f}** (Macro F1 = **{f1_hung:.4f}**)  
**Raw Blended Balanced Accuracy**: **{acc_raw:.4f}**  
**Fold Scores (Hungarian)**: {[round(s, 4) for s in fold_scores_hungarian]} (Std: ±{np.std(fold_scores_hungarian):.4f})  
**Runtime**: {total_elapsed:.1f}s  

---

## 1. Executive Summary & Impact Analysis
- **EXP_27 Official 4-View SVC**: `{balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_off, axis=1)):.4f}` BalAcc
- **EXP_28 Open Pool 4-View SVC**: `{balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_pool, axis=1)):.4f}` BalAcc
- **EXP_29 Open Pool + Prompt Residual + Hungarian**: **`{acc_hung:.4f}`** BalAcc
- **Net Gain vs Official Baseline**: **`{acc_hung - balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_off, axis=1)):+.4f}`**

## 2. Per-Class Recall Forensics
| Model Class | Recall | F1-Score |
| :--- | :---: | :---: |
""" + "\n".join([f"| `{c}` | {cls_rep[c]['recall']:.2%} | {cls_rep[c]['f1-score']:.4f} |" for c in classes]) + f"""
""")

print(f"Report written to {report_path}")

# %% — Register in experiment_log.jsonl
log_path = os.path.join(EXPERIMENTS_DIR, "experiment_log.jsonl")
record = {
    "experiment_id": "29_open_pool_residual",
    "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "research_question": "Does fusing Open Pool SVC with prompt residuals and Hungarian matching maximize accuracy?",
    "hypothesis": "Combining external n-gram coverage with prompt residual centering and Hungarian assignment resolves both data sparsity and topic confounding",
    "research_reference": "Phase 4 Synthesis + Wolpert Stacked Generalization",
    "dataset_version": "Official v1 + Open Pool Clean (36,000)",
    "feature_method": "Multi-Paradigm Fusion (Open Pool SVC + Prompt Residuals + Sibling Specialists + Hungarian Matching)",
    "model": "Nested Convex Log-Loss Optimized Meta-Ensemble + Hungarian Matching",
    "hyperparameters": {"n_splits": 5, "candidates_count": M, "hungarian": True},
    "cv_strategy": "GroupKFold(n_splits=5, group=prompt_id) (Strict Prompt-Disjoint Nested)",
    "group_column": "prompt_id",
    "random_seed": 42,
    "fold_scores": [float(s) for s in fold_scores_hungarian],
    "mean_balanced_accuracy": float(acc_hung),
    "std_balanced_accuracy": float(np.std(fold_scores_hungarian)),
    "macro_f1": float(f1_hung),
    "per_class_recall": {c: float(cls_rep[c]["recall"]) for c in classes},
    "training_time_sec": round(total_elapsed, 2),
    "status": "completed",
    "conclusion": f"EXP_29 achieved {acc_hung:.4f} Balanced Accuracy ({acc_hung - balanced_accuracy_score(y_true_idx, np.argmax(oof_svc_off, axis=1)):+.4f} over official SVC).",
    "next_recommendation": "Phase 6 & 7: Prompt Recovery & Shared-Prompt Gaussian Heads.",
}
with open(log_path, "a") as f:
    f.write(json.dumps(record) + "\n")

print("Registered EXP_29_open_pool_residual in experiment_log.jsonl")
