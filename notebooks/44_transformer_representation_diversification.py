"""
# 44 — EXP_44: Transformer Representation Diversification for Relational Attribution
EXPERIMENT_ID: EXP_44_transformer_representation_diversification
DATE: 2026-11-06
RESEARCH_QUESTION: Does replacing or enriching the MiniLM-L3 representation used by the relational GAT
                   with stronger semantic representations (DeBERTa-v3-small, ModernBERT-base, Hybrid)
                   produce a stronger complementary attribution signal that improves upon the frozen EXP42C baseline?
CV_STRATEGY: Strict 5-Fold GroupKFold(group=prompt_id) (0.0% Prompt Leakage)
CANONICAL BASELINE CONTROL: EXP42C (0.95 EXP36 + 0.05 EXP40 GAT -> Hungarian BA = 0.6134 ± 0.0363)
"""

import os
import sys
import time
import json
import random
import datetime
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, spearmanr
from scipy.special import softmax
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Pinned Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = "/home/rj2826/name-the-llm"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.submission_pipeline import LABELS, LABEL_TO_IDX, IDX_TO_LABEL, recover_prompt_clusters, hungarian_assignment

DATA_PATH = os.path.join(BASE_DIR, "data/train.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Executing EXP_44 on compute device: {DEVICE}")

# ======================================================================
# Architectural Definitions: GAT Network matching EXP_40
# ======================================================================
class PromptGATLayer(nn.Module):
    def __init__(self, d_in, d_out, n_heads=4, dropout=0.15):
        super().__init__()
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.q_proj = nn.Linear(d_in, d_out, bias=False)
        self.k_proj = nn.Linear(d_in, d_out, bias=False)
        self.v_proj = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.skip_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.ffn = nn.Sequential(
            nn.Linear(d_out, d_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
            nn.Dropout(dropout)
        )
        self.dropout = dropout

    def forward(self, x, mask=None):
        B, K, _ = x.shape
        q = self.q_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, self.d_out)
        h = self.norm1(self.skip_proj(x) + self.out_proj(attn_out))
        out = self.norm2(h + self.ffn(h))
        return out


class PromptResidualGAT(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=18, n_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gat1 = PromptGATLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.gat2 = PromptGATLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.gat1(h, mask=mask)
        h = self.gat2(h, mask=mask)
        logits = self.classifier(h)
        return logits


class ResidualMLPBaseline(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=18, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================================
# Graph Construction Utilities
# ======================================================================
def compute_prompt_residuals(X, prompt_labels):
    R = np.zeros_like(X)
    for p in np.unique(prompt_labels):
        mask = (prompt_labels == p)
        mean_p = np.mean(X[mask], axis=0, keepdims=True)
        R[mask] = X[mask] - mean_p
    return R


def pack_prompt_graph_batch(node_features, prompt_labels, node_targets=None, max_k=18):
    unique_prompts = np.unique(prompt_labels)
    n_prompts = len(unique_prompts)
    D = node_features.shape[1]

    batch_x = np.zeros((n_prompts, max_k, D), dtype=np.float32)
    batch_mask = np.zeros((n_prompts, 1, max_k, max_k), dtype=bool)
    batch_y = np.full((n_prompts, max_k), -1, dtype=np.int64)
    batch_valid = np.zeros((n_prompts, max_k), dtype=bool)
    index_mapping = []

    for p_idx, p in enumerate(unique_prompts):
        orig_indices = np.where(prompt_labels == p)[0]
        k = min(len(orig_indices), max_k)
        batch_x[p_idx, :k] = node_features[orig_indices[:k]]
        batch_mask[p_idx, 0, :k, :k] = True
        batch_valid[p_idx, :k] = True
        if node_targets is not None:
            batch_y[p_idx, :k] = node_targets[orig_indices[:k]]
        index_mapping.append((p, orig_indices[:k]))

    return (
        torch.tensor(batch_x, dtype=torch.float32),
        torch.tensor(batch_mask, dtype=torch.bool),
        torch.tensor(batch_y, dtype=torch.long),
        torch.tensor(batch_valid, dtype=torch.bool),
        index_mapping
    )


# ======================================================================
# Main Execution Pipeline
# ======================================================================
def main():
    print("=" * 80)
    print("EXP_44: TRANSFORMER REPRESENTATION DIVERSIFICATION BENCHMARK")
    print("=" * 80)

    # 1. Load Data & Validated Baseline Artifacts
    train_df = pd.read_csv(DATA_PATH)
    texts = train_df["text"].values
    prompts = train_df["prompt_id"].values
    y_raw = train_df["model"].values
    y_idx = np.array([LABEL_TO_IDX[m] for m in y_raw])
    n_samples = len(train_df)
    n_classes = len(LABELS)
    unique_prompts = np.unique(prompts)
    n_prompts = len(unique_prompts)

    p_36 = np.load(os.path.join(OUTPUTS_DIR, "oof_exp36_predictions.npy"))
    p_gat_exp40 = np.load(os.path.join(OUTPUTS_DIR, "oof_exp40_gat_predictions.npy"))

    rec_clusters = recover_prompt_clusters(texts)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(train_df, y_idx, groups=prompts))

    # Verify Frozen Baselines
    h_36 = hungarian_assignment(p_36, rec_clusters)
    p_42c = 0.95 * p_36 + 0.05 * p_gat_exp40
    h_42c = hungarian_assignment(p_42c, rec_clusters)

    exp36_fold_bas = [balanced_accuracy_score(y_idx[val], hungarian_assignment(p_36[val], rec_clusters[val])) for tr, val in splits]
    exp42c_fold_bas = [balanced_accuracy_score(y_idx[val], hungarian_assignment(p_42c[val], rec_clusters[val])) for tr, val in splits]

    print(f"Verified EXP_36 Baseline: Fold Mean = {np.mean(exp36_fold_bas):.4f} ± {np.std(exp36_fold_bas):.4f} | Global = {balanced_accuracy_score(y_idx, h_36):.4f}")
    print(f"Verified EXP_42C Baseline: Fold Mean = {np.mean(exp42c_fold_bas):.4f} ± {np.std(exp42c_fold_bas):.4f} | Global = {balanced_accuracy_score(y_idx, h_42c):.4f}")

    # ======================================================================
    # 2. Load Representations
    # ======================================================================
    print("\n" + "-" * 80)
    print("LOADING MULTIMODAL TRANSFORMER REPRESENTATIONS")
    print("-" * 80)
    feat_minilm = np.load(os.path.join(OUTPUTS_DIR, "minilm_layer3_mean.npy")) # 384d
    feat_deb = np.load(os.path.join(OUTPUTS_DIR, "exp44_deberta_embeddings.npy")) # 768d
    feat_mod = np.load(os.path.join(OUTPUTS_DIR, "exp44_modernbert_embeddings.npy")) # 768d
    feat_sty = np.load(os.path.join(OUTPUTS_DIR, "features_reference_stylometry.npy")) # 144d

    print(f"MiniLM-L3:        {feat_minilm.shape}")
    print(f"DeBERTa-v3-small: {feat_deb.shape}")
    print(f"ModernBERT-base:  {feat_mod.shape}")
    print(f"Stylometry:       {feat_sty.shape}")

    # Construct feature branches (embedding + stylometry)
    branch_features = {
        "EXP40_MiniLM": np.hstack([feat_minilm, feat_sty]).astype(np.float32),      # 528d
        "EXP44_A_DeBERTa": np.hstack([feat_deb, feat_sty]).astype(np.float32),       # 912d
        "EXP44_B_ModernBERT": np.hstack([feat_mod, feat_sty]).astype(np.float32),   # 912d
        "EXP44_C_Hybrid": np.hstack([feat_minilm, feat_deb, feat_sty]).astype(np.float32) # 1296d
    }

    # ======================================================================
    # 3. Step 7: Representation-Only Controls (LinearSVC / LogisticRegression)
    # ======================================================================
    print("\n" + "-" * 80)
    print("STEP 7: REPRESENTATION-ONLY CONTROLS (PROMPT-DISJOINT CV)")
    print("-" * 80)
    rep_controls = {
        "MiniLM-L3": feat_minilm,
        "DeBERTa-v3-small": feat_deb,
        "ModernBERT-base": feat_mod,
        "MiniLM + DeBERTa": np.hstack([feat_minilm, feat_deb])
    }

    rep_ctrl_results = []
    for name, X in rep_controls.items():
        oof_preds = np.zeros(n_samples, dtype=int)
        for tr_idx, val_idx in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_val = scaler.transform(X[val_idx])
            clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
            clf.fit(X_tr, y_idx[tr_idx])
            oof_preds[val_idx] = clf.predict(X_val)

        ba = float(balanced_accuracy_score(y_idx, oof_preds))
        f1 = float(f1_score(y_idx, oof_preds, average="macro"))
        rec = {"representation": name, "dim": X.shape[1], "standalone_raw_ba": ba, "macro_f1": f1}
        rep_ctrl_results.append(rec)
        print(f"{name:20s} (Dim: {X.shape[1]:4d}) | Standalone Raw BA = {ba:.4f} | Macro F1 = {f1:.4f}")

    # ======================================================================
    # 4. Step 8: Train Prompt-Residual GAT Branches
    # ======================================================================
    print("\n" + "-" * 80)
    print("STEP 8: TRAINING PROMPT-RESIDUAL GAT BRANCHES (5 FOLDS)")
    print("-" * 80)

    gat_oof_probs = {}

    for branch_name, X_dense in branch_features.items():
        t0_branch = time.time()
        in_dim = X_dense.shape[1]
        print(f"\n--- Training GAT Branch: {branch_name} (in_dim={in_dim}) ---")

        oof_branch_probs = np.zeros((n_samples, n_classes), dtype=np.float32)

        for fold, (tr_idx, val_idx) in enumerate(splits, 1):
            f_t0 = time.time()
            tr_prompts = prompts[tr_idx]
            val_prompts = prompts[val_idx]
            y_tr = y_idx[tr_idx]
            y_val = y_idx[val_idx]

            # Leak-Free Feature Preprocessing: Prompt Centering
            R_tr = compute_prompt_residuals(X_dense[tr_idx], tr_prompts)
            scaler = StandardScaler()
            R_tr_s = scaler.fit_transform(R_tr).astype(np.float32)

            # Blind Validation Residuals
            val_texts = texts[val_idx]
            rec_clusters_val = recover_prompt_clusters(val_texts)
            R_val_blind = compute_prompt_residuals(X_dense[val_idx], rec_clusters_val)
            R_val_blind_s = scaler.transform(R_val_blind).astype(np.float32)

            # Pack Graphs
            bx_tr, bmask_tr, by_tr, bvalid_tr, _ = pack_prompt_graph_batch(R_tr_s, tr_prompts, y_tr)
            bx_tr, bmask_tr, by_tr, bvalid_tr = bx_tr.to(DEVICE), bmask_tr.to(DEVICE), by_tr.to(DEVICE), bvalid_tr.to(DEVICE)

            bx_val, bmask_val, _, bvalid_val, map_val = pack_prompt_graph_batch(R_val_blind_s, rec_clusters_val)
            bx_val, bmask_val, bvalid_val = bx_val.to(DEVICE), bmask_val.to(DEVICE), bvalid_val.to(DEVICE)

            # Init Model with explicit seed per fold
            torch.manual_seed(SEED + fold)
            gat_model = PromptResidualGAT(in_features=in_dim, hidden_dim=128, num_classes=n_classes, n_heads=4, dropout=0.2).to(DEVICE)
            opt_gat = AdamW(gat_model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched_gat = CosineAnnealingLR(opt_gat, T_max=40, eta_min=1e-5)

            gat_model.train()
            for epoch in range(40):
                opt_gat.zero_grad()
                logits = gat_model(bx_tr, mask=bmask_tr)
                loss = F.cross_entropy(logits[bvalid_tr], by_tr[bvalid_tr])
                loss.backward()
                opt_gat.step()
                sched_gat.step()

            gat_model.eval()
            with torch.no_grad():
                logits_val = gat_model(bx_val, mask=bmask_val)
                probs_val = F.softmax(logits_val, dim=-1).cpu().numpy()
                for p_idx, (_, orig_idx) in enumerate(map_val):
                    oof_branch_probs[val_idx[orig_idx]] = probs_val[p_idx, :len(orig_idx)]

            f_ba = balanced_accuracy_score(y_val, np.argmax(oof_branch_probs[val_idx], axis=-1))
            print(f"  Fold {fold} in {time.time() - f_t0:.1f}s | GAT Raw BA = {f_ba:.4f}")

        gat_oof_probs[branch_name] = oof_branch_probs
        branch_raw_ba = balanced_accuracy_score(y_idx, np.argmax(oof_branch_probs, axis=-1))
        branch_h_ba = balanced_accuracy_score(y_idx, hungarian_assignment(oof_branch_probs, rec_clusters))
        print(f">> {branch_name} Full OOF: Raw BA = {branch_raw_ba:.4f} | Hungarian BA = {branch_h_ba:.4f} in {time.time() - t0_branch:.1f}s")

    # Save generated GAT predictions
    np.save(os.path.join(OUTPUTS_DIR, "oof_exp44_deberta_gat_predictions.npy"), gat_oof_probs["EXP44_A_DeBERTa"])
    np.save(os.path.join(OUTPUTS_DIR, "oof_exp44_modernbert_gat_predictions.npy"), gat_oof_probs["EXP44_B_ModernBERT"])
    np.save(os.path.join(OUTPUTS_DIR, "oof_exp44_hybrid_gat_predictions.npy"), gat_oof_probs["EXP44_C_Hybrid"])
    print(">> Saved all GAT prediction arrays to outputs/")

    # ======================================================================
    # 5. Step 9: Complementarity & Error Overlap Analysis
    # ======================================================================
    print("\n" + "-" * 80)
    print("STEP 9: COMPLEMENTARITY ANALYSIS AGAINST EXP_36 BASELINE")
    print("-" * 80)

    preds_36 = np.argmax(p_36, axis=1)
    corr_36 = (preds_36 == y_idx)
    err_36 = (preds_36 != y_idx)

    comp_records = []
    for branch_name, probs in gat_oof_probs.items():
        preds_gat = np.argmax(probs, axis=1)
        corr_gat = (preds_gat == y_idx)
        err_gat = (preds_gat != y_idx)

        h_preds_gat = hungarian_assignment(probs, rec_clusters)
        h_ba_gat = float(balanced_accuracy_score(y_idx, h_preds_gat))
        raw_ba_gat = float(balanced_accuracy_score(y_idx, preds_gat))

        agreement = float(np.mean(preds_36 == preds_gat))
        spearman_corr = float(np.mean([spearmanr(p_36[i], probs[i]).correlation for i in range(n_samples)]))

        both_corr = int(np.sum(corr_36 & corr_gat))
        both_err = int(np.sum(err_36 & err_gat))
        exp36_only_corr = int(np.sum(corr_36 & err_gat))
        gat_only_corr = int(np.sum(err_36 & corr_gat))

        rec = {
            "branch": branch_name,
            "raw_ba": raw_ba_gat,
            "hungarian_ba": h_ba_gat,
            "agreement_with_exp36": agreement,
            "spearman_with_exp36": spearman_corr,
            "both_correct": both_corr,
            "both_wrong": both_err,
            "exp36_only_correct": exp36_only_corr,
            "gat_unique_correct": gat_only_corr
        }
        comp_records.append(rec)
        print(f"{branch_name:20s} | Standalone H BA = {h_ba_gat:.4f} | Agreement = {agreement:.2%} | Spearman = {spearman_corr:.4f} | GAT-Unique-Correct = {gat_only_corr} (EXP40 Ref: 285)")

    # ======================================================================
    # 6. Step 10 & 11: Fusion Test & Comparison Table
    # ======================================================================
    print("\n" + "-" * 80)
    print("STEP 10 & 11: PROBABILITY FUSION BENCHMARK (NESTED VS FIXED ALPHA)")
    print("-" * 80)

    alpha_grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
    comparison_records = []

    # Include existing EXP40 as verified control
    all_branches = {
        "EXP40 MiniLM-GAT": p_gat_exp40,
        "EXP44-A DeBERTa-GAT": gat_oof_probs["EXP44_A_DeBERTa"],
        "EXP44-B ModernBERT-GAT": gat_oof_probs["EXP44_B_ModernBERT"],
        "EXP44-C MiniLM+DeBERTa-GAT": gat_oof_probs["EXP44_C_Hybrid"]
    }

    for sys_name, p_gat_cand in all_branches.items():
        # Standalone Hungarian BA
        sa_h_preds = hungarian_assignment(p_gat_cand, rec_clusters)
        sa_h_ba = float(balanced_accuracy_score(y_idx, sa_h_preds))

        # Fixed alpha = 0.05
        p_fixed = 0.95 * p_36 + 0.05 * p_gat_cand
        h_fixed_preds = np.zeros(n_samples, dtype=int)
        fixed_fold_bas = []
        for tr, val in splits:
            hf = hungarian_assignment(p_fixed[val], rec_clusters[val])
            h_fixed_preds[val] = hf
            fixed_fold_bas.append(float(balanced_accuracy_score(y_idx[val], hf)))
        fixed_05_ba = float(balanced_accuracy_score(y_idx, h_fixed_preds))
        fixed_fold_mean = float(np.mean(fixed_fold_bas))
        fixed_fold_std = float(np.std(fixed_fold_bas))

        # Nested OOF alpha selection
        oof_nested_preds = np.zeros(n_samples, dtype=int)
        chosen_alphas = []
        nested_fold_bas = []

        for tr, val in splits:
            inner_gkf = GroupKFold(n_splits=4)
            best_a, best_inner_ba = 0.05, -1.0
            for a in alpha_grid:
                inner_bas = []
                for in_tr, in_val in inner_gkf.split(tr, y_idx[tr], groups=prompts[tr]):
                    v_idx = tr[in_val]
                    p_in = (1.0 - a) * p_36[v_idx] + a * p_gat_cand[v_idx]
                    h_in = hungarian_assignment(p_in, rec_clusters[v_idx])
                    inner_bas.append(balanced_accuracy_score(y_idx[v_idx], h_in))
                m_ba = np.mean(inner_bas)
                if m_ba > best_inner_ba:
                    best_inner_ba = m_ba
                    best_a = a
            chosen_alphas.append(best_a)
            p_val = (1.0 - best_a) * p_36[val] + best_a * p_gat_cand[val]
            hf = hungarian_assignment(p_val, rec_clusters[val])
            oof_nested_preds[val] = hf
            nested_fold_bas.append(float(balanced_accuracy_score(y_idx[val], hf)))

        nested_fused_ba = float(balanced_accuracy_score(y_idx, oof_nested_preds))
        nested_fold_mean = float(np.mean(nested_fold_bas))
        nested_fold_std = float(np.std(nested_fold_bas))
        delta_vs_exp42c = nested_fused_ba - 0.6134

        rec = {
            "system": sys_name,
            "standalone_h_ba": sa_h_ba,
            "chosen_alphas": str(chosen_alphas),
            "mean_best_alpha": float(np.mean(chosen_alphas)),
            "nested_fused_ba": nested_fused_ba,
            "nested_fold_mean": nested_fold_mean,
            "nested_fold_std": nested_fold_std,
            "fixed_alpha_05_ba": fixed_05_ba,
            "fixed_fold_mean": fixed_fold_mean,
            "fixed_fold_std": fixed_fold_std,
            "delta_vs_exp42c": delta_vs_exp42c
        }
        comparison_records.append(rec)
        print(f"{sys_name:28s} | Standalone: {sa_h_ba:.4f} | Best alpha: {np.mean(chosen_alphas):.3f} | Nested Fused BA: {nested_fused_ba:.4f} | Fixed 0.05 BA: {fixed_05_ba:.4f} | Delta vs EXP42C: {delta_vs_exp42c:+.4f}")

    comp_df = pd.DataFrame(comparison_records)
    comp_df.to_csv(os.path.join(OUTPUTS_DIR, "exp44_comparison.csv"), index=False)
    print(f">> Saved comparison table to outputs/exp44_comparison.csv")

    # ======================================================================
    # 7. Step 12: Statistical Validation (Prompt-Level Cluster Bootstrap)
    # ======================================================================
    print("\n" + "-" * 80)
    print("STEP 12: STATISTICAL VALIDATION (PROMPT-LEVEL CLUSTER BOOTSTRAP)")
    print("-" * 80)

    # Identify candidate with highest fused score
    best_cand_name = max(comparison_records, key=lambda x: x["nested_fused_ba"])["system"]
    p_best_gat = all_branches[best_cand_name]
    best_mean_a = next(r["mean_best_alpha"] for r in comparison_records if r["system"] == best_cand_name)
    p_best_fused = (1.0 - best_mean_a) * p_36 + best_mean_a * p_best_gat
    h_best_fused = hungarian_assignment(p_best_fused, rec_clusters)

    # Evaluate prompt-level differences vs EXP42C
    prompt_to_indices = {pid: np.where(prompts == pid)[0] for pid in unique_prompts}
    prompt_deltas_vs_42c = []
    imp_prompts, wrs_prompts, tie_prompts = 0, 0, 0

    for pid in unique_prompts:
        mask = (prompts == pid)
        acc_42c = np.mean(h_42c[mask] == y_idx[mask])
        acc_best = np.mean(h_best_fused[mask] == y_idx[mask])
        d = float(acc_best - acc_42c)
        prompt_deltas_vs_42c.append(d)
        if d > 1e-6:
            imp_prompts += 1
        elif d < -1e-6:
            wrs_prompts += 1
        else:
            tie_prompts += 1

    prompt_deltas_vs_42c = np.array(prompt_deltas_vs_42c)
    mean_prompt_d = float(np.mean(prompt_deltas_vs_42c))
    median_prompt_d = float(np.median(prompt_deltas_vs_42c))

    # Cluster Bootstrap
    n_boot = 2000
    boot_deltas = []
    rng_boot = np.random.RandomState(42)

    for b in range(n_boot):
        boot_pids = rng_boot.choice(unique_prompts, size=n_prompts, replace=True)
        boot_idx = np.concatenate([prompt_to_indices[pid] for pid in boot_pids])
        ba_42c_b = balanced_accuracy_score(y_idx[boot_idx], h_42c[boot_idx])
        ba_best_b = balanced_accuracy_score(y_idx[boot_idx], h_best_fused[boot_idx])
        boot_deltas.append(ba_best_b - ba_42c_b)

    boot_deltas = np.array(boot_deltas)
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    p_pos = float(np.mean(boot_deltas > 0))

    print(f"Best Candidate vs EXP42C: {best_cand_name}")
    print(f"  Improved prompt groups: {imp_prompts} / {n_prompts} ({imp_prompts/n_prompts:.1%})")
    print(f"  Worsened prompt groups: {wrs_prompts} / {n_prompts} ({wrs_prompts/n_prompts:.1%})")
    print(f"  Tied prompt groups:     {tie_prompts} / {n_prompts} ({tie_prompts/n_prompts:.1%})")
    print(f"  Mean per-prompt delta:  {mean_prompt_d:+.4f}")
    print(f"  Median per-prompt delta:{median_prompt_d:+.4f}")
    print(f"  Prompt Bootstrap 95% CI:[{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  P(Bootstrap delta > 0): {p_pos:.4f}")

    # ======================================================================
    # 8. Step 13: Uncertainty Analysis (Quintiles vs EXP42C)
    # ======================================================================
    print("\n" + "-" * 80)
    print("STEP 13: UNCERTAINTY-CONDITIONAL ANALYSIS (QUINTILES VS EXP42C)")
    print("-" * 80)

    conf_36 = np.max(p_36, axis=1)
    q_labels = ["Q5_lowest", "Q4_low", "Q3_mid", "Q2_high", "Q1_highest"]
    q_bins = pd.qcut(conf_36, q=5, labels=q_labels)

    uncertainty_records = []
    for q in q_labels:
        mask = np.array(q_bins == q)
        n_q = int(np.sum(mask))
        ba_42c_q = float(balanced_accuracy_score(y_idx[mask], h_42c[mask]))
        ba_best_q = float(balanced_accuracy_score(y_idx[mask], h_best_fused[mask]))
        acc_42c_q = float(accuracy_score(y_idx[mask], h_42c[mask]))
        acc_best_q = float(accuracy_score(y_idx[mask], h_best_fused[mask]))
        gain_ba = ba_best_q - ba_42c_q
        pct_diff = float(np.mean(h_42c[mask] != h_best_fused[mask]))

        rec = {
            "bin": q,
            "n_samples": n_q,
            "exp42c_acc": acc_42c_q,
            "exp44_acc": acc_best_q,
            "exp42c_ba": ba_42c_q,
            "exp44_ba": ba_best_q,
            "ba_gain": gain_ba,
            "pct_diff": pct_diff
        }
        uncertainty_records.append(rec)
        print(f"{q:10s} (N={n_q}): EXP42C BA={ba_42c_q:.4f} | EXP44 BA={ba_best_q:.4f} | Gain vs 42C={gain_ba:+.4f} | Diff={pct_diff:.2%}")

    # ======================================================================
    # 9. Step 14: Error Overlap Analysis Across All GAT Branches
    # ======================================================================
    print("\n" + "-" * 80)
    print("STEP 14: ERROR OVERLAP MATRIX ACROSS ALL GAT MODELS")
    print("-" * 80)

    branch_keys = list(all_branches.keys())
    error_overlap_matrix = np.zeros((len(branch_keys), len(branch_keys)), dtype=float)
    corr_matrix = np.zeros((len(branch_keys), len(branch_keys)), dtype=float)

    for i, k1 in enumerate(branch_keys):
        preds1 = np.argmax(all_branches[k1], axis=1)
        err1 = (preds1 != y_idx)
        for j, k2 in enumerate(branch_keys):
            preds2 = np.argmax(all_branches[k2], axis=1)
            err2 = (preds2 != y_idx)
            # Jaccard error overlap
            jaccard_err = float(np.sum(err1 & err2) / max(np.sum(err1 | err2), 1))
            error_overlap_matrix[i, j] = jaccard_err
            # Mean Spearman correlation between probability predictions
            s_corr = float(np.mean([spearmanr(all_branches[k1][idx], all_branches[k2][idx]).correlation for idx in range(n_samples)]))
            corr_matrix[i, j] = s_corr

    print("Jaccard Error Overlap Matrix:")
    print(pd.DataFrame(error_overlap_matrix, index=branch_keys, columns=branch_keys).round(4))
    print("\nMean Prediction Spearman Correlation Matrix:")
    print(pd.DataFrame(corr_matrix, index=branch_keys, columns=branch_keys).round(4))

    # ======================================================================
    # 10. Step 15: Ablation Study for the Best Architecture
    # ======================================================================
    print("\n" + "-" * 80)
    print("STEP 15: ABLATION STUDY")
    print("-" * 80)

    # Ablations:
    # A. Transformer only (DeBERTa 768d linear)
    # B. Transformer + stylometry (912d linear)
    # C. Transformer residual only (ResMLP on 912d)
    # D. Transformer residual + GAT (standalone GAT)
    # E. EXP36 + Transformer-GAT (0.95 EXP36 + 0.05 DeBERTa-GAT)
    # F. EXP42C + Transformer-GAT (Three-way blend: 0.95 EXP36 + 0.025 MiniLM-GAT + 0.025 DeBERTa-GAT)

    # Run C: ResMLP on DeBERTa residuals
    X_deb_sty = branch_features["EXP44_A_DeBERTa"]
    oof_deb_mlp = np.zeros((n_samples, n_classes), dtype=np.float32)
    for fold, (tr_idx, val_idx) in enumerate(splits, 1):
        tr_p = prompts[tr_idx]
        val_texts = texts[val_idx]
        rec_cl = recover_prompt_clusters(val_texts)
        R_tr = compute_prompt_residuals(X_deb_sty[tr_idx], tr_p)
        scaler = StandardScaler()
        R_tr_s = scaler.fit_transform(R_tr).astype(np.float32)
        R_val = compute_prompt_residuals(X_deb_sty[val_idx], rec_cl)
        R_val_s = scaler.transform(R_val).astype(np.float32)

        mlp_deb = ResidualMLPBaseline(in_features=912, hidden_dim=128, num_classes=n_classes, dropout=0.2).to(DEVICE)
        opt_mlp = AdamW(mlp_deb.parameters(), lr=1e-3, weight_decay=1e-4)
        t_x_tr = torch.tensor(R_tr_s, dtype=torch.float32).to(DEVICE)
        t_y_tr = torch.tensor(y_idx[tr_idx], dtype=torch.long).to(DEVICE)
        t_x_val = torch.tensor(R_val_s, dtype=torch.float32).to(DEVICE)

        mlp_deb.train()
        for ep in range(40):
            opt_mlp.zero_grad()
            l = F.cross_entropy(mlp_deb(t_x_tr), t_y_tr)
            l.backward()
            opt_mlp.step()

        mlp_deb.eval()
        with torch.no_grad():
            oof_deb_mlp[val_idx] = F.softmax(mlp_deb(t_x_val), dim=-1).cpu().numpy()

    ba_deb_mlp = float(balanced_accuracy_score(y_idx, np.argmax(oof_deb_mlp, axis=-1)))

    # F: Three-way blend
    p_threeway = 0.95 * p_36 + 0.025 * p_gat_exp40 + 0.025 * gat_oof_probs["EXP44_A_DeBERTa"]
    h_threeway = hungarian_assignment(p_threeway, rec_clusters)
    ba_threeway = float(balanced_accuracy_score(y_idx, h_threeway))

    # Run B: Logistic Regression on DeBERTa + Stylometry
    oof_deb_sty_lin = np.zeros(n_samples, dtype=int)
    for tr_idx, val_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_deb_sty[tr_idx])
        X_val = scaler.transform(X_deb_sty[val_idx])
        clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        clf.fit(X_tr, y_idx[tr_idx])
        oof_deb_sty_lin[val_idx] = clf.predict(X_val)
    ba_deb_sty_lin = float(balanced_accuracy_score(y_idx, oof_deb_sty_lin))

    ablation_records = [
        {"configuration": "A. DeBERTa Transformer Only (Linear)", "hungarian_ba": next(r["standalone_raw_ba"] for r in rep_ctrl_results if r["representation"] == "DeBERTa-v3-small"), "delta_vs_exp36": next(r["standalone_raw_ba"] for r in rep_ctrl_results if r["representation"] == "DeBERTa-v3-small") - 0.5978},
        {"configuration": "B. DeBERTa + Stylometry (Linear)", "hungarian_ba": ba_deb_sty_lin, "delta_vs_exp36": ba_deb_sty_lin - 0.5978},
        {"configuration": "C. DeBERTa Residual Only (MLP, Blind)", "hungarian_ba": ba_deb_mlp, "delta_vs_exp36": ba_deb_mlp - 0.5978},
        {"configuration": "D. DeBERTa Residual + GAT (Standalone)", "hungarian_ba": next(r["standalone_h_ba"] for r in comparison_records if "DeBERTa" in r["system"]), "delta_vs_exp36": next(r["standalone_h_ba"] for r in comparison_records if "DeBERTa" in r["system"]) - 0.5978},
        {"configuration": "E. EXP_36 + DeBERTa-GAT (0.95 + 0.05)", "hungarian_ba": next(r["fixed_alpha_05_ba"] for r in comparison_records if "DeBERTa" in r["system"]), "delta_vs_exp36": next(r["fixed_alpha_05_ba"] for r in comparison_records if "DeBERTa" in r["system"]) - 0.5978},
        {"configuration": "F. EXP_42C + DeBERTa-GAT (Three-Way Blend)", "hungarian_ba": ba_threeway, "delta_vs_exp36": ba_threeway - 0.5978}
    ]

    for ab in ablation_records:
        print(f"{ab['configuration']:48s} | BA = {ab['hungarian_ba']:.4f} | Delta vs EXP36 = {ab['delta_vs_exp36']:+.4f}")

    # ======================================================================
    # 11. Save Metadata Artifacts
    # ======================================================================
    metadata = {
        "experiment_id": "EXP_44_transformer_representation_diversification",
        "date": datetime.datetime.now().isoformat(),
        "frozen_baseline": {
            "exp36_ba": 0.5978,
            "exp42c_ba": 0.6134,
            "status": "frozen_canonical"
        },
        "representation_controls": rep_ctrl_results,
        "gat_branches": comparison_records,
        "complementarity_metrics": comp_records,
        "uncertainty_analysis": uncertainty_records,
        "error_overlap_jaccard": error_overlap_matrix.tolist(),
        "prediction_spearman_corr": corr_matrix.tolist(),
        "statistical_bootstrap_vs_42c": {
            "best_system": best_cand_name,
            "mean_delta": mean_prompt_d,
            "median_delta": median_prompt_d,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "p_positive": p_pos,
            "improved_prompts": imp_prompts,
            "worsened_prompts": wrs_prompts,
            "tied_prompts": tie_prompts
        },
        "ablation_results": ablation_records
    }

    with open(os.path.join(OUTPUTS_DIR, "exp44_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(">> Saved metadata to outputs/exp44_metadata.json")
    print("=" * 80)
    print("EXP_44 BENCHMARK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
