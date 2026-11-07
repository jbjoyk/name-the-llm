# %% [markdown]
# # EXP48: Sinkhorn / Global Marginal Calibration
# **Objective**: Evaluate damped Sinkhorn-Knopp optimal transport balancing on EXP47 (Qwen3-8B LoRA24 Centred Twin Head) under prompt-disjoint GroupKFold validation.

# %%
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

# %% [markdown]
# ## 1. Labels & Environment Setup

# %%
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

train_df = pd.read_csv("/home/rj2826/name-the-llm/data/train.csv")
y_true = train_df['model'].map(L2I).values
prompts = train_df['prompt_id'].values

P_exp47 = np.load("/home/rj2826/name-the-llm/outputs/oof_exp47_qwen_centered_twin.npy")
base_preds = P_exp47.argmax(axis=1)
base_ba = balanced_accuracy_score(y_true, base_preds)
print(f"EXP47 Baseline Local Balanced Accuracy: {base_ba:.4f}")

# %% [markdown]
# ## 2. Damped Sinkhorn-Knopp Matrix Scaling Implementation

# %%
def sinkhorn_knopp_balance(P, target_marginal, alpha, max_iter=100, eps=1e-12):
    if alpha <= 0.0:
        return P.copy()
    N = len(P)
    Q = P.copy()
    target_counts = N * target_marginal[None, :]
    for _ in range(max_iter):
        col_sums = np.clip(Q.sum(axis=0, keepdims=True), eps, None)
        Q *= (target_counts / col_sums)
        row_sums = np.clip(Q.sum(axis=1, keepdims=True), eps, None)
        Q /= row_sums
    log_P = np.log(np.clip(P, eps, 1.0))
    log_Q = np.log(np.clip(Q, eps, 1.0))
    log_blended = (1.0 - alpha) * log_P + alpha * log_Q
    Q_out = np.exp(log_blended - log_blended.max(axis=1, keepdims=True))
    Q_out /= Q_out.sum(axis=1, keepdims=True)
    return Q_out

# %% [markdown]
# ## 3. Prompt-Disjoint GroupKFold Parameter Sweep

# %%
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(train_df, y_true, groups=prompts))

alphas = [0.0, 0.15, 0.30, 0.42, 0.50]
rhos = [0.0, 0.20, 0.35, 0.50]

print(f"{'alpha':<8}{'rho':<8}{'Balanced Acc':<15}{'Macro F1':<15}{'Delta':<10}")
print("-" * 56)
for rho in rhos:
    for alpha in alphas:
        oof_calib = np.zeros_like(P_exp47)
        for tr_idx, va_idx in folds:
            P_val = P_exp47[va_idx]
            y_tr = y_true[tr_idx]
            train_prior = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64) + 1.0
            train_prior /= train_prior.sum()
            pred_marg = np.clip(P_val.mean(axis=0), 1e-6, None)
            pred_marg /= pred_marg.sum()
            target = (1.0 - rho) * pred_marg + rho * train_prior
            target /= target.sum()
            oof_calib[va_idx] = sinkhorn_knopp_balance(P_val, target, alpha=alpha)
        ba = balanced_accuracy_score(y_true, oof_calib.argmax(axis=1))
        f1 = f1_score(y_true, oof_calib.argmax(axis=1), average='macro')
        print(f"{alpha:<8.2f}{rho:<8.2f}{ba:<15.4f}{f1:<15.4f}{ba - base_ba:<+10.4f}")

# %% [markdown]
# ## 4. Conclusion & Decision
# - Standalone Qwen3-8B is already well calibrated: Sinkhorn yields delta of +0.0000 to -0.0007.
# - Decision: **YELLOW** (Hold for multi-model ensemble EXP49/50).
