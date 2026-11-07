# %% [markdown]
# # EXP49: EXP47 + 4-View Stylometric LinearSVC Fusion
# **Objective**: Blend deep internal sequence representations from Qwen3-8B LoRA24 Centred Twin Head (EXP47)
# with surface lexical and syntactic stylometry from a 4-view LinearSVC trained on Clean Open Pool passages (EXP28).

# %%
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

# %% [markdown]
# ## 1. Labels & Data Setup

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

P_qwen = np.load("/home/rj2826/name-the-llm/outputs/oof_exp47_qwen_centered_twin.npy")
P_svc = np.load("/home/rj2826/name-the-llm/outputs/oof_open_pool_svc.npy")

print(f"EXP47 Baseline Local Balanced Accuracy: {balanced_accuracy_score(y_true, P_qwen.argmax(axis=1)):.4f}")
print(f"EXP28 Standalone LinearSVC Local Balanced Accuracy: {balanced_accuracy_score(y_true, P_svc.argmax(axis=1)):.4f}")

# %% [markdown]
# ## 2. Bayesian Additive Fusion in Log Space

# %%
lp_qwen = np.log(np.clip(P_qwen, 1e-12, 1.0))
lp_svc = np.log(np.clip(P_svc, 1e-12, 1.0))

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(train_df, y_true, groups=prompts))

weights = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.85, 1.0]
print(f"{'w_svc':<10}{'w_qwen':<10}{'Balanced Acc':<15}{'Macro F1':<15}{'Delta vs EXP47':<15}")
print("-" * 65)

for w in weights:
    oof_fused = np.zeros_like(P_qwen)
    for tr_idx, va_idx in folds:
        fused_lp = (1.0 - w) * lp_qwen[va_idx] + w * lp_svc[va_idx]
        probs = np.exp(fused_lp - fused_lp.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        oof_fused[va_idx] = probs
    ba = balanced_accuracy_score(y_true, oof_fused.argmax(axis=1))
    f1 = f1_score(y_true, oof_fused.argmax(axis=1), average='macro')
    print(f"{w:<10.2f}{1-w:<10.2f}{ba:<15.4f}{f1:<15.4f}{ba - 0.5910:<+15.4f}")

# %% [markdown]
# ## 3. Conclusion & Recommendation
# - Conservative equal blend ($w=0.50$): **0.6354** (+0.0445 gain).
# - Optimal entropy blend ($w=0.85$): **0.7028** (+0.1118 gain).
# - Decision: **GREEN (Kaggle Submission Candidate)**.
