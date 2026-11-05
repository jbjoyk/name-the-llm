# %% [markdown]
"""
# 05 — Supervised Contrastive Learning for LLM Attribution
EXPERIMENT_ID: 05_contrastive_learning
DATE: 2026-11-05
RESEARCH_QUESTION: Can supervised contrastive learning (SupCon) train a projection head
                   on pretrained sentence embeddings to disentangle LLM style from prompt
                   topic, overcoming the semantic collapse observed in Experiment 03?
HYPOTHESIS: Raw sentence embeddings achieved only 0.1363 because representations cluster by
            topic (prompt) rather than author style. Supervised Contrastive Loss (Khosla et al., 2020)
            explicitly pulls embeddings from the SAME LLM (across diverse prompts) closer together
            while pushing embeddings from DIFFERENT LLMs (even on the same prompt) apart.
            A linear classifier on these SupCon-adapted embeddings will dramatically outperform
            raw embeddings (0.1363) and provide a clean, non-confounded signal for fusion.
DATA_VERSION: v1 (original Kaggle release)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id)
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODELS: SupCon Projection Head (384 -> 256 -> 128) + Linear Classifier; SupCon + TF-IDF Fusion
FEATURES: 384-d MiniLM Embeddings transformed via SupCon Metric Learning
HYPERPARAMETERS: temperature=0.07, lr=1e-3, epochs=40, batch_size=128
COMPUTE_CONFIGURATION: GPU (RTX 3060 12GB, PyTorch CUDA)
PREVIOUS_BEST_SCORE: 0.3178 (04_04b_full_fusion_dense)
"""

# %% — Imports & Setup
import sys, time, json, datetime, os
sys.path.insert(0, "/home/rj2826/name-the-llm/src")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.sparse import hstack, csr_matrix

TRAIN_PATH = "/home/rj2826/name-the-llm/data/train.csv"
EXPERIMENTS_DIR = "/home/rj2826/name-the-llm/experiments"
OUTPUTS_DIR = "/home/rj2826/name-the-llm/outputs"
RANDOM_SEED = 42
N_SPLITS = 5
PREV_BEST = 0.3178

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Previous Best: {PREV_BEST:.4f}")

# %% — Load data & cached embeddings
train = pd.read_csv(TRAIN_PATH)
y_raw = train["model"].values
groups = train["prompt_id"].values

dense_emb = np.load(os.path.join(OUTPUTS_DIR, "embeddings_minilm.npy"))
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(le.classes_)
print(f"Loaded {len(train)} samples with {n_classes} classes. Embeddings: {dense_emb.shape}")

# %% — Supervised Contrastive Loss Implementation
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning (Khosla et al., NeurIPS 2020).
    Pulls embeddings with the same label together, pushes differing labels apart.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (batch_size, dim), L2-normalized
        # labels: (batch_size,)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Self-contrast mask (exclude diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute dot-product similarity
        similarity = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # Numerical stability: subtract max per row
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Denominator: exp of all except self
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)
        
        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
        loss = -mean_log_prob_pos[mask.sum(1) > 0].mean()
        return loss

# %% — Projection Head Architecture
class StyleProjectionHead(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        feat = self.net(x)
        return F.normalize(feat, p=2, dim=1)

# %% — GroupKFold Evaluation of SupCon Projection Head
kf = GroupKFold(n_splits=N_SPLITS)
splits = list(kf.split(train, y, groups=groups))

def train_supcon_fold(X_tr, y_tr, epochs=40, lr=1e-3, batch_size=128):
    """Train SupCon projection head on training fold only."""
    model = StyleProjectionHead(in_dim=384, hidden_dim=256, out_dim=128).to(device)
    criterion = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            features = model(batch_x)
            loss = criterion(features, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

print("=" * 60)
print("EXP 05a — Supervised Contrastive Head + Linear Classifier")
print("=" * 60)

fold_bal_accs, fold_f1s = [], []
supcon_embeddings = np.zeros((len(train), 128))
t0 = time.time()

for fold_i, (tr_idx, val_idx) in enumerate(splits):
    X_tr, X_val = dense_emb[tr_idx], dense_emb[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    # Standardize embeddings per fold
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    
    # Train SupCon head strictly on training fold
    head = train_supcon_fold(X_tr_s, y_tr, epochs=45, lr=2e-3, batch_size=128)
    
    # Extract contrastive projections
    with torch.no_grad():
        proj_tr = head(torch.tensor(X_tr_s, dtype=torch.float32).to(device)).cpu().numpy()
        proj_val = head(torch.tensor(X_val_s, dtype=torch.float32).to(device)).cpu().numpy()
        
    supcon_embeddings[val_idx] = proj_val
    
    # Fit classifier on contrastive features
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(proj_tr, y_tr)
    y_pred = clf.predict(proj_val)
    
    ba = balanced_accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    fold_bal_accs.append(ba)
    fold_f1s.append(f1)
    print(f"  [SupCon_LR] Fold {fold_i+1}/{N_SPLITS} | BalAcc={ba:.4f} | MacroF1={f1:.4f}")

elapsed = time.time() - t0
mean_ba = float(np.mean(fold_bal_accs))
std_ba = float(np.std(fold_bal_accs))
macro_f1 = float(np.mean(fold_f1s))
print(f"\n  → SupCon_LR: BalAcc={mean_ba:.4f} ± {std_ba:.4f} | F1={macro_f1:.4f} | vs raw MiniLM (0.1363): {'+' if mean_ba > 0.1363 else ''}{mean_ba - 0.1363:.4f} | {elapsed:.1f}s\n")

# Save SupCon OOF projections
np.save(os.path.join(OUTPUTS_DIR, "embeddings_supcon_oof.npy"), supcon_embeddings)

# %% — Summary & Record Saving
record = {
    "experiment_id": "05_supcon_projection_lr",
    "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "research_question": "Can Supervised Contrastive Learning (SupCon) overcome semantic collapse for LLM attribution?",
    "hypothesis": "SupCon optimizes embedding geometry to cluster by model author rather than prompt topic",
    "research_reference": "Khosla et al. (2020) Supervised Contrastive Learning; Uchendu et al. (2020)",
    "dataset_version": "v1",
    "feature_method": "MiniLM-384 -> SupCon Head (256-128) -> LR",
    "model": "SupCon_Head_LR",
    "hyperparameters": {"temp": 0.07, "epochs": 45, "lr": 2e-3, "n_splits": N_SPLITS, "random_seed": RANDOM_SEED},
    "cv_strategy": "GroupKFold(n_splits=5, group=prompt_id)",
    "group_column": "prompt_id",
    "random_seed": RANDOM_SEED,
    "fold_scores": fold_bal_accs,
    "mean_balanced_accuracy": mean_ba,
    "std_balanced_accuracy": std_ba,
    "macro_f1": macro_f1,
    "per_class_recall": None,
    "training_time_sec": round(elapsed, 1),
    "status": "completed",
    "conclusion": f"BalAcc={mean_ba:.4f} ± {std_ba:.4f}",
    "next_recommendation": "If SupCon beat raw MiniLM (0.1363), test fusion of SupCon representations with TF-IDF and Stylometry.",
}

log_path = os.path.join(EXPERIMENTS_DIR, "experiment_log.jsonl")
with open(log_path, "a") as f:
    f.write(json.dumps(record) + "\n")

print(f"✅ Saved experiment record to {log_path}")
