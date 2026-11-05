# %% [markdown]
"""
# 06 — Transformer Representation Analysis: Layers & Pooling
EXPERIMENT_ID: 06_transformer_layers
DATE: 2026-11-05
RESEARCH_QUESTION: Which transformer layers (early vs middle vs late) and pooling methods
                   (mean, standard deviation, max, multi-pool) retain the most discriminative
                   stylistic signal for LLM attribution on unseen prompts?
HYPOTHESIS: Higher transformer layers are optimized for task-invariant semantics, which
            conflates LLM style with prompt topic. Intermediate layers (Layer 2–3) and
            token-variance pooling (mean + std pooling) preserve syntactic rhythms and
            token burstiness, providing higher balanced accuracy under GroupKFold than
            final-layer mean-pooling alone.
DATA_VERSION: v1 (original Kaggle release)
CV_STRATEGY: GroupKFold(n_splits=5, group=prompt_id)
GROUP_COLUMN: prompt_id
RANDOM_SEED: 42
MODELS: LogisticRegression (class_weight='balanced')
FEATURES: Multi-layer hidden states from sentence-transformers/all-MiniLM-L6-v2:
          - Layer 1 (Early/Morphological)
          - Layer 3 (Intermediate/Syntactic)
          - Layer 6 (Late/Semantic)
          - Multi-pooling (Mean + Std Pooling)
HYPERPARAMETERS: C=1.0, solver='lbfgs', max_iter=2000
COMPUTE_CONFIGURATION: GPU (RTX 3060 12GB) for layer extraction + CPU multi-core for CV
PREVIOUS_BEST_SCORE: 0.3178 (04_04b_full_fusion_dense)
"""

# %% — Imports & Setup
import sys, time, json, datetime, os
sys.path.insert(0, "/home/rj2826/name-the-llm/src")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score

TRAIN_PATH = "/home/rj2826/name-the-llm/data/train.csv"
EXPERIMENTS_DIR = "/home/rj2826/name-the-llm/experiments"
OUTPUTS_DIR = "/home/rj2826/name-the-llm/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
PREV_BEST = 0.3178

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Previous Best: {PREV_BEST:.4f}")

# %% — Load dataset
train = pd.read_csv(TRAIN_PATH)
texts = train["text"].tolist()
y = train["model"].values
groups = train["prompt_id"].values
classes = sorted(np.unique(y))
print(f"Loaded {len(train)} samples across {len(classes)} classes.")

# %% — Multi-Layer Hidden State Extraction Function
CACHE_PATH_L1 = os.path.join(OUTPUTS_DIR, "minilm_layer1_mean.npy")
CACHE_PATH_L3 = os.path.join(OUTPUTS_DIR, "minilm_layer3_mean.npy")
CACHE_PATH_L6 = os.path.join(OUTPUTS_DIR, "minilm_layer6_mean.npy")
CACHE_PATH_L6_MEAN_STD = os.path.join(OUTPUTS_DIR, "minilm_layer6_mean_std.npy")

if (os.path.exists(CACHE_PATH_L1) and os.path.exists(CACHE_PATH_L3) and 
    os.path.exists(CACHE_PATH_L6) and os.path.exists(CACHE_PATH_L6_MEAN_STD)):
    print("Loading cached layer representations...")
    feat_l1 = np.load(CACHE_PATH_L1)
    feat_l3 = np.load(CACHE_PATH_L3)
    feat_l6 = np.load(CACHE_PATH_L6)
    feat_l6_mean_std = np.load(CACHE_PATH_L6_MEAN_STD)
    print("Loaded all representations successfully.")
else:
    print("Extracting intermediate hidden states from sentence-transformers/all-MiniLM-L6-v2 on GPU...")
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    
    batch_size = 64
    all_l1, all_l3, all_l6, all_l6_mean_std = [], [], [], []
    t0 = time.time()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tok(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            mask = encoded["attention_mask"].unsqueeze(-1)
            
            outputs = model(**encoded, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Helper for masked mean pooling
            def mean_pool(layer_tensor):
                masked = layer_tensor * mask
                return (masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-7)).cpu().numpy()
            
            # Helper for masked mean + std pooling
            def mean_std_pool(layer_tensor):
                masked = layer_tensor * mask
                mean = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-7)
                diff = (layer_tensor - mean.unsqueeze(1)) * mask
                var = (diff ** 2).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-7)
                std = torch.sqrt(var + 1e-7)
                return torch.cat([mean, std], dim=1).cpu().numpy()
            
            all_l1.append(mean_pool(hidden_states[1]))
            all_l3.append(mean_pool(hidden_states[3]))
            all_l6.append(mean_pool(hidden_states[6]))
            all_l6_mean_std.append(mean_std_pool(hidden_states[6]))
            
    feat_l1 = np.vstack(all_l1)
    feat_l3 = np.vstack(all_l3)
    feat_l6 = np.vstack(all_l6)
    feat_l6_mean_std = np.vstack(all_l6_mean_std)
    
    np.save(CACHE_PATH_L1, feat_l1)
    np.save(CACHE_PATH_L3, feat_l3)
    np.save(CACHE_PATH_L6, feat_l6)
    np.save(CACHE_PATH_L6_MEAN_STD, feat_l6_mean_std)
    print(f"Extraction complete in {time.time() - t0:.2f}s | Saved all representations.")

# %% — Evaluation function
kf = GroupKFold(n_splits=N_SPLITS)
splits = list(kf.split(train, y, groups=groups))

def evaluate_layer_features(features, label):
    fold_bal_accs, fold_f1s = [], []
    t0 = time.time()
    
    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = features[tr_idx], features[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_val_s)
        
        ba = balanced_accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        fold_bal_accs.append(ba)
        fold_f1s.append(f1)
        print(f"  [{label}] Fold {fold_i+1}/{N_SPLITS} | BalAcc={ba:.4f} | MacroF1={f1:.4f}")
        
    elapsed = time.time() - t0
    result = {
        "label": label,
        "fold_scores": fold_bal_accs,
        "mean_balanced_accuracy": float(np.mean(fold_bal_accs)),
        "std_balanced_accuracy": float(np.std(fold_bal_accs)),
        "macro_f1": float(np.mean(fold_f1s)),
        "training_time_sec": round(elapsed, 1),
    }
    print(f"  → {label}: BalAcc={result['mean_balanced_accuracy']:.4f} ± {result['std_balanced_accuracy']:.4f} | F1={result['macro_f1']:.4f} | {elapsed:.1f}s\n")
    return result

results = {}

# %% ─────────────────────────────────────────────────
# EXP 06a: Layer 1 (Early Morphological Representation)
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 06a — Layer 1 (Early Morphological / Surface Features)")
print("=" * 60)
results["06a_layer1"] = evaluate_layer_features(feat_l1, "MiniLM_Layer1_Mean")

# %% ─────────────────────────────────────────────────
# EXP 06b: Layer 3 (Intermediate Syntactic Representation)
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 06b — Layer 3 (Intermediate Syntactic Structure)")
print("=" * 60)
results["06b_layer3"] = evaluate_layer_features(feat_l3, "MiniLM_Layer3_Mean")

# %% ─────────────────────────────────────────────────
# EXP 06c: Layer 6 (Final Semantic Representation)
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 06c — Layer 6 (Final Semantic Layer)")
print("=" * 60)
results["06c_layer6"] = evaluate_layer_features(feat_l6, "MiniLM_Layer6_Mean")

# %% ─────────────────────────────────────────────────
# EXP 06d: Layer 6 Multi-Pooling (Mean + Std Pooling, 768-dim)
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 06d — Layer 6 Multi-Pooling (Mean + Std Token Variance, 768-d)")
print("=" * 60)
results["06d_mean_std_pool"] = evaluate_layer_features(feat_l6_mean_std, "MiniLM_Layer6_MeanStd")

# %% ─────────────────────────────────────────────────
# EXP 06e: Layer 3 + Layer 6 Concatenation (Syntax + Semantics)
# ─────────────────────────────────────────────────────
print("=" * 60)
print("EXP 06e — Multi-Layer Fusion (Layer 3 Syntax + Layer 6 Semantics, 768-d)")
print("=" * 60)
feat_l3_l6 = np.hstack([feat_l3, feat_l6])
results["06e_layer3_layer6_concat"] = evaluate_layer_features(feat_l3_l6, "MiniLM_Layer3_Layer6_Concat")

# %% — Results Summary
print("=" * 75)
print("RESULTS SUMMARY (06_transformer_layers)")
print("=" * 75)
print(f"{'Model / Layer':<35} {'BalAcc':>8} {'±':>7}  {'MacroF1':>8}  {'Time(s)':>8}")
print("-" * 75)
for k, r in results.items():
    print(f"{r['label']:<35} {r['mean_balanced_accuracy']:>8.4f} ±{r['std_balanced_accuracy']:.4f}  "
          f"{r['macro_f1']:>8.4f}  {r['training_time_sec']:>7.1f}")

best_key = max(results, key=lambda k: results[k]["mean_balanced_accuracy"])
best = results[best_key]
print(f"\n🏆 Best in 06: {best['label']} | BalAcc={best['mean_balanced_accuracy']:.4f} ± {best['std_balanced_accuracy']:.4f}")

# Append to experiment_log.jsonl
log_path = os.path.join(EXPERIMENTS_DIR, "experiment_log.jsonl")
existing_ids = set()
if os.path.exists(log_path):
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                existing_ids.add(json.loads(line).get("experiment_id", ""))

saved = 0
with open(log_path, "a") as f:
    for key, r in results.items():
        exp_id = f"06_{key}"
        if exp_id not in existing_ids:
            record = {
                "experiment_id": exp_id,
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "research_question": "Which transformer layers and pooling mechanisms preserve LLM stylistic signals?",
                "hypothesis": "Intermediate layers and standard deviation pooling retain token activation variance",
                "research_reference": "Jawahar et al. (2019) What does BERT learn about surface syntax?",
                "dataset_version": "v1",
                "feature_method": r["label"],
                "model": "LogisticRegression(balanced)",
                "hyperparameters": {"n_splits": N_SPLITS, "random_seed": RANDOM_SEED},
                "cv_strategy": "GroupKFold(n_splits=5, group=prompt_id)",
                "group_column": "prompt_id",
                "random_seed": RANDOM_SEED,
                "fold_scores": r["fold_scores"],
                "mean_balanced_accuracy": r["mean_balanced_accuracy"],
                "std_balanced_accuracy": r["std_balanced_accuracy"],
                "macro_f1": r["macro_f1"],
                "per_class_recall": None,
                "training_time_sec": r["training_time_sec"],
                "status": "completed",
                "conclusion": f"BalAcc={r['mean_balanced_accuracy']:.4f} ± {r['std_balanced_accuracy']:.4f}",
                "next_recommendation": "Compare layer representations vs TF-IDF to design optimal ensembling in 08_ensemble.py",
            }
            f.write(json.dumps(record) + "\n")
            saved += 1

print(f"\n✅ Saved {saved} experiment records → {log_path}")
