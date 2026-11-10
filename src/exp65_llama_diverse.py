"""
================================================================================
EXP65: LLAMA-3.1-8B DIVERSE BACKBONE COMPLEMENTARITY EXPERIMENT
================================================================================
Investigates whether a genuinely distinct foundation model family
(Llama-3.1-8B Base via QLoRA) provides complementary attribution signals
that the current Qwen3-8B + Qwen3-14B dual-backbone ensemble (EXP52)
does not already contain.

Safety Rules:
- ZERO Kaggle submissions.
- EXP52 model weights, code, and artifacts remain 100% frozen and unmodified.
- EXP64 scientific results remain unchanged.
- STRICT SINGLE-SAMPLE INFERENCE: Zero prompt IDs, zero prompt centroids,
  zero LOO centering, zero Hungarian matching, zero pairwise peer comparisons.
- Strict 5-fold prompt-disjoint cross-validation (150 prompts, 30 prompts/fold).
- Nested cross-validation (5 outer folds x 4 inner folds) for selection-safe evaluation.
- Paired bootstrap statistical tests (2,000 resamples).
- Publication-quality 300-DPI figures.
================================================================================
"""

import os
import sys
import gc
import json
import time
import hashlib
import pathlib
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, log_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Set offline environment variables for strict reproducibility
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Headless matplotlib
plt.switch_backend("Agg")

# Paths
REPO_DIR = pathlib.Path("/home/rj2826/name-that-llm")
DATA_DIR = REPO_DIR / "data"
OUTPUTS_DIR = REPO_DIR / "outputs"
MODELS_DIR = REPO_DIR / "models"
EXP_DIR = REPO_DIR / "experiments" / "EXP65_llama_diverse"
PLOTS_DIR = REPO_DIR / "plots"
REPORTS_DIR = REPO_DIR / "reports"

EXP_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(EXP_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def compute_sha256(file_path: pathlib.Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def balanced_oracle(pred_list: list, y: np.ndarray, n_classes: int = 18) -> float:
    """Calculates macro-averaged balanced accuracy of an oracle chooser among models."""
    correct_any = np.zeros(len(y), dtype=bool)
    for pred in pred_list:
        correct_any = correct_any | (pred == y)
    recalls = []
    for k in range(n_classes):
        mask = (y == k)
        if mask.sum() > 0:
            recalls.append(float(correct_any[mask].mean()))
        else:
            recalls.append(0.0)
    return float(np.mean(recalls))


def paired_bootstrap(y_true: np.ndarray, pred_base: np.ndarray, pred_test: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """Computes paired bootstrap confidence interval for delta = BA(test) - BA(base)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ba_b = balanced_accuracy_score(y_true[idx], pred_base[idx])
        ba_t = balanced_accuracy_score(y_true[idx], pred_test[idx])
        deltas[b] = ba_t - ba_b
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_positive = float(np.mean(deltas > 0))
    mean_delta = float(np.mean(deltas))
    return mean_delta, ci_lower, ci_upper, p_positive, deltas


def evaluate_calibration(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """Evaluates NLL, Brier score, ECE, and Balanced Accuracy for multiclass probabilities."""
    nll = float(log_loss(y_true, probs))
    n_classes = probs.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot)**2, axis=1)))

    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs, bin_accs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        cnt = mask.sum()
        bin_counts.append(int(cnt))
        if cnt > 0:
            mean_acc = float(acc[mask].mean())
            mean_conf = float(conf[mask].mean())
            ece += (cnt / len(y_true)) * abs(mean_acc - mean_conf)
            bin_accs.append(mean_acc)
            bin_confs.append(mean_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.5 * (bins[i] + bins[i + 1]))

    ba = float(balanced_accuracy_score(y_true, preds))
    return {
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "ba": ba,
        "bin_confs": bin_confs,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts
    }


def main():
    start_total_time = time.time()
    log("=====================================================================")
    log("EXP65: LLAMA-3.1-8B DIVERSE BACKBONE COMPLEMENTARITY STARTED")
    log("=====================================================================")

    # -------------------------------------------------------------------------
    # PHASE 1 & 29: REPOSITORY & ARTIFACT AUDIT (PRE-EXPERIMENT INTEGRITY)
    # -------------------------------------------------------------------------
    log("\n--- Phase 1: Repository & Baseline Integrity Audit ---")
    artifact_paths = {
        "train_csv": DATA_DIR / "train.csv",
        "test_csv": DATA_DIR / "test.csv",
        "sample_sub_parquet": DATA_DIR / "sample_submission.parquet",
        "open_pool_clean_parquet": DATA_DIR / "external" / "open_pool_clean.parquet",
        "q14_oof_csv": OUTPUTS_DIR / "oof_q14v5_antopil_lora.csv",
        "q8_twin_oof_npy": OUTPUTS_DIR / "oof_exp47_qwen_centered_twin.npy",
        "q8_plain_oof_npy": OUTPUTS_DIR / "oof_exp46_qwen_lora.npy",
        "exp52_kernel_nb": REPO_DIR / "notebooks" / "52_dual_lora_kernel.ipynb",
        "exp52_metrics_json": REPO_DIR / "experiments" / "EXP52_dual_lora" / "metrics.json",
        "exp64_metrics_json": REPO_DIR / "experiments" / "EXP64_confidence_router" / "metrics.json",
    }

    artifact_hashes = {}
    for key, path in artifact_paths.items():
        if path.exists():
            sha = compute_sha256(path)
            size = path.stat().st_size
            artifact_hashes[key] = {
                "path": str(path),
                "size_bytes": size,
                "sha256": sha
            }
            log(f"Verified {key}: {size:,} bytes | sha256={sha[:16]}...")
        else:
            log(f"WARNING: Missing expected artifact {key} at {path}")

    with open(EXP_DIR / "artifact_hashes.json", "w") as f:
        json.dump(artifact_hashes, f, indent=2)

    # -------------------------------------------------------------------------
    # PHASE 4: DATA AUDIT & LOAD BASELINE PREDICTIONS
    # -------------------------------------------------------------------------
    log("\n--- Phase 4: Data & Baseline Verification ---")
    train_df = pd.read_csv(artifact_paths["train_csv"])
    test_df = pd.read_csv(artifact_paths["test_csv"])
    q14_df = pd.read_csv(artifact_paths["q14_oof_csv"])
    p8_twin_raw = np.load(artifact_paths["q8_twin_oof_npy"])

    n_samples = len(train_df)
    assert n_samples == 2699, f"Expected 2699 train rows, got {n_samples}"
    LABELS = sorted(train_df["model"].unique())
    assert len(LABELS) == 18, f"Expected 18 classes, got {len(LABELS)}"
    L2I = {m: i for i, m in enumerate(LABELS)}
    y_true = np.array([L2I[m] for m in train_df["model"].values])
    prompts = train_df["prompt_id"].values
    outer_folds = q14_df["fold"].values
    eps = 1e-15

    # Normalize frozen baseline probabilities
    p8 = p8_twin_raw / p8_twin_raw.sum(axis=1, keepdims=True)
    p14 = q14_df[LABELS].values / q14_df[LABELS].values.sum(axis=1, keepdims=True)
    lp8 = np.log(np.clip(p8, eps, 1.0))
    lp14 = np.log(np.clip(p14, eps, 1.0))

    pred_q8 = lp8.argmax(axis=1)
    pred_q14 = lp14.argmax(axis=1)
    lp52 = 0.60 * lp8 + 0.40 * lp14
    pred_exp52 = lp52.argmax(axis=1)
    p52 = scipy.special.softmax(lp52, axis=1)

    ba_q8 = float(balanced_accuracy_score(y_true, pred_q8))
    ba_q14 = float(balanced_accuracy_score(y_true, pred_q14))
    ba_exp52 = float(balanced_accuracy_score(y_true, pred_exp52))
    f1_exp52 = float(f1_score(y_true, pred_exp52, average="macro"))

    log(f"EXP52 Baseline Reproduction: OOF BA = {ba_exp52:.6f} (Macro-F1: {f1_exp52:.6f})")
    log(f"Q8 Twin Baseline: BA = {ba_q8:.6f} | Q14 Baseline: BA = {ba_q14:.6f}")
    assert abs(ba_exp52 - 0.787313) < 1e-5, f"EXP52 reproduction mismatch: {ba_exp52}"

    # -------------------------------------------------------------------------
    # PHASE 6: PROMPT-DISJOINT CV VERIFICATION & SAVE
    # -------------------------------------------------------------------------
    log("\n--- Phase 6: Prompt-Disjoint Cross-Validation Setup ---")
    fold_df = pd.DataFrame({
        "sample_id": train_df["id"],
        "prompt_id": train_df["prompt_id"],
        "fold": outer_folds,
        "model": train_df["model"]
    })
    fold_df.to_csv(EXP_DIR / "fold_assignments.csv", index=False)

    for f in range(5):
        p_train = set(fold_df[fold_df["fold"] != f]["prompt_id"].unique())
        p_val = set(fold_df[fold_df["fold"] == f]["prompt_id"].unique())
        overlap = p_train.intersection(p_val)
        assert len(overlap) == 0, f"Fold {f} prompt leakage detected: {overlap}"
        assert len(p_val) == 30, f"Fold {f} expected 30 prompts, got {len(p_val)}"
    log("All 5 folds strictly prompt-disjoint (30 prompts per fold, 0 leakage).")

    # -------------------------------------------------------------------------
    # PHASE 8 & 10: 5-FOLD LLAMA-3.1-8B TRAINING & OOF PREDICTION
    # -------------------------------------------------------------------------
    log("\n--- Phase 8 & 10: 5-Fold Llama-3.1-8B QLoRA Training ---")
    model_id = "meta-llama/Llama-3.1-8B"
    max_len = 384
    batch_size = 4
    grad_accum_steps = 2
    epochs = 2
    lr = 2e-4

    oof_logits = np.zeros((n_samples, 18), dtype=float)
    oof_probs = np.zeros((n_samples, 18), dtype=float)
    oof_preds = np.zeros(n_samples, dtype=int)
    fold_metrics = []

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Pre-tokenize full dataset once to save time
    log(f"Pre-tokenizing {n_samples} passages with max_length={max_len}...")
    encoded_all = tok(
        train_df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    max_memory = {0: "8GiB", 1: "10GiB"}

    for fold_idx in range(5):
        log(f"\n=======================================================")
        log(f"STARTING FOLD {fold_idx + 1}/5 (Prompt-Disjoint)")
        log(f"=======================================================")
        fold_start = time.time()

        val_mask = (outer_folds == fold_idx)
        train_mask = ~val_mask

        val_indices = np.where(val_mask)[0]
        train_indices = np.where(train_mask)[0]

        log(f"Train samples: {len(train_indices)} (from 120 prompts) | Val samples: {len(val_indices)} (from 30 prompts)")

        # Create Model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=18,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            dtype=torch.bfloat16,
            local_files_only=True
        )
        model.config.pad_token_id = tok.pad_token_id
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)

        # Datasets & Loaders
        class AttributionDataset(torch.utils.data.Dataset):
            def __init__(self, indices):
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                return {
                    "input_ids": encoded_all["input_ids"][real_idx],
                    "attention_mask": encoded_all["attention_mask"][real_idx],
                    "labels": torch.tensor(y_true[real_idx], dtype=torch.long)
                }

        train_dataset = AttributionDataset(train_indices)
        val_dataset = AttributionDataset(val_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            drop_last=False
        )

        total_steps = (len(train_loader) // grad_accum_steps) * epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        # Training loop
        model.train()
        global_step = 0
        optimizer.zero_grad()

        for ep in range(epochs):
            ep_loss = 0.0
            n_batches = 0
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
                loss.backward()

                ep_loss += outputs.loss.item()
                n_batches += 1

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (step + 1) % 100 == 0:
                    log(f"Fold {fold_idx + 1} | Epoch {ep + 1}/{epochs} | Step {step + 1}/{len(train_loader)} | Mean Loss: {ep_loss / n_batches:.4f}")

            log(f"Fold {fold_idx + 1} | Epoch {ep + 1}/{epochs} Complete. Average Loss: {ep_loss / n_batches:.4f}")

        # Save LoRA adapter checkpoint for reproducibility
        ckpt_dir = EXP_DIR / "checkpoints" / f"fold_{fold_idx}"
        model.save_pretrained(ckpt_dir)
        log(f"Saved Fold {fold_idx + 1} checkpoint to {ckpt_dir}")

        # Validation Inference
        model.eval()
        val_logits_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
                out = model(**batch)
                val_logits_list.append(out.logits.detach().cpu().numpy())

        val_logits = np.concatenate(val_logits_list, axis=0)
        val_probs = scipy.special.softmax(val_logits, axis=1)
        val_preds = val_probs.argmax(axis=1)

        oof_logits[val_indices] = val_logits
        oof_probs[val_indices] = val_probs
        oof_preds[val_indices] = val_preds

        f_ba = float(balanced_accuracy_score(y_true[val_indices], val_preds))
        f_f1 = float(f1_score(y_true[val_indices], val_preds, average="macro"))
        f_acc = float(accuracy_score(y_true[val_indices], val_preds))
        f_52_ba = float(balanced_accuracy_score(y_true[val_indices], pred_exp52[val_indices]))
        f_q8_ba = float(balanced_accuracy_score(y_true[val_indices], pred_q8[val_indices]))
        f_q14_ba = float(balanced_accuracy_score(y_true[val_indices], pred_q14[val_indices]))

        elapsed_fold = time.time() - fold_start
        log(f"Fold {fold_idx + 1} Evaluation: Llama BA = {f_ba:.4f} (F1: {f_f1:.4f}, Acc: {f_acc:.4f}) | EXP52 BA = {f_52_ba:.4f} [Elapsed: {elapsed_fold:.1f}s]")

        fold_metrics.append({
            "fold": fold_idx,
            "llama_ba": f_ba,
            "llama_f1": f_f1,
            "llama_acc": f_acc,
            "exp52_ba": f_52_ba,
            "q8_ba": f_q8_ba,
            "q14_ba": f_q14_ba,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "duration_seconds": elapsed_fold
        })

        # Memory Cleanup
        del model, optimizer, lr_scheduler, train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()

    # Save Fold Results CSV
    df_folds = pd.DataFrame(fold_metrics)
    df_folds.to_csv(EXP_DIR / "fold_results.csv", index=False)
    log(f"\nSaved fold_results.csv:")
    print(df_folds)

    # -------------------------------------------------------------------------
    # PHASE 10: OVERALL OOF EVALUATION & METRICS
    # -------------------------------------------------------------------------
    log("\n--- Phase 10: Out-of-Fold Evaluation ---")
    ba_llama = float(balanced_accuracy_score(y_true, oof_preds))
    f1_llama = float(f1_score(y_true, oof_preds, average="macro"))
    acc_llama = float(accuracy_score(y_true, oof_preds))
    nll_llama = float(log_loss(y_true, oof_probs))

    conf_llama = oof_probs.max(axis=1)
    sort_probs = np.sort(oof_probs, axis=1)
    margin_llama = sort_probs[:, -1] - sort_probs[:, -2]
    ent_llama = -np.sum(oof_probs * np.log(np.clip(oof_probs, eps, 1.0)), axis=1)

    log(f"Llama-3.1-8B Standalone OOF: BA = {ba_llama:.4f} | Macro-F1 = {f1_llama:.4f} | Acc = {acc_llama:.4f} | NLL = {nll_llama:.4f}")
    log(f"EXP52 Champion Baseline:    BA = {ba_exp52:.4f} | Macro-F1 = {f1_exp52:.4f}")

    # Build and Save oof_predictions.csv
    oof_dict = {
        "sample_id": train_df["id"].values,
        "prompt_id": prompts,
        "fold": outer_folds,
        "true_label": [LABELS[y] for y in y_true],
        "true_class_idx": y_true,
    }
    for c_idx, c_name in enumerate(LABELS):
        oof_dict[f"llama_logits_{c_idx}"] = oof_logits[:, c_idx]
        oof_dict[f"llama_prob_{c_idx}"] = oof_probs[:, c_idx]
    oof_dict["llama_pred"] = [LABELS[p] for p in oof_preds]
    oof_dict["llama_pred_idx"] = oof_preds
    oof_dict["llama_confidence"] = conf_llama
    oof_dict["llama_margin"] = margin_llama
    oof_dict["llama_entropy"] = ent_llama

    df_oof = pd.DataFrame(oof_dict)
    df_oof.to_csv(EXP_DIR / "oof_predictions.csv", index=False)
    log(f"Saved oof_predictions.csv ({len(df_oof)} rows).")

    # -------------------------------------------------------------------------
    # PHASE 11: PRIMARY COMPLEMENTARITY TEST (MEASURED ERROR CORRELATION)
    # -------------------------------------------------------------------------
    log("\n--- Phase 11: Complementarity & Error Correlation Analysis ---")
    err_q8 = (pred_q8 != y_true).astype(float)
    err_q14 = (pred_q14 != y_true).astype(float)
    err_52 = (pred_exp52 != y_true).astype(float)
    err_llama = (oof_preds != y_true).astype(float)

    corr_q8_llama, p_q8_llama = scipy.stats.pearsonr(err_q8, err_llama)
    corr_q14_llama, p_q14_llama = scipy.stats.pearsonr(err_q14, err_llama)
    corr_52_llama, p_52_llama = scipy.stats.pearsonr(err_52, err_llama)
    corr_q8_q14, p_q8_q14 = scipy.stats.pearsonr(err_q8, err_q14)

    log(f"Measured Error Correlation (Q8 vs Q14):     r = {corr_q8_q14:.4f}")
    log(f"Measured Error Correlation (Q8 vs Llama):   r = {corr_q8_llama:.4f} (p={p_q8_llama:.2e})")
    log(f"Measured Error Correlation (Q14 vs Llama):  r = {corr_q14_llama:.4f} (p={p_q14_llama:.2e})")
    log(f"Measured Error Correlation (EXP52 vs Llama):r = {corr_52_llama:.4f} (p={p_52_llama:.2e})")

    # -------------------------------------------------------------------------
    # PHASE 12: UNIQUE CORRECTION CAPACITY
    # -------------------------------------------------------------------------
    log("\n--- Phase 12: Unique Correction Capacity ---")
    llama_corr_52_wrong = int(((oof_preds == y_true) & (pred_exp52 != y_true)).sum())
    llama_wrong_52_corr = int(((oof_preds != y_true) & (pred_exp52 == y_true)).sum())
    both_correct = int(((oof_preds == y_true) & (pred_exp52 == y_true)).sum())
    both_wrong = int(((oof_preds != y_true) & (pred_exp52 != y_true)).sum())
    net_correction = llama_corr_52_wrong - llama_wrong_52_corr

    llama_corr_q8_wrong = int(((oof_preds == y_true) & (pred_q8 != y_true)).sum())
    llama_corr_q14_wrong = int(((oof_preds == y_true) & (pred_q14 != y_true)).sum())
    llama_corr_both_qwen_wrong = int(((oof_preds == y_true) & (pred_q8 != y_true) & (pred_q14 != y_true)).sum())

    log(f"Llama Unique Corrections (Llama correct & EXP52 wrong): {llama_corr_52_wrong} ({llama_corr_52_wrong/n_samples*100:.2f}%)")
    log(f"Llama Degradations (Llama wrong & EXP52 correct):        {llama_wrong_52_corr} ({llama_wrong_52_corr/n_samples*100:.2f}%)")
    log(f"Net Correction over EXP52:                              {net_correction:+d} ({net_correction/n_samples*100:+.2f}%)")
    log(f"Llama correct when Q8 wrong:                            {llama_corr_q8_wrong}")
    log(f"Llama correct when Q14 wrong:                           {llama_corr_q14_wrong}")
    log(f"Llama correct when BOTH Qwen wrong:                     {llama_corr_both_qwen_wrong}")

    df_comp = pd.DataFrame([{
        "metric": "error_corr_q8_llama", "value": corr_q8_llama, "p_value": p_q8_llama
    }, {
        "metric": "error_corr_q14_llama", "value": corr_q14_llama, "p_value": p_q14_llama
    }, {
        "metric": "error_corr_exp52_llama", "value": corr_52_llama, "p_value": p_52_llama
    }, {
        "metric": "error_corr_q8_q14", "value": corr_q8_q14, "p_value": p_q8_q14
    }, {
        "metric": "llama_unique_corrections", "value": llama_corr_52_wrong, "proportion": llama_corr_52_wrong / n_samples
    }, {
        "metric": "llama_degradations", "value": llama_wrong_52_corr, "proportion": llama_wrong_52_corr / n_samples
    }, {
        "metric": "net_correction", "value": net_correction, "proportion": net_correction / n_samples
    }, {
        "metric": "llama_corr_both_qwen_wrong", "value": llama_corr_both_qwen_wrong, "proportion": llama_corr_both_qwen_wrong / n_samples
    }])
    df_comp.to_csv(EXP_DIR / "complementarity.csv", index=False)

    # -------------------------------------------------------------------------
    # PHASE 13: THREE-MODEL ORACLE ANALYSIS
    # -------------------------------------------------------------------------
    log("\n--- Phase 13: Three-Model Oracle Analysis ---")
    oracle_q8_q14 = balanced_oracle([pred_q8, pred_q14], y_true)
    oracle_3model = balanced_oracle([pred_q8, pred_q14, oof_preds], y_true)
    oracle_exp52_llama = balanced_oracle([pred_exp52, oof_preds], y_true)
    oracle_improvement = oracle_3model - oracle_q8_q14

    log(f"Oracle(Q8, Q14):           BA = {oracle_q8_q14:.4f}")
    log(f"Oracle(Q8, Q14, Llama):     BA = {oracle_3model:.4f}")
    log(f"Oracle(EXP52, Llama):       BA = {oracle_exp52_llama:.4f}")
    log(f"Three-Model Oracle Gain:   Delta = {oracle_improvement:+.4f}")

    df_oracle = pd.DataFrame([
        {"setup": "Oracle(Q8, Q14)", "balanced_accuracy": oracle_q8_q14},
        {"setup": "Oracle(Q8, Q14, Llama)", "balanced_accuracy": oracle_3model},
        {"setup": "Oracle(EXP52, Llama)", "balanced_accuracy": oracle_exp52_llama},
        {"setup": "Oracle_Gain_over_Q8_Q14", "balanced_accuracy": oracle_improvement}
    ])
    df_oracle.to_csv(EXP_DIR / "oracle_results.csv", index=False)

    # -------------------------------------------------------------------------
    # PHASE 14: EXP64 DISAGREEMENT SUBSET (N = 1612)
    # -------------------------------------------------------------------------
    log("\n--- Phase 14: EXP64 Disagreement Subset Analysis (N = 1612) ---")
    disagree_mask = (pred_q8 != pred_q14)
    n_dis = int(disagree_mask.sum())
    assert n_dis == 1612, f"Expected exactly 1612 disagreement samples, found {n_dis}"

    y_dis = y_true[disagree_mask]
    q8_dis_ba = float(balanced_accuracy_score(y_dis, pred_q8[disagree_mask]))
    q14_dis_ba = float(balanced_accuracy_score(y_dis, pred_q14[disagree_mask]))
    exp52_dis_ba = float(balanced_accuracy_score(y_dis, pred_exp52[disagree_mask]))
    llama_dis_ba = float(balanced_accuracy_score(y_dis, oof_preds[disagree_mask]))
    oracle_dis_3model = balanced_oracle([pred_q8[disagree_mask], pred_q14[disagree_mask], oof_preds[disagree_mask]], y_dis)

    fixes_in_dis = int(((oof_preds[disagree_mask] == y_dis) & (pred_exp52[disagree_mask] != y_dis)).sum())
    harms_in_dis = int(((oof_preds[disagree_mask] != y_dis) & (pred_exp52[disagree_mask] == y_dis)).sum())
    net_in_dis = fixes_in_dis - harms_in_dis

    log(f"Disagreement Subset (N = {n_dis}):")
    log(f"  Q8 BA:     {q8_dis_ba:.4f}")
    log(f"  Q14 BA:    {q14_dis_ba:.4f}")
    log(f"  EXP52 BA:  {exp52_dis_ba:.4f}")
    log(f"  Llama BA:  {llama_dis_ba:.4f}")
    log(f"  Oracle BA: {oracle_dis_3model:.4f}")
    log(f"  Llama fixes EXP52: {fixes_in_dis} | Llama harms EXP52: {harms_in_dis} | Net: {net_in_dis:+d}")

    df_dis = pd.DataFrame([{
        "n_samples": n_dis,
        "q8_ba": q8_dis_ba,
        "q14_ba": q14_dis_ba,
        "exp52_ba": exp52_dis_ba,
        "llama_ba": llama_dis_ba,
        "oracle_3model_ba": oracle_dis_3model,
        "llama_fixes_exp52": fixes_in_dis,
        "llama_harms_exp52": harms_in_dis,
        "net_disagreement_correction": net_in_dis
    }])
    df_dis.to_csv(EXP_DIR / "disagreement_analysis.csv", index=False)

    # -------------------------------------------------------------------------
    # PHASE 15: CONFIDENCE ANALYSIS
    # -------------------------------------------------------------------------
    log("\n--- Phase 15: Confidence & Margin Distributions ---")
    groups = {
        "Both Correct": (oof_preds == y_true) & (pred_exp52 == y_true),
        "Llama Only Correct": (oof_preds == y_true) & (pred_exp52 != y_true),
        "EXP52 Only Correct": (oof_preds != y_true) & (pred_exp52 == y_true),
        "Both Wrong": (oof_preds != y_true) & (pred_exp52 != y_true)
    }

    conf_stats = []
    for g_name, g_mask in groups.items():
        cnt = int(g_mask.sum())
        if cnt > 0:
            c_mean = float(conf_llama[g_mask].mean())
            m_mean = float(margin_llama[g_mask].mean())
            e_mean = float(ent_llama[g_mask].mean())
            c52_mean = float(p52.max(axis=1)[g_mask].mean())
        else:
            c_mean, m_mean, e_mean, c52_mean = 0.0, 0.0, 0.0, 0.0
        log(f"Group '{g_name:18s}' (N={cnt:4d}): Llama Conf={c_mean:.4f}, Margin={m_mean:.4f}, EXP52 Conf={c52_mean:.4f}")
        conf_stats.append({
            "group": g_name, "count": cnt, "llama_mean_conf": c_mean,
            "llama_mean_margin": m_mean, "llama_mean_entropy": e_mean,
            "exp52_mean_conf": c52_mean
        })
    df_conf = pd.DataFrame(conf_stats)

    # -------------------------------------------------------------------------
    # PHASE 16: STATIC FUSION SWEEP
    # -------------------------------------------------------------------------
    log("\n--- Phase 16: Static Probability Fusion Sweep ---")
    weights = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    lp_llama = np.log(np.clip(oof_probs, eps, 1.0))
    fusion_results = []

    for w in weights:
        lp_fused = (1.0 - w) * lp52 + w * lp_llama
        pred_fused = lp_fused.argmax(axis=1)
        f_ba = float(balanced_accuracy_score(y_true, pred_fused))
        f_f1 = float(f1_score(y_true, pred_fused, average="macro"))
        delta = f_ba - ba_exp52
        fusion_results.append({
            "weight_llama": w,
            "weight_exp52": 1.0 - w,
            "balanced_accuracy": f_ba,
            "macro_f1": f_f1,
            "delta_vs_exp52": delta
        })
        log(f"Weight Llama = {w:.2f}: OOF BA = {f_ba:.4f} (Delta: {delta:+.4f})")

    df_fusion = pd.DataFrame(fusion_results)
    df_fusion.to_csv(EXP_DIR / "fusion_results.csv", index=False)

    # -------------------------------------------------------------------------
    # PHASE 17: NESTED CROSS-VALIDATION (5 OUTER x 4 INNER FOLDS)
    # -------------------------------------------------------------------------
    log("\n--- Phase 17: Selection-Safe Nested Cross-Validation ---")
    outer_nested_ba = []
    outer_selected_w = []
    outer_fused_preds = np.zeros(n_samples, dtype=int)

    for out_fold in range(5):
        val_idx = (outer_folds == out_fold)
        tr_idx = ~val_idx

        # Inner cross-validation on 4 inner folds (prompt-disjoint)
        inner_prompts = np.unique(prompts[tr_idx])
        inner_prompt_folds = {}
        np.random.seed(42)
        shuffled = np.random.permutation(inner_prompts)
        for i_p_idx, p in enumerate(shuffled):
            inner_prompt_folds[p] = i_p_idx % 4

        inner_folds_arr = np.array([inner_prompt_folds[p] for p in prompts[tr_idx]])

        best_w = 0.0
        best_inner_ba = -1.0

        for w_cand in weights:
            inner_ba_list = []
            for inn_f in range(4):
                inn_val = (inner_folds_arr == inn_f)
                inn_tr = ~inn_val
                # Evaluate on inner val
                lp_cand = (1.0 - w_cand) * lp52[tr_idx][inn_val] + w_cand * lp_llama[tr_idx][inn_val]
                p_cand = lp_cand.argmax(axis=1)
                ba_cand = balanced_accuracy_score(y_true[tr_idx][inn_val], p_cand)
                inner_ba_list.append(ba_cand)
            mean_inn_ba = float(np.mean(inner_ba_list))
            if mean_inn_ba > best_inner_ba:
                best_inner_ba = mean_inn_ba
                best_w = w_cand

        # Evaluate selected weight on held-out outer fold
        lp_out = (1.0 - best_w) * lp52[val_idx] + best_w * lp_llama[val_idx]
        pred_out = lp_out.argmax(axis=1)
        ba_out = float(balanced_accuracy_score(y_true[val_idx], pred_out))

        outer_nested_ba.append(ba_out)
        outer_selected_w.append(best_w)
        outer_fused_preds[val_idx] = pred_out

        log(f"Outer Fold {out_fold + 1}: Selected w* = {best_w:.2f} | Outer BA = {ba_out:.4f} (EXP52 = {fold_metrics[out_fold]['exp52_ba']:.4f})")

    pooled_nested_ba = float(balanced_accuracy_score(y_true, outer_fused_preds))
    mean_nested_ba = float(np.mean(outer_nested_ba))
    std_nested_ba = float(np.std(outer_nested_ba))
    delta_nested = pooled_nested_ba - ba_exp52

    log(f"Nested CV Result: Mean Fold BA = {mean_nested_ba:.4f} ± {std_nested_ba:.4f}")
    log(f"Pooled Nested OOF BA = {pooled_nested_ba:.4f} (Delta vs EXP52: {delta_nested:+.4f})")

    # -------------------------------------------------------------------------
    # PHASE 19: CLASS-WISE ANALYSIS
    # -------------------------------------------------------------------------
    log("\n--- Phase 19: Class-Wise Performance & Recall Analysis ---")
    class_stats = []
    for c_idx, c_name in enumerate(LABELS):
        c_mask = (y_true == c_idx)
        n_c = int(c_mask.sum())
        rec_q8 = float((pred_q8[c_mask] == c_idx).mean())
        rec_q14 = float((pred_q14[c_mask] == c_idx).mean())
        rec_52 = float((pred_exp52[c_mask] == c_idx).mean())
        rec_llama = float((oof_preds[c_mask] == c_idx).mean())
        rec_nested = float((outer_fused_preds[c_mask] == c_idx).mean())

        uniq_corr = int(((oof_preds[c_mask] == c_idx) & (pred_exp52[c_mask] != c_idx)).sum())
        degrads = int(((oof_preds[c_mask] != c_idx) & (pred_exp52[c_mask] == c_idx)).sum())

        class_stats.append({
            "class_name": c_name,
            "n_samples": n_c,
            "q8_recall": rec_q8,
            "q14_recall": rec_q14,
            "exp52_recall": rec_52,
            "llama_recall": rec_llama,
            "nested_recall": rec_nested,
            "llama_unique_corrections": uniq_corr,
            "llama_degradations": degrads,
            "net_class_delta": uniq_corr - degrads
        })
    df_classes = pd.DataFrame(class_stats)
    df_classes.to_csv(EXP_DIR / "class_results.csv", index=False)

    # -------------------------------------------------------------------------
    # PHASE 20: STATISTICAL TESTING (PAIRED BOOTSTRAP B = 2000)
    # -------------------------------------------------------------------------
    log("\n--- Phase 20: Paired Bootstrap Statistical Testing (B = 2,000) ---")
    mean_d, ci_l, ci_u, p_pos, boot_deltas = paired_bootstrap(
        y_true, pred_exp52, outer_fused_preds, n_boot=2000, seed=42
    )
    log(f"Bootstrap Results: Mean Delta = {mean_d:+.4f} | 95% CI = [{ci_l:+.4f}, {ci_u:+.4f}] | P(Delta > 0) = {p_pos*100:.1f}%")

    bootstrap_summary = {
        "mean_delta": mean_d,
        "ci_lower_95": ci_l,
        "ci_upper_95": ci_u,
        "p_positive": p_pos,
        "pooled_exp52_ba": ba_exp52,
        "pooled_nested_ba": pooled_nested_ba,
        "outer_fold_deltas": [float(outer_nested_ba[f] - fold_metrics[f]["exp52_ba"]) for f in range(5)]
    }
    with open(EXP_DIR / "bootstrap_results.json", "w") as f:
        json.dump(bootstrap_summary, f, indent=2)

    # -------------------------------------------------------------------------
    # PHASE 21: EXP65 DECISION RULE
    # -------------------------------------------------------------------------
    log("\n--- Phase 21: Formal Decision Rule Evaluation ---")
    folds_improving = sum(1 for d in bootstrap_summary["outer_fold_deltas"] if d > 0)

    if delta_nested >= 0.005 and ci_l > 0.0 and folds_improving >= 4 and net_correction > 50 and oracle_improvement > 0.02:
        verdict = "GREEN"
        status_note = "Strong complementary signal with statistically significant nested gain."
    elif delta_nested >= 0.002 and (oracle_improvement > 0.01 or net_correction > 0):
        verdict = "YELLOW"
        status_note = "Marginal or uncertain improvement; fails significance threshold. Retain EXP52."
    else:
        verdict = "RED"
        status_note = "No meaningful complementary improvement or degradation over EXP52. Retain EXP52."

    log(f"EXP65 VERDICT: {verdict} ({status_note})")

    # -------------------------------------------------------------------------
    # PHASE 24: PUBLICATION-QUALITY FIGURES (8 FIGURES, 300 DPI)
    # -------------------------------------------------------------------------
    log("\n--- Phase 24: Generating Publication-Quality Figures (300 DPI) ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Model Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Qwen3-8B", "Qwen3-14B", "EXP52 Champion", "Llama-3.1-8B", "Nested Fusion"]
    scores = [ba_q8, ba_q14, ba_exp52, ba_llama, pooled_nested_ba]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    bars = ax.bar(models, scores, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Prompt-Disjoint Balanced Accuracy")
    ax.set_title("EXP65: Foundation Model Standalone & Ensemble Comparison", fontweight="bold")
    ax.set_ylim([0.50, 0.85])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp65_model_comparison.png", dpi=300)
    plt.close()

    # 2. Error Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_mat = np.array([
        [1.0, corr_q8_q14, corr_q8_llama],
        [corr_q8_q14, 1.0, corr_q14_llama],
        [corr_q8_llama, corr_q14_llama, 1.0]
    ])
    labels_corr = ["Qwen3-8B", "Qwen3-14B", "Llama-3.1-8B"]
    sns.heatmap(corr_mat, annot=True, fmt=".3f", cmap="Blues", xticklabels=labels_corr, yticklabels=labels_corr, vmin=0, vmax=1, ax=ax)
    ax.set_title("EXP65: Measured Error Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp65_error_correlation.png", dpi=300)
    plt.close()

    # 3. Disagreement Subset Performance
    fig, ax = plt.subplots(figsize=(7, 5))
    dis_models = ["Qwen3-8B", "Qwen3-14B", "EXP52", "Llama-3.1-8B", "3-Model Oracle"]
    dis_scores = [q8_dis_ba, q14_dis_ba, exp52_dis_ba, llama_dis_ba, oracle_dis_3model]
    ax.bar(dis_models, dis_scores, color=["#64B5F6", "#81C784", "#E57373", "#BA68C8", "#FFD54F"], edgecolor="black")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"Performance on Disagreement Subset (N = {n_dis})", fontweight="bold")
    ax.set_ylim([0.45, 0.95])
    for idx, v in enumerate(dis_scores):
        ax.text(idx, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp65_disagreement_analysis.png", dpi=300)
    plt.close()

    # 4. Unique Corrections vs Degradations
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_types = ["Corrections\n(Llama right, EXP52 wrong)", "Degradations\n(Llama wrong, EXP52 right)", "Net Impact"]
    counts = [llama_corr_52_wrong, -llama_wrong_52_corr, net_correction]
    colors_c = ["forestgreen", "crimson", "steelblue"]
    bars = ax.bar(corr_types, counts, color=colors_c, edgecolor="black", width=0.5)
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_ylabel("Number of Samples (out of 2,699)")
    ax.set_title("EXP65: Unique Correction Capacity vs EXP52", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2.0, h + (10 if h >= 0 else -25), f"{int(h):+d}", ha="center", va=va, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp65_unique_corrections.png", dpi=300)
    plt.close()

    # 5. Oracle Comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    oracles = ["EXP52 Actual", "Oracle(Q8, Q14)", "Oracle(EXP52, Llama)", "Oracle(Q8, Q14, Llama)"]
    o_vals = [ba_exp52, oracle_q8_q14, oracle_exp52_llama, oracle_3model]
    ax.bar(oracles, o_vals, color=["#34495E", "#2980B9", "#27AE60", "#E67E22"], edgecolor="black", width=0.55)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("EXP65: Theoretical Ensembling Ceiling (Oracle)", fontweight="bold")
    ax.set_ylim([0.70, 0.95])
    for idx, v in enumerate(o_vals):
        ax.text(idx, v + 0.006, f"{v:.4f}", ha="center", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp65_oracle_comparison.png", dpi=300)
    plt.close()

    # 6. Class Recall Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(18)
    w_bar = 0.35
    ax.bar(x_pos - w_bar/2, df_classes["exp52_recall"], width=w_bar, label="EXP52 Champion", color="#C44E52", edgecolor="black")
    ax.bar(x_pos + w_bar/2, df_classes["llama_recall"], width=w_bar, label="Llama-3.1-8B", color="#8172B3", edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall: EXP52 vs Llama-3.1-8B", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp65_class_recall_comparison.png", dpi=300)
    plt.close()

    # 7. Static Fusion Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_fusion["weight_llama"], df_fusion["balanced_accuracy"], marker="o", color="purple", lw=2, label="Fusion (EXP52 + Llama)")
    ax.axhline(ba_exp52, color="red", linestyle="--", lw=1.5, label=f"EXP52 Baseline ({ba_exp52:.4f})")
    ax.set_xlabel("Llama Fusion Weight $w$")
    ax.set_ylabel("Prompt-Disjoint Balanced Accuracy")
    ax.set_title("EXP65: Static Probability Fusion Curve", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp65_fusion_curve.png", dpi=300)
    plt.close()

    # 8. Confidence Analysis
    fig, ax = plt.subplots(figsize=(7, 5))
    df_conf_plot = df_conf.copy()
    ax.bar(df_conf_plot["group"], df_conf_plot["llama_mean_conf"], color=["#2ECC71", "#3498DB", "#E74C3C", "#95A5A6"], edgecolor="black", width=0.5)
    ax.set_ylabel("Mean Llama Top-1 Confidence")
    ax.set_title("Confidence Calibration Across Error Agreement Groups", fontweight="bold")
    ax.set_ylim([0.0, 1.0])
    for idx, v in enumerate(df_conf_plot["llama_mean_conf"]):
        ax.text(idx, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp65_confidence_analysis.png", dpi=300)
    plt.close()

    log("Saved all 8 publication-grade 300-DPI plots to plots/.")

    # -------------------------------------------------------------------------
    # PHASE 25 & 26: CONFIG, README, SUMMARY JSON
    # -------------------------------------------------------------------------
    training_summary = {
        "experiment_id": "EXP65_llama_diverse",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": "meta-llama/Llama-3.1-8B",
        "architecture": "Llama-3.1-8B + QLoRA 4-bit (r=16, alpha=32)",
        "hardware": "2x NVIDIA GeForce RTX 3060 (12GB VRAM each)",
        "verdict": verdict,
        "status_note": status_note,
        "exp52_frozen": True,
        "metrics": {
            "q8_ba": ba_q8,
            "q14_ba": ba_q14,
            "exp52_ba": ba_exp52,
            "llama_ba": ba_llama,
            "nested_fusion_ba": pooled_nested_ba,
            "nested_delta": delta_nested,
            "bootstrap_mean_delta": mean_d,
            "ci_lower_95": ci_l,
            "ci_upper_95": ci_u,
            "p_positive": p_pos,
            "error_corr_q8_llama": corr_q8_llama,
            "error_corr_q14_llama": corr_q14_llama,
            "error_corr_exp52_llama": corr_52_llama,
            "llama_unique_corrections": llama_corr_52_wrong,
            "llama_degradations": llama_wrong_52_corr,
            "net_correction": net_correction,
            "oracle_q8_q14": oracle_q8_q14,
            "oracle_3model": oracle_3model,
            "oracle_improvement": oracle_improvement
        }
    }
    with open(EXP_DIR / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    config_data = {
        "experiment_id": "EXP65_llama_diverse",
        "description": "Llama-3.1-8B Diverse Foundation Backbone Attribution",
        "base_model": "meta-llama/Llama-3.1-8B",
        "max_length": max_len,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "learning_rate": lr,
        "epochs": epochs,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "quantization": "4-bit NF4 double quant",
        "cv_folds": 5,
        "cv_scheme": "Prompt-disjoint GroupKFold (30 prompts/fold)"
    }
    with open(EXP_DIR / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    readme_content = f"""# EXP65: Llama-3.1-8B Diverse Foundation Backbone Attribution

- **Experiment ID**: `EXP65_llama_diverse`
- **Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Base Model**: `meta-llama/Llama-3.1-8B`
- **Status**: {verdict} ({status_note})
- **Frozen Champion**: EXP52 (0.60 Q8 + 0.40 Q14, OOF BA = 0.7873, Public LB = 0.632)
- **EXP52 Remains Frozen**: YES
- **Production Changed**: NO
- **Kaggle Submission Made**: NO

## Key Results
- **Qwen3-8B Standalone**: {ba_q8:.4f} BA
- **Qwen3-14B Standalone**: {ba_q14:.4f} BA
- **EXP52 Champion Baseline**: {ba_exp52:.4f} BA
- **Llama-3.1-8B Standalone**: {ba_llama:.4f} BA
- **Nested EXP52 + Llama**: {pooled_nested_ba:.4f} BA ($\Delta = {delta_nested:+.4f}$, 95% CI [{ci_l:+.4f}, {ci_u:+.4f}], $P(\Delta > 0) = {p_pos*100:.1f}\%$)
- **Measured Error Correlation (EXP52 vs Llama)**: $r = {corr_52_llama:.4f}$
- **Unique Corrections**: {llama_corr_52_wrong} samples | **Degradations**: {llama_wrong_52_corr} samples | **Net**: {net_correction:+d} samples
- **Three-Model Oracle**: {oracle_3model:.4f} BA (Gain over Q8+Q14: {oracle_improvement:+.4f})
"""
    with open(EXP_DIR / "README.md", "w") as f:
        f.write(readme_content)

    total_duration = time.time() - start_total_time
    log(f"\n=====================================================================")
    log(f"EXP65 PIPELINE COMPLETE in {total_duration/60:.2f} minutes")
    log(f"=====================================================================")


if __name__ == "__main__":
    main()
