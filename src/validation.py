"""
Validation utilities for Name That LLM.
Enforces leakage-safe prompt-grouped cross-validation throughout.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix


RANDOM_SEED = 42
N_SPLITS = 5
GROUP_COLUMN = "prompt_id"
TARGET_COLUMN = "model"

# Path constants
DATA_DIR = "/home/rj2826/name-the-llm/data"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.parquet"
VENV_PYTHON = (
    "/home/rj2826/compitition/"
    "04. Competition (due Nov 20)-20261028T163421Z-1-001/"
    "04. Competition (due Nov 20)/.venv/bin/python"
)


def load_data():
    """Load train and test sets."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def get_folds(train: pd.DataFrame, strategy: str = "group", random_state: int = RANDOM_SEED):
    """
    Return list of (train_idx, val_idx) tuples using leakage-safe split.

    strategy: 'group'  → GroupKFold on prompt_id (default, safest)
              'stratgroup' → StratifiedGroupKFold (stratifies by label within groups)
    """
    groups = train[GROUP_COLUMN].values
    y = train[TARGET_COLUMN].values

    if strategy == "stratgroup":
        kf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        splits = list(kf.split(train, y, groups=groups))
    else:
        kf = GroupKFold(n_splits=N_SPLITS)
        splits = list(kf.split(train, y, groups=groups))

    return splits


def evaluate(y_true, y_pred, classes=None):
    """Return dict with standard metrics."""
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return {
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(f1),
        "per_class_recall": per_class.tolist() if classes is not None else per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def cross_validate(
    pipeline_fn,  # fn(X_train, y_train, X_val) → y_pred array
    X: np.ndarray | pd.Series,
    y: np.ndarray,
    groups: np.ndarray,
    strategy: str = "group",
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
):
    """
    Run cross-validation with leakage-safe prompt-grouped splits.

    pipeline_fn:  Callable(X_train, y_train, X_val) → np.ndarray of predicted labels
    X:            Feature array or pd.Series of raw texts
    y:            Label array
    groups:       Group array (prompt_id values)
    Returns dict with fold_scores, mean, std, per_class, etc.
    """
    classes = sorted(np.unique(y))

    if strategy == "stratgroup":
        kf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        splits = list(kf.split(X, y, groups=groups))
    else:
        kf = GroupKFold(n_splits=N_SPLITS)
        splits = list(kf.split(X, y, groups=groups))

    fold_bal_accs, fold_f1s = [], []
    all_prc = []
    all_cm = np.zeros((len(classes), len(classes)), dtype=int)

    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        X_tr = X[tr_idx] if isinstance(X, np.ndarray) else X.iloc[tr_idx]
        X_val = X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        y_pred = pipeline_fn(X_tr, y_tr, X_val)
        metrics = evaluate(y_val, y_pred, classes=classes)

        fold_bal_accs.append(metrics["balanced_accuracy"])
        fold_f1s.append(metrics["macro_f1"])
        all_prc.append(metrics["per_class_recall"])
        all_cm += np.array(metrics["confusion_matrix"])

        if verbose:
            print(f"  Fold {fold_i+1}/{N_SPLITS} | BalAcc={metrics['balanced_accuracy']:.4f} | F1={metrics['macro_f1']:.4f}")

    mean_prc = np.mean(all_prc, axis=0).tolist()
    result = {
        "fold_scores": fold_bal_accs,
        "mean_balanced_accuracy": float(np.mean(fold_bal_accs)),
        "std_balanced_accuracy": float(np.std(fold_bal_accs)),
        "fold_f1s": fold_f1s,
        "macro_f1": float(np.mean(fold_f1s)),
        "per_class_recall": mean_prc,
        "classes": classes,
        "confusion_matrix": all_cm.tolist(),
    }

    if verbose:
        print(f"\n  BalAcc: {result['mean_balanced_accuracy']:.4f} ± {result['std_balanced_accuracy']:.4f}")
        print(f"  MacroF1: {result['macro_f1']:.4f}")

    return result
