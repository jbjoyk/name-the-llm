"""
Experiment tracker for Name That LLM competition.
Appends JSON records to experiments/experiment_log.jsonl.
Never overwrites existing records.
"""

import json
import os
import time
import datetime
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
LOG_FILE = EXPERIMENTS_DIR / "experiment_log.jsonl"
LEADERBOARD_FILE = EXPERIMENTS_DIR / "leaderboard.csv"

REQUIRED_FIELDS = [
    "experiment_id", "date", "research_question", "hypothesis",
    "dataset_version", "cv_strategy", "group_column", "random_seed",
    "model", "features", "hyperparameters", "compute_config",
    "fold_scores", "mean_balanced_accuracy", "std_balanced_accuracy",
    "macro_f1", "per_class_recall", "training_time_sec",
    "inference_time_sec", "feature_dimensionality", "status",
    "conclusion", "next_recommendation",
]


def load_all() -> list[dict]:
    """Return all recorded experiments."""
    if not LOG_FILE.exists():
        return []
    records = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_current_best() -> dict | None:
    """Return the experiment with highest mean_balanced_accuracy."""
    records = [r for r in load_all() if r.get("status") == "completed"]
    if not records:
        return None
    return max(records, key=lambda r: r.get("mean_balanced_accuracy", 0.0))


def experiment_exists(experiment_id: str) -> bool:
    return any(r["experiment_id"] == experiment_id for r in load_all())


def save_experiment(record: dict) -> None:
    """Append experiment record. Raises if ID already exists."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = record.get("experiment_id", "")
    if experiment_exists(exp_id):
        raise ValueError(
            f"Experiment '{exp_id}' already recorded. "
            "Create a new experiment_id for a new run."
        )
    record.setdefault("date", datetime.datetime.utcnow().isoformat())
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    _update_leaderboard()
    print(f"✅ Experiment '{exp_id}' saved → {LOG_FILE}")


def _update_leaderboard() -> None:
    """Write a human-readable CSV leaderboard sorted by mean_balanced_accuracy."""
    import csv
    records = sorted(
        [r for r in load_all() if r.get("status") == "completed"],
        key=lambda r: r.get("mean_balanced_accuracy", 0.0),
        reverse=True,
    )
    cols = [
        "experiment_id", "model", "features",
        "mean_balanced_accuracy", "std_balanced_accuracy",
        "macro_f1", "cv_strategy", "date", "conclusion",
    ]
    with open(LEADERBOARD_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def print_leaderboard(n: int = 10) -> None:
    records = sorted(
        [r for r in load_all() if r.get("status") == "completed"],
        key=lambda r: r.get("mean_balanced_accuracy", 0.0),
        reverse=True,
    )[:n]
    print(f"\n{'='*80}")
    print(f"{'EXPERIMENT LEADERBOARD':^80}")
    print(f"{'='*80}")
    header = f"{'Rank':<5} {'ID':<30} {'Model':<25} {'BalAcc':>8} {'±':>6} {'F1':>7}"
    print(header)
    print("-" * 80)
    for i, r in enumerate(records, 1):
        print(
            f"{i:<5} {r['experiment_id'][:30]:<30} {str(r['model'])[:25]:<25} "
            f"{r.get('mean_balanced_accuracy', 0):.4f}  "
            f"±{r.get('std_balanced_accuracy', 0):.4f} "
            f"{r.get('macro_f1', 0):.4f}"
        )
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print_leaderboard()
    best = get_current_best()
    if best:
        print(f"Current best: {best['experiment_id']} | BalAcc={best['mean_balanced_accuracy']:.4f}")
    else:
        print("No completed experiments yet.")
