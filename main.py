"""Sleep staging self-improvement loop.

The loop proposes one hyperparameter configuration, evaluates it on the fixed
validation split, and keeps only configurations that improve the primary metric.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DIR = PROJECT_ROOT / "tasks" / "sleep_edf"
CONFIG_FILE = TASK_DIR / "model_config.py"
DATA_FILE = TASK_DIR / "data" / "prepared" / "SC4001E0.npz"
RUN_HISTORY = PROJECT_ROOT / "logs" / "sleep_run_history.jsonl"

load_dotenv(PROJECT_ROOT / ".env")

from harness.agent import generate_sleep_strategy
from harness.git_manager import GitManager
from harness.sandbox import run_python_script


def write_config(config_file: Path, hyperparams: dict) -> None:
    """Write the validated hyperparameter surface consumed by train.py."""
    content = f'''"""TinySleepNet hyperparameters managed by the sleep harness."""

LEARNING_RATE = {hyperparams.get("learning_rate", 0.0001)}
BATCH_SIZE = {hyperparams.get("batch_size", 32)}
NUM_EPOCHS = {hyperparams.get("num_epochs", 5)}
PATIENCE = {hyperparams.get("patience", 3)}

LSTM_HIDDEN_SIZE = {hyperparams.get("lstm_hidden_size", 128)}
DROPOUT = {hyperparams.get("dropout", 0.5)}
WEIGHT_DECAY = {hyperparams.get("weight_decay", 0.0001)}

INPUT_SIZE = 3000
NUM_CLASSES = 5
'''
    config_file.write_text(content, encoding="utf-8")


def run_experiment(
    output_dir: Path,
    timeout_seconds: int = 900,
) -> dict:
    """Train once and evaluate the saved best checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_model.pt"
    evaluation_file = output_dir / "evaluation.json"

    train = run_python_script(
        TASK_DIR / "train.py",
        [
            "--data-file",
            str(DATA_FILE),
            "--output-dir",
            str(output_dir),
            "--seed",
            "42",
        ],
        cwd=TASK_DIR,
        timeout=timeout_seconds,
    )
    print(train["stdout"], end="")
    if not train["success"]:
        raise RuntimeError(f"训练失败: {train['error']}")

    evaluation = run_python_script(
        PROJECT_ROOT / "harness" / "evaluate.py",
        [
            "--checkpoint",
            str(checkpoint),
            "--data-file",
            str(DATA_FILE),
            "--output",
            str(evaluation_file),
        ],
        cwd=PROJECT_ROOT,
        timeout=300,
    )
    if not evaluation["success"]:
        raise RuntimeError(f"评估失败: {evaluation['error']}")

    result = json.loads(evaluation_file.read_text(encoding="utf-8"))
    return {
        "balanced_accuracy": result["balanced_accuracy"],
        "macro_f1": result["macro_f1"],
        "val_loss": result["val_loss"],
        "checkpoint_epoch": result["checkpoint_epoch"],
    }


def is_better(new: dict, best: dict) -> bool:
    """Lexicographic metric comparison: BalAcc, Macro-F1, then loss."""
    return (
        new["balanced_accuracy"] > best["balanced_accuracy"]
        or (
            new["balanced_accuracy"] == best["balanced_accuracy"]
            and new["macro_f1"] > best["macro_f1"]
        )
        or (
            new["balanced_accuracy"] == best["balanced_accuracy"]
            and new["macro_f1"] == best["macro_f1"]
            and new["val_loss"] < best["val_loss"]
        )
    )


def append_history(entry: dict) -> None:
    RUN_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with RUN_HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main_loop(max_rounds: int) -> None:
    print("=" * 70)
    print("Sleep staging self-improvement")
    print("=" * 70)

    config_before_round = CONFIG_FILE.read_text(encoding="utf-8")
    initial_metrics = run_experiment(TASK_DIR / "outputs" / "round_0_initial")
    best_metrics = initial_metrics.copy()
    current_metrics = initial_metrics.copy()
    best_round = 0
    append_history({"round": 0, "action": "init", **initial_metrics})

    git = GitManager(PROJECT_ROOT)
    config_relative = str(CONFIG_FILE.relative_to(PROJECT_ROOT))

    for round_num in range(1, max_rounds + 1):
        print(f"\nRound {round_num}/{max_rounds}")
        config_before_round = CONFIG_FILE.read_text(encoding="utf-8")
        try:
            proposal = generate_sleep_strategy(
                current_metrics=current_metrics,
                best_metrics=best_metrics,
                round_num=round_num,
                project_root=PROJECT_ROOT,
            )
            write_config(CONFIG_FILE, proposal)

            config_after_round = CONFIG_FILE.read_text(encoding="utf-8")
            if config_after_round == config_before_round:
                print("[SKIP] Agent 没有产生配置变化")
                continue

            new_metrics = run_experiment(TASK_DIR / "outputs" / f"round_{round_num}")
            action = "accept" if is_better(new_metrics, best_metrics) else "rollback"

            if action == "accept":
                git.commit_changes(
                    f"[Round {round_num}] accept sleep config\n\n"
                    f"Balanced Accuracy: {new_metrics['balanced_accuracy']:.6f}\n"
                    f"Reasoning: {proposal.get('reasoning', '')}",
                    paths=[config_relative],
                )
                best_metrics = new_metrics.copy()
                best_round = round_num
            else:
                CONFIG_FILE.write_text(config_before_round, encoding="utf-8")

            current_metrics = new_metrics.copy()
            append_history(
                {
                    "round": round_num,
                    "action": action,
                    **new_metrics,
                    "proposal": proposal,
                }
            )
        except KeyboardInterrupt:
            print("用户中断")
            CONFIG_FILE.write_text(config_before_round, encoding="utf-8")
            break
        except Exception as exc:
            CONFIG_FILE.write_text(config_before_round, encoding="utf-8")
            print(f"[ERROR] Round {round_num}: {exc}")
            append_history(
                {"round": round_num, "action": "error", "error": str(exc)}
            )

        if round_num < max_rounds:
            time.sleep(2)

    print("\nBest round:", best_round)
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))
    print("History:", RUN_HISTORY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the sleep staging optimization loop")
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    main_loop(args.rounds)
