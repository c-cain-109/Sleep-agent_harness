"""Sleep staging self-improvement loop with layered experiment memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DIR = PROJECT_ROOT / "tasks" / "sleep_edf"
CONFIG_FILE = TASK_DIR / "model_config.py"
DATA_DIR = TASK_DIR / "data" / "sleep-edf-sc-full"
SPLIT_FILE = DATA_DIR / "subject_split.json"
RUN_HISTORY = PROJECT_ROOT / "logs" / "sleep_run_history.jsonl"
MEMORY_V2 = PROJECT_ROOT / "logs" / "experiment_memory_v2.jsonl"

load_dotenv(PROJECT_ROOT / ".env")

from harness.agent import (  # noqa: E402
    AGENT_MODEL,
    AGENT_TEMPERATURE,
    PROMPT_VERSION,
    generate_sleep_strategy,
)
from harness.git_manager import GitManager  # noqa: E402
from harness.memory_store import (  # noqa: E402
    PARAMETER_KEYS,
    TASK_ID,
    append_run_record,
    dataset_version,
    diff_configs,
    evaluator_version,
    git_commit,
    new_run_id,
    parse_model_config,
    split_id,
    utc_now,
)
from harness.sandbox import run_python_script  # noqa: E402


class ExperimentError(RuntimeError):
    """Training/evaluation failure with metadata needed by the fact record."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        status: str,
        duration_seconds: float,
        log_path: str | None,
        checkpoint_path: str | None,
        phase: str,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.status = status
        self.duration_seconds = duration_seconds
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        self.phase = phase


def write_config(config_file: Path, hyperparams: dict) -> None:
    """Write only the validated hyperparameter surface consumed by train.py."""
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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _metric_payload(metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "val_loss": metrics.get("val_loss"),
    }


def _proposal_parameters(proposal: dict[str, Any] | None) -> dict[str, Any] | None:
    if proposal is None:
        return None
    return {
        key: proposal[key]
        for key in PARAMETER_KEYS
        if key in proposal
    }


def run_experiment(
    output_dir: Path,
    timeout_seconds: int = 3600,
    *,
    seed: int = 42,
) -> dict:
    """Train once and evaluate the saved checkpoint on the fixed validation split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_model.pt"
    evaluation_file = output_dir / "evaluation.json"
    metrics_log = output_dir / "metrics.jsonl"
    started_at = time.perf_counter()

    train = run_python_script(
        TASK_DIR / "train.py",
        [
            "--data-dir",
            str(DATA_DIR),
            "--split-file",
            str(SPLIT_FILE),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda",
            "--seed",
            str(seed),
        ],
        cwd=TASK_DIR,
        timeout=timeout_seconds,
    )
    print(train["stdout"], end="")
    if not train["success"]:
        timed_out = train.get("returncode") is None and "超时" in str(
            train.get("error", "")
        )
        raise ExperimentError(
            f"训练失败: {train['error']}",
            error_type="TrainingTimeout" if timed_out else "TrainingError",
            status="timeout" if timed_out else "error",
            duration_seconds=time.perf_counter() - started_at,
            log_path=_relative(metrics_log),
            checkpoint_path=_relative(checkpoint),
            phase="training",
        )

    evaluation = run_python_script(
        PROJECT_ROOT / "harness" / "evaluate.py",
        [
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(DATA_DIR),
            "--split-file",
            str(SPLIT_FILE),
            "--output",
            str(evaluation_file),
        ],
        cwd=PROJECT_ROOT,
        timeout=300,
    )
    if not evaluation["success"]:
        timed_out = evaluation.get("returncode") is None and "超时" in str(
            evaluation.get("error", "")
        )
        raise ExperimentError(
            f"评估失败: {evaluation['error']}",
            error_type="EvaluationTimeout" if timed_out else "EvaluationError",
            status="timeout" if timed_out else "error",
            duration_seconds=time.perf_counter() - started_at,
            log_path=_relative(metrics_log),
            checkpoint_path=_relative(checkpoint),
            phase="evaluation",
        )

    result = json.loads(evaluation_file.read_text(encoding="utf-8"))
    return {
        "balanced_accuracy": result["balanced_accuracy"],
        "macro_f1": result["macro_f1"],
        "val_loss": result["val_loss"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "duration_seconds": time.perf_counter() - started_at,
        "log_path": _relative(metrics_log),
        "checkpoint_path": _relative(checkpoint),
        "evaluation_log_path": _relative(evaluation_file),
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
    """Append the legacy format for backward-compatible readers."""
    RUN_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with RUN_HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )


def append_v2(record: dict[str, Any]) -> None:
    append_run_record(MEMORY_V2, record)


def make_terminal_record(
    *,
    run_id: str,
    round_num: int,
    config_before: dict[str, Any],
    config_after: dict[str, Any],
    proposal: dict[str, Any] | None,
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any] | None,
    decision: str,
    status: str,
    reason: str,
    error_type: str | None = None,
    log_path: str | None = None,
    checkpoint_path: str | None = None,
    commit_hash: str | None = None,
    duration_seconds: float = 0.0,
    agent_metadata: dict[str, Any] | None = None,
    memory_mode: str = "layered",
    seed: int = 42,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Build exactly one v2 fact record for a terminal Harness path."""
    metadata = agent_metadata or {}
    proposal_parameters = _proposal_parameters(proposal)
    metrics = _metric_payload(metrics_after)
    return {
        "schema_version": 2,
        "run_id": run_id,
        "round": round_num,
        "timestamp": utc_now(),
        "task_id": TASK_ID,
        "dataset_version": dataset_version(PROJECT_ROOT),
        "split_id": split_id(PROJECT_ROOT),
        "seed": seed,
        "code_commit": code_commit or git_commit(PROJECT_ROOT),
        "evaluator_version": evaluator_version(PROJECT_ROOT),
        "agent_model": metadata.get("agent_model", AGENT_MODEL),
        "prompt_version": metadata.get("prompt_version", PROMPT_VERSION),
        "temperature": metadata.get("temperature", AGENT_TEMPERATURE),
        "config_before": config_before,
        "agent_proposal": proposal,
        "config_after": config_after,
        "config_diff": diff_configs(config_before, proposal_parameters or {}),
        "applied_config_diff": diff_configs(config_before, config_after),
        "metrics_before": _metric_payload(metrics_before),
        "val_balanced_accuracy": metrics["balanced_accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_loss": metrics["val_loss"],
        "duration_seconds": float(duration_seconds),
        "token_usage": metadata.get("token_usage", {}),
        "decision": decision,
        "status": status,
        "reason": reason,
        "error_type": error_type,
        "log_path": log_path,
        "checkpoint_path": checkpoint_path,
        "commit_hash": commit_hash,
        "source": "native",
        "memory_mode": memory_mode,
    }


def _restore_config(config_text: str) -> None:
    CONFIG_FILE.write_text(config_text, encoding="utf-8")


def validate_dataset_inputs() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"预处理数据目录不存在：{DATA_DIR}")
    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(f"受试者划分文件不存在：{SPLIT_FILE}")

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if split.get("feasible") is not True:
        raise ValueError(f"受试者划分不可用：{split.get('reason')}")

    groups = ("train", "val", "test")
    subjects = {name: set(split["subjects"][name]) for name in groups}
    records = {name: list(split["records"][name]) for name in groups}

    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            if subjects[left] & subjects[right]:
                raise ValueError(f"受试者泄漏：{left} 与 {right} 有交集")
            if set(records[left]) & set(records[right]):
                raise ValueError(f"记录重复：{left} 与 {right} 有交集")

    all_records = [record for name in groups for record in records[name]]
    if len(all_records) != len(set(all_records)):
        raise ValueError("subject_split.json 存在重复记录")

    missing = [rid for rid in all_records if not (DATA_DIR / f"{rid}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"划分清单引用了缺失 NPZ：{missing[:10]}")

    print("数据集：Sleep-EDF Expanded / Sleep Cassette")
    print(f"记录数：{split.get('n_records', len(all_records))}")
    print(f"受试者数：{split.get('n_subjects', len(set.union(*subjects.values())))}")
    print(f"Epoch 数：{split.get('n_epochs', 'unknown')}")
    for name in groups:
        print(f"{name}: {len(subjects[name])} subjects, {len(records[name])} records")
    print("受试者级数据划分检查通过；本轮优化只使用 train/val，test 保持隔离。")


def main_loop(
    max_rounds: int,
    *,
    memory_mode: str = "layered",
    seed: int = 42,
) -> None:
    print("=" * 70)
    print("Sleep staging self-improvement")
    print(f"Memory mode: {memory_mode}")
    print("=" * 70)

    validate_dataset_inputs()
    code_commit_at_start = git_commit(PROJECT_ROOT)
    git = GitManager(PROJECT_ROOT)
    config_relative = str(CONFIG_FILE.relative_to(PROJECT_ROOT))

    baseline_config = parse_model_config(CONFIG_FILE)
    baseline_run_id = new_run_id(0)
    baseline_started = time.perf_counter()
    try:
        initial_metrics = run_experiment(
            TASK_DIR / "outputs" / "round_0_initial",
            seed=seed,
        )
    except KeyboardInterrupt:
        reason = "用户中断"
        duration = time.perf_counter() - baseline_started
        append_history({"round": 0, "action": "error", "error": reason})
        append_v2(
            make_terminal_record(
                run_id=baseline_run_id,
                round_num=0,
                config_before=baseline_config,
                config_after=baseline_config,
                proposal=None,
                metrics_before={},
                metrics_after=None,
                decision="Error",
                status="error",
                reason=reason,
                error_type="KeyboardInterrupt",
                duration_seconds=duration,
                memory_mode=memory_mode,
                seed=seed,
                code_commit=code_commit_at_start,
            )
        )
        print("[ERROR] Baseline: 用户中断")
        return
    except Exception as exc:
        error_type = getattr(exc, "error_type", type(exc).__name__)
        status = getattr(exc, "status", "error")
        duration = getattr(
            exc,
            "duration_seconds",
            time.perf_counter() - baseline_started,
        )
        reason = str(exc)
        append_history({"round": 0, "action": "error", "error": reason})
        append_v2(
            make_terminal_record(
                run_id=baseline_run_id,
                round_num=0,
                config_before=baseline_config,
                config_after=baseline_config,
                proposal=None,
                metrics_before={},
                metrics_after=None,
                decision="Error",
                status=status,
                reason=reason,
                error_type=error_type,
                log_path=getattr(exc, "log_path", None),
                checkpoint_path=getattr(exc, "checkpoint_path", None),
                duration_seconds=duration,
                memory_mode=memory_mode,
                seed=seed,
                code_commit=code_commit_at_start,
            )
        )
        print(f"[ERROR] Baseline: {reason}")
        return

    best_metrics = _metric_payload(initial_metrics)
    current_metrics = best_metrics.copy()
    best_round = 0
    last_decision = "Baseline"
    append_history(
        {
            "round": 0,
            "action": "init",
            "balanced_accuracy": initial_metrics["balanced_accuracy"],
            "macro_f1": initial_metrics["macro_f1"],
            "val_loss": initial_metrics["val_loss"],
            "checkpoint_epoch": initial_metrics["checkpoint_epoch"],
        }
    )
    append_v2(
        make_terminal_record(
            run_id=baseline_run_id,
            round_num=0,
            config_before=baseline_config,
            config_after=baseline_config,
            proposal=None,
            metrics_before={},
            metrics_after=initial_metrics,
            decision="Baseline",
            status="completed",
            reason="Initial baseline trained and evaluated on validation split.",
            log_path=initial_metrics["log_path"],
            checkpoint_path=initial_metrics["checkpoint_path"],
            commit_hash=code_commit_at_start,
            duration_seconds=initial_metrics["duration_seconds"],
            memory_mode=memory_mode,
            seed=seed,
            code_commit=code_commit_at_start,
        )
    )

    for round_num in range(1, max_rounds + 1):
        print(f"\nRound {round_num}/{max_rounds}")
        run_id = new_run_id(round_num)
        round_started = time.perf_counter()
        config_before_text = CONFIG_FILE.read_text(encoding="utf-8")
        config_before = parse_model_config(CONFIG_FILE)
        comparison_metrics = current_metrics.copy()
        proposal = None
        agent_metadata: dict[str, Any] = {}
        log_path: str | None = None
        checkpoint_path: str | None = None
        commit_hash: str | None = None
        evaluated_metrics: dict[str, Any] | None = None
        error_type: str | None = None
        status = "error"
        decision = "Error"
        duration = 0.0

        try:
            proposal = generate_sleep_strategy(
                current_metrics=current_metrics,
                best_metrics=best_metrics,
                round_num=round_num,
                project_root=PROJECT_ROOT,
                memory_mode=memory_mode,
                last_decision=last_decision,
                run_id=run_id,
            )
            agent_metadata = proposal.pop("_agent_metadata", {})
            write_config(CONFIG_FILE, proposal)
            proposed_config = parse_model_config(CONFIG_FILE)

            if proposed_config == config_before:
                duration = time.perf_counter() - round_started
                reason = "Agent 没有产生配置变化"
                append_history(
                    {
                        "round": round_num,
                        "action": "skip",
                        "reason": reason,
                        "proposal": proposal,
                    }
                )
                append_v2(
                    make_terminal_record(
                        run_id=run_id,
                        round_num=round_num,
                        config_before=config_before,
                        config_after=config_before,
                        proposal=proposal,
                        metrics_before=comparison_metrics,
                        metrics_after=None,
                        decision="Skip",
                        status="skipped",
                        reason=reason,
                        duration_seconds=duration,
                        agent_metadata=agent_metadata,
                        memory_mode=memory_mode,
                        seed=seed,
                        code_commit=code_commit_at_start,
                    )
                )
                last_decision = "Skip"
                print("[SKIP] Agent 没有产生配置变化")
                if round_num < max_rounds:
                    time.sleep(2)
                continue

            new_metrics = run_experiment(
                TASK_DIR / "outputs" / f"round_{round_num}",
                seed=seed,
            )
            evaluated_metrics = new_metrics
            log_path = new_metrics["log_path"]
            checkpoint_path = new_metrics["checkpoint_path"]
            duration = time.perf_counter() - round_started
            action = "accept" if is_better(new_metrics, best_metrics) else "rollback"

            if action == "accept":
                git.commit_changes(
                    f"[Round {round_num}] accept sleep config\n\n"
                    f"Balanced Accuracy: {new_metrics['balanced_accuracy']:.6f}\n"
                    f"Reasoning: {proposal.get('reasoning', '')}",
                    paths=[config_relative],
                )
                commit_hash = git_commit(PROJECT_ROOT)
                config_after = parse_model_config(CONFIG_FILE)
                best_metrics = _metric_payload(new_metrics)
                best_round = round_num
                current_metrics = best_metrics.copy()
                decision = "Accept"
                status = "completed"
            else:
                _restore_config(config_before_text)
                config_after = parse_model_config(CONFIG_FILE)
                current_metrics = best_metrics.copy()
                decision = "Rollback"
                status = "completed"

            append_history(
                {
                    "round": round_num,
                    "action": action,
                    "balanced_accuracy": new_metrics["balanced_accuracy"],
                    "macro_f1": new_metrics["macro_f1"],
                    "val_loss": new_metrics["val_loss"],
                    "checkpoint_epoch": new_metrics["checkpoint_epoch"],
                    "proposal": proposal,
                }
            )
            append_v2(
                make_terminal_record(
                    run_id=run_id,
                    round_num=round_num,
                    config_before=config_before,
                    config_after=config_after,
                    proposal=proposal,
                    metrics_before=comparison_metrics,
                    metrics_after=new_metrics,
                    decision=decision,
                    status=status,
                    reason=str(proposal.get("reasoning", "")),
                    log_path=log_path,
                    checkpoint_path=checkpoint_path,
                    commit_hash=commit_hash,
                    duration_seconds=duration,
                    agent_metadata=agent_metadata,
                    memory_mode=memory_mode,
                    seed=seed,
                    code_commit=code_commit_at_start,
                )
            )
            last_decision = decision
        except KeyboardInterrupt:
            _restore_config(config_before_text)
            reason = "用户中断"
            duration = time.perf_counter() - round_started
            append_history({"round": round_num, "action": "error", "error": reason})
            append_v2(
                make_terminal_record(
                    run_id=run_id,
                    round_num=round_num,
                    config_before=config_before,
                    config_after=parse_model_config(CONFIG_FILE),
                    proposal=proposal,
                    metrics_before=comparison_metrics,
                    metrics_after=evaluated_metrics,
                    decision="Error",
                    status="error",
                    reason=reason,
                    error_type="KeyboardInterrupt",
                    log_path=log_path,
                    checkpoint_path=checkpoint_path,
                    duration_seconds=duration,
                    agent_metadata=agent_metadata,
                    memory_mode=memory_mode,
                    seed=seed,
                    code_commit=code_commit_at_start,
                )
            )
            print("\n用户中断")
            break
        except Exception as exc:
            _restore_config(config_before_text)
            reason = str(exc)
            error_type = getattr(exc, "error_type", type(exc).__name__)
            status = getattr(exc, "status", "error")
            duration = getattr(
                exc,
                "duration_seconds",
                time.perf_counter() - round_started,
            )
            log_path = getattr(exc, "log_path", log_path)
            checkpoint_path = getattr(exc, "checkpoint_path", checkpoint_path)
            append_history(
                {"round": round_num, "action": "error", "error": reason}
            )
            append_v2(
                make_terminal_record(
                    run_id=run_id,
                    round_num=round_num,
                    config_before=config_before,
                    config_after=parse_model_config(CONFIG_FILE),
                    proposal=proposal,
                    metrics_before=comparison_metrics,
                    metrics_after=evaluated_metrics,
                    decision="Error",
                    status=status,
                    reason=reason,
                    error_type=error_type,
                    log_path=log_path,
                    checkpoint_path=checkpoint_path,
                    duration_seconds=duration,
                    agent_metadata=agent_metadata,
                    memory_mode=memory_mode,
                    seed=seed,
                    code_commit=code_commit_at_start,
                )
            )
            last_decision = "Error"
            print(f"[ERROR] Round {round_num}: {reason}")

        if round_num < max_rounds:
            time.sleep(2)

    print("\nBest round:", best_round)
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))
    print("Legacy history:", RUN_HISTORY)
    print("Structured v2 memory:", MEMORY_V2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the sleep staging optimization loop"
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument(
        "--memory-mode",
        choices=["none", "legacy", "layered"],
        default="layered",
        help="A/none, B/legacy, or C/layered Agent memory",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    main_loop(args.rounds, memory_mode=args.memory_mode, seed=args.seed)
