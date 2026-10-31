"""Evaluate a saved TinySleepNet checkpoint on the validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = PROJECT_ROOT / "tasks" / "sleep_edf"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 确保直接运行 evaluate.py 时也能找到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.sleep_edf.model import CLASS_NAMES, build_model
from tasks.sleep_edf.data_loader import load_subject_split


def load_validation_split(data_file: Path, val_ratio: float = 0.2) -> TensorDataset:
    """Load the same chronological split used by ``train.py``."""
    with np.load(data_file, allow_pickle=False) as data:
        required = ("x", "y", "epoch_index")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{data_file.name} 缺少字段: {missing}")
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64).reshape(-1)
        epoch_index = np.asarray(data["epoch_index"])

    if x.ndim != 2 or x.shape[1] != 3000:
        raise ValueError(f"x 形状应为 (N, 3000)，实际为 {x.shape}")
    if len(x) != len(y):
        raise ValueError("x 和 y 的样本数不一致")
    if not np.isfinite(x).all():
        raise ValueError("x 包含 NaN 或 Inf")
    if not np.isin(y, [0, 1, 2, 3, 4]).all():
        raise ValueError("y 包含 0-4 之外的标签")
    if not (np.diff(epoch_index) > 0).all():
        raise ValueError("epoch_index 不是严格递增，无法复现时间切分")

    n_val = int(round(len(y) * val_ratio))
    if n_val < 1 or n_val >= len(y):
        raise ValueError("val_ratio 无法产生非空训练集和验证集")
    val_x = x[len(y) - n_val :, None, :]
    val_y = y[len(y) - n_val :]
    return TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_file: Path | None = None,
    val_ratio: float = 0.2,
    batch_size: int = 32,
    device_name: str = "cpu",
    data_dir: Path | None = None,
    split_file: Path | None = None,
) -> dict[str, Any]:
    """Load the best checkpoint and return validation metrics."""
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = build_model(
        seed=42,
        lstm_hidden_size=int(config.get("lstm_hidden_size", 128)),
        dropout=float(config.get("dropout", 0.5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if data_dir is not None or split_file is not None:
        if data_dir is None or split_file is None or data_file is not None:
            raise ValueError("全量评估必须同时提供 data_dir 和 split_file，且不能传 data_file")
        dataset = load_subject_split(data_dir, split_file, "val")
    elif data_file is not None:
        dataset = load_validation_split(data_file, val_ratio)
    else:
        raise ValueError("必须提供 --data-file，或同时提供 --data-dir 和 --split-file")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            predictions = logits.argmax(dim=1)
            total_loss += float(loss.item()) * len(y)
            total_samples += len(y)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "n_samples": int(total_samples),
        "val_loss": total_loss / max(1, total_samples),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "class_names": list(CLASS_NAMES),
        "confusion_matrix": matrix.tolist(),
        "label_counts": {
            name: int((y_true == index).sum())
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.data_file is not None and (args.data_dir is not None or args.split_file is not None):
        parser.error("--data-file 不能与 --data-dir/--split-file 混用")
    if (args.data_dir is None) != (args.split_file is None):
        parser.error("--data-dir 和 --split-file 必须同时提供")
    if args.data_file is None and args.data_dir is None:
        parser.error("必须提供 --data-file，或同时提供 --data-dir 和 --split-file")

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_file=args.data_file,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        device_name=args.device,
        data_dir=args.data_dir,
        split_file=args.split_file,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
