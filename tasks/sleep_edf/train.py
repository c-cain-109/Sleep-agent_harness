
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, f1_score

from model import build_model, CLASS_NAMES


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_npz_directory(directory):
    """读取一个目录中的记录，每个 NPZ 包含 x 和 y。"""
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"目录中没有 NPZ 文件：{directory}"
        )

    all_x = []
    all_y = []

    for path in files:
        with np.load(path, allow_pickle=True) as data:
            if "x" not in data or "y" not in data:
                raise ValueError(f"{path.name} 缺少 x 或 y")

            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.int64).reshape(-1)

        if x.ndim != 2 or x.shape[1] != 3000:
            raise ValueError(
                f"{path.name} 的 x 形状应为 (N, 3000)，"
                f"实际为 {x.shape}"
            )

        if len(x) != len(y):
            raise ValueError(f"{path.name} 中 x 和 y 数量不一致")

        if not np.isfinite(x).all():
            raise ValueError(f"{path.name} 的 EEG 包含 NaN 或无穷值")

        if not np.isin(y, [0, 1, 2, 3, 4]).all():
            raise ValueError(f"{path.name} 存在非法标签")

        all_x.append(x)
        all_y.append(y)

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"读取目录: {directory}")
    print(f"记录数量: {len(files)}")
    print(f"Epoch 数量: {len(y)}")
    print("类别分布:", {
        name: int((y == i).sum())
        for i, name in enumerate(CLASS_NAMES)
    })

    # (N, 3000) -> (N, 1, 3000)
    X = torch.from_numpy(x[:, None, :])
    Y = torch.from_numpy(y)

    return TensorDataset(X, Y)


def load_single_record_time_split(path, val_ratio=0.2):
    """单条记录按时间顺序切分：前段训练、后段验证。

    同一受试者的数据同时出现在 train 和 val 中，结果只能证明管线可用，
    不能作为模型泛化能力的证据。
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        for key in ("x", "y", "epoch_index", "record_id", "subject_id"):
            if key not in data:
                raise ValueError(f"{path.name} 缺少字段 {key}")
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64).reshape(-1)
        epoch_index = np.asarray(data["epoch_index"])
        record_id = str(data["record_id"])
        subject_id = str(data["subject_id"])

    if x.ndim != 2 or x.shape[1] != 3000:
        raise ValueError(f"{path.name} 的 x 形状应为 (N, 3000)，实际为 {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"{path.name} 中 x 和 y 数量不一致")
    if not np.isfinite(x).all():
        raise ValueError(f"{path.name} 的 EEG 包含 NaN 或无穷值")
    if not np.isin(y, [0, 1, 2, 3, 4]).all():
        raise ValueError(f"{path.name} 存在非法标签")
    if not (np.diff(epoch_index) > 0).all():
        raise ValueError(f"{path.name} 的 epoch_index 不是递增的，无法按时间切分")

    n_val = int(round(len(y) * val_ratio))
    if n_val < 1 or n_val >= len(y):
        raise ValueError(f"val_ratio={val_ratio} 无法切出非空的训练集和验证集")
    cut = len(y) - n_val

    def describe(name, ys):
        dist = {c: int((ys == i).sum()) for i, c in enumerate(CLASS_NAMES)}
        print(f"  {name}: {len(ys)} 个 epoch，类别分布 {dist}")
        missing = [c for c, n in dist.items() if n == 0]
        if missing:
            print(f"    注意：{name} 中缺少类别 {missing}，这些类别的指标没有意义")

    print(f"读取文件: {path}")
    print(f"记录 {record_id}，受试者 {subject_id}，共 {len(y)} 个 epoch")
    print(f"按时间切分：前 {cut} 个训练，后 {n_val} 个验证")
    describe("训练集", y[:cut])
    describe("验证集", y[cut:])
    print("⚠️  train/val 来自同一受试者同一晚的不同时间段，")
    print("   指标只说明管线能跑通，不代表模型的泛化能力。")

    def to_dataset(xs, ys):
        return TensorDataset(torch.from_numpy(xs[:, None, :]), torch.from_numpy(ys))

    return to_dataset(x[:cut], y[:cut]), to_dataset(x[cut:], y[cut:])


def run_training_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_samples += len(y)

    return total_loss / total_samples


def run_validation(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * len(y)
            total_samples += len(y)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    return {
        "val_loss": total_loss / total_samples,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            average="macro",
            zero_division=0,
        ),
    }


def make_smoke_test_data():
    """生成模拟数据，只用于测试训练流程，不代表真实睡眠数据。"""
    rng = np.random.default_rng(42)

    train_x = rng.standard_normal((10, 3000)).astype(np.float32)
    train_y = np.tile(np.arange(5), 2).astype(np.int64)

    val_x = rng.standard_normal((5, 3000)).astype(np.float32)
    val_y = np.arange(5, dtype=np.int64)

    train_dataset = TensorDataset(
        torch.from_numpy(train_x[:, None, :]),
        torch.from_numpy(train_y),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_x[:, None, :]),
        torch.from_numpy(val_y),
    )

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument(
        "--data-file", type=Path,
        help="单条记录 NPZ，按时间切分 train/val（仅用于管线验证）",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "baseline",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs 和 batch-size 必须为正整数")

    if not args.smoke_test and args.data_file is None and (
        args.train_dir is None or args.val_dir is None
    ):
        parser.error("真实训练必须提供 --data-file，或同时提供 --train-dir 和 --val-dir")

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("当前 PyTorch 环境无法使用 MPS，请改用 --device cpu")

    set_seed(args.seed)
    device = torch.device(args.device)

    if args.smoke_test:
        print("模式：模拟数据训练测试")
        train_dataset, val_dataset = make_smoke_test_data()
        epochs = 1
    elif args.data_file is not None:
        print("模式：单条记录管线验证（按时间切分）")
        train_dataset, val_dataset = load_single_record_time_split(
            args.data_file, args.val_ratio
        )
        epochs = args.epochs
    else:
        print("模式：真实数据训练")
        train_dataset = load_npz_directory(args.train_dir)
        val_dataset = load_npz_directory(args.val_dir)
        epochs = args.epochs

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(seed=args.seed).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    checkpoint_path = args.output_dir / "best_model.pt"

    best_score = -1.0

    for epoch in range(1, epochs + 1):
        train_loss = run_training_epoch(
            model, train_loader, criterion, optimizer, device
        )
        metrics = run_validation(
            model, val_loader, criterion, device
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        # 只根据验证集 Balanced Accuracy 保存最佳模型
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": record,
                    "class_names": CLASS_NAMES,
                },
                checkpoint_path,
            )

    print("\n训练流程结束。")
    print("日志文件:", log_path)
    print("最佳模型:", checkpoint_path)

    if args.smoke_test:
        print("注意：本次使用随机模拟数据，不代表模型具备睡眠分期能力。")


if __name__ == "__main__":
    main()
