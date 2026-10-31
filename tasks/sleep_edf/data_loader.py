"""Load Sleep-EDF NPZ records according to subject_split.json.

This module is shared by tasks/sleep_edf/train.py and harness/evaluate.py.
It deliberately loads only the requested split (train, val, or test).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import ConcatDataset, TensorDataset

CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")
SPLIT_NAMES = ("train", "val", "test")


def _scalar_text(value: object, field_name: str, file_name: str) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{file_name}: {field_name} 应为标量，实际形状 {arr.shape}")
    result = arr.reshape(()).item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return str(result)


def read_and_validate_split(split_file: Path) -> dict:
    """Read split manifest and reject invalid or overlapping subject splits."""
    split_file = Path(split_file)
    if not split_file.is_file():
        raise FileNotFoundError(f"找不到受试者划分文件：{split_file}")

    split = json.loads(split_file.read_text(encoding="utf-8"))
    if split.get("feasible") is not True:
        raise ValueError(
            f"subject_split.json 标记为不可用：{split.get('reason', 'unknown')}"
        )

    for key in ("subjects", "records"):
        if not isinstance(split.get(key), dict):
            raise ValueError(f"{split_file.name} 缺少有效的 {key} 字段")
        missing = set(SPLIT_NAMES) - set(split[key])
        if missing:
            raise ValueError(f"{key} 缺少集合：{sorted(missing)}")

    subject_sets = {name: set(split["subjects"][name]) for name in SPLIT_NAMES}
    record_sets = {name: set(split["records"][name]) for name in SPLIT_NAMES}

    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1:]:
            overlap_subjects = subject_sets[left] & subject_sets[right]
            overlap_records = record_sets[left] & record_sets[right]
            if overlap_subjects:
                raise ValueError(
                    f"受试者泄漏：{left}/{right} 重叠 {sorted(overlap_subjects)}"
                )
            if overlap_records:
                raise ValueError(
                    f"记录重复划分：{left}/{right} 重叠 {sorted(overlap_records)}"
                )

    all_records = set().union(*record_sets.values())
    if len(all_records) != sum(len(v) for v in record_sets.values()):
        raise ValueError("subject_split.json 中存在重复记录")

    if split.get("n_records") is not None and len(all_records) != int(split["n_records"]):
        raise ValueError(
            f"划分记录数不一致：清单声明 {split['n_records']}，实际 {len(all_records)}"
        )
    if split.get("n_subjects") is not None:
        all_subjects = set().union(*subject_sets.values())
        if len(all_subjects) != int(split["n_subjects"]):
            raise ValueError(
                f"划分受试者数不一致：清单声明 {split['n_subjects']}，实际 {len(all_subjects)}"
            )

    return split


def load_subject_split(
    data_dir: Path,
    split_file: Path,
    split_name: Literal["train", "val", "test"],
    max_records: int | None = None,
) -> ConcatDataset:
    """Load NPZ records assigned to one subject-level split.

    NPZ naming convention: <record_id>.npz, e.g. SC4001E0.npz.
    Expected arrays: x (N, 3000), y (N,), subject_id, record_id.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name 必须是 {SPLIT_NAMES} 之一")

    data_dir = Path(data_dir)
    split = read_and_validate_split(Path(split_file))
    if not data_dir.is_dir():
        raise FileNotFoundError(f"找不到预处理数据目录：{data_dir}")

    record_ids = list(split["records"][split_name])
    expected_subjects = set(split["subjects"][split_name])
    if not record_ids:
        raise ValueError(f"{split_name} 集没有记录")
    if max_records is not None:
        if max_records < 1:
            raise ValueError("max_records 必须是正整数")
        record_ids = record_ids[:max_records]

    parts: list[TensorDataset] = []
    observed_subjects: set[str] = set()
    label_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    total_epochs = 0

    for record_id in record_ids:
        path = data_dir / f"{record_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"{split_name} 划分引用了不存在的记录文件：{path}"
            )

        # These NPZ files are generated locally by the project's preprocessing
        # script. allow_pickle=True permits scalar metadata saved as object arrays.
        with np.load(path, allow_pickle=True) as data:
            required = {"x", "y", "subject_id", "record_id"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path.name} 缺少字段：{sorted(missing)}")

            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.int64).reshape(-1)
            saved_subject = _scalar_text(data["subject_id"], "subject_id", path.name)
            saved_record = _scalar_text(data["record_id"], "record_id", path.name)

            if "fs" in data.files and int(np.asarray(data["fs"]).item()) != 100:
                raise ValueError(f"{path.name} 的采样率不是 100 Hz")
            if "epoch_duration" in data.files and int(np.asarray(data["epoch_duration"]).item()) != 30:
                raise ValueError(f"{path.name} 的 epoch_duration 不是 30 秒")

        if saved_record != record_id:
            raise ValueError(
                f"{path.name} 内 record_id={saved_record}，与划分清单 {record_id} 不一致"
            )
        if saved_subject not in expected_subjects:
            raise ValueError(
                f"受试者泄漏/映射错误：{record_id} 属于 {saved_subject}，"
                f"但 {split_name} 清单受试者为 {saved_subjects}"
            )
        if x.ndim != 2 or x.shape[1] != 3000:
            raise ValueError(f"{path.name} 的 x 应为 (N, 3000)，实际为 {x.shape}")
        if y.ndim != 1 or len(x) != len(y):
            raise ValueError(f"{path.name} 的 x/y 数量不一致")
        if len(y) == 0:
            raise ValueError(f"{path.name} 没有有效 epoch")
        if not np.isfinite(x).all():
            raise ValueError(f"{path.name} 的 EEG 包含 NaN 或 Inf")
        if not np.isin(y, np.arange(len(CLASS_NAMES))).all():
            raise ValueError(f"{path.name} 存在 0-4 之外的标签")

        x = np.ascontiguousarray(x[:, None, :])
        y = np.ascontiguousarray(y)
        parts.append(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)))
        observed_subjects.add(saved_subject)
        label_counts += np.bincount(y, minlength=len(CLASS_NAMES))
        total_epochs += len(y)

    if max_records is None and observed_subjects != expected_subjects:
        missing_subjects = expected_subjects - observed_subjects
        extra_subjects = observed_subjects - expected_subjects
        raise ValueError(
            f"{split_name} 受试者清单与 NPZ 不一致；"
            f"缺少={sorted(missing_subjects)}，多出={sorted(extra_subjects)}"
        )

    suffix = " (sample load only)" if max_records is not None else ""
    print(
        f"[{split_name}] records_loaded={len(record_ids)}, "
        f"subjects_loaded={len(observed_subjects)}, epochs={total_epochs}{suffix}"
    )
    print(
        f"[{split_name}] class_counts="
        f"{dict(zip(CLASS_NAMES, label_counts.astype(int).tolist()))}"
    )
    return ConcatDataset(parts)
