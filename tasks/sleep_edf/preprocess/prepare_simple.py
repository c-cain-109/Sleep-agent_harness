#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sleep-EDF 预处理脚本

功能：
1. 读取单条或批量 PSG + Hypnogram EDF 文件
2. 提取 EEG Fpz-Cz 通道，重采样到 100 Hz
3. 切分为 30 秒 epoch（3000 采样点）
4. 映射五分类标签：W=0, N1=1, N2=2, N3=3, REM=4
5. 保存为 NPZ 格式（x: N×3000, y: N）

用法：
    # 处理单条记录
    python prepare_simple.py \
        --psg path/to/SC4001E0-PSG.edf \
        --hypno path/to/SC4001EC-Hypnogram.edf \
        --output data/prepared/SC4001E0.npz

    # 批量处理目录
    python prepare_simple.py \
        --input-dir data/raw \
        --output-dir data/prepared
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import signal as sp_signal
import pyedflib


# 五分类标签映射
LABEL_MAP = {
    "Sleep stage W": 0,      # 清醒
    "Sleep stage 1": 1,      # N1
    "Sleep stage 2": 2,      # N2
    "Sleep stage 3": 3,      # N3（原 S3 + S4 合并）
    "Sleep stage 4": 3,
    "Sleep stage R": 4,      # REM
}

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def read_eeg_channel(psg_path: Path, channel_name: str = "EEG Fpz-Cz") -> Tuple[np.ndarray, float]:
    """读取 PSG 文件中的指定 EEG 通道

    Args:
        psg_path: PSG EDF 文件路径
        channel_name: 通道名称

    Returns:
        (信号数据, 采样率)
    """
    with pyedflib.EdfReader(str(psg_path)) as f:
        channel_labels = f.getSignalLabels()

        # 查找通道索引
        try:
            ch_idx = channel_labels.index(channel_name)
        except ValueError:
            raise ValueError(
                f"未找到通道 '{channel_name}'，可用通道：{channel_labels}"
            )

        signal_data = f.readSignal(ch_idx)
        sample_rate = f.getSampleFrequency(ch_idx)

    return signal_data, sample_rate


def resample_signal(signal_data: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """重采样到目标采样率

    Args:
        signal_data: 原始信号
        orig_fs: 原始采样率
        target_fs: 目标采样率

    Returns:
        重采样后的信号
    """
    if abs(orig_fs - target_fs) < 0.01:
        return signal_data

    num_samples = int(len(signal_data) * target_fs / orig_fs)
    resampled = sp_signal.resample(signal_data, num_samples)
    return resampled.astype(np.float32)


def read_hypnogram(hypno_path: Path) -> List[Tuple[float, str]]:
    """读取 Hypnogram 标注文件

    Args:
        hypno_path: Hypnogram EDF 文件路径

    Returns:
        [(开始时间秒, 睡眠阶段文本), ...]
    """
    annotations = []

    with pyedflib.EdfReader(str(hypno_path)) as f:
        # Hypnogram 的标注存储在 annotations 中
        onset_times = f.readAnnotations()[0]  # 开始时间（秒）
        durations = f.readAnnotations()[1]     # 持续时间（秒）
        texts = f.readAnnotations()[2]         # 标注文本

        for onset, duration, text in zip(onset_times, durations, texts):
            annotations.append((float(onset), text.strip()))

    return annotations


def make_epochs(
    signal_data: np.ndarray,
    annotations: List[Tuple[float, str]],
    fs: float,
    epoch_sec: int = 30,
    trim_edge_wake: bool = True,
    edge_wake_minutes: int = 30
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """将信号切分为 epoch 并分配标签

    Args:
        signal_data: EEG 信号（已重采样）
        annotations: 标注列表
        fs: 采样率
        epoch_sec: epoch 长度（秒）
        trim_edge_wake: 是否裁掉开头/结尾的长清醒段
        edge_wake_minutes: 裁剪阈值（分钟）

    Returns:
        (x: N×samples, y: N, stats: 统计信息)
    """
    samples_per_epoch = int(fs * epoch_sec)
    n_epochs = len(signal_data) // samples_per_epoch

    # 切分 epoch
    signal_data = signal_data[:n_epochs * samples_per_epoch]
    x = signal_data.reshape(n_epochs, samples_per_epoch).astype(np.float32)

    # 为每个 epoch 分配标签
    y = np.full(n_epochs, -1, dtype=np.int32)

    for i in range(n_epochs):
        epoch_start = i * epoch_sec
        epoch_center = epoch_start + epoch_sec / 2

        # 找到覆盖该 epoch 中心点的标注
        for onset, text in annotations:
            if onset <= epoch_center < onset + epoch_sec:
                if text in LABEL_MAP:
                    y[i] = LABEL_MAP[text]
                break

    # 统计
    stats = {
        "total_epochs": n_epochs,
        "unlabeled": int((y == -1).sum()),
    }

    # 移除未标注的 epoch
    valid_mask = (y >= 0)
    x = x[valid_mask]
    y = y[valid_mask]

    # 裁掉边缘长清醒段（可选）
    if trim_edge_wake and len(y) > 0:
        edge_epochs = int(edge_wake_minutes * 60 / epoch_sec)

        # 找到第一个非 W epoch
        first_sleep = np.where(y != 0)[0]
        if len(first_sleep) > 0:
            trim_start = max(0, first_sleep[0] - edge_epochs)
        else:
            trim_start = 0

        # 找到最后一个非 W epoch
        last_sleep = np.where(y != 0)[0]
        if len(last_sleep) > 0:
            trim_end = min(len(y), last_sleep[-1] + edge_epochs + 1)
        else:
            trim_end = len(y)

        stats["edge_trimmed"] = (len(y) - (trim_end - trim_start))
        x = x[trim_start:trim_end]
        y = y[trim_start:trim_end]
    else:
        stats["edge_trimmed"] = 0

    # 检查非有限值
    nonfinite_mask = ~np.isfinite(x).all(axis=1)
    if nonfinite_mask.any():
        print(f"  ⚠️  警告：发现 {nonfinite_mask.sum()} 个包含 NaN/Inf 的 epoch，已移除")
        x = x[~nonfinite_mask]
        y = y[~nonfinite_mask]
        stats["nonfinite"] = int(nonfinite_mask.sum())
    else:
        stats["nonfinite"] = 0

    stats["final_epochs"] = len(y)

    # 标签分布
    stats["label_counts"] = {
        CLASS_NAMES[i]: int((y == i).sum())
        for i in range(5)
    }

    return x, y, stats


def process_one_record(
    psg_path: Path,
    hypno_path: Path,
    output_path: Path,
    channel: str = "EEG Fpz-Cz",
    target_fs: float = 100.0,
    epoch_sec: int = 30,
    trim_edge_wake: bool = True
):
    """处理单条记录

    Args:
        psg_path: PSG 文件路径
        hypno_path: Hypnogram 文件路径
        output_path: 输出 NPZ 路径
        channel: EEG 通道名
        target_fs: 目标采样率
        epoch_sec: epoch 长度（秒）
        trim_edge_wake: 是否裁边缘清醒
    """
    record_id = psg_path.stem.replace("-PSG", "")
    print(f"\n{'='*60}")
    print(f"处理记录: {record_id}")
    print(f"  PSG      : {psg_path.name}")
    print(f"  Hypnogram: {hypno_path.name}")

    # 1. 读取 EEG
    signal_data, orig_fs = read_eeg_channel(psg_path, channel)
    print(f"  读取信号: {len(signal_data)} 采样点，采样率 {orig_fs:.1f} Hz")

    # 2. 重采样
    if abs(orig_fs - target_fs) > 0.01:
        signal_data = resample_signal(signal_data, orig_fs, target_fs)
        print(f"  重采样到: {target_fs:.1f} Hz，{len(signal_data)} 采样点")

    # 3. 读取标注
    annotations = read_hypnogram(hypno_path)
    print(f"  读取标注: {len(annotations)} 条")

    # 4. 切分 epoch
    x, y, stats = make_epochs(
        signal_data, annotations, target_fs, epoch_sec, trim_edge_wake
    )

    # 5. 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        record_id=record_id,
        subject_id=record_id[:6],  # SC4001E0 -> SC4001
        fs=target_fs,
        epoch_duration=epoch_sec,
        ch_label=channel,
    )

    print(f"  ✅ 保存: {output_path.name}")
    print(f"  总 epoch: {stats['total_epochs']} → 最终: {stats['final_epochs']}")
    print(f"    未标注: {stats['unlabeled']}, 边缘裁剪: {stats['edge_trimmed']}, "
          f"非有限值: {stats['nonfinite']}")
    print(f"  标签分布: {stats['label_counts']}")


def find_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    """在目录中查找 PSG-Hypnogram 配对

    Args:
        input_dir: 输入目录

    Returns:
        [(PSG路径, Hypnogram路径), ...]
    """
    psg_files = sorted(input_dir.rglob("*-PSG.edf"))
    pairs = []

    for psg in psg_files:
        # 推测 Hypnogram 文件名（SC4001E0-PSG.edf → SC4001EC-Hypnogram.edf）
        base = psg.stem.replace("-PSG", "")

        # Sleep-EDF 规则：SC4001E0 → SC4001EC (E→E, 数字→C)
        if base.endswith("E0"):
            hypno_base = base[:-2] + "EC"
        else:
            hypno_base = base

        hypno_candidates = [
            psg.parent / f"{hypno_base}-Hypnogram.edf",
            psg.parent / f"{base}-Hypnogram.edf",
            psg.parent / f"{base}C-Hypnogram.edf",
        ]

        for hypno in hypno_candidates:
            if hypno.exists():
                pairs.append((psg, hypno))
                break
        else:
            print(f"⚠️  警告：未找到 {psg.name} 的配对 Hypnogram，跳过", file=sys.stderr)

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Sleep-EDF 预处理（简化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 单条记录模式
    parser.add_argument("--psg", type=Path, help="PSG EDF 文件")
    parser.add_argument("--hypno", type=Path, help="Hypnogram EDF 文件")
    parser.add_argument("--output", type=Path, help="输出 NPZ 文件")

    # 批量模式
    parser.add_argument("--input-dir", type=Path, help="输入目录（自动查找 PSG-Hypnogram 配对）")
    parser.add_argument("--output-dir", type=Path, help="输出目录")

    # 参数
    parser.add_argument("--channel", default="EEG Fpz-Cz", help="EEG 通道名 [默认: EEG Fpz-Cz]")
    parser.add_argument("--target-fs", type=float, default=100.0, help="目标采样率 [默认: 100]")
    parser.add_argument("--epoch-sec", type=int, default=30, help="Epoch 长度（秒）[默认: 30]")
    parser.add_argument("--no-trim-edge-wake", action="store_true", help="不裁剪边缘清醒段")

    args = parser.parse_args()

    # 检查模式
    if args.psg and args.hypno and args.output:
        # 单条记录模式
        process_one_record(
            args.psg, args.hypno, args.output,
            args.channel, args.target_fs, args.epoch_sec,
            not args.no_trim_edge_wake
        )

    elif args.input_dir and args.output_dir:
        # 批量模式
        pairs = find_pairs(args.input_dir)

        if not pairs:
            print(f"❌ 错误：在 {args.input_dir} 中未找到任何 PSG-Hypnogram 配对", file=sys.stderr)
            return 1

        print(f"发现 {len(pairs)} 对文件")

        for i, (psg, hypno) in enumerate(pairs, 1):
            record_id = psg.stem.replace("-PSG", "")
            output_path = args.output_dir / f"{record_id}.npz"

            print(f"\n[{i}/{len(pairs)}]")
            try:
                process_one_record(
                    psg, hypno, output_path,
                    args.channel, args.target_fs, args.epoch_sec,
                    not args.no_trim_edge_wake
                )
            except Exception as e:
                print(f"  ❌ 处理失败: {e}", file=sys.stderr)
                continue

        print(f"\n{'='*60}")
        print(f"批量处理完成：成功 {len(pairs)} 条记录")

    else:
        parser.error(
            "必须提供以下之一：\n"
            "  单条记录: --psg + --hypno + --output\n"
            "  批量处理: --input-dir + --output-dir"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
