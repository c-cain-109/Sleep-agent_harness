
from pathlib import Path

import mne

# 以当前脚本所在目录为基准，避免工作目录不同导致找不到文件
data_dir = (
    Path(__file__).resolve().parent
    / "data"
    / "raw"
    / "sleep-edf"
    / "physionet-sleep-data"
)

psg_path = data_dir / "SC4001E0-PSG.edf"
hyp_path = data_dir / "SC4001EC-Hypnogram.edf"

# 1. 检查两个文件是否存在
print("数据目录:", data_dir)

for path in (psg_path, hyp_path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}\n"
            "请确认两个 EDF 文件都放在 physionet-sleep-data 文件夹中。"
        )

    print(f"文件存在: {path.name} ({path.stat().st_size:,} bytes)")

# 2. 读取 PSG 脑电信号
print("\n正在读取 PSG 文件……")

raw = mne.io.read_raw_edf(
    str(psg_path),
    preload=False,
    verbose="ERROR",
)

print("\n--- EEG 信号信息 ---")
print("通道名称:", raw.ch_names)
print("采样率:", raw.info["sfreq"], "Hz")
print(
    "记录时长:",
    round(raw.n_times / raw.info["sfreq"] / 3600, 2),
    "小时",
)

# 3. 读取睡眠阶段标注
print("\n正在读取 Hypnogram 文件……")

annotations = mne.read_annotations(str(hyp_path))

print("\n--- 睡眠阶段标注 ---")
print("标注数量:", len(annotations))
print("标注类别:", sorted(set(annotations.description)))

print("\n前 10 条标注：")
for onset, duration, description in zip(
    annotations.onset[:10],
    annotations.duration[:10],
    annotations.description[:10],
):
    print(
        f"开始时间={onset:8.1f}s | "
        f"持续时间={duration:5.1f}s | "
        f"阶段={description}"
    )

# 4. 检查目标 EEG 通道
if "EEG Fpz-Cz" not in raw.ch_names:
    raise ValueError(
        f"未找到 EEG Fpz-Cz，实际通道为：{raw.ch_names}"
    )

print("\n验证通过：本地 PSG 信号和 Hypnogram 标注均已读取！")

