
from pathlib import Path
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/sleep-edf-sc-full"

split = json.loads(
    (DATA_DIR / "subject_split.json").read_text(encoding="utf-8")
)

split_records = {
    name: set(split["records"][name])
    for name in ("train", "val", "test")
}

split_subjects = {
    name: set(split["subjects"][name])
    for name in ("train", "val", "test")
}

files = sorted(DATA_DIR.glob("*.npz"))

total_epochs = 0
class_counts = np.zeros(5, dtype=np.int64)
split_class_counts = {
    name: np.zeros(5, dtype=np.int64)
    for name in ("train", "val", "test")
}

errors = []

print("NPZ 文件数量:", len(files))

if len(files) != 153:
    errors.append(f"预期 153 个 NPZ，实际为 {len(files)}")

all_split_records = (
    split_records["train"]
    | split_records["val"]
    | split_records["test"]
)

if not (
    split_records["train"].isdisjoint(split_records["val"])
    and split_records["train"].isdisjoint(split_records["test"])
    and split_records["val"].isdisjoint(split_records["test"])
):
    errors.append("记录划分存在交叉")

for path in files:
    try:
        with np.load(path, allow_pickle=False) as data:
            x = data["x"]
            y = data["y"]

            if x.ndim != 2 or x.shape[1] != 3000:
                errors.append(f"{path.name}: x 形状错误 {x.shape}")

            if y.ndim != 1 or len(x) != len(y):
                errors.append(f"{path.name}: x/y 数量或维度不一致")

            if x.dtype != np.float32:
                errors.append(f"{path.name}: x 不是 float32")

            if not np.isfinite(x).all():
                errors.append(f"{path.name}: EEG 存在 NaN/Inf")

            if not np.isin(y, [0, 1, 2, 3, 4]).all():
                errors.append(f"{path.name}: 存在非法标签")
                continue

            # 从记录编号提取受试者编号
            match = re.match(r"^(SC4\d{2})", path.stem)
            if not match:
                errors.append(f"{path.name}: 无法识别受试者编号")
                continue

            subject_id = match.group(1)

            # 检查该记录是否在划分清单中
            membership = [
                name for name, records in split_records.items()
                if path.stem in records
            ]

            if len(membership) != 1:
                errors.append(f"{path.name}: 记录划分不唯一")
                continue

            group = membership[0]

            if subject_id not in split_subjects[group]:
                errors.append(f"{path.name}: 受试者与划分清单不一致")

            if "subject_id" in data:
                saved_subject = str(data["subject_id"].item())
                if saved_subject != subject_id:
                    errors.append(
                        f"{path.name}: NPZ 内 subject_id 不正确"
                    )

            if "fs" in data and int(data["fs"].item()) != 100:
                errors.append(f"{path.name}: 采样率不是 100 Hz")

            if "epoch_duration" in data:
                if int(data["epoch_duration"].item()) != 30:
                    errors.append(f"{path.name}: epoch 不是 30 秒")

            total_epochs += len(y)
            class_counts += np.bincount(y, minlength=5)
            split_class_counts[group] += np.bincount(y, minlength=5)

    except Exception as exc:
        errors.append(f"{path.name}: 读取失败：{exc}")

print("\n===== 数据统计 =====")
print("有效 Epoch 数量:", total_epochs)

for i, name in enumerate(["W", "N1", "N2", "N3", "REM"]):
    print(f"{name}: {class_counts[i]:,}")

print("\n===== 各集合标签分布 =====")
for group, counts in split_class_counts.items():
    print(group, {
        name: int(counts[i])
        for i, name in enumerate(["W", "N1", "N2", "N3", "REM"])
    })

print("\n===== 检查结果 =====")
print("错误数量:", len(errors))

for error in errors[:30]:
    print("ERROR:", error)

if total_epochs != split["n_epochs"]:
    errors.append("NPZ Epoch 总数与 subject_split.json 不一致")

if errors:
    print("\n数据质量检查未通过，请先修复问题。")
    raise SystemExit(1)

print("\n全部数据质量检查通过！")
