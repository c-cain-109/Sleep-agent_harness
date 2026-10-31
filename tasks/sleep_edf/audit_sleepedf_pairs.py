
from pathlib import Path
from collections import defaultdict

DATA_DIR = (
    Path(__file__).resolve().parent
    / "data/raw/sleep-edf/physionet-sleep-data"
)

psg_files = list(DATA_DIR.glob("*-PSG.edf"))
hyp_files = list(DATA_DIR.glob("*-Hypnogram.edf"))


def make_key(path, suffix):
    # 去掉文件类型后缀，再去掉末尾的设备/评分标识
    stem = path.name.removesuffix(suffix)
    return stem[:-1]


psg_map = defaultdict(list)
hyp_map = defaultdict(list)

for f in psg_files:
    psg_map[make_key(f, "-PSG.edf")].append(f)

for f in hyp_files:
    hyp_map[make_key(f, "-Hypnogram.edf")].append(f)

psg_keys = set(psg_map)
hyp_keys = set(hyp_map)

matched = psg_keys & hyp_keys
missing_hyp = psg_keys - hyp_keys
missing_psg = hyp_keys - psg_keys

# 统计受试者
subjects = set()
for key in matched:
    if key.startswith("SC4") and len(key) >= 7:
        subjects.add(key[3:5])

print("PSG 文件数量:", len(psg_files))
print("Hypnogram 文件数量:", len(hyp_files))
print("成功配对记录:", len(matched))
print("缺少标注文件:", len(missing_hyp))
print("缺少脑电文件:", len(missing_psg))
print("成功配对的受试者数:", len(subjects))

if missing_hyp:
    print("缺少标注的记录:", sorted(missing_hyp))

if missing_psg:
    print("缺少脑电的记录:", sorted(missing_psg))

if any(len(v) != 1 for v in psg_map.values()):
    print("警告：存在重复 PSG 配对键")

if any(len(v) != 1 for v in hyp_map.values()):
    print("警告：存在重复 Hypnogram 配对键")

if (
    len(psg_files) == 153
    and len(hyp_files) == 153
    and len(matched) == 153
    and not missing_hyp
    and not missing_psg
    and all(len(v) == 1 for v in psg_map.values())
    and all(len(v) == 1 for v in hyp_map.values())
):
    print("\n文件配对检查通过！")
else:
    print("\n文件配对检查未通过，请检查异常记录。")
