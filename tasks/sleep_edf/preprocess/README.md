# Sleep-EDF 预处理模块（Sleep-Harness 数据管线）

把 Sleep-EDF Expanded（Sleep Cassette）的原始 EDF 记录转换成 TinySleepNet 训练/评估所需的
30 秒、五分类、单通道 EEG epoch 数据集。

本模块**只做数据预处理**，不训练模型，也不改动 Harness 的评估器、Agent、Git 回滚逻辑或
API 调用代码。

---

## 0. 三步上手

**第 1 步：装依赖**（Python 3.11 建议）

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r tasks/sleep_edf/requirements.txt
```

**第 2 步：把数据放好**

把 `*-PSG.edf` 和 `*-Hypnogram.edf` 放进项目下的约定目录：

```
<project>/data/sleep-edf/
├── SC4001E0-PSG.edf
└── SC4001EC-Hypnogram.edf
```

程序按**文件头的开始时间**自动配对，不依赖文件名，所以文件夹里放多少条记录都行。

**第 3 步：运行**

```bash
# 什么都不用传（PyCharm 里直接点绿三角也是这个效果）
python tasks/sleep_edf/prepare_sleepedf.py
```

两个目录都会自动解析：

| 参数 | 不传时会怎样 |
|---|---|
| `--input-dir` | 依次探测 `$SLEEPEDF_DATA_DIR`、`<cwd>/data/sleep-edf`、`<project>/data/sleep-edf` 等位置，找到含 `*-PSG.edf` 的目录就用它 |
| `--output-dir` | 默认 `<project>/outputs/prepared`，不需要你指定 |

日志开头会明确写出每个路径是**自动发现**还是**你指定的**，不会悄悄读一个你没预期的目录。
如果一个 EDF 都找不到，程序不会只丢一行错误，而是列出**搜索过哪些目录**并给出三种解决办法。

当然也可以显式指定：

```bash
# 处理整个文件夹
python tasks/sleep_edf/prepare_sleepedf.py \
  --input-dir  my_data \
  --output-dir my_out

# 只处理一对文件
python tasks/sleep_edf/prepare_sleepedf.py \
  --psg       my_data/SC4001E0-PSG.edf \
  --hypnogram my_data/SC4001EC-Hypnogram.edf
```

结果全在输出目录里：

```
outputs/prepared/
├── SC4001E0.npz           # 模型直接读这个：x (N,3000) float32 + y (N,) int32
├── dataset_manifest.json  # 处理前后的样本数、五类分布、排除了什么、为什么
├── epoch_metadata.csv     # 每个 epoch 的时间戳和标签，与 x/y 逐行对应
├── preprocessing.log      # 完整运行日志
└── subject_split.json     # 仅加 --make-split 且受试者足够时才生成
```

读数据的代码（PyTorch 里就这么用）：

```python
import numpy as np, torch
d = np.load("outputs/prepared/SC4001E0.npz")
x = torch.from_numpy(d["x"])                      # (N, 3000) float32
y = torch.from_numpy(d["y"]).long()               # (N,) 分类标签 0-4
print(d["record_id"], d["subject_id"], d["fs"])   # SC4001E0 SC4001 100
```

> 默认会裁掉睡眠前后各超过 30 分钟的清醒段（对齐 TinySleepNet 官方脚本），所以本例是 841 段。
> 想保留全部有标签的 epoch（2650 段），加 `--no-trim-edge-wake`。两种结果的对比见第 11 节。

### 在 PyCharm 里运行

项目已附带三个运行配置（右上角下拉框直接选）：

| 配置 | 作用 |
|---|---|
| `1. prepare_sleepedf (zero-config)` | 零参数，用自动发现的目录，输出到 `outputs/prepared` |
| `2. prepare_sleepedf (keep all epochs)` | 同上但不裁边缘清醒，输出到 `outputs/prepared_no_trim` |
| `3. verify_alignment (check pairing)` | 校验 `outputs/prepared` 里每段脑电与标签是否严格对应 |

自己新建配置时，**唯一必须注意的是 Script parameters 可以留空**——两个目录都有默认值。
如果数据不在 `data/sleep-edf/`，在 Script parameters 里填 `--input-dir "你的目录"`，
或在 Environment variables 里设 `SLEEPEDF_DATA_DIR=你的目录`。

> 第二次点运行会提示"输出目录已有结果"并**拒绝覆盖**（防止误删上次结果）。
> 想重跑就加 `--overwrite`，或换一个 `--output-dir`。

---

## 1. 设计要点

| 要求 | 实现方式 |
|---|---|
| 不写死路径 | 输入/输出目录全部来自命令行或配置文件；源码中没有任何绝对路径 |
| 不破坏原始数据 | 源 EDF 以只读方式打开；程序对输入目录只读，不写入 |
| 不自动清空输出目录 | 目标目录已有本工具产物时直接报错退出，必须显式 `--overwrite` |
| 可批量 | 递归扫描目录，按"开始时间"自动配对 PSG 与 Hypnogram，单条失败不影响其余记录 |
| 可审计 | 每个被剔除的 epoch 都有计数和原因，写入 `dataset_manifest.json` 与 `preprocessing.log` |
| 可复现 | 无随机数；`.npz` 使用固定 zip 时间戳，重复运行字节级一致 |
| 可移植 | 依赖仅 `numpy` / `scipy` / `mne`（`PyYAML` 可选），内置纯 NumPy 的 EDF 读取器兜底 |
| 对齐可自证 | 每条 epoch 保留 `epoch_index`，配独立校验器逐字节复核脑电与标签是否真的一一对应 |

---

## 2. 目录结构

```
tasks/sleep_edf/
├── prepare_sleepedf.py          # 主程序（单文件，可独立运行）
├── verify_alignment.py          # 独立对齐校验器（不依赖主程序，用于自证）
├── requirements.txt             # 依赖清单
├── config.example.yaml          # 配置文件示例
├── README.md                    # 本文件
├── TESTLOG.md                   # 实际测试日志与验收对照
└── tests/
    └── test_prepare_sleepedf.py # 单元测试 + 真实数据集成测试
```

输出目录（由 `--output-dir` 指定）内容：

```
<output_dir>/
├── SC4001E0.npz                 # 每条记录一个
├── dataset_manifest.json        # 数据来源、配对、参数、分布、异常
├── epoch_metadata.csv           # 逐 epoch 元数据，行序与 x/y 一致
├── preprocessing.log            # 本次运行完整日志
└── subject_split.json           # 仅 --make-split 时生成
```

---

## 3. 环境准备

参考解释器：**Python 3.11**（3.9+ 亦可）。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r tasks/sleep_edf/requirements.txt
```

依赖说明：

- `numpy`：数值计算与 `.npz` 容器（**唯一必需**依赖）
- `mne`：EDF 读取主路径，同时作为内置解析器的独立交叉校验（建议安装）
- `scipy`：仅重采样路径需要（源文件不是目标采样率时才用），MNE 也依赖它
- `PyYAML`：**可选**，只用 `.yaml` 配置时需要；`.json` 配置无需

测试过的版本组合：numpy 2.4.6 / mne 1.13.2 / scipy 1.17.1 / PyYAML 6.0.3（Python 3.11.9）。

> **没有 MNE 也能跑**：程序内置纯 NumPy 的 EDF/EDF+ 解析器（头信息、TAL 注解、int16 采样解码），
> 用 `--reader native` 或让它自动降级即可。MNE 在导入或读取阶段失败时会记录原因并自动降级，
> 不会静默产出空结果。
>
> 已知不可用的组合（构建本模块时实际踩到）：`mne 1.6.1 + scipy>=1.15`
> （`cannot import name 'sph_harm'`）、`mne 1.13 + numpy 1.26`
> （`module 'numpy.dtypes' has no attribute 'StringDType'`）。`requirements.txt` 已给出可用区间。

---

## 4. 快速开始

### 4.1 单条记录

```bash
python tasks/sleep_edf/prepare_sleepedf.py \
  --psg       "/path/to/SC4001E0-PSG.edf" \
  --hypnogram "/path/to/SC4001EC-Hypnogram.edf" \
  --output-dir ./outputs/sleep_edf
```

### 4.2 批量目录（本次交付的示例就是这条命令生成的）

```bash
python tasks/sleep_edf/prepare_sleepedf.py \
  --input-dir  ./data/sleep-edf/sleep-cassette \
  --output-dir ./outputs/sleep_edf \
  --make-split
```

`--input-dir` 会被递归扫描，所有 `*-PSG.edf` / `*-Hypnogram.edf` 都会尝试配对。配对失败的
情况会写进 `dataset_manifest.json` 的 `errors`，并让进程以退出码 1 结束，但已成功的记录
仍然会正常落盘。

### 4.3 用配置文件

```bash
python tasks/sleep_edf/prepare_sleepedf.py --config tasks/sleep_edf/config.example.yaml
```

命令行显式给出的参数**优先于**配置文件，因此可以拿配置文件当基线、临时覆盖个别参数。

### 4.4 重复运行

如果输出目录里已经有本工具的产物，程序会拒绝运行且**不删除任何文件**：

```
CONFIG ERROR: Output directory already contains results: ...
  Nothing was deleted or overwritten. Re-run with --overwrite ...
```

确认要重跑时加 `--overwrite`（只会替换本工具自己的文件名，不会碰其它文件）。

---

## 5. 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| （无参数） | — | 直接运行即可：输入自动发现、输出到 `outputs/prepared` |
| `--input-dir` | 自动发现 | 递归扫描 EDF 的目录（与 `--psg/--hypnogram` 二选一） |
| `--psg` | 无 | 单条模式的 PSG 文件 |
| `--hypnogram` | 无 | 单条模式的标注文件，必须与 `--psg` 同时给出 |
| `--output-dir` | `<project>/outputs/prepared` | 输出目录 |
| `--config` | 无 | `.yaml` / `.yml` / `.json` 配置文件 |
| `--overwrite` | False | 允许替换本工具之前写入的文件 |
| `--channel` | `EEG Fpz-Cz` | 使用的 EEG 通道名 |
| `--target-fs` | `100` | 目标采样率（Hz） |
| `--epoch-duration` | `30` | epoch 长度（秒），必须为整数 |
| `--reader` | `auto` | `auto` / `mne` / `native` |
| `--no-trim-edge-wake` | False | 关闭边缘清醒裁剪，保留全部已评分 epoch |
| `--edge-wake-epochs` | `60` | 睡眠期前后各保留多少个 W epoch（60 × 30s = 30 分钟） |
| `--make-split` | False | 额外生成 `subject_split.json` |
| `--split-seed` | `42` | 划分随机种子 |
| `--subject-id-regex` | 无 | 提取受试者 ID 的正则（含 1 个捕获组） |
| `--strict` | False | 标注缺口/重叠视为致命错误 |
| `--log-level` | `INFO` | 控制台日志级别（文件日志始终 DEBUG） |

输入目录的自动探测顺序（第一个找到含 `*-PSG.edf` 的就用）：

1. `$SLEEPEDF_DATA_DIR`
2. `<cwd>/data/sleep-edf`、`<cwd>/data/sleep_edf`、`<cwd>/data/sleepedf`、`<cwd>/data`、`<cwd>`
3. `<project>/data/sleep-edf`、`<project>/data/sleep_edf`、`<project>/data/sleepedf`、`<project>/data`、`<project>`

退出码：`0` 全部成功，`1` 有记录/配对失败，`2` 参数或配置问题（含"找不到数据"）。

---

## 6. 配置文件参数

见 `config.example.yaml`，键名与默认值：

```yaml
# 两个路径都可省略：省略时走与命令行相同的自动解析规则
input_dir / output_dir / psg / hypnogram
channel: "EEG Fpz-Cz"
target_fs: 100
epoch_duration: 30
reader: "auto"
overwrite: false

label_map:            # 原始标注文本 -> 标签编号
  "Sleep stage W": 0
  "Sleep stage 1": 1
  "Sleep stage 2": 2
  "Sleep stage 3": 3
  "Sleep stage 4": 3
  "Sleep stage R": 4
excluded_stages:      # 识别但不参与训练；5/6 仅为内部占位码
  "Movement time": 5
  "Sleep stage ?": 6

trim_edge_wake: true
edge_wake_epochs: 60
on_annotation_gap: "drop"      # drop | error
on_stage_conflict: "error"     # error | drop
drop_nonfinite_epochs: true

split_seed: 42
split_ratios: {train: 0.70, val: 0.15, test: 0.15}
subject_id_regex: null
log_level: "INFO"
```

`label_map` / `excluded_stages` 就是"不做隐式映射"的开关：任何**未出现在这两张表里**的
标注文本都会直接报错并终止该记录，而不是被猜成某个睡眠期。

---

## 7. 输出文件

### 7.1 `<record_id>.npz`

必含字段（契约）：

| key | 类型 | 形状/值 | 说明 |
|---|---|---|---|
| `x` | `float32` | `(N, 3000)` | EEG 原始幅值，单位 µV，未滤波、未归一化 |
| `y` | `int32` | `(N,)` | 标签，取值 0–4 |
| `fs` | `int32` | `100` | 采样率 |
| `ch_label` | `str` | `EEG Fpz-Cz` | 通道名 |
| `epoch_duration` | `int32` | `30` | epoch 秒数 |
| `record_id` | `str` | `SC4001E0` | 记录 ID |
| `subject_id` | `str` | `SC4001` | 受试者 ID |

附带字段（不属于最低契约，但便于追溯与直接喂给 TinySleepNet）：

| key | 形状 | 说明 |
|---|---|---|
| `epoch_index` | `(N,)` | 该 epoch 在**原始记录**中的序号，可据此定位回原始时间轴 |
| `label_names` | `(5,)` | `["W","N1","N2","N3","REM"]` |
| `start_datetime` | 标量 | 记录开始时间（ISO 8601） |
| `preprocessing_version` | 标量 | 生成该文件的脚本版本 |
| `n_epochs` | 标量 | 本文件中 epoch 数（= `N`），同名键在参考实现的 npz 里也有 |
| `n_all_epochs` | 标量 | 原始记录的总 epoch 数（裁剪前） |
| `file_duration` | 标量 | 原始信号总时长（秒） |

读写方式：

```python
import numpy as np
with np.load("SC4001E0.npz") as d:
    x, y = d["x"], d["y"]        # (N, 3000) float32, (N,) int32
    print(d["fs"], d["ch_label"], d["record_id"], d["subject_id"])
```

已是深度学习可直接使用的形式：`x` 为 C 连续 `float32`（对应 `torch.float32`），
`y` 为 `int32`（`torch.tensor(y, dtype=torch.long)` 即分类标签），无需额外整形。

`.npz` 用固定 zip 时间戳写出，同样的输入重复运行会得到**字节完全一致**的文件。

### 7.2 `dataset_manifest.json`

主要字段：

| 字段 | 内容 |
|---|---|
| `tool` | 脚本版本、Python / numpy / MNE 版本、参考实现地址 |
| `dataset` | 数据集名、输入目录、扫描到的文件列表、成功/失败记录数、受试者数 |
| `processing_parameters` | 通道、采样率、epoch 长度、标签映射、裁剪策略、对齐规则原文 |
| `pairing` | 每条记录的 PSG/Hypnogram 文件名、配对方式、两个文件的 SHA-256 |
| `records[]` | 每条记录的完整统计（见下） |
| `records[].epoch_accounting` | epoch 账目：信号长度内总数、未被标注覆盖数、超出信号被截断数、边缘裁剪数、Movement 数、未评分数、非有限值数、最终数 |
| `records[].label_distribution` | 五类样本数与占比 |
| `records[].checks` | 逐项校验结果（文件身份、通道、采样率、时间范围……） |
| `records[].notes` | 该记录的所有异常说明 |
| `records[].signal_stats` | 幅值 min/max/mean/std、全平 epoch 数 |
| `totals` | 全体 epoch 数与标签分布 |
| `exclusions` | 各类剔除原因的汇总计数 |
| `anomalies` | 存在异常的记录清单 |
| `errors` | 配对或处理失败的记录及原因 |
| `subject_split` | 受试者划分结果（未请求时为 `null`） |

### 7.3 `epoch_metadata.csv`

一行一个 epoch，**行序与 `x`/`y` 的索引一一对应**：

```
record_id,subject_id,epoch_seq,epoch_index,start_time_sec,start_datetime,label,label_name,raw_stage
SC4001E0,SC4001,0,961,28830,1989-04-25 00:13:30,0,W,Sleep stage W
```

- `epoch_seq`：在本记录输出中的序号（0..N-1），等于 `x` 的行号
- `epoch_index`：在原始记录 30 秒网格中的序号
- `start_time_sec` / `start_datetime`：该 epoch 的起点（相对记录起点 / 绝对时间）
- `raw_stage`：原始标注文本，便于和 hypnogram 直接对照

### 7.4 `preprocessing.log`

完整处理日志：运行环境、配对过程、每项校验、每次剔除的原因与数量、标签分布、异常告警。
`--log-level` 只影响控制台输出，文件日志始终包含 DEBUG 级信息。

---

## 8. 处理规则

### 8.1 读取、配对与身份校验

- PSG 用 MNE（主）与内置解析器（校验）读取；Hypnogram 用内置 EDF+ TAL 解析器读取。
- 配对优先级：
  1. **开始时间完全相同**（EDF 头里的 `dd.mm.yy` + `hh.mm.ss`）——这是记录身份最可靠的证据；
     若同一开始时间对应多份 PSG，再用患者标识串消歧；
  2. 开始时间对不上时，退回文件名前缀匹配（Sleep-EDF 的 `SC4001E0` vs `SC4001EC`），
     并在日志中以 WARNING 明确标注"退化为弱匹配"。
- 无法配对的文件不会被静默忽略，而是写入 `errors` 并让退出码为 1。
- 记录的校验项（结果写进 `records[].checks`）：

| 校验项 | 失败处理 |
|---|---|
| PSG 含真实信号通道 | 致命 |
| Hypnogram 是纯标注文件（有 `EDF Annotations` 通道） | 致命 |
| 两者开始时间一致 | 致命 |
| 患者标识串一致 | 仅告警 |
| 目标通道存在 | 致命（并列出所有可用通道） |
| 每记录时长 > 0、采样率与"采样点数 / 每记录时长"一致 | 致命 |
| 标注时间范围未超出信号时间范围 | 仅告警 + 截断（Sleep-EDF 常见：标注覆盖 24h，PSG 只有 22h） |
| 首个标注从 0 秒开始 | 仅告警 |
| 每个 epoch 采样点数 == `epoch_duration × target_fs`（默认 30 × 100 = 3000） | 致命 |

### 8.2 时间对齐规则（关键）

- **epoch 网格锚定在 PSG 的第一个采样点**：epoch `k` 覆盖 `[k·30s, (k+1)·30s)`。
- 一条标注 `(onset, duration)` 标注的是被 `[onset, onset+duration)` **完整覆盖**的那些 epoch。
- 接受条件：`onset` 和 `duration` 都必须是 `epoch_duration` 的整数倍。这样任何标注边界都
  不可能落在 epoch 内部，从根本上排除"半个 epoch 属于两个睡眠期"的错位。
  不满足则该标注覆盖的 epoch 全部剔除，并在 `notes` 里写明是哪个标注、偏移多少。
- 没有任何标注覆盖的 epoch（标注缺口）→ 剔除并计数；`--strict` 下直接报错。
- 标注超出信号末尾的部分 → 截断并计数。
- 标注之间有重叠且阶段不同 → 默认报错（`on_stage_conflict: error`）。

### 8.3 五分类标签映射

| 原始睡眠阶段 | 标签编号 | 类别 |
|---|---:|---|
| `Sleep stage W` | 0 | W |
| `Sleep stage 1` | 1 | N1 |
| `Sleep stage 2` | 2 | N2 |
| `Sleep stage 3` | 3 | N3 |
| `Sleep stage 4` | 3 | N3（按 AASM 与 S3 合并） |
| `Sleep stage R` | 4 | REM |
| `Movement time` | 排除 | 不参与训练 |
| `Sleep stage ?` | 排除 | 未评分阶段 |

- 未在 `label_map` / `excluded_stages` 中出现的标注文本 → **报错终止该记录**，绝不猜测。
- `Movement time` 和 `Sleep stage ?` 在内部先记为占位码 5 / 6，因为参考实现把它们算作
  "非 W"，会影响边缘裁剪边界；裁剪结束后再整体剔除，所以最终 `y` 只会是 0–4。

### 8.4 边缘裁剪（与参考实现一致，**这是唯一会超出"只排除无标签/Movement"的规则**）

Sleep-EDF 的 hypnogram 通常从下午开始就记 W，睡眠期前后各有数小时清醒。若原样保留，
W 会压倒性主导类别分布。因此默认启用与 TinySleepNet 相同的裁剪：

- 找到第一个和最后一个**非 W** epoch，向前、向后各保留 `edge_wake_epochs = 60` 个 epoch
  （60 × 30s = 30 分钟）；
- 窗口外的 epoch 全部丢弃，计入 `n_dropped_by_edge_trim`；
- 若整条记录没有非 W epoch，则跳过裁剪并告警（不会崩）。

对应参考实现里的 `w_edge_mins = 30` 与 `nw_idx[0] - w_edge_mins * 2`（按 30 秒 epoch 换算
即 60 个 epoch）。

**注意这与"每 30 秒一个样本"的字面理解有差别。** 本记录两种口径的实际结果：

| 口径 | 命令 | epoch 数 | W 占比 | 说明 |
|---|---|---:|---:|---|
| 默认（参考实现，边缘裁剪） | 不加参数 | **841** | 22.4% | 前后长段清醒被裁掉，五类较均衡 |
| 字面规则（只排除无标签/M） | `--no-trim-edge-wake` | **2650** | 75.4% | 保留全部有标签 epoch，W 压倒性 |

两种都正确、都可复现，只是用途不同：训练分类模型通常用前者（否则模型倾向于全猜 W），
做完整性核对或想自行控制采样策略时用后者。切换只需一个开关，不需要改代码。

### 8.5 信号处理

- **只取原始幅值**：不做滤波、不做归一化、不做任何变换。
- 单位是 **µV**，与 pyedflib / 参考实现一致（EDF 头里该通道的物理单位）。
- 采样率已经是 100 Hz 时不做重采样；若源文件不是 100 Hz，才用多项滤波重采样到
  `target_fs`，并在日志中记录 `原采样率 -> 目标采样率`。
- 换算公式（EDF 标准）：
  `phys = phys_min + (digital - dig_min) × (phys_max - phys_min) / (dig_max - dig_min)`
- **双读取器交叉校验**：`--reader` 选定的读取器产出数据后，另用内置解析器再算一遍并比较，
  差异写进日志（本次实测 `max |MNE - native| = 2.84e-14 µV`）。两条独立实现互证，
  可避免单个读取库的静默错误。
- 输出前检查：`NaN` / `Inf`（默认整 epoch 剔除并计数）、全平 epoch（仅告警）、
  幅值范围（超 ±10000 µV 视为可疑并写入 `notes`）。

### 8.6 剔除账目

`records[].epoch_accounting` 记录：

```
n_epochs_in_signal                     信号长度内的 epoch 总数
n_epochs_uncovered_by_annotation       无标注覆盖
n_epochs_annotation_beyond_signal      标注超出信号被截断
n_epochs_eligible_before_trim          裁剪前可用 = 总数 - 未被覆盖
n_dropped_by_edge_trim                 边缘清醒裁剪
n_dropped_movement                     Movement time
n_dropped_unknown_stage                Sleep stage ?
n_dropped_nonfinite                    NaN / Inf
n_epochs_final                         最终样本数
n_epochs_final_check_ok                账目是否配平（程序内置一致性断言）
```

---

## 9. 受试者级数据划分（`--make-split`）

用于未来多受试者实验，规则：

1. 先按受试者聚合记录，**epoch 层面从不跨集合打散**；
2. 受试者列表排序后用 `numpy.random.default_rng(seed)`（默认 seed=42）打乱；
3. 按 70% / 15% / 15% 的**受试者比例**切分，每个集合至少 1 个受试者；
4. 输出后自检：三个集合两两无交集、并集等于全部受试者；自检失败直接抛错；
5. 受试者不足 3 人时**不会伪造划分**，`subject_split.json` 中写
   `"feasible": false` 并给出原因，三个集合均为空列表。

`subject_split.json` 结构：

```json
{
  "feasible": true,
  "seed": 42,
  "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
  "n_subjects": 20, "n_records": 40, "n_epochs": 34000,
  "subjects": {"train": ["SC4001", "..."], "val": ["..."], "test": ["..."]},
  "records":  {"train": ["SC4001E0", "..."], "val": ["..."], "test": ["..."]},
  "per_subject": {"SC4001": {"records": ["SC4001E0"], "n_epochs": 841}},
  "stats": {"train": {"n_subjects": 14, "n_epochs": 21000, "epoch_share": 0.62,
                      "label_distribution": {"W": 0, "N1": 0, "N2": 0, "N3": 0, "REM": 0}}},
  "checks": {"disjoint": true, "covers_all": true, "no_epoch_shuffling": true}
}
```

> 当前只有 1 位受试者，因此本次运行的 `feasible` 为 `false`。这是**预期行为**：
> 单受试者验证预处理，正式划分要等更多受试者数据。

---

## 10. 可复现性

- 预处理全程无随机数；唯一的随机环节是受试者划分，种子固定为 42。
- `.npz` 以固定 zip 时间戳 + 固定字段顺序写出，重复运行**字节一致**。
- 输入文件的 SHA-256 记录在 manifest 的 `pairing` 里，可确认两次运行用的是同一份数据。
- 日志开头记录 Python / numpy / MNE 版本，便于复现环境。
- 单元测试 `TestNpzRoundTrip` 会断言两次写出的 `.npz` 字节相同。

---

## 11. 本次实测结果（SC4001E0，单条记录）

数据：`SC4001E0-PSG.edf` + `SC4001EC-Hypnogram.edf`

### 11.1 对齐验证（最关键的一项）

用**完全绕开本模块**的独立脚本复核（自写 EDF 字节偏移计算 + 正则解析 hypnogram 文本）：

| 检查 | 结果 |
|---|---|
| `x` 每一行是否等于该 epoch 在原始 EDF 中的那 3000 个采样点 | **841 / 841 逐位一致**（float32 精确相等） |
| `y` 每个标签是否等于该 30 秒窗口在 hypnogram 文本里对应的阶段 | **841 / 841 一致** |
| 标注边界是否都落在 30 秒网格上 | 154 条标注，**0 条越界** |
| 入睡瞬间标签是否精确切换 | 30 630 s（epoch 1021）由 W(0) 切到 N1(1)，与标注完全吻合 |
| `epoch_index` 是否连续无重复 | 961–1801 连续，无重复 |

即"脑电片段与睡眠标签一一对应"这一点是被独立证明的，不是靠程序自我声明。
该验证已固化为测试 `TestAlignmentAgainstRawBytes`，每次跑测试都会重新执行。

### 11.2 处理统计

| 项目 | 结果 |
|---|---|
| 记录身份 | 患者 `X F X Female_33yr`，开始 1989-04-24 16:13:00（PSG 与 Hypnogram 一致） |
| PSG 通道 | 7 个；取 `EEG Fpz-Cz`（index 0） |
| 采样率 | 100 Hz（3000 样本 / 30 秒记录，增益 0.09377289 µV/count） |
| 读取器交叉校验 | `max |MNE - native| = 2.84e-14 µV` |
| 信号时长 | 79 500 s（22.08 h，2650 个 30 秒记录） |
| 标注 | 154 条，0 → 86 400 s，无缺口、无重叠 |
| 标注超出信号 | `Sleep stage ?` 一段 6900 s 落在信号之外，截断 230 个 epoch |
| 原始 W 占比 | 75.36%（记录前后各有长段清醒） |
| 最终样本（默认 / 不裁剪） | **841** / **2650** |

默认口径（边缘裁剪）的 epoch 账目：

```
2650 信号内 epoch
  - 0  无标注覆盖
  - 0  Movement time
  - 0  Sleep stage ?
  - 0  NaN / Inf
  - 1809 边缘裁剪（前后长段清醒）
= 841 最终样本
```

标签分布：

| 类别 | 默认（841） | 占比 | 不裁剪（2650） | 占比 |
|---|---:|---:|---:|---:|
| W | 188 | 22.35% | 1997 | 75.36% |
| N1 | 58 | 6.90% | 58 | 2.19% |
| N2 | 250 | 29.73% | 250 | 9.43% |
| N3 | 220 | 26.16% | 220 | 8.30% |
| REM | 125 | 14.86% | 125 | 4.72% |
| **合计** | **841** | 100% | **2650** | 100% |

注意 N1/N2/N3/REM 的绝对数量在两种口径下完全相同（58/250/220/125），差异全部来自 W，
这与"只裁掉前后长段清醒"的规则完全自洽——可以反过来验证裁剪没有误删睡眠段。

数据质量：

| 项目 | 结果 |
|---|---|
| `x` | `(841, 3000)`，`float32`，C 连续 |
| `y` | `(841,)`，`int32`，取值 0–4 全覆盖 |
| NaN / Inf | 0 / 0 |
| 全平 epoch | 0 |
| 幅值范围 | −192.0 … +170.6 µV，均值 0.17 µV |
| npz SHA-256 | `18b94081…4beb66d`（6 次运行、两种读取器均一致） |
| 受试者划分 | 1 位受试者 → `feasible: false`（需要 ≥3），未伪造 |

睡眠期落在记录内的 30 630 s → 52 260 s（约 6 小时），裁剪后保留了 28 830 s → 54 060 s
（含前后各 30 分钟 W）。

---

## 12. 自测

### 12.1 独立对齐校验（推荐每次跑完都执行一次）

```bash
python tasks/sleep_edf/verify_alignment.py \
  --psg       my_data/SC4001E0-PSG.edf \
  --hypnogram my_data/SC4001EC-Hypnogram.edf \
  --npz       my_out/SC4001E0.npz
```

这个脚本**不 import 主程序**，而是自己从零算 EDF 字节偏移、自己正则解析 hypnogram，
然后逐 epoch 断言：

- `x` 那一行是否等于该 epoch 在原始 EDF 里的那 3000 个采样点（float32 精确相等）
- `y` 那个标签是否等于该 30 秒窗口在 hypnogram 里对应的阶段
- 所有标注边界是否都落在 30 秒网格上

输出 `RESULT: PASS` 且退出码为 0 才说明配对无误。实际运行结果（本仓库数据）：

```
sample identity : 841/841 epochs match the raw EDF bytes
label identity  : 841/841 epochs match the hypnogram
annotations on the 30s grid : 154/154

RESULT: PASS - every epoch's samples and label are correctly paired
```

校验器自身也做了"能报错"的证明：把标签整体错位一个 epoch、把脑电整体错位一个 epoch、
把标签随机打乱、把 `epoch_index` 整体 +1，四种篡改全部被判为 FAIL。

### 12.2 单元 / 集成测试

```bash
# 只跑不依赖真实数据的单元测试
python -m unittest tasks.sleep_edf.tests.test_prepare_sleepedf -v

# 连真实 EDF 一起跑（集成测试）
set SLEEPEDF_DATA_DIR=C:\path\to\edf        # Windows
export SLEEPEDF_DATA_DIR=/path/to/edf       # Linux / macOS
python -m unittest tasks.sleep_edf.tests.test_prepare_sleepedf -v
```

测试覆盖：

- 标签映射（S3/S4 合并、M/? 排除、五类只含 0–4）
- epoch 对齐（精确/错位/缺口/超出信号/未知阶段）
- **逐字节对齐验证**（`TestAlignmentAgainstRawBytes`）：自写字节偏移独立复算，断言
  `x` 每行与原始 EDF 采样点 float32 精确相等、`y` 与 hypnogram 文本一致
- `epoch_metadata.csv` 与 `x`/`y` 逐行一一对应
- 受试者划分（单受试者拒绝、无重叠、同受试者不被拆开、种子确定性）
- EDF 头解析、`.npz` 字节确定性、输出目录保护
- 真实数据端到端契约（形状、类型、0–4、无 NaN、重跑被拒）
- `--no-trim-edge-wake` 下 epoch 账目必须配平
- **零参数运行**：输出目录回落到默认、`$SLEEPEDF_DATA_DIR` 生效、约定目录能被发现、
  找不到数据时抛 `UsageError` 并给出可操作提示（不再是一行 `CONFIG ERROR`）

本次实际跑出的结果（**32 项全部通过**）与 7 个负面场景的完整输出，见 **`TESTLOG.md`**。

---

## 13. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `CANNOT START: input data not found` | 没找到任何 `*-PSG.edf`。屏幕上的提示会列出搜索过的目录，按其中一种方式指路即可（`--input-dir` / `SLEEPEDF_DATA_DIR` / 把文件放进 `data/sleep-edf/`） |
| `Output directory already has results from a previous run` | 上次的结果还在。加 `--overwrite` 覆盖，或换 `--output-dir` |
| `No *-PSG.edf files found under ...` | `--input-dir` 指错了，指到存放 EDF 的目录 |
| `Channel 'EEG Fpz-Cz' not found ... Available channels: [...]` | 通道名不同，用报错里列出的名字改 `--channel` |
| `Unmapped hypnogram stage '...'` | 出现了表外的标注文本。确认它该归到哪一类，再补进 `label_map` 或 `excluded_stages`；程序不会替你猜 |
| `Hypnogram is not contiguous` | 标注有缺口且开了 `--strict`；去掉 `--strict` 即可改为"剔除并报告" |
| `start_datetime_matches failed` | PSG 与 Hypnogram 不是同一次记录，配对错了 |
| MNE 读取报错 | 自动降级到内置读取器（`--reader auto`）；也可直接 `--reader native` |

---

## 14. 已知限制

- 第一版只处理 `EEG Fpz-Cz` 单通道；多通道（如 `EEG Pz-Oz`、EOG、EMG）需要扩展
  `channel` 为列表并调整输出形状，当前未实现。
- 只支持 AASM 五分类；N1/N2/N3/REM 的更细划分（如 R 期分阶段）不在范围内。
- 不做信号级伪迹检测（如肌电/眼动污染、导联脱落），只做全平与幅值范围检查；这类清洗
  按任务要求留给后续版本，且必须单独说明方法、参数和理由。
- 边缘裁剪默认开、参数与参考实现对齐；若实验设计需要"全 W 保留"，用
  `--no-trim-edge-wake`（本次数据从 841 段变 2650 段），但类别分布会明显偏向 W。
- 受试者划分按受试者**数量**比例，不按 epoch 数量比例；数据量小时集合间 epoch 数可能不均衡，
  `subject_split.json` 的 `stats.epoch_share` 会如实给出。
- 幅值单位固定为 µV（EDF 头声明的物理单位，与 pyedflib / 参考实现一致）。若下游模型期望
  其他量纲（如 V），需在 dataloader 里自行换算，本模块不做隐式缩放。
