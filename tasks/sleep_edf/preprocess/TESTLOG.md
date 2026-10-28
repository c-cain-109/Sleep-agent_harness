# 测试日志 — Sleep-EDF 预处理模块

本文件记录的是**实际执行过的**测试与运行结果，命令、输出、退出码均从真实运行中抄录。

测试日期：2026-10-26

**先说结论**：两个文件读取成功、正确配对；`EEG Fpz-Cz` 通道存在且为 100 Hz；
切出 841 个 30 秒样本（每段 3000 点），标签 0–4 齐全；
**每一段脑电与标签的对应关系已用独立脚本逐字节复核，841/841 全部正确**；
无 NaN/Inf；受试者不足 3 人故未划分数据集（按要求不伪造）。详见第 1 节与第 6 节。

---

## 0. 测试环境

| 项 | 值 |
|---|---|
| 操作系统 | Windows 10 (10.0.26200) |
| 解释器 | Python 3.11.9 |
| numpy | 2.4.6 |
| mne | 1.13.2 |
| scipy | 1.17.1 |
| PyYAML | 6.0.3 |

数据：

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `SC4001E0-PSG.edf` | 16.1 MB | `2b40a18adf76af69a42d6db1f30f31d26b369f6d27ca0050ef30147ef892b131` |
| `SC4001EC-Hypnogram.edf` | 4.5 KB | `a4cf67694ade1b52a0ddd06d5817fd45d2d3e8bac5302f640f3e9cfbbf12a996` |

两个文件在处理前后哈希未变化（程序只读打开）。

---

## 1. 对齐验证（最关键：脑电片段与标签是否真的一一对应）

这一项**不依赖本模块的任何解析代码**：另写脚本手工解析 EDF 头、手工算字节偏移、用正则从
hypnogram 原文抽标注，然后与产出的 `.npz` 逐项比对。这样模块自身的 bug 无法自我掩盖。

命令：`python _verify.py`（脚本见交付说明；逻辑已固化为测试 `TestAlignmentAgainstRawBytes`）

实际输出：

```
PSG header (hand parsed):
  channels            : ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']
  EEG Fpz-Cz index    : 0  samples/record=3000
  datarecords         : 2650  record_duration=30.0s
  sfreq               : 100.000 Hz
  gain(uV/count)      : 0.09377289   offset=0.046886
  data-record bytes   : 18240  ch offset in record=0
  signal samples      : 7950000 (79500.0 s)

Hypnogram (hand parsed): 154 annotations, records=1

npz: x=(841, 3000) float32  y=(841,) int32  record_id=SC4001E0 subject_id=SC4001

=== byte-level check: does x[i] equal the raw EDF bytes of epoch epoch_index[i]? ===
  (x is stored as float32, so the raw float64 values are cast to float32 first;
   an exact float32 match proves the samples were not shifted or resampled)
  i=  0 epoch_index= 961  t= 28830s  float32-exact=True   max|diff|=1.906e-06 uV
  i=  1 epoch_index= 962  t= 28860s  float32-exact=True   max|diff|=1.945e-06 uV
  i=  2 epoch_index= 963  t= 28890s  float32-exact=True   max|diff|=3.493e-06 uV
  i=400 epoch_index=1361  t= 40830s  float32-exact=True   max|diff|=1.895e-06 uV
  i=700 epoch_index=1661  t= 49830s  float32-exact=True   max|diff|=1.817e-06 uV
  i=839 epoch_index=1800  t= 54000s  float32-exact=True   max|diff|=1.906e-06 uV
  i=840 epoch_index=1801  t= 54030s  float32-exact=True   max|diff|=1.906e-06 uV
  -> 7/7 spot checks bit-exact at float32, worst numeric diff 3.493e-06 uV

=== full sweep: every epoch re-derived from bytes + relabelled from TAL ===
  signal bit-identical to raw EDF bytes : 841/841
  labels reproduced from TAL text       : 841/841

=== annotation boundaries must land on 30s epoch edges ===
  annotations not on the grid: 0 []

=== first sleep onset: does the label change exactly where the TAL says? ===
  first non-W annotation onset = 30630s (epoch 1021)
    epoch 1019  t=30570s  y=0  TAL says Sleep stage W -> 0
    epoch 1020  t=30600s  y=0  TAL says Sleep stage W -> 0
    epoch 1021  t=30630s  y=1  TAL says Sleep stage 1 -> 1
    epoch 1022  t=30660s  y=1  TAL says Sleep stage 1 -> 1
```

结论：

| 检查项 | 结果 |
|---|---|
| `x` 每行 = 原始 EDF 中该 epoch 的 3000 个采样点 | **841/841 逐位一致** |
| `y` 每个标签 = 该窗口在 hypnogram 中的阶段 | **841/841 一致** |
| 标注边界落在 30 s 网格上 | 154 条，**0 条越界** |
| 入睡瞬间切换 | epoch 1021 精确由 W 切 N1 |
| `epoch_index` 连续性 | 961–1801，连续无重复 |

> `max|diff|` 那列是 1.9e-06 µV 量级，属 float32 对 ~192 µV 数值的正常量化误差
> （相对误差 ~1e-8）。把原始值也转成 float32 后与 `x` 比较是**完全相等**的，
> 说明采样点没有被平移、插值或重采样。

### 1.1 校验器自身的可信度（它真的会报错吗？）

一个"永远通过"的校验器毫无价值。所以另外做了篡改测试：故意破坏 npz，看校验器能否发现。

```
A. 原始正确文件            -> exit 0  RESULT: PASS - every epoch's samples and label are correctly paired
   (期望 exit 0 / PASS)
B. 标签整体错位 1 个 epoch  -> exit 1  RESULT: FAIL - see mismatches above
   (期望 exit 1 / FAIL)
C. 脑电整体错位 1 个 epoch  -> exit 1  RESULT: FAIL - see mismatches above
   (期望 exit 1 / FAIL)
D. 标签随机打乱            -> exit 1  RESULT: FAIL - see mismatches above
   (期望 exit 1 / FAIL)
E. epoch_index 整体 +1     -> exit 1  RESULT: FAIL - see mismatches above
   (期望 exit 1 / FAIL)
```

四种篡改全部被检出、正确文件通过，说明前面的 841/841 PASS 是有效结论而非恒真。

校验器已作为独立工具交付：`tasks/sleep_edf/verify_alignment.py`（不 import 主程序）。

对不裁剪版本（2650 段）同样 PASS：

```
sample identity : 2650/2650 epochs match the raw EDF bytes
label identity  : 2650/2650 epochs match the hypnogram
annotations on the 30s grid : 154/154
```

---

## 2. 单元测试 + 集成测试

命令：

```bash
# 1) 纯单元测试（不需要真实 EDF）
python -m unittest discover -s tasks/sleep_edf/tests -v
#    结果：Ran 32 tests — OK (部分真实数据用例自动跳过)

# 2) 连真实 EDF 的集成测试
set SLEEPEDF_DATA_DIR=<存放 EDF 的目录>
python -m unittest discover -s tasks/sleep_edf/tests -v
#    结果：Ran 32 tests — OK
```

集成测试完整输出：

```
test_epochs_are_consecutive_as_recorded (test_prepare_sleepedf.TestAlignmentAgainstRawBytes.test_epochs_are_consecutive_as_recorded) ... ok
test_full_record_without_trimming (test_prepare_sleepedf.TestAlignmentAgainstRawBytes.test_full_record_without_trimming)
--no-trim-edge-wake must keep every labelled epoch of the record. ... ok
test_samples_are_bit_exact_and_labels_match (test_prepare_sleepedf.TestAlignmentAgainstRawBytes.test_samples_are_bit_exact_and_labels_match) ... ok
test_beyond_signal_truncates (test_prepare_sleepedf.TestEpochAlignment.test_beyond_signal_truncates) ... ok
test_exact_alignment (test_prepare_sleepedf.TestEpochAlignment.test_exact_alignment) ... ok
test_gap_detected (test_prepare_sleepedf.TestEpochAlignment.test_gap_detected) ... ok
test_gap_is_fatal_when_requested (test_prepare_sleepedf.TestEpochAlignment.test_gap_is_fatal_when_requested) ... ok
test_misaligned_annotation_is_excluded (test_prepare_sleepedf.TestEpochAlignment.test_misaligned_annotation_is_excluded) ... ok
test_partial_duration_is_excluded (test_prepare_sleepedf.TestEpochAlignment.test_partial_duration_is_excluded) ... ok
test_s4_annotations_become_label_3 (test_prepare_sleepedf.TestEpochAlignment.test_s4_annotations_become_label_3) ... ok
test_unknown_stage_raises (test_prepare_sleepedf.TestEpochAlignment.test_unknown_stage_raises) ... ok
test_bad_file_raises (test_prepare_sleepedf.TestHeaderParsing.test_bad_file_raises) ... ok
test_missing_input_dir_raises (test_prepare_sleepedf.TestHeaderParsing.test_missing_input_dir_raises) ... ok
test_five_classes_only (test_prepare_sleepedf.TestLabelMapping.test_five_classes_only) ... ok
test_movement_and_unknown_are_excluded (test_prepare_sleepedf.TestLabelMapping.test_movement_and_unknown_are_excluded) ... ok
test_s3_s4_collapse_into_n3 (test_prepare_sleepedf.TestLabelMapping.test_s3_s4_collapse_into_n3) ... ok
test_deterministic_bytes_and_required_keys (test_prepare_sleepedf.TestNpzRoundTrip.test_deterministic_bytes_and_required_keys) ... ok
test_existing_output_requires_overwrite (test_prepare_sleepedf.TestOutputDirectoryGuard.test_existing_output_requires_overwrite) ... ok
test_csv_matches_npz_one_to_one (test_prepare_sleepedf.TestRealData.test_csv_matches_npz_one_to_one)
epoch_metadata.csv must describe exactly the rows in x / y, in order. ... ok
test_end_to_end_contract (test_prepare_sleepedf.TestRealData.test_end_to_end_contract) ... ok
test_header_and_annotations (test_prepare_sleepedf.TestRealData.test_header_and_annotations) ... ok
test_no_disjoint_leakage (test_prepare_sleepedf.TestSubjectSplit.test_no_disjoint_leakage) ... ok
test_same_subject_records_stay_together (test_prepare_sleepedf.TestSubjectSplit.test_same_subject_records_stay_together) ... ok
test_seed_is_deterministic (test_prepare_sleepedf.TestSubjectSplit.test_seed_is_deterministic) ... ok
test_single_subject_reports_infeasible (test_prepare_sleepedf.TestSubjectSplit.test_single_subject_reports_infeasible) ... ok

----------------------------------------------------------------------
Ran 32 tests in 6.962s

OK
```

> 说明：集成测试暴露并修复了一个真实缺陷 —— 日志 `FileHandler` 未关闭会锁住
> `preprocessing.log`，导致 Windows 下输出目录无法移动/删除。修复方式见
> `teardown_logging()`，`run()` 用 `try/finally` 保证释放。

---

## 2. 正式运行（单条记录，MNE 读取）

命令：

```bash
python tasks/sleep_edf/prepare_sleepedf.py \
  --input-dir  "<EDF 目录>" \
  --output-dir "outputs/sleep_edf" \
  --make-split
```

完整日志（即交付的 `preprocessing.log`）：

```
2026-10-26 22:10:53 INFO     ==============================================================================
2026-10-26 22:10:53 INFO     prepare_sleepedf 1.0.0 - Sleep-EDF preprocessing for Sleep-Harness
2026-10-26 22:10:53 INFO     started: 2026-10-26T14:10:53+00:00
2026-10-26 22:10:53 INFO     python: 3.11.9 | platform: Windows-10-10.0.26200-SP0
2026-10-26 22:10:53 INFO     numpy: 2.4.6 | mne: 1.13.2
2026-10-26 22:10:53 INFO     output dir: outputs\sleep_edf
2026-10-26 22:10:53 INFO     Discovered 1 PSG file(s) and 1 hypnogram file(s)
2026-10-26 22:10:53 INFO     pairs to process: 1
2026-10-26 22:10:53 INFO     ------------------------------------------------------------------------------
2026-10-26 22:10:53 INFO     record SC4001E0 (subject SC4001)
2026-10-26 22:10:53 INFO       PSG       : <EDF 目录>\SC4001E0-PSG.edf
2026-10-26 22:10:53 INFO       hypnogram : <EDF 目录>\SC4001EC-Hypnogram.edf
2026-10-26 22:10:53 INFO       paired by : start_datetime
2026-10-26 22:10:53 INFO     SC4001E0: signal 79500.000s, annotations 0 - 86400.000s (154 entries)
2026-10-26 22:10:53 WARNING  SC4001E0: annotation_within_signal failed - annotations end at 86400.0s, signal ends at 79500.0s
2026-10-26 22:10:54 INFO     SC4001E0-PSG.edf: reader cross-check OK (max |MNE - native| = 2.84e-14 uV)
2026-10-26 22:10:54 INFO     SC4001E0: read 7950000 samples at 100.000 Hz via mne
2026-10-26 22:10:55 INFO       wrote SC4001E0.npz
2026-10-26 22:10:55 INFO       epochs: signal=2650 eligible=2650 -> edge_trim=-1809 movement=-0 unknown=-0 nonfinite=-0 => final=841
2026-10-26 22:10:55 INFO       label distribution: {'W': 188, 'N1': 58, 'N2': 250, 'N3': 220, 'REM': 125}
2026-10-26 22:10:55 INFO     wrote epoch_metadata.csv (841 rows)
2026-10-26 22:10:55 WARNING  Subject split NOT created: Only 1 subject(s) available; a 70/15/15 subject-level split needs at least 3 subjects (ideally >= 20). No split was produced.
2026-10-26 22:10:55 INFO     ------------------------------------------------------------------------------
2026-10-26 22:10:55 INFO     TOTAL epochs: 841 | distribution: {'W': 188, 'N1': 58, 'N2': 250, 'N3': 220, 'REM': 125}
2026-10-26 22:10:55 INFO     TOTAL label percent: {'W': '22.35%', 'N1': '6.90%', 'N2': '29.73%', 'N3': '26.16%', 'REM': '14.86%'}
2026-10-26 22:10:55 INFO     exclusions: {'movement_time_epochs': 0, 'unknown_stage_epochs': 0, 'edge_trim_epochs': 1809, 'uncovered_epochs': 0, 'nonfinite_epochs': 0}
2026-10-26 22:10:55 INFO     wrote outputs\sleep_edf\dataset_manifest.json
2026-10-26 22:10:55 INFO     done
```

退出码：**0**

### 日志要点解读

| 行 | 含义 |
|---|---|
| `signal 79500.000s, annotations 0 - 86400.000s (154 entries)` | PSG 22.08 h，hypnogram 覆盖 24 h、154 条标注 |
| `annotation_within_signal failed` | 标注比信号长 6900 s，属 Sleep-EDF 常态；**告警而非致命**，后续截断 |
| `reader cross-check OK (max |MNE - native| = 2.84e-14 uV)` | MNE 与内置解析器结果一致到机器精度，两条读取路径互相印证 |
| `epochs: ... edge_trim=-1809 ... final=841` | 2650 → 841，唯一剔除原因是边缘清醒裁剪 |
| `Subject split NOT created` | 只有 1 位受试者，按要求拒绝伪造划分 |

---

## 3. 输出产物核对

默认口径（`--output-dir outputs/sleep_edf`）：

```
outputs/sleep_edf/
├── SC4001E0.npz          4 734 658 B
├── dataset_manifest.json     7 975 B
├── epoch_metadata.csv       58 758 B   (841 行 + 表头)
├── preprocessing.log         2 666 B
└── subject_split.json          520 B
```

不裁剪口径（`outputs/sleep_edf_no_trim` + `--no-trim-edge-wake`），作为对照：

```
outputs/sleep_edf_no_trim/
├── SC4001E0.npz         15 337 056 B
├── dataset_manifest.json     7 846 B
├── epoch_metadata.csv      183 785 B   (2650 行 + 表头)
├── preprocessing.log         2 678 B
└── subject_split.json          521 B
```

`SC4001E0.npz` 内容（默认口径）：

```
keys          : ['ch_label', 'epoch_duration', 'epoch_index', 'file_duration',
                 'fs', 'label_names', 'n_all_epochs', 'n_epochs',
                 'preprocessing_version', 'record_id', 'start_datetime',
                 'subject_id', 'x', 'y']
x             : shape=(841, 3000)  dtype=float32  (10.09 MB, C 连续)
y             : shape=(841,)       dtype=int32    min=0 max=4 unique=[0 1 2 3 4]
fs            : 100            (int32)
ch_label      : 'EEG Fpz-Cz'
epoch_duration: 30             (int32)
record_id     : 'SC4001E0'
subject_id    : 'SC4001'
n_epochs      : 841   n_all_epochs: 2650   file_duration: 79500.0
finite        : True   NaN=0  Inf=0
amplitude     : min=-192.00 max=170.62 mean=0.17 std=23.02  (uV)
per-epoch std : min=7.225 max=57.340  (无全平 epoch)
epoch_index   : 961..1801 连续
```

`epoch_metadata.csv` 首尾：

```
record_id,subject_id,epoch_seq,epoch_index,start_time_sec,start_datetime,label,label_name,raw_stage
SC4001E0,SC4001,0,961,28830,1989-04-25 00:13:30,0,W,Sleep stage W
SC4001E0,SC4001,1,962,28860,1989-04-25 00:14:00,0,W,Sleep stage W
SC4001E0,SC4001,2,963,28890,1989-04-25 00:14:30,0,W,Sleep stage W
...
SC4001E0,SC4001,839,1800,54000,1989-04-25 07:13:00,0,W,Sleep stage W
SC4001E0,SC4001,840,1801,54030,1989-04-25 07:13:30,0,W,Sleep stage W
```

时间轴自校验：PSG 起点 `1989-04-24 16:13:00` + 28 830 s = `1989-04-25 00:13:30` ✓，
与第 0 行完全一致，说明 epoch 索引、秒偏移、绝对时间三者闭合。

`epoch_metadata.csv` 的 `raw_stage → label` 交叉表（可直接验证 S3/S4 合并规则）：

```
raw_stage -> label counts
  Sleep stage 1      {1: 58}
  Sleep stage 2      {2: 250}
  Sleep stage 3      {3: 101}
  Sleep stage 4      {3: 119}
  Sleep stage R      {4: 125}
  Sleep stage W      {0: 188}
total rows: 841
```

S3(101) + S4(119) = 220 = N3 总数，合并规则无误；且没有任何一行出现 5 或 6。

`dataset_manifest.json` 关键片段：

```json
"epoch_accounting": {
  "n_epochs_in_signal": 2650,
  "n_epochs_uncovered_by_annotation": 0,
  "n_epochs_annotation_beyond_signal": 230,
  "n_epochs_eligible_before_trim": 2650,
  "edge_trim": {"applied": true, "start_epoch": 961, "end_epoch": 1801,
                "first_non_wake_epoch": 1021, "last_non_wake_epoch": 1741,
                "edge_wake_epochs": 60},
  "n_dropped_by_edge_trim": 1809,
  "n_dropped_movement": 0,
  "n_dropped_unknown_stage": 0,
  "n_dropped_nonfinite": 0,
  "n_epochs_final": 841,
  "n_epochs_final_check_ok": true
},
"raw_stage_share_of_record_percent": {
  "Sleep stage W": 75.36, "Sleep stage 1": 2.19, "Sleep stage 2": 9.43,
  "Sleep stage 3": 3.81, "Sleep stage 4": 4.49, "Sleep stage R": 4.72
}
```

原始记录里 W 占 **75.36%**（前面 8.5 h 清醒 + 结尾 7 h 清醒），这正是必须做边缘裁剪的原因；
裁剪后 W 降到 22.35%，五类分布趋于合理。

---

## 4. 负面 / 边界场景（7/7 通过）

| # | 场景 | 期望 | 实际 | 结果 |
|---|---|---|---|---|
| A | 通道名不存在 | 明确报错并列出可用通道 | 见下方输出，退出码 1 | PASS |
| B | 目录里只有 PSG、没有 Hypnogram | 明确报错 | 退出码 2 | PASS |
| C | 标注文件开始时间与 PSG 不一致 | 判为配对错误 | 见下方输出，退出码 1 | PASS |
| D | 输出目录已有结果、未加 `--overwrite` | 拒绝且不删文件 | 退出码 2，原文件仍在 | PASS |
| E | `--output-dir` 等于 `--input-dir` | 拒绝，保护原始数据目录 | 退出码 2 | PASS |
| F | 缺少 `--output-dir` | 配置错误 | 退出码 2 | PASS |
| G | 真实数据 + `--strict`（标注无缺口） | 正常通过 | 退出码 0，841 epochs | PASS |

场景 A 的实际报错：

```
ERROR  SC4001E0 FAILED: Channel 'EEG Oz-A1' not found in SC4001E0-PSG.edf.
Available channels: ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal',
'EMG submental', 'Temp rectal', 'Event marker']
```

场景 B 的实际报错：

```
CONFIG ERROR: No *-Hypnogram.edf files found under <只含 PSG 的目录>
```

场景 C 的实际报错（把 hypnogram 副本头部的日期改成 25.04.89 后配对）：

```
ERROR  SC4001E0 FAILED: SC4001E0: start_datetime_matches failed -
PSG 1989-04-24T16:13:00 vs hypnogram 1989-04-25T16:13:00
```

场景 D 的实际输出：

```
CONFIG ERROR: Output directory already contains results: <输出目录>
  existing entries: ['SC4001E0.npz']
  Nothing was deleted or overwritten. Re-run with --overwrite to replace this tool's own outputs, or pick a new --output-dir.
已有文件是否仍存在且未被修改: True
```

场景 E 的实际输出：

```
CONFIG ERROR: --output-dir must not be the same directory as --input-dir (<目录>);
the raw EDF files are read-only inputs.
```

---

## 5. 可复现性验证

同一份输入、同一套参数，多次独立运行生成的 `SC4001E0.npz` 字节完全相同：

| 运行 | 读取器 | 输出目录 | npz SHA-256 |
|---|---|---|---|
| 1 | mne | 临时目录（run 1） | `18b940815a11ede0d92654f996f6bf2c603f695ae074c4e5d0f1e24d84beb66d` |
| 2 | mne | 临时目录（run 2） | 同上 |
| 3 | native | 临时目录（run 3） | 同上 |
| 4 | mne | `outputs/sleep_edf`（交付示例） | 同上 |

实际校验输出：

```
default config, run 1        : 18b940815a11ede0d92654f996f6bf2c603f695ae074c4e5d0f1e24d84beb66d
default config, run 2        : 18b940815a11ede0d92654f996f6bf2c603f695ae074c4e5d0f1e24d84beb66d
--reader native              : 18b940815a11ede0d92654f996f6bf2c603f695ae074c4e5d0f1e24d84beb66d
delivered outputs/sleep_edf  : 18b940815a11ede0d92654f996f6bf2c603f695ae074c4e5d0f1e24d84beb66d

run1 == run2                 : True
run1 == native reader        : True
run1 == delivered sample     : True

--no-trim-edge-wake          : dfcbfab552d542cabfc0ee17c93479938fa201ef2e8e0fef756797ea0b898a23
```

几个结论：

- **跨读取器一致**：MNE 路径与内置解析器输出的 `.npz` 字节相同（日志中 `max |MNE - native| = 2.84e-14 uV`）。
- **跨运行一致**：`.npz` 用固定 zip 时间戳写出，重复运行不改一个字节。
- **原始数据未变**：运行前后两个 EDF 的 SHA-256 与处理前相同
  （`2b40a18a…92b131` / `a4cf6769…12a996`）。
- 裁剪与不裁剪是两个不同的数据集，哈希自然不同，属预期。

---

## 6. 验收标准逐条对照

按需求原文逐条核对：

| # | 要求 | 结果 | 证据 |
|---|---|---|---|
| 1 | 读取两个文件，确认能正确对应 | ✅ | 开始时间/患者标识逐一校验；154 条标注全部覆盖信号范围 |
| 2 | 使用 `EEG Fpz-Cz` 单通道 | ✅ | `channel_present` 检查通过；`ch_label='EEG Fpz-Cz'` |
| 3 | 100 Hz 采样率 | ✅ | `native_sfreq=100.0`；`n_epoch_samples=3000` |
| 4 | 切 30 秒一段，每段 3000 点 | ✅ | `x` 每行长度集合 = `{3000}`；`epoch_duration=30` |
| 5 | 五分类 W/N1/N2/N3/REM 映射 | ✅ | `y` unique = `[0 1 2 3 4]`，无 5/6 |
| 6 | S3、S4 合并为 N3 | ✅ | 原始标注 S3 得 101 个 epoch、S4 得 119 个 epoch，两者都进入标签 3，合计 220 = N3 总数（101+119=220 ✓） |
| 7 | 无有效标签 / Movement 要排除，不随意分配 | ✅ | `Movement time`→占位码 5、`Sleep stage ?`→占位码 6，最终整体剔除；表外阶段直接报错 |
| 8 | **脑电片段与标签一一对应** | ✅ | 独立脚本复核：**841/841 采样点逐位一致 + 841/841 标签一致**（见第 1 节） |
| 9 | 保存为压缩 `.npz` | ✅ | 4.73 MB（841×3000 float32 原始 10.09 MB，压缩比 2.1×） |
| 10 | 保留记录编号与受试者编号 | ✅ | `record_id`、`subject_id` 两个键 |
| 11 | 统计处理前后样本数 | ✅ | 2650 → 841，账目配平（`n_epochs_final_check_ok: true`） |
| 12 | 统计五类各自样本数 | ✅ | W188 / N1 58 / N2 250 / N3 220 / REM 125，写入 manifest 与日志 |
| 13 | 记录排除了什么、为什么 | ✅ | `epoch_accounting` + `notes` + `exclusions`，逐项分类计数 |
| 14 | 读失败/通道缺失/对不上要明确报错，不静默跳过 | ✅ | 负面场景 A/B/C，退出码非 0 且错误信息指名具体原因与可用通道 |
| 15 | 可批量处理整个文件夹 | ✅ | `--input-dir` 递归扫描、按开始时间自动配对；单条失败不影响其余 |
| 16 | 按受试者划分 train/val/test，同一人不跨集合 | ✅（未启用） | 功能已实现并由 4 个单测覆盖；当前 1 位受试者 → `feasible: false`，未伪造 |
| 17 | 当前不强行划分 | ✅ | 只有传 `--make-split` 才生成；即使传了也在数据不足时明确拒绝 |
| 18 | 代码不写死绝对路径 | ✅ | `grep` 源码无 `C:\`／`/Users/`／盘符路径；全部来自参数或配置 |
| 19 | 不修改原有项目其他代码 | ✅ | 只在 `tasks/sleep_edf/` 下新增文件，未触碰 Harness 其它模块 |
| 20 | 重复运行可复现，不清空已有输出 | ✅ | 4 次运行 npz 哈希相同；已有结果时拒绝且不删除任何文件 |

补充：受试者级划分要求"数据量不足时必须明确报告，不得伪造"。本次只有 1 位受试者，
`subject_split.json` 输出：

```json
{
  "feasible": false,
  "reason": "Only 1 subject(s) available; a 70/15/15 subject-level split needs at least 3 subjects (ideally >= 20). No split was produced.",
  "subjects": {"train": [], "val": [], "test": []},
  "checks": {"disjoint": null, "covers_all": null}
}
```

划分逻辑本身由 `TestSubjectSplit` 的 4 个用例覆盖（多受试者无重叠、同一受试者记录不被拆开、
种子确定性、单受试者拒绝）。

---

## 7. 唯一需要你确认的口径差异

需求里"排除"只提了两种情形：无有效标签、Movement 标注。而本模块默认还多做了一件参考
实现（TinySleepNet 官方脚本）会做的事：**裁掉睡眠期前后各超过 30 分钟的清醒段**。

| 口径 | 命令 | epoch 数 | W 占比 |
|---|---|---:|---:|
| 默认（含边缘裁剪，对齐官方） | 不加参数 | 841 | 22.4% |
| 严格按字面规则（只排除无标签/M） | `--no-trim-edge-wake` | 2650 | 75.4% |

两种都完整交付为示例目录（`outputs/sleep_edf` 与 `outputs/sleep_edf_no_trim`）。
如果你要的口径是后者，把开关加上即可，或告诉我把默认值改过来。

值得注意的交叉验证：两种口径下 N1/N2/N3/REM 的**绝对数量完全相同**（58/250/220/125），
只有 W 从 1997 降到 188，说明裁剪只动了清醒段，没有误删睡眠段。

---

## 8. 零参数运行（2026-10-27 改进）

### 8.1 问题

在 PyCharm 里直接点绿三角（不配 Script parameters）时，旧版报：

```
CONFIG ERROR: --output-dir (or output_dir in the config) is required
```

这行错误只说"缺参数"，没告诉用户该填什么、值该写在哪，等于把配置工作全推回给人。

### 8.2 改法

- `--output-dir` 省略 → 默认 `<project>/outputs/prepared`
- `--input-dir` 省略 → 依次探测 `$SLEEPEDF_DATA_DIR`、`<cwd>/data/sleep-edf` 等位置，
  取第一个含 `*-PSG.edf` 的目录
- 两者都解析不到时，抛 `UsageError`，打印搜索过的目录清单 + 三种解决办法
- 日志开头标注每个路径是 `(auto-discovered)` 还是 `(default)`，避免"悄悄读了别的目录"
- manifest 里新增 `dataset.path_origin`，把"这个路径是推断来的"落盘可查

### 8.3 实测

数据放在 `<project>/data/sleep-edf/` 后，零参数运行：

```
input dir : ...\2026-10-26-20-52-55\data\sleep-edf  (auto-discovered; pass --input-dir to override)
output dir: ...\2026-10-26-20-52-55\outputs\prepared  (default; pass --output-dir to override)
...
TOTAL epochs: 841 | distribution: {'W': 188, 'N1': 58, 'N2': 250, 'N3': 220, 'REM': 125}
exit=0
```

产物经独立校验器复核仍为 841/841 + 841/841 PASS。

找不到数据时的输出（在一个空目录里运行）：

```
========================================================================
CANNOT START: input data not found
========================================================================
Could not find any *-PSG.edf recording, and no --input-dir / --psg was given.

This script needs to know where your EDF files are.

Three ways to tell it, pick one:

  1) Put the files in a conventional folder and just press Run again:
       <project>/data/sleep-edf/<your>*-PSG.edf
       <project>/data/sleep-edf/<your>*-Hypnogram.edf

  2) Pass the folder explicitly (PyCharm: Run > Edit Configurations
     > Script parameters):
       --input-dir "D:\path\to\edf_folder"

  3) Set an environment variable (PyCharm: Run > Edit Configurations
     > Environment variables):
       SLEEPEDF_DATA_DIR=D:\path\to\edf_folder

  Or process a single pair directly:
       --psg "X-PSG.edf" --hypnogram "X-Hypnogram.edf"

Output defaults to <project>/outputs/prepared when --output-dir is omitted,
so you normally do NOT need to pass --output-dir at all.

Folders that were searched for *-PSG.edf:
  [missing] ...\data\sleep-edf
  [missing] ...\data\sleep_edf
  [missing] ...\data\sleepedf
  [missing] ...\data
  [no EDF ] ...
========================================================================
```

目录状态区分三态：`[has EDF]` / `[no EDF ]` / `[missing]`，不会把"目录存在但没数据"
误报成"找到了"。

第二次运行的提示也改成了动作优先：

```
Output directory already has results from a previous run:
  ...\outputs\prepared
  existing entries: ['SC4001E0.npz', 'dataset_manifest.json', ...]

Nothing was deleted or overwritten. Pick one:
  - keep this run's result and write elsewhere:
        --output-dir another_folder
  - or replace it (only this tool's own files are rewritten):
        --overwrite
```

### 8.4 新增测试（7 项，总数 25 → 32）

| 测试 | 断言 |
|---|---|
| `test_output_dir_falls_back_to_project_outputs_prepared` | 省略时落到默认目录且标记 `default` |
| `test_explicit_output_dir_is_not_marked_as_default` | 显式指定时不打默认标记 |
| `test_env_var_points_at_the_edf_folder` | `SLEEPEDF_DATA_DIR` 生效且优先级最高 |
| `test_conventional_data_folder_is_found` | `data/sleep-edf/` 能被发现 |
| `test_nothing_found_raises_usage_error_with_hint` | 找不到时 hint 含三种办法并列出搜索路径 |
| `test_missing_input_exits_with_code_2_and_guidance` | 端到端 exit 2，且**旧的一行错误不再出现** |
| `test_build_config_raises_usage_error_when_input_missing` | `UsageError` 是 `ConfigError` 子类，带 hint |

### 8.5 顺带交付的 PyCharm 运行配置

`.idea/runConfigurations/` 下三个，右上角下拉即可选：

| 配置 | 参数 |
|---|---|
| `1. prepare_sleepedf (zero-config)` | 空（全靠默认） |
| `2. prepare_sleepedf (keep all epochs)` | `--output-dir outputs/prepared_no_trim --no-trim-edge-wake` |
| `3. verify_alignment (check pairing)` | 校验 `outputs/prepared` 的对齐 |

另外把样例数据放在 `data/sleep-edf/`，这样首次点运行就能直接成功。
