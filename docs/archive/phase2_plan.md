# 接入睡眠分期任务计划

## 目标
将 partner 的预处理模块整合到 Sleep-Harness，完成第一次真实睡眠数据的训练管线验证（不追求泛化性能）。

## 当前状态
- ✅ Harness 基础框架（白名单、子进程执行、Git、Agent API）
- ✅ TinySleepNet 模型接口（PyTorch）
- ✅ 玩具任务（二次函数优化）已跑通
- ✅ 单条 Sleep-EDF 记录（SC4001）已下载
- ✅ Partner 预处理代码已交付（`pretreat/`）
- ❌ 预处理输出（`.npz`）尚未生成
- ❌ `train.py` 未在真实数据上验证

## 核心问题
1. **单受试者限制**：只有 1 位受试者，无法按受试者划分 train/val
   - **方案**：按时间切分（前 80% / 后 20%），仅验证管线，不作为实验结果
   - 输出中明确标注"同一受试者，指标不代表泛化能力"

2. **路径不匹配**：
   - `test_task/` 已迁移到 `tasks/test_task/`，但 `main.py`、`harness/evaluate.py`、`whitelist.json` 仍引用旧路径
   - 完整 EDF 文件在 `tasks/sleep_edf/data/physionet-sleep-data/`，不在 `data/raw/sleep-edf/`

3. **输出格式不明确**：
   - Partner 脚本有 150 行英文注释，25 条日志
   - 输出目录结构、文件名、NPZ 键值需要核查

## 实施步骤

### 1. 整合预处理模块 (30 分钟)
- [x] 移动 `pretreat/` → `tasks/sleep_edf/preprocess/`
- [ ] 翻译核心注释为中文（保留技术术语英文）
- [ ] 改进日志输出：
  - 处理进度（x/y 条记录）
  - epoch 数量统计（总数 / 各标签分布）
  - 输出文件路径
- [ ] 调整默认路径：
  - `--input-dir` 默认指向 `tasks/sleep_edf/data/physionet-sleep-data/`
  - `--output-dir` 默认指向 `tasks/sleep_edf/data/prepared/`

### 2. 生成预处理数据 (5 分钟)
```bash
cd tasks/sleep_edf/preprocess
python prepare_sleepedf.py \
  --input-dir ../data/physionet-sleep-data \
  --output-dir ../data/prepared \
  --overwrite
```
- [ ] 确认输出：
  - `SC4001E0.npz`：包含 `x` (N, 3000) 和 `y` (N,)
  - `manifest.json` 或 `subject_split.json`：记录元数据
- [ ] 检查标签分布（5 类：W/N1/N2/N3/REM）

### 3. 适配 train.py 数据加载 (20 分钟)
当前 `load_npz_directory()` 假设 train/val 在不同目录，需改为：
- [ ] 读取单个 `.npz` 文件
- [ ] 按时间索引切分（前 80% 训练 / 后 20% 验证）
- [ ] 添加数据检查：
  - 形状验证（`x`: (N, 3000), `y`: (N,)）
  - 标签范围验证（0-4）
  - 有限数值验证（无 inf/nan）
- [ ] 在输出中明确标注：
  ```python
  print("⚠️  警告：train/val 来自同一受试者的不同时间段")
  print("   仅用于验证数据管线，不代表模型泛化能力")
  ```

### 4. 运行第一次真实训练 (10 分钟)
```bash
cd tasks/sleep_edf
python train.py \
  --data-file data/prepared/SC4001E0.npz \
  --epochs 5 \
  --batch-size 32 \
  --device cpu \
  --output-dir outputs/pipeline_test
```
- [ ] 验证目标：
  - 数据加载成功
  - 模型前向/反向计算正常
  - 输出指标（Balanced Acc、Macro F1）
  - 保存 checkpoint 和 `metrics.jsonl`
- [ ] 不验证：
  - 绝对性能（单受试者无意义）
  - 过拟合/欠拟合（无独立测试集）

### 5. 修复旧路径引用 (10 分钟)
- [ ] `main.py:78`：`test_task` → `tasks/test_task`
- [ ] `harness/evaluate.py:56-57`：同上
- [ ] `whitelist.json`：`test_task/model.py` → `tasks/test_task/model.py`
- [ ] 提交修复：`git commit -m "修复 test_task 迁移后的路径引用"`

### 6. 文档和交接 (15 分钟)
- [ ] 更新 `tasks/sleep_edf/README.md`：
  - 预处理命令
  - 训练命令
  - 当前限制（单受试者）
- [ ] 记录本次管线测试结果：
  - 数据量（epoch 数）
  - 训练指标（epoch 1 vs epoch 5）
  - 已验证 vs 未验证的内容
- [ ] 标记下一步：下载更多受试者数据

## 不做的事（本计划范围外）
- ❌ 下载更多 Sleep-EDF 数据（等管线跑通后再批量处理）
- ❌ 修改 Harness 核心逻辑（白名单、子进程执行、Git）
- ❌ 接入 Agent 自动优化（先确保训练流程正确）
- ❌ 实现睡眠结构指标（TST、WASO、效率等）
- ❌ 优化模型性能（当前只验证管线）

## 验收标准
- [x] 用户确认单受试者方案
- [ ] 预处理脚本跑通，生成 `.npz`
- [ ] `train.py` 在真实数据上完成 5 epoch
- [ ] 输出 `metrics.jsonl` 和 `best_model.pt`
- [ ] 旧路径引用全部修复
- [ ] Git 提交记录清晰

## 风险
1. **NPZ 格式不符预期**：partner 可能输出了额外字段或不同形状
   - 缓解：先用 `np.load()` 检查键值，再适配
2. **内存不足**：单条记录约 800-2600 epoch，batch=32 可能占用 ~500 MB
   - 缓解：降低 batch size 到 16
3. **依赖冲突**：partner 的 `requirements.txt` 可能与现有环境冲突
   - 缓解：先检查 diff，按需安装

## 时间估算
总计约 90 分钟（不含等待时间）。
