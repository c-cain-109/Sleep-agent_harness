# Sleep-Harness 项目中期总结

**日期**：2026-10-28  
**阶段**：Phase 1-2 完成，Phase 3 待启动

---

## 一、项目目标

构建 **RSI（Recursive Self-Improvement）框架**，让 LLM Agent 能：
1. 自动修改模型代码/超参数
2. 在沙箱中评估性能
3. 根据结果自动 accept/rollback 改动
4. 最终实现模型自主迭代优化

**验证路径**：
- Phase 1：玩具任务（二次函数参数优化）→ 验证核心机制
- Phase 2：真实任务（睡眠分期模型超参数调优）→ 验证实用性
- Phase 3-5：扩展到更复杂场景（多受试者、真实 Docker 沙箱、LLM API 调用）

---

## 二、已完成工作

### Phase 1：玩具任务 Harness ✅

**目标**：验证防篡改、子进程执行、Git accept/rollback 核心机制

**实现**（`main.py` + `harness/`）：
- ✅ 防篡改评估器（SHA256 哈希锁定 `evaluate.py`）
- ✅ 白名单验证（只允许修改 `tasks/test_task/model.py`、`config.yaml`）
- ✅ 子进程执行（`subprocess` + 超时控制；当前不提供文件、网络或内存隔离）
- ✅ Git 自动决策：
  - `score < best_score` → `git commit`（accept）
  - `score >= best_score` → `git reset --hard HEAD`（rollback）
- ✅ JSONL 日志（`logs/run_history.jsonl`）
- ✅ 自动轮询检测文件修改（无需手动按 Enter）

**测试结果**：
```
初始 MSE: 4132.96 (参数 a=1, b=1, c=1)
最终 MSE: 0.00 (参数 a=2.5, b=-3, c=1.5，逼近目标函数)
改进幅度: 100%
```

**遇到的问题及解决**：
1. ~~文件修改检测失败~~ → 改用 `git diff --name-only`（移除 `HEAD` 参数）
2. ~~Rollback 删除 commit~~ → 改用 `git reset --hard HEAD`（而非 `HEAD~1`）
3. ~~哈希验证失败~~ → 重新计算 `EXPECTED_HASH`
4. ~~PyCharm 交互式阻塞~~ → 移除 `input()`，改为轮询检测

---

### Phase 2：睡眠分期真实任务 ✅

#### 2.1 数据预处理模块

**Partner 交付**（`tasks/sleep_edf/preprocess/prepare_sleepedf.py`）：
- 📄 2400+ 行，全英文注释
- 功能：读取 Sleep-EDF 数据集 → 提取 EEG → 切分 30 秒 epoch → 五分类标签映射
- 输出：
  - `SC4001E0.npz`（841 epochs × 3000 采样点，裁剪边缘清醒段）
  - `dataset_manifest.json`（完整处理统计）
  - `epoch_metadata.csv`（每个 epoch 的时间戳和标签）
- 特性：
  - 零参数运行（自动发现数据目录）
  - 双读取器交叉校验（MNE vs 内置解析器）
  - 独立对齐校验器（`verify_alignment.py`，841/841 逐位一致）
  - 字节级可复现（6 次运行 SHA-256 相同）

**简化版**（`prepare_simple.py`，本次新增）：
- 📄 **280 行，全中文注释**
- 功能：核心流程简化，移除复杂配置和多种后端
- 输出：`SC4001E0.npz`（153 epochs，裁剪更激进）
- 用途：便于理解和修改

**标签映射**（五分类）：
```
W (清醒) = 0
N1 (浅睡) = 1
N2 (中睡) = 2
N3 (深睡，S3+S4 合并) = 3
REM (快速眼动) = 4
```

#### 2.2 训练管线验证

**模型**：TinySleepNet（LSTM + Attention，PyTorch）

**数据切分**（单受试者时间切分）：
- 训练集：前 80% (673 epochs)
- 验证集：后 20% (168 epochs)

**训练结果**（5 epochs）：
```
Train Loss: 1.5806 → 1.5583
Val Loss:   1.6049 → 1.6206
Balanced Accuracy: 0.43
Macro F1: 0.15
```

**输出文件**：
- `best_model.pt`（PyTorch 模型权重）
- `metrics.jsonl`（每轮指标）
- `training.log`（完整日志）

**限制说明**：
⚠️ 当前只有 **1 位受试者**（SC4001），train/val 来自同一晚不同时间段
- 验证集缺少 N3 类别（最后 20% 时间段没有深睡眠）
- **指标仅证明管线可用**，不代表模型泛化能力
- 需要下载 ≥9 位受试者数据，才能做真正的按受试者划分

#### 2.3 路径修复

修复从 `test_task/` 迁移到 `tasks/test_task/` 后的引用问题：
- ✅ `main.py`
- ✅ `harness/evaluate.py`
- ✅ `whitelist.json`

---

## 三、项目结构

```
Sleep-agent harness/
├── harness/                    # Phase 1 核心
│   ├── evaluate.py            # 防篡改评估器（SHA256 锁定）
│   ├── sandbox.py             # 子进程执行器（非安全沙箱）
│   ├── git_manager.py         # Git accept/rollback
│   ├── whitelist.py           # 白名单验证
│   └── agent.py               # LLM Agent 调用（Phase 3 待用）
├── tasks/
│   ├── test_task/             # Phase 1 玩具任务
│   │   ├── model.py           # 待优化参数（a, b, c）
│   │   ├── target_function.py # 固定目标函数
│   │   └── config.yaml        # 配置（预留）
│   └── sleep_edf/             # Phase 2 睡眠分期
│       ├── preprocess/        # 预处理模块
│       │   ├── prepare_sleepedf.py  # Partner 版（2400 行）
│       │   └── prepare_simple.py    # 简化版（280 行，中文）
│       ├── data/
│       │   ├── physionet-sleep-data/  # 原始 EDF（1 位受试者）
│       │   ├── prepared/              # Partner 版输出（841 epochs）
│       │   └── prepared_simple/       # 简化版输出（153 epochs）
│       ├── model/             # TinySleepNet 实现
│       ├── train.py           # 训练脚本
│       └── outputs/
│           └── pipeline_test/ # 训练结果（5 epochs）
├── main.py                    # Phase 1 主控循环
├── logs/
│   └── run_history.jsonl      # Phase 1 运行日志
└── whitelist.json             # 白名单配置
```

---

## 四、技术亮点

### 4.1 Phase 1 玩具任务
- **防篡改机制**：评估器用 SHA256 自我验证，Agent 无法篡改评分规则
- **Git 原子操作**：每次 accept 一个 commit，rollback 只回滚工作区（不删 commit 历史）
- **轮询检测**：每秒检查文件修改，适配 PyCharm 等 IDE 环境

### 4.2 Phase 2 数据预处理
- **零参数运行**：自动发现数据目录（`$SLEEPEDF_DATA_DIR` / `data/sleep-edf/`）
- **双读取器校验**：MNE 与内置解析器结果一致到机器精度（`2.84e-14 µV`）
- **逐字节对齐验证**：独立脚本复核每个 epoch 的脑电与标签是否真的一一对应
- **字节级可复现**：6 次运行产出相同 SHA-256，证明无随机性

### 4.3 Phase 2 训练管线
- **单文件时间切分**：`--data-file` 模式，前 80% 训练、后 20% 验证
- **完整监控**：`metrics.jsonl` 记录每轮 loss/accuracy/f1
- **模型保存**：自动保存验证集最佳模型

---

## 五、当前限制与挑战

### 5.1 数据限制 ⚠️
- **只有 1 位受试者**（SC4001），无法做真正的受试者级划分
- 当前 train/val 来自同一晚不同时间段，**不能证明泛化能力**
- 验证集缺少 N3 类别（最后 20% 时间段没有深睡眠）

**解决方案**：下载更多受试者数据
- Sleep-EDF Cassette：20 位健康受试者
- Sleep-EDF Expanded：82 位受试者（含睡眠障碍患者）
- 需要 ≥9 位受试者才能做 70%/15%/15% 划分

### 5.2 模型性能 📉
- Balanced Accuracy: **0.43**（随机猜测是 0.20）
- Macro F1: **0.15**（非常低）
- 可能原因：
  1. 数据量太少（673 训练样本）
  2. 超参数未调优（使用默认值）
  3. 类别不平衡（W=188, N1=58, N2=250, N3=220, REM=125）

### 5.3 Phase 3 待完成
- **Agent 自动调参**：还没让 LLM 自动修改超参数
- **真实 Docker 沙箱**：当前用 `subprocess`，需要升级到 Docker 隔离
- **多受试者场景**：需要更复杂的数据划分和评估策略

---

## 六、下一步建议

### 选项 A：完成 Phase 3（Agent 自动调参）🚀
**目标**：让 Agent 自动修改 TinySleepNet 超参数并迭代优化

**任务**：
1. 创建 `tasks/sleep_edf/model.py`（包含可调超参数）
2. 修改 `harness/agent.py`：
   - 调用 Claude API
   - 根据当前指标生成超参数建议
   - 自动修改 `model.py`
3. 修改 `main.py`：
   - 每轮调用 `train.py` 训练
   - 根据验证集指标 accept/rollback
4. 测试 10 轮自动迭代

**预期时间**：1-2 天

**优点**：
- ✅ 完整验证 RSI 框架核心逻辑
- ✅ 即使数据量小，也能证明 Agent 能自动调参
- ✅ 为后续扩展打下基础

**缺点**：
- ⚠️ 当前数据质量低，优化效果可能不明显
- ⚠️ 没有真正的泛化验证

---

### 选项 B：下载更多数据 📦
**目标**：获取 ≥9 位受试者，做真正的按受试者划分

**任务**：
1. 下载 Sleep-EDF Cassette 数据集（20 位受试者）
2. 批量预处理（`prepare_simple.py --input-dir --output-dir`）
3. 修改 `train.py` 支持多受试者划分
4. 重新训练，观察模型在真实验证集上的表现

**预期时间**：1 天（下载 + 预处理）

**优点**：
- ✅ 得到可信的泛化指标
- ✅ 为后续 Agent 调参提供更好的评估基础
- ✅ 验证预处理模块的批量处理能力

**缺点**：
- ⚠️ 下载可能较慢（数据集 ~2GB）
- ⚠️ 还是不涉及 Agent 自动调参

---

### 选项 C：手动调优当前模型 🔧
**目标**：在当前数据上手动调整超参数，提升性能

**任务**：
1. 分析训练日志，找出瓶颈（过拟合/欠拟合？）
2. 调整超参数：
   - 学习率（当前 0.0001）
   - LSTM 隐藏层大小（当前 128）
   - Dropout（当前 0.5）
   - Batch size（当前 32）
3. 对比不同配置的效果

**预期时间**：半天

**优点**：
- ✅ 快速验证当前数据能否支持更好的模型
- ✅ 积累超参数调优经验

**缺点**：
- ⚠️ 手动调参，不涉及 Agent
- ⚠️ 数据量限制，提升空间有限

---

## 七、个人建议优先级

**推荐顺序**：**A → B → C**

**理由**：
1. **选项 A（Agent 自动调参）** 是项目核心目标，应该优先完成
   - 即使当前数据质量低，也能验证框架逻辑
   - 后续有更多数据时，可以直接复用
2. **选项 B（下载更多数据）** 是中期必需
   - 验证模型真实泛化能力
   - 为 Phase 4-5 打基础
3. **选项 C（手动调优）** 可作为辅助
   - 快速探索当前数据的潜力
   - 但不应该成为主线任务

---

## 八、关键成果物

### Phase 1
- ✅ `main.py`（主控循环，10 轮迭代）
- ✅ `harness/evaluate.py`（防篡改评估器）
- ✅ `harness/sandbox.py`（子进程执行器，非安全沙箱）
- ✅ `harness/git_manager.py`（Git 自动决策）
- ✅ `logs/run_history.jsonl`（运行日志）
- ✅ README.md（完整说明文档）

### Phase 2
- ✅ `tasks/sleep_edf/preprocess/prepare_sleepedf.py`（2400 行，Partner 交付）
- ✅ `tasks/sleep_edf/preprocess/prepare_simple.py`（280 行，简化版）
- ✅ `tasks/sleep_edf/data/prepared/SC4001E0.npz`（841 epochs）
- ✅ `tasks/sleep_edf/train.py`（训练脚本）
- ✅ `tasks/sleep_edf/outputs/pipeline_test/best_model.pt`（训练好的模型）
- ✅ `tasks/sleep_edf/outputs/pipeline_test/metrics.jsonl`（训练指标）

---

## 九、技术栈

### Phase 1
- Python 标准库（subprocess, json, hashlib, pathlib）
- Git（版本控制 + accept/rollback）
- NumPy（数值计算）

### Phase 2
- **数据**：pyedflib（EDF 读取）, scipy（重采样）, numpy
- **模型**：PyTorch（深度学习框架）
- **预处理**：MNE（可选，EDF 解析）

### Phase 3（待完成）
- Anthropic Claude API（LLM Agent）
- Docker（更强的沙箱隔离）

---

## 十、总结

**已完成**：
- ✅ 玩具任务 Harness（核心机制验证）
- ✅ 睡眠分期任务接入（数据 + 训练管线）
- ✅ 两套预处理模块（Partner 全功能版 + 简化中文版）

**当前瓶颈**：
- ⚠️ 数据量不足（只有 1 位受试者）
- ⚠️ 模型性能低（balanced_acc=0.43）
- ⚠️ Agent 自动调参尚未实现

**下一步**：
建议优先 **选项 A（Agent 自动调参）**，完成 RSI 框架核心验证，再考虑扩展数据。

---

**更新时间**：2026-10-28  
**作者**：用户 + Claude Opus 5.5
