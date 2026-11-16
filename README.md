# Sleep-Agent Harness

面向睡眠分期任务的 Agent 辅助实验优化框架。项目以 TinySleepNet 为目标模型，使用 Sleep-EDF Expanded 数据集，通过 Agent 提出超参数调整方案，由 Harness 执行训练、验证，并根据验证集指标决定接受新配置或回滚。

## 1. 项目目标

本项目希望构建一个能够持续开展机器学习实验的 Harness：

1. 使用固定的数据集划分训练睡眠分期模型。
2. 使用验证集评价当前模型表现。
3. Agent 根据当前配置、验证指标和历史实验记录提出超参数调整方案。
4. Harness 校验调整结果并执行新一轮训练。
5. 新配置优于历史最佳配置时接受修改，否则恢复原配置。
6. 保存实验记录，为后续实验提供参考。

**当前研究任务：** 基于 EEG 信号的五分类睡眠分期。

## 2. 系统工作流程

```text
Sleep-EDF Expanded
        │
        ▼
数据预处理与受试者级划分
        │
        ▼
训练集 / 验证集 / 测试集
        │
        ▼
TinySleepNet 基线训练
        │
        ▼
验证集指标计算
        │
        ▼
Agent 读取配置、历史记录和实验策略
        │
        ▼
提出超参数调整方案
        │
        ▼
Harness 校验并执行新实验
        │
        ▼
比较验证集指标
        │
        ├── 优于历史最佳：Accept
        │
        └── 未达到提升条件：Rollback
        │
        ▼
更新实验记录，进入下一轮
```

核心模块：

| 模块 | 位置 | 职责 |
|---|---|---|
| 主控程序 | `main.py` | 管理训练、评估和迭代流程 |
| Agent | `harness/agent.py` | 生成超参数调整方案 |
| Sandbox | `harness/sandbox.py` | 在受控子进程中执行 Python 脚本 |
| Evaluator | `harness/evaluate.py` | 计算验证集评价指标 |
| Git Manager | `harness/git_manager.py` | 接受配置修改或执行回滚 |
| Memory Store | `harness/memory_store.py` | v2 记录校验、旧日志兼容、相关案例检索与四层上下文 |
| Memory Reflection | `harness/memory_reflection.py` | 离线经验候选、事实核验和人工审核 |
| Sleep-EDF Task | `tasks/sleep_edf/` | 模型、数据加载和训练代码 |

## 3. 数据集与任务定义

使用 Sleep-EDF Expanded 数据集中的 Sleep Cassette（SC）子集。

目前本地已完成的数据准备结果：

| 项目 | 数量或设置 |
|---|---:|
| 独立受试者 | 78 |
| PSG/Hypnogram 记录对 | 153 |
| Epoch 总数 | 199,352 |
| 训练集受试者 | 54 |
| 验证集受试者 | 12 |
| 测试集受试者 | 12 |
| Epoch 时长 | 30 秒 |
| 采样率 | 100 Hz |
| 每个 Epoch 采样点 | 3,000 |
| EEG 通道 | Fpz-Cz |
| 分类类别 | 5 |

标签映射：

| 标签 | 类别 |
|---:|---|
| 0 | W（Wake） |
| 1 | N1 |
| 2 | N2 |
| 3 | N3（合并原始 S3、S4） |
| 4 | REM |

**数据划分原则：** 按受试者划分训练集、验证集和测试集，避免同一受试者的数据跨集合出现。训练只使用训练集，Agent 的优化依据是验证集指标，测试集保持隔离。

本地预处理数据位于：

```text
tasks/sleep_edf/data/sleep-edf-sc-full/
```

目录中应包含各记录的 NPZ 文件、`subject_split.json` 及数据集清单等文件。

注意：原始 EDF 和预处理后的数据文件属于本地数据，不随 Git 仓库克隆。更换电脑时需要单独迁移预处理数据。

## 4. 模型与评价指标

### 模型

使用 MedMaxLab 的 SelfEEG v0.2.2 提供的 PyTorch TinySleepNet 实现。

- 输入：`(batch_size, 1, 3000)`
- 输出：5 个睡眠阶段的分类 logits
- 类别顺序：W、N1、N2、N3、REM

源码与版本：

[SelfEEG v0.2.2](https://github.com/MedMaxLab/selfEEG/tree/v0.2.2)

### 评价指标

当前使用三个验证集指标：

1. **Balanced Accuracy**：主要优化指标。
2. **Macro F1**：次级比较指标。
3. **Validation Loss**：第三比较指标。

配置接受规则按字典序比较：优先比较 Balanced Accuracy；相同时比较 Macro F1；两者都相同时比较 Validation Loss，损失更低者更优。

## 5. 项目目录

```text
Sleep-agent_harness/
│
├── main.py
├── README.md
├── requirements.txt
├── strategy.md
├── whitelist.json
│
├── harness/
│   ├── agent.py
│   ├── evaluate.py
│   ├── git_manager.py
│   ├── memory_store.py
│   ├── sandbox.py
│   └── whitelist.py
│
├── tasks/
│   ├── sleep_edf/
│   │   ├── model.py
│   │   ├── model_config.py
│   │   ├── data_loader.py
│   │   ├── train.py
│   │   ├── preprocess/
│   │   └── data/
│   │
│   └── test_task/
│
├── docs/
│   ├── README.md
│   └── experiment_memory/
│
└── requirements.txt
```

本地运行时还会产生训练输出目录和实验日志。它们不属于需要提交到 Git 的源代码。

## 6. 环境配置

推荐使用 Python 3.11 环境。

创建 Conda 环境：

```bash
conda create -n sleep-harness python=3.11 -y
conda activate sleep-harness
```

安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

项目依赖包括 PyTorch、SelfEEG、MNE、pyEDFlib、NumPy、SciPy、scikit-learn、pandas 等。SelfEEG 版本固定为 `0.2.2`。

Agent 使用 `ANTHROPIC_API_KEY` 环境变量读取 API Key。请在项目根目录创建本地 `.env` 文件：

```text
ANTHROPIC_API_KEY=替换为自己的API_Key
```

`.env` 已加入 `.gitignore`，不要将密钥提交到 GitHub。

当前 API 接口地址和模型名称配置在 `harness/agent.py` 中；更换 API 服务时需要检查该模块。

## 7. 数据准备

如果已经获得预处理好的完整数据集，将其放到：

```text
tasks/sleep_edf/data/sleep-edf-sc-full/
```

至少需要：

- 每条记录对应的 `.npz` 文件；
- `subject_split.json`。

其他数据清单和预处理日志建议一并保留，以便审计和复现。

如果需要从原始 EDF 重新生成数据，请参考：

- [Sleep-EDF 预处理说明](tasks/sleep_edf/preprocess/README.md)
- [预处理测试记录](tasks/sleep_edf/preprocess/TESTLOG.md)

原始 EDF 文件不要求放入代码仓库。

## 8. 运行

在项目根目录执行。

### 第一步：语法检查

```bash
python -m py_compile main.py harness/evaluate.py tasks/sleep_edf/train.py
```

该命令只检查 Python 语法，不会启动模型训练。

### 第二步：单轮试运行

```bash
python main.py --rounds 1
```

该命令会先执行一次基线训练和验证，再最多执行一轮 Agent 优化实验。因此它不是纯粹的语法测试，会实际训练模型并调用 Agent API。

### 第三步：多轮实验

```bash
python main.py --rounds 10
```

不指定 `--rounds` 时，默认执行 10 轮 Agent 优化。

### 输出位置

模型检查点和各轮评估结果保存在：

```text
tasks/sleep_edf/outputs/
```

旧格式实验历史继续保存在：

```text
logs/sleep_run_history.jsonl
```

每次运行终态另追加一条 v2 结构化事实到：

```text
logs/experiment_memory_v2.jsonl
```

相关案例检索追踪保存在：

```text
logs/memory_retrieval_trace.jsonl
```

这两个新日志同样属于本地运行记录，不提交到 Git。Baseline、Accept、
Rollback、Skip、Error 和 Timeout 都会写入唯一 `run_id` 的 v2 记录。

Agent 使用的调参策略保存在：

```text
strategy.md
```

### 四层记忆模式

```bash
python main.py --rounds 1 --memory-mode layered
```

可选模式：

- `none`：对应 A 组，只提供基础层与实时层。
- `legacy`：对应 B 组，提供策略和最近 5 条旧格式记录。
- `layered`：对应 C 组，提供四层上下文、Top-3 相关案例和最近 2 条补充记录。

这些模式只准备运行入口，不代表已经完成 A/B/C 正式对照实验。

### 测试

```bash
python -m unittest discover -s tests -v
```

测试不启动真实训练，也不调用 Agent API。

### 离线反思与人工审核

每累计约 5 轮实验后运行一次反思，而不是每轮训练后调用：

```bash
python -m harness.memory_reflection --batch-size 5
```

候选与拒绝结果分别写入 `docs/experiment_memory/reflection_queue/`。只有人工执行
批准命令后，核验通过的候选才会进入 `experience_memory.md`：

```bash
python -m harness.memory_reflection `
  --batch-dir docs/experiment_memory/reflection_queue/<batch_id> `
  --approve <candidate_id> `
  --reviewer <reviewer>
```

`strategy.md` 仍由人工维护，反思程序不会自动改写策略。

## 9. 文档索引

| 文档 | 内容 |
|---|---|
| [文档导航](docs/README.md) | 项目文档入口 |
| [Agent 调参策略](strategy.md) | Agent 的实验调整参考 |
| [分层实验记忆系统](docs/experiment_memory/README_revised.md) | v2 记录、检索、反思和审核 |
| [预处理说明](tasks/sleep_edf/preprocess/README.md) | EDF 数据预处理方法 |
| [预处理测试记录](tasks/sleep_edf/preprocess/TESTLOG.md) | 数据预处理测试记录 |

## 10. 当前项目状态

### 已完成

- Sleep-EDF Expanded SC 子集下载与预处理。
- 153 条记录的配对与质量检查。
- 受试者级数据集划分。
- SelfEEG PyTorch TinySleepNet 环境配置。
- 模型前向传播、反向传播和优化器更新测试。
- 训练与评估脚本对完整数据集的适配。
- Harness 的全量数据路径和受试者划分检查。
- 旧 Harness 的 8 轮验证集调参闭环已写入旧格式日志。
- Round 8 之后的独立 test 评估已锁定，不作为本轮记忆策略选择依据。
- 旧日志读取与策略注入。
- v2 结构化实验事实写入，覆盖全部运行终态。
- 可解释 Top-3 案例检索和四层 Prompt 组装。
- 离线反思候选、程序事实核验和人工审核流程。
- `none / legacy / layered` 三种记忆模式入口。
- 不训练、不调用 API 的单元测试。

### 待完成

- 用新记忆机制运行至少一次真实单轮闭环。
- 生成并人工审核第一份正式经验。
- A/B/C 公平对照实验与稳定性分析。

**注意：** 旧格式历史包含不同 Epoch 预算的 legacy 记录，不能直接作为公平
A/B/C 对照样本。当前 v2 记忆机制已经通过功能测试，但尚无用新机制完成的真实
闭环，因此不能把功能测试结果写成模型性能提高的证据。

---

项目定位：研究与实验用途，不作为临床诊断工具。
