# 分层实验记忆系统

本目录实现 Layered Experiment Memory（LEM）。事实指标、运行状态和
Accept / Rollback 决策由 Harness 与固定 Evaluator 产生；LLM 只能提出超参数
提案和离线经验候选，不能改写事实日志。

## 1. 接入状态

| 功能 | 状态 | 实现 |
|---|---|---|
| 旧日志读取与兼容 | 已接入 | `load_run_history()`、`legacy_to_v2()` |
| v2 结构化事实写入 | 已接入 | `append_run_record()`、`main.make_terminal_record()` |
| 全部终态记录 | 已接入 | Baseline / Accept / Rollback / Skip / Error / Timeout |
| 重复 `run_id` 防护 | 已接入 | 追加前扫描 v2 日志 |
| 四层 Prompt | 已接入 | `build_agent_memory_context()` |
| Top-3 规则检索 | 已接入 | `retrieve_relevant_experiments()` |
| 检索追踪 | 已接入 | `logs/memory_retrieval_trace.jsonl` |
| 离线反思与事实核验 | 已接入 | `harness/memory_reflection.py` |
| 人工审核入正式经验层 | 已接入 | `experience_memory.md` |
| A/B/C 正式对照结果 | 尚未运行 | 已提供 `none / legacy / layered` 模式入口 |

## 2. 数据文件

| 文件 | 角色 | 是否权威 |
|---|---|---|
| `logs/sleep_run_history.jsonl` | 旧格式日志，继续追加以兼容现有读取器 | 是，旧格式事实 |
| `logs/sleep_run_history_before_memory_v2.jsonl` | Phase 0 冻结备份 | 是，历史快照 |
| `logs/experiment_memory_v2.jsonl` | 新格式追加式实验事实 | 是，主事实日志 |
| `logs/memory_retrieval_trace.jsonl` | 检索命中、来源和 Token 估算 | 追踪记录 |
| `docs/experiment_memory/task_profile.md` | 只读任务约束 | 固定上下文 |
| `strategy.md` | 人工维护策略 | 参考 |
| `docs/experiment_memory/experience_memory.md` | 人工审核后的正式经验 | 已核验经验 |
| `docs/experiment_memory/reflection_queue/` | 未审核候选与拒绝原因 | 候选，不是事实 |

`logs/*.jsonl` 已被 `.gitignore` 忽略。它们是本地运行结果，不随代码提交。

## 3. v2 记录语义

每条 v2 记录由一个唯一 `run_id` 标识，并包含：

- 环境：`dataset_version`、`split_id`、`seed`、`code_commit`、
  `evaluator_version`、Agent 模型和 Prompt 版本。
- 配置：`config_before`、`agent_proposal`、`config_after`、
  `config_diff`、`applied_config_diff`。
- validation 事实：BA、Macro F1、Loss、耗时、Token 用量和日志/检查点位置。
- 终态：`decision`、`status`、`reason`、`error_type`。
- 追溯：`source=native|legacy_import` 和有效 `run_id`。

Rollback 记录的 `config_after` 是回滚后实际生效的旧配置；
`config_diff` 保留本轮尝试的参数变化，`applied_config_diff` 为空。

Skip 记录不包含 validation 指标，因为它没有执行训练。Error / Timeout 记录
也不得伪装成“指标下降的 Rollback”。

## 4. Prompt 顺序

1. Foundation：`task_profile.md` 中的固定任务、数据和禁止 test 泄漏约束。
2. Strategy：根目录人工维护的 `strategy.md`。
3. Experience：
   - C 组使用正式经验、Top-3 相关案例和最近 2 条补充记录；
   - B 组使用最近 5 条旧格式记录；
   - A 组不提供历史经验。
4. Real-time：当前配置、当前/最佳 validation 指标、轮次和上一轮决策。

案例中的 legacy 记录明确标记为 `legacy_uncontrolled`，只能作为弱证据。

## 5. 检索规则

第一版不引入向量数据库，按以下信息排序：

1. 同任务；native 记录还必须同数据版本和 split。
2. 历史参数差异与当前关注参数的匹配程度。
3. 当前配置与历史实际生效配置的接近程度。
4. Harness 决策状态与新近程度作为辅助。

返回最多 3 条案例，再补充最近 2 条未重复记录；没有匹配时明确写“无相关案例”。
每次提示组装都追加一条检索追踪，包含 Top-K 来源、命中数和 Token 估算。

## 6. 离线反思

```bash
python -m harness.memory_reflection --batch-size 5
```

流程：

1. 只读取 v2 validation 事实，构造脱敏最小输入。
2. LLM 输出经验候选及逐条 evidence。
3. 程序核对 `run_id`、decision、status、result type、参数差异和全部指标。
4. 通过核验的候选进入 `reflection_queue/<batch>/candidates.jsonl`。
5. 人工审核后进入 `experience_memory.md`。

程序会拒绝不存在的 `run_id`、错误指标，以及把 Skip 写成训练失败等状态错配。

批准命令：

```bash
python -m harness.memory_reflection `
  --batch-dir docs/experiment_memory/reflection_queue/<batch_id> `
  --approve <candidate_id> `
  --reviewer <reviewer>
```

## 7. 测试

```bash
python -m unittest discover -s tests -v
```

现有测试覆盖 Schema、重复 `run_id`、旧日志转换、检索排序、四层 Prompt、
无 test 指标泄漏、反思事实核验，以及 Baseline / Accept / Rollback / Skip /
Timeout 的单条记录行为。测试不运行真实训练或外部 API。

## 8. 结论边界

当前完成的是 Phase 1–5 的实现和功能测试。尚无使用新记忆机制完成的真实闭环
和 A/B/C 对照，因此不能据此宣称模型性能已经提高。正式实验只能使用 validation
进行调参和组别选择；此前独立 test 结果不得用于本轮记忆策略选择。
