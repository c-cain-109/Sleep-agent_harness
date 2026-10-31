# 实验记忆系统

本目录存放 Sleep-Agent Harness 的实验记忆设计文档。该机制用于让 Agent 参考历史实验事实和策略摘要，减少重复试错；实验指标和接受/回滚决策仍由 Harness 与固定评估器产生。

## 1. 当前接入状态

| 功能 | 状态 | 当前实现 |
|---|---|---|
| 读取 `strategy.md` | 已接入 | `harness/memory_store.py` |
| 读取实验历史 JSONL | 已接入 | 读取 `logs/sleep_run_history.jsonl` |
| 将最近 5 条记录注入 Agent Prompt | 已接入 | `harness/agent.py` 调用 `build_agent_memory_context()` |
| 由 Harness 写入旧格式实验记录 | 已接入 | `main.py` 的 `append_history()` |
| 按新 Schema 写入结构化记录 | 尚未接入 | `append_run_record()` 尚未由主流程调用 |
| Agent 根据实验结果自动更新策略摘要 | 尚未接入 | 当前没有自动更新 `strategy.md` 的流程 |

**注意：** 当前已经具备“读取策略和历史记录”的基础记忆能力，但还没有完成结构化实验记录的完整写入闭环。不要把 Schema 或模板描述成已经在运行中生效。

## 2. 文件说明

| 文件 | 用途 |
|---|---|
| `experiment_memory_schema.json` | 结构化实验记录的建议字段规范 |
| `AGENT_MEMORY_CONTRACT.md` | Agent 与 Harness 的职责边界和建议流程 |
| `strategy_memory_template.md` | 可参考的策略记忆模板，不会自动覆盖根目录的 `strategy.md` |

实际代码位于：

- `harness/memory_store.py`：读取策略、读取 JSONL、构建 Agent 记忆上下文，以及预留的结构化记录追加函数。
- `harness/agent.py`：读取记忆上下文并加入 Agent Prompt。
- `main.py`：执行实验并向 `logs/sleep_run_history.jsonl` 追加当前格式的运行记录。

## 3. 当前日志约定

当前运行历史文件：

```text
logs/sleep_run_history.jsonl
```

目前 `main.py` 写入的记录以 `round`、`action` 和评估指标等字段为主。`memory_store.py` 的上下文构建函数会读取 JSONL 中的记录，并选取最近 5 条提供给 Agent。

`experiment_memory_schema.json` 定义的是后续结构化记录的建议格式，与现有日志字段并不完全相同。正式迁移前，应设计兼容方案或采用独立版本化日志；不要直接覆盖旧日志，也不要让新旧格式混用后无法识别。

## 4. 设计原则

1. **指标由 Harness 产生。** Agent 可以解释实验结果，但不得编造或修改 Balanced Accuracy、Macro F1、Validation Loss 等实际指标。
2. **保留失败记录。** 指标未提升、程序异常和超时应区分记录，不应删除失败实验。
3. **策略摘要不等于事实日志。** `strategy.md` 用于保存经验总结；具体指标、运行状态和决策应以 Harness 写入的实验日志为准。
4. **实验可追溯。** 后续结构化记录应尽可能关联代码提交、数据划分、配置和评估器版本。
5. **渐进式集成。** 先保证真实数据训练与验证闭环稳定，再完善结构化写入、最佳实验检索和自动策略更新。

## 5. 后续工作

- [ ] 确定旧格式 `sleep_run_history.jsonl` 与新 Schema 的兼容或迁移方案。
- [ ] 将结构化记录写入接入 `main.py`，并确保异常、回滚和无配置变化等情况都有清晰状态。
- [ ] 为实验记录补充稳定的 `run_id`、时间戳、代码提交和数据划分标识。
- [ ] 在有真实 Sleep-EDF 实验记录后，再维护可追溯的成功经验和失败经验。
- [ ] 评估是否需要由 Agent 提出 `strategy.md` 更新建议，并由 Harness 控制实际写入。

在上述工作完成前，`experiment_memory_schema.json` 和 `strategy_memory_template.md` 均作为设计参考，不应视为完整启用的结构化记忆系统。
