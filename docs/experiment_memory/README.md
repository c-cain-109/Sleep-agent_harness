# Sleep-Harness 实验记忆 starter pack

本包用于在不重写现有 Harness 的前提下，为 Agent 增加持久化实验记忆。

## 文件

- `strategy_memory_template.md`：可复制为项目根目录的 `strategy.md` 的参考模板。不要直接覆盖现有 strategy.md；先备份并合并已有内容。
- `experiment_memory_schema.json`：建议的 JSONL 单条实验记录字段规范。
- `AGENT_MEMORY_CONTRACT.md`：Agent 与 Harness 之间的职责边界和每轮流程。
- `memory_store.py`：读取 strategy、读取 JSONL、追加记录及构建 Agent prompt 上下文的轻量工具函数。

## 建议放置位置

```text
Sleep-agent harness/
├── harness/
│   ├── agent.py
│   └── memory_store.py       # 可放这里，或单独放 harness/memory/
├── logs/
│   └── run_history.jsonl     # 保留既有历史，不要覆盖
└── strategy.md               # 备份后按模板合并
```

## 重要集成前检查

1. 先备份现有 `strategy.md` 和 `logs/run_history.jsonl`。
2. 检查现有 `run_history.jsonl` 的实际字段。不要直接将新 schema 强加到旧记录，也不要清空旧日志。
3. 如果旧日志格式不同，增加兼容读取或另建版本化日志文件；不要让读取函数静默忽略旧记录。
4. 在 `harness/agent.py` 中，调用 `build_agent_memory_context(...)`，把返回文本加入 Agent 的任务上下文。
5. 在评估完成后，由 Harness 调用 `append_run_record(...)` 追加实际运行结果。
6. 指标必须由固定评估器产生；Agent 只能总结经验，不得填写或更改真实指标。

本包是集成起点，不是已经接入现有项目的补丁。需要查看当前 `agent.py`、`main.py` 和日志格式后，才能安全完成精确集成。
