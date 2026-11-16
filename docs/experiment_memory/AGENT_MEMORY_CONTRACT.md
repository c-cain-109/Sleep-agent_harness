# Agent 实验记忆接入约定

## 职责边界

- Harness 执行训练、调用固定 Evaluator、比较 validation 指标并决定
  Accept / Rollback。
- Harness 写入旧格式日志和 v2 结构化事实；Agent 不能修改事实日志。
- Agent 提出一个受 Schema 和参数范围约束的超参数候选。
- 离线反思器只能生成经验候选；程序核验后仍需人工审核。
- `strategy.md` 由人工维护，Agent 和反思器都不能自动改写。

## 每轮输入

Agent Prompt 按固定顺序接收：

1. 基础层：`task_profile.md` 中的只读任务与评估约束。
2. 策略层：`strategy.md`。
3. 经验层：已审核经验、相关案例和近期补充；A/B 模式按实验设计裁剪。
4. 实时层：当前配置、当前/最佳 validation 指标、轮次和上一轮决策。

每条案例必须带 `run_id`。legacy 记录必须标记为 `legacy_uncontrolled`，不得
作为严格同协议对照。没有足够相关证据时，Prompt 明确写“无相关案例”。

## Agent 输出

Agent 返回全部七个超参数与 `reasoning`。程序校验：

- 必需字段齐全且类型正确；
- 参数在允许范围内；
- 离散参数为整数；
- 每轮原则上只改变 1–2 个参数；
- reasoning 说明假设、依据、预期影响和风险。

Agent 输出的 reasoning 是提案解释，不是实验事实。

## Harness 必须记录的终态

每次运行结束恰写一条 v2 记录：

| 终态 | decision | status | 指标 |
|---|---|---|---|
| Baseline 完成 | `Baseline` | `completed` | validation 事实 |
| 超过最佳 | `Accept` | `completed` | validation 事实 |
| 正常评估但未超过最佳 | `Rollback` | `completed` | validation 事实 + 回滚后配置 |
| 候选未产生变化 | `Skip` | `skipped` | null |
| API/训练/评估异常 | `Error` | `error` | null |
| 执行超时 | `Error` | `timeout` | null |

同时保留旧格式 `sleep_run_history.jsonl` 追加记录，以兼容现有读取器。

## 防重复与防伪造

- `run_id` 必须唯一，追加前扫描已有 v2 日志。
- validation 指标只能来自 Evaluator 输出文件。
- Rollback 必须同时保存失败提案的结果和回滚后实际生效配置。
- Skip 与 Error 不得描述为训练指标下降。
- 反思候选中的 `run_id`、decision、status、result type、参数差异和指标必须
  与原始记录一致，否则拒绝入队。

## test 集隔离

- 调参、检索排序、反思、经验审核和 A/B/C 组别选择只使用 validation。
- test 指标、test 标签统计或由 test 结果推导的策略不得进入 Prompt、v2
  日志或正式经验。
- 此前完成的独立 test 结果不能作为本次记忆机制改进的对照证据。
