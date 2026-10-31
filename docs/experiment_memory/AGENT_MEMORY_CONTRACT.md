# Agent 实验记忆接入约定

## 目标

让每一轮 Agent 能使用历史实验事实和总结经验，减少重复试错，同时确保 Agent 不能自行伪造评估指标。

## 文件职责

- `strategy.md`：供 Agent 阅读的压缩经验摘要、当前假设和下一步计划；可以由 Agent 在实验结束后提出更新。
- `logs/run_history.jsonl`：追加式实验事实日志；只能由 Harness 根据实际执行结果写入，不允许 Agent 直接编辑或重写。
- 模型 checkpoint / Git commit：保存可恢复的代码和模型状态；`strategy.md` 不代替版本控制。

## 每轮开始前，Agent 应读取

1. 完整的 `strategy.md`。
2. 最近 5 条实验记录（或最近 5 条有效 JSONL 记录）。
3. 当前最佳验证集 run 的记录。
4. 当前代码 commit、配置 hash、数据划分 ID 和 evaluator hash。

如果历史记录格式不符合预期，先报告并停止记忆解析，不得默默把解析失败的记录当作不存在。

## 每轮开始时，Agent 必须输出

- `hypothesis`：一个可检验的假设。
- `rationale`：依据哪些历史 run_id 或观察提出该假设。
- `planned_changes`：准备修改哪些白名单文件。
- `expected_effect`：预计影响哪个指标及方向。
- `risk_and_confounders`：可能的风险和混杂因素。
- `success_criterion`：实验前确定的接受条件。

## 每轮结束后，Harness 必须先写事实

Harness 从执行器和固定评估器获取：

- run_id、时间、运行状态、超时或异常信息。
- hypothesis、实际修改文件、代码 commit、配置 hash。
- 数据划分 ID、评估器 hash、随机种子。
- 实际 train/validation 指标。
- accept / rollback / no_change 决策。

这些数据由程序写入 `run_history.jsonl`。Agent 不得自行提供或改写指标数值。

## 然后再更新 strategy.md

可让 Agent 根据新 run_id 更新：

- 已验证经验；
- 没有改善或失败的方向；
- 证据可信程度；
- 下一轮待验证假设。

每条经验都必须引用 run_id。单次实验只能标为“初步”，不能写成已确定规律。

## 防止重复试错的最低规则

- 如果最近记录中相同代码改动、相同配置和相同数据划分已失败，不要原样重复。
- 如确实要重试，必须说明改变了哪个条件，以及为什么这次结果可能不同。
- 指标未改善不等于程序运行失败；二者必须分别记录。
- 不允许因为验证指标波动而删除失败记录。

## 建议的渐进式集成

第一阶段只把 `strategy.md` 和最近 5 条 run history 注入 Agent prompt。
第二阶段再增加“当前最佳 run”和历史失败检索。
第三阶段才考虑自动重启 Agent 实例和代际经验继承。

不要在真实数据训练闭环尚未稳定时，同时引入复杂的向量数据库、多 Agent 或自动重启机制。
