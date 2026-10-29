# Phase 3: Agent 自动调参 - 快速启动指南

## 前置条件

✅ 已完成 Phase 2（数据预处理 + 训练管线验证）  
✅ 已有数据文件：`tasks/sleep_edf/data/prepared/SC4001E0.npz`  
✅ 已配置环境变量：`ANTHROPIC_API_KEY`

---

## 运行方式

### 1. 完整 10 轮自动调优

```bash
cd "/Users/550w/PycharmProjects/Sleep-agent harness"
python main.py --rounds 10
```

**流程**：
1. 初始训练（Round 0），记录基线指标
2. Agent 生成超参数调整建议（调用 Claude API）
3. 写入 `tasks/sleep_edf/model_config.py`
4. 重新训练
5. 根据指标 accept/rollback
6. 重复 10 轮

**预期时间**：每轮 1-2 分钟（取决于 API 响应 + 训练时间）

---

### 2. 测试单轮（验证流程）

```bash
# 1. 初始训练
cd tasks/sleep_edf
python train.py --data-file data/prepared/SC4001E0.npz --output-dir outputs/test_round_0 --seed 42

# 2. 查看指标
tail -1 outputs/test_round_0/metrics.jsonl

# 3. 手动调整配置（模拟 Agent）
# 编辑 model_config.py，如 LEARNING_RATE = 0.0005

# 4. 重新训练
python train.py --data-file data/prepared/SC4001E0.npz --output-dir outputs/test_round_1 --seed 42

# 5. 对比指标
tail -1 outputs/test_round_1/metrics.jsonl
```

---

## 关键文件

| 文件 | 作用 |
|---|---|
| `main.py` | 主控循环（10 轮迭代） |
| `harness/agent.py` | Agent 模块（调用 LLM 接口） |
| `tasks/sleep_edf/model_config.py` | 可调超参数（Agent 修改此文件） |
| `tasks/sleep_edf/train.py` | 训练脚本（读取 model_config） |
| `logs/run_history.jsonl` | 运行日志 |
| `strategy.md` | 调参策略文档（Agent 会读取） |
| `whitelist.json` | 白名单（允许修改的文件） |

---

## 预期输出

### 控制台输出示例

```
======================================================================
睡眠分期自动调优 - 初始化
======================================================================
[RUN] python train.py --data-file data/prepared/SC4001E0.npz --output-dir outputs/round_0_initial --seed 42
[SUCCESS] 训练完成
[INITIAL] Balanced Acc=0.4300, Macro F1=0.1500, Val Loss=1.6206

======================================================================
Round 1/10
======================================================================
[AGENT] 生成超参数调整建议...
[AGENT] 建议: {
  "learning_rate": 0.0005,
  "dropout": 0.6,
  "reasoning": "提升学习率加快收敛，增大 dropout 缓解过拟合"
}
[CONFIG] 已更新 model_config.py
[RUN] python train.py ...
[SUCCESS] 训练完成
[METRICS] Balanced Acc=0.4500, Macro F1=0.1800, Val Loss=1.5900
[GIT] ✅ Accept: 已提交改动

======================================================================
Round 2/10
======================================================================
...
```

### 日志文件（logs/run_history.jsonl）

```json
{"round": 0, "action": "init", "balanced_accuracy": 0.43, "macro_f1": 0.15, "val_loss": 1.6206}
{"round": 1, "action": "accept", "balanced_accuracy": 0.45, "macro_f1": 0.18, "val_loss": 1.59, "reasoning": "..."}
{"round": 2, "action": "rollback", "balanced_accuracy": 0.44, "macro_f1": 0.17, "val_loss": 1.62, "reasoning": "..."}
```

---

## 验收指标

### Phase 3 核心目标

✅ **Agent 能自动调参**：10 轮中至少 1 轮成功调用 Claude API 并生成合法配置  
✅ **Accept/Rollback 正确**：性能提升时 commit，未提升时 rollback  
✅ **日志完整**：`run_history.jsonl` 记录每轮指标和操作  
✅ **Git 历史清晰**：accept 的改动有 commit，rollback 的改动不存在

### 性能改善（次要）

- Balanced Accuracy > 0.43（初始基线）
- Macro F1 > 0.15（初始基线）

⚠️ **注意**：当前数据量小（1 位受试者），性能提升有限，主要验证框架机制

---

## 常见问题

### Q1: API 调用失败
```
ValueError: API 调用失败（3 次重试后）
```

**解决**：
1. 检查环境变量：`echo $ANTHROPIC_API_KEY`
2. 检查网络：`curl https://www.sui-xiang.net/v1/models`
3. 查看详细错误：修改 `harness/agent.py` 添加打印

### Q2: 训练超时
```
subprocess.TimeoutExpired: 600 seconds
```

**解决**：
- 减小 `NUM_EPOCHS`（5 → 3）
- 或增大 `main.py` 中的 `timeout_seconds`（900 → 1200）

### Q3: 指标文件不存在
```
FileNotFoundError: 指标文件不存在
```

**解决**：
- 检查 `tasks/sleep_edf/data/prepared/SC4001E0.npz` 是否存在
- 手动运行一次训练确认流程正常

---

## 下一步

完成 Phase 3 后：
- **Phase 4**：下载更多受试者数据（9+ 人），真正的泛化验证
- **Phase 5**：Docker 沙箱隔离，防止恶意代码
- **Phase 6**：多任务扩展（其他深度学习任务）

---

**更新时间**：2026-10-28
