# Sleep-Harness Phase 1: 玩具任务验证

**项目目标**: 通过玩具任务（优化二次函数系数）验证 RSI（Recursive Self-Improvement）框架的核心防篡改机制、沙箱执行和 Git accept/rollback 流程。

## 快速开始

### 环境要求
- Python 3.8+
- Git
- 标准库（无需额外依赖）

### 初始化项目

```bash
cd "/Users/550w/PycharmProjects/Sleep-agent harness"
git init
git add -A
git commit -m "Initial commit"
```

### 首次运行：生成评估器哈希

```bash
python harness/evaluate.py
```

将输出的哈希值复制到 `harness/evaluate.py` 的 `EXPECTED_HASH` 常量中。

### 运行主循环

```bash
python main.py
```

按提示修改 `test_task` 中的参数 `a, b, c`，目标是逼近目标函数 `2.5*x² - 3.0*x + 1.5`。

## 项目结构

```
/Users/550w/PycharmProjects/Sleep-agent harness/
├── harness/
│   ├── evaluate.py          # 评估器（SHA256 哈希锁定）
│   ├── sandbox.py           # 沙箱执行（subprocess + 超时限制）
│   ├── git_manager.py       # Git 操作（accept/rollback）
│   └── whitelist.py         # 白名单验证
├── toy_task/
│   ├── model.py             # 待优化参数（a, b, c）
│   ├── config.yaml          # 配置文件（预留）
│   └── target_function.py   # 目标函数（固定）
├── logs/
│   └── run_history.jsonl    # 运行日志（自动生成）
├── main.py                  # 主控循环
├── whitelist.json           # 白名单配置
├── README.md                # 本文档
└── .gitignore
```

## 核心机制

### 1. 防篡改：SHA256 哈希验证

`harness/evaluate.py` 在每次运行时验证自身完整性：
- 计算文件 SHA256 哈希
- 与 `EXPECTED_HASH` 常量对比
- 不匹配则抛出 `SecurityError`

**验证方法**：
```bash
# 修改 evaluate.py（添加任意内容）
python main.py  # 预期：抛出 "评估器已被篡改"
```

### 2. 沙箱执行

`harness/sandbox.py` 使用 `subprocess.Popen` 隔离执行评估器：
- **超时限制**：300 秒
- **内存限制**：理论 4GB（macOS 上 `resource.setrlimit` 支持有限）
- **异常捕获**：超时、评估器完整性失败

返回格式：
```python
{
    "score": float,      # MSE 分数
    "duration": float,   # 执行时长
    "success": bool,     # 是否成功
    "error": str or None # 错误信息
}
```

### 3. 白名单验证

`harness/whitelist.py` 检查修改的文件是否在白名单中：
- 允许修改：`test_task`, `test_task`
- 其他文件修改 → 拒绝评估

**验证方法**：
```bash
# 修改非白名单文件（如 harness/sandbox.py）
python main.py  # 预期：拒绝评估，日志记录 "whitelist_violation"
```

### 4. Git Accept/Rollback

`harness/git_manager.py` 根据性能提升决策：
- **Accept**: `score < best_score` → `git add -A && git commit`
- **Rollback**: `score >= best_score` → `git reset --hard HEAD~1`

日志记录每轮的 `action` 字段（`accept` / `rollback`）。

## 测试指南

### 基本功能测试（10 轮）

| 轮次 | 操作 | 预期结果 |
|------|------|----------|
| 0 | 基线评估 | MSE ≈ 112.5（初始 `a=b=c=1.0`） |
| 1-3 | 调整 `a` → 2.5 | Accept，MSE 下降 |
| 4-6 | 调整 `b` → -3.0 | Accept，MSE 下降 |
| 7-9 | 调整 `c` → 1.5 | Accept，MSE 接近 0 |
| 10 | 恶化参数（如 `a=0`） | Rollback |

### 检查日志

```bash
cat logs/run_history.jsonl | jq .
git log --oneline
```

每次 accept 应对应一个 Git commit。

### 防篡改测试

```bash
# 1. 修改 harness/evaluate.py（添加 print("hacked")）
# 2. 运行 python main.py
# 预期：SecurityError: 评估器已被篡改
```

### 白名单测试

```bash
# 1. 修改 harness/sandbox.py（非白名单文件）
# 2. 运行 python main.py
# 预期：拒绝评估，日志 action="reject"
```

## 故障排查

### 问题：`EXPECTED_HASH` 为空
**解决**：运行 `python harness/evaluate.py`，将输出的哈希值填入 `EXPECTED_HASH`。

### 问题：Git 报错 "not a git repository"
**解决**：
```bash
git init
git add -A
git commit -m "Initial commit"
```

### 问题：没有检测到文件修改
**解决**：确保修改了 `test_task` 并保存。

### 问题：沙箱执行超时
**解决**：检查 `model.py` 是否有死循环或长时间计算。

## Phase 2 展望

Phase 1 验证了核心机制，Phase 2 将扩展为：
- **真实任务**：睡眠分期模型优化（基于 EEG 信号）
- **LLM Agent**：用 Claude API 替代手动修改参数
- **Docker 沙箱**：替换 `subprocess`，实现网络隔离
- **配置扩展**：使用 `test_task` 管理超参数

## 技术规格

- **评估指标**：MSE (Mean Squared Error)
- **测试点**：x ∈ [-10, 10]，100 个均匀采样点
- **日志格式**：JSONL（可用 `jq` 分析）
- **Git 策略**：每次 accept 一个 commit，rollback 回退一个 commit

## 参考资料

- 完整 5-phase 计划：`/Users/550w/.claude/plans/harness-hazy-river.md`
- 输出目录约定：`~/Documents/Claude/<项目名>/`
