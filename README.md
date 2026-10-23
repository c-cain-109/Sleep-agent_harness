# Sleep-Harness Phase 1: 玩具任务验证

**项目目标**: 通过玩具任务（优化二次函数系数）验证 RSI（Recursive Self-Improvement）框架的核心防篡改机制、沙箱执行和 Git accept/rollback 流程。

## Phase 1 实现总结

### 实现内容

Phase 1 成功验证了 RSI 框架的核心机制：

1. **防篡改评估器**：SHA256 哈希验证，确保评估逻辑不被篡改
2. **沙箱执行**：subprocess 隔离执行，限制超时和资源
3. **白名单验证**：只允许修改指定文件（model.py、config.yaml）
4. **Git 版本控制**：自动 accept/rollback 决策，保留完整历史
5. **自动轮询检测**：每秒检测文件修改，无需手动交互
6. **JSONL 日志**：记录每轮评估结果、分数、操作类型

### 遇到的主要问题及解决方案

#### 问题 1：文件修改检测失败
**现象**：第一轮、第二轮检测不到 model.py 的修改

**原因**：
- `git diff --name-only HEAD` 只检测 staged/committed 的修改
- 应该用 `git diff --name-only` 检测工作区未暂存的修改

**解决**：修改 `harness/git_manager.py` 第 20 行，移除 `HEAD` 参数
```python
# 修改前
["git", "diff", "--name-only", "HEAD"]

# 修改后
["git", "diff", "--name-only"]
```

#### 问题 2：Rollback 删除 commit
**现象**：性能未提升时回滚，导致之前接受的 commit 被删除

**原因**：
- `git reset --hard HEAD~1` 会删除最后一个 commit
- Rollback 应该只回滚工作区修改，不删除 commit 历史

**解决**：修改 `harness/git_manager.py` 第 43 行
```python
# 修改前
["git", "reset", "--hard", "HEAD~1"]

# 修改后
["git", "reset", "--hard", "HEAD"]
```

#### 问题 3：哈希验证失败
**现象**：修改 git_manager.py 后，评估器报告 "已被篡改"

**原因**：
- EXPECTED_HASH 过期，需要重新计算

**解决**：更新 `harness/evaluate.py` 第 14 行的 EXPECTED_HASH

#### 问题 4：交互式输入阻塞
**现象**：PyCharm 运行时卡在第一轮，提示 `EOFError`

**原因**：
- `main.py` 使用 `input()` 等待用户按 Enter
- 后台运行或非交互环境无法提供输入

**解决**：修改 `main.py` 第 73-87 行，移除 `input()` 阻塞，改为自动轮询检测文件修改
```python
# 修改前
input("   修改完成后按 Enter 继续...")

# 修改后
while True:
    changed_files = get_changed_files()
    if changed_files:
        print(f"✓ 检测到修改: {', '.join(changed_files)}")
        break
    time.sleep(1)
```

#### 问题 5：PyCharm 干扰 Git 检测
**现象**：在 PyCharm 运行正常，但命令行测试时检测到 main.py 被修改

**原因**：
- 修改 main.py 后未提交，导致白名单验证失败
- PyCharm 的 .idea/ 文件已在 .gitignore 中，不影响检测

**解决**：提交 main.py 的修改，保持工作区干净

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

### 运行 10 轮测试

```bash
python main.py
```

程序会自动轮询检测 `test_task/model.py` 的修改。每次保存文件后，程序会在 1 秒内检测到修改并自动评估。

**优化目标**：将参数 `a, b, c` 从初始值 `1.0, 1.0, 1.0` 逐步调整到最优解 `2.5, -3.0, 1.5`，使 MSE 接近 0。

## 项目结构

```
/Users/550w/PycharmProjects/Sleep-agent harness/
├── harness/
│   ├── evaluate.py          # 评估器（SHA256 哈希锁定）
│   ├── sandbox.py           # 沙箱执行（subprocess + 超时限制）
│   ├── git_manager.py       # Git 操作（accept/rollback）
│   └── whitelist.py         # 白名单验证
├── test_task/
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
- 计算文件 SHA256 哈希（排除 EXPECTED_HASH 行）
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
- 允许修改：`test_task/model.py`, `test_task/config.yaml`
- 其他文件修改 → 拒绝评估

**验证方法**：
```bash
# 修改非白名单文件（如 harness/sandbox.py）
python main.py  # 预期：拒绝评估，日志记录 "whitelist_violation"
```

### 4. Git Accept/Rollback

`harness/git_manager.py` 根据性能提升决策：
- **Accept**: `score < best_score` → `git add -A && git commit`
- **Rollback**: `score >= best_score` → `git reset --hard HEAD`（只回滚工作区，不删除 commit）

日志记录每轮的 `action` 字段（`accept` / `rollback` / `reject`）。

### 5. 自动轮询检测

`main.py` 每秒检查工作区文件修改：
- 检测到修改 → 白名单验证 → 沙箱评估 → accept/rollback
- 无需手动按 Enter，适合 PyCharm 等 IDE 环境

## 测试指南

### 基本功能测试（10 轮）

| 轮次 | 操作 | 预期结果 |
|------|------|----------|
| 0 | 基线评估 | MSE ≈ 4132.96（初始 `a=b=c=1.0`） |
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
**解决**：
1. 确保修改了 `test_task/model.py` 并保存
2. 检查工作区是否干净（`git status`）
3. 确认修改的参数与上次 commit 不同

### 问题：沙箱执行超时
**解决**：检查 `model.py` 是否有死循环或长时间计算。

### 问题：PyCharm 运行时 EOFError
**解决**：已修复，现在使用轮询检测替代 `input()` 阻塞。

## Phase 2 展望

Phase 1 验证了核心机制，Phase 2 将实现：
- **LLM API 调用**：用 Claude API 替代手动修改参数，让 Agent 自动生成优化策略
- **真实任务**：睡眠分期模型优化（基于 EEG 信号）
- **Docker 沙箱**：替换 `subprocess`，实现网络隔离
- **配置扩展**：使用 `test_task/config.yaml` 管理超参数

## 技术规格

- **评估指标**：MSE (Mean Squared Error)
- **测试点**：x ∈ [-10, 10]，100 个均匀采样点
- **日志格式**：JSONL（可用 `jq` 分析）
- **Git 策略**：每次 accept 一个 commit，rollback 只回滚工作区
- **轮询间隔**：1 秒

## 参考资料

- 完整 5-phase 计划：`/Users/550w/.claude/plans/harness-hazy-river.md`
- 输出目录约定：`~/Documents/Claude/<项目名>/`
