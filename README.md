# Sleep-Agent Harness

这是一个面向 Sleep-EDF 睡眠分期任务的自动化实验框架。主流程为：

```text
prepare_sleepedf.py
  → tasks/sleep_edf/data/prepared/SC4001E0.npz
  → train.py
  → best_model.pt + metrics.jsonl
  → harness/evaluate.py
  → agent 提议超参数
  → accept / rollback
```

TinySleepNet 使用 [MedMaxLab/selfEEG v0.2.2](https://github.com/MedMaxLab/selfEEG/tree/v0.2.2)
提供的官方 PyTorch 实现，依赖版本固定为 `selfeeg==0.2.2`。

## 目录约定

```text
main.py                 最终睡眠分期自动调优入口
harness/                睡眠任务的 Agent、评估器、子进程执行、白名单和 Git 管理
tasks/sleep_edf/        当前主任务
tasks/test_task/        Phase 1 玩具任务归档，不参与主流程
logs/                   运行记录
```

## 环境

推荐使用已经验证过的 Python 3.11 环境：

```bash
/opt/anaconda3/envs/sleep-harness/bin/python --version
```

重新安装依赖：

```bash
python -m pip install -r requirements.txt
```

## Phase 2：生成并验证演示结果

仓库已经提供经过独立校验的预处理数据：

```text
tasks/sleep_edf/data/prepared/SC4001E0.npz
```

重新生成训练与评估结果：

```bash
cd tasks/sleep_edf
/opt/anaconda3/envs/sleep-harness/bin/python train.py \
  --data-file data/prepared/SC4001E0.npz \
  --output-dir outputs/phase2_demo \
  --seed 42

cd ../..
/opt/anaconda3/envs/sleep-harness/bin/python harness/evaluate.py \
  --checkpoint tasks/sleep_edf/outputs/phase2_demo/best_model.pt \
  --data-file tasks/sleep_edf/data/prepared/SC4001E0.npz \
  --output tasks/sleep_edf/outputs/phase2_demo/evaluation.json
```

Phase 2 只验证“数据预处理 → TinySleepNet 训练 → 指标输出”管线。
当前只有一名受试者，训练集和验证集来自同一晚，不能据此评价跨受试者泛化能力。

## Phase 3：自动调参

```bash
/opt/anaconda3/envs/sleep-harness/bin/python main.py --rounds 10
```

`main.py` 只提交 `tasks/sleep_edf/model_config.py`，未提升时恢复本轮开始前的配置，不再使用全仓库 `git reset --hard`。

## 数据文档

- [预处理说明](tasks/sleep_edf/preprocess/README.md)
- [预处理测试记录](tasks/sleep_edf/preprocess/TESTLOG.md)
- [调参策略](strategy.md)
- [Phase 3 运行说明](docs/phase3_guide.md)
- [历史进度记录](docs/progress_history.md)

