# 项目文档导航

本目录用于存放 Sleep-Agent Harness 的项目设计、实验记忆机制和开发历史文档。

如果是第一次接触项目，建议先阅读根目录的 [README.md](../README.md)。

## 1. 推荐阅读顺序

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [项目 README](../README.md) | 项目目标、架构、数据集、环境和运行入口 |
| 2 | [Agent 调参策略](../strategy.md) | Agent 调整模型超参数时使用的策略参考 |
| 3 | [实验记忆系统](experiment_memory/README.md) | 实验记忆的存储、读取与使用方式 |
| 4 | [Sleep-EDF 预处理说明](../tasks/sleep_edf/preprocess/README.md) | 原始 EDF 数据的预处理流程 |
| 5 | [预处理测试记录](../tasks/sleep_edf/preprocess/TESTLOG.md) | 预处理模块的测试过程和结果 |

## 2. 实验记忆系统

`experiment_memory/` 存放 Agent 结构化实验记忆机制的设计文档。

| 文件 | 说明 |
|---|---|
| [README.md](experiment_memory/README.md) | 实验记忆系统介绍 |
| [AGENT_MEMORY_CONTRACT.md](experiment_memory/AGENT_MEMORY_CONTRACT.md) | Agent 使用实验记忆的约定 |
| [experiment_memory_schema.json](experiment_memory/experiment_memory_schema.json) | 实验记录的数据结构定义 |
| [strategy_memory_template.md](experiment_memory/strategy_memory_template.md) | 策略记忆模板 |

## 3. 开发历史与阶段计划

以下文档主要用于了解项目演进过程，不应直接作为当前运行操作指南。

| 文件 | 内容 | 使用说明 |
|---|---|---|
| [Phase 2 初期计划](archive/phase2_plan.md) | Phase 2 接入 Sleep-EDF 的初期计划 | 历史设计参考 |
| [Phase 3 旧版运行指南](archive/phase3_guide_legacy.md) | 睡眠分期自动调参运行说明 | 早期运行指南，部分路径和参数可能已经过时 |
| [项目历史进展](archive/progress_history.md) | Phase 1 至 Phase 2 的开发过程 | 历史记录 |

## 4. 文档维护约定

为避免文档重复和信息不一致，项目文档遵循以下约定：

1. **根目录 README 是项目总入口。** 项目目标、总体流程、数据要求和运行方式以此为准。
2. **代码是实际运行行为的依据。** 涉及命令行参数、文件路径和程序逻辑时，应以当前代码为准。
3. **专题文档只描述各自负责的内容。** 预处理文档不重复介绍整个 Harness，实验记忆文档不重复介绍模型训练流程。
4. **历史文档保留历史价值。** 已经过时的方案和实验过程可以保留，但应明确标注，不与当前操作指南混淆。
5. **实验结果必须可追溯。** 真实实验指标应来自实际运行记录，不得将模拟数据或 Smoke Test 结果写成正式模型性能。
