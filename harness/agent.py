"""LLM Agent 模块 - 睡眠分期任务版

功能：
1. 调用 Claude API 生成超参数优化策略
2. 解析并严格校验 LLM 返回的参数
3. 读取历史日志和策略文档，避免重复错误
4. 返回验证后的配置，由主控循环写入文件
"""

import os
import re
import json
import time
import requests
from pathlib import Path
from typing import Optional


def _load_history(log_file: Path, last_n: int = 3) -> str:
    """读取最近 N 轮历史记录

    Args:
        log_file: run_history.jsonl 路径
        last_n: 读取最近几轮

    Returns:
        格式化的历史字符串
    """
    if not log_file.exists():
        return "无历史记录"

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    history = []
    for line in lines[-last_n:]:
        try:
            entry = json.loads(line)
            if entry.get("action") in ("accept", "rollback"):
                history.append(
                    f"轮次 {entry['round']}: "
                    f"balanced_acc={entry.get('balanced_accuracy', 0):.4f}, "
                    f"macro_f1={entry.get('macro_f1', 0):.4f}, "
                    f"val_loss={entry.get('val_loss', 0):.4f}, "
                    f"操作={entry['action']}"
                )
        except (json.JSONDecodeError, KeyError):
            continue

    return "\n".join(history) if history else "无有效历史"


def _load_strategy_doc(doc_file: Path) -> str:
    """读取策略文档（如果存在）

    Args:
        doc_file: strategy.md 路径

    Returns:
        文档内容或空字符串
    """
    if doc_file.exists():
        with open(doc_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def _extract_json_safely(text: str) -> Optional[dict]:
    """安全提取 JSON，支持 markdown 代码块

    Args:
        text: LLM 返回的原始文本

    Returns:
        解析后的字典，失败返回 None
    """
    # 尝试正则提取 markdown 中的 JSON
    patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(.*?)\n```',      # ``` ... ```
        r'\{.*\}'                    # 直接 JSON
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1) if match.lastindex else match.group(0)
            try:
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                continue

    # 最后尝试直接解析整个文本
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _validate_hyperparams(result: dict) -> dict:
    """严格校验超参数安全性

    Args:
        result: 解析后的字典

    Returns:
        校验后的字典

    Raises:
        ValueError: 参数不合法
    """
    # 必需字段
    required_keys = {
        "learning_rate", "batch_size", "num_epochs",
        "lstm_hidden_size", "dropout", "reasoning"
    }
    missing = required_keys - result.keys()
    if missing:
        raise ValueError(f"缺少必需字段: {missing}")

    # 参数范围约束（防注入）
    constraints = {
        "learning_rate": (1e-5, 1e-2),
        "batch_size": (16, 128),
        "num_epochs": (5, 20),
        "lstm_hidden_size": (64, 256),
        "dropout": (0.3, 0.7),
        "weight_decay": (0, 0.001),
        "patience": (2, 10),
    }

    validated = {}
    for key, (min_val, max_val) in constraints.items():
        if key not in result:
            continue  # 可选参数

        val = result[key]
        if not isinstance(val, (int, float)):
            raise ValueError(f"参数 {key}={val} 不是数值类型")
        if key in {"batch_size", "num_epochs", "lstm_hidden_size", "patience"}:
            if isinstance(val, bool) or int(val) != val:
                raise ValueError(f"参数 {key}={val} 必须是整数")
        if not (min_val <= val <= max_val):
            raise ValueError(f"参数 {key}={val} 超出范围 [{min_val}, {max_val}]")

        validated[key] = int(val) if key in {"batch_size", "num_epochs", "lstm_hidden_size", "patience"} else val

    # 保留 reasoning
    validated["reasoning"] = str(result.get("reasoning", ""))

    return validated


def generate_sleep_strategy(
    current_metrics: dict,
    best_metrics: dict,
    round_num: int,
    project_root: Path
) -> dict:
    """调用 Claude API 生成超参数优化策略

    Args:
        current_metrics: 当前指标 {
            "balanced_accuracy": float,
            "macro_f1": float,
            "val_loss": float
        }
        best_metrics: 历史最佳指标（同结构）
        round_num: 当前轮次
        project_root: 项目根目录

    Returns:
        dict: {
            "learning_rate": float,
            "batch_size": int,
            "num_epochs": int,
            "lstm_hidden_size": int,
            "dropout": float,
            "weight_decay": float (可选),
            "patience": int (可选),
            "reasoning": str
        }

    Raises:
        ValueError: API 调用失败或返回格式错误
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("未设置 ANTHROPIC_API_KEY 环境变量")

    # 加载历史
    log_file = project_root / "logs" / "run_history.jsonl"
    history = _load_history(log_file, last_n=3)

    strategy_doc_file = project_root / "strategy.md"
    strategy_doc = _load_strategy_doc(strategy_doc_file)

    # 读取当前配置
    config_file = project_root / "tasks" / "sleep_edf" / "model_config.py"
    current_config = {}
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("LEARNING_RATE"):
                    current_config["learning_rate"] = float(line.split("=")[1].strip())
                elif line.startswith("BATCH_SIZE"):
                    current_config["batch_size"] = int(line.split("=")[1].strip())
                elif line.startswith("NUM_EPOCHS"):
                    current_config["num_epochs"] = int(line.split("=")[1].strip())
                elif line.startswith("LSTM_HIDDEN_SIZE"):
                    current_config["lstm_hidden_size"] = int(line.split("=")[1].strip())
                elif line.startswith("DROPOUT"):
                    current_config["dropout"] = float(line.split("=")[1].strip())
                elif line.startswith("WEIGHT_DECAY"):
                    current_config["weight_decay"] = float(line.split("=")[1].strip())
                elif line.startswith("PATIENCE"):
                    current_config["patience"] = int(line.split("=")[1].strip())

    # 构造 Prompt
    prompt = f"""你是深度学习超参数调优专家。任务：优化 TinySleepNet 模型的超参数以提升睡眠分期准确率。

## 当前状态
数据集: Sleep-EDF (单受试者，前 80% 训练，后 20% 验证)
模型: TinySleepNet (LSTM + Attention)
轮次: {round_num}/10

当前配置:
{json.dumps(current_config, indent=2, ensure_ascii=False)}

当前指标:
- Balanced Accuracy: {current_metrics.get('balanced_accuracy', 0):.4f}
- Macro F1: {current_metrics.get('macro_f1', 0):.4f}
- Val Loss: {current_metrics.get('val_loss', 0):.4f}

历史最佳指标:
- Balanced Accuracy: {best_metrics.get('balanced_accuracy', 0):.4f}
- Macro F1: {best_metrics.get('macro_f1', 0):.4f}
- Val Loss: {best_metrics.get('val_loss', 0):.4f}

## 历史记录（最近 3 轮）
{history}

## 策略文档
{strategy_doc if strategy_doc else "无"}

## 可调参数及范围
- learning_rate: [1e-5, 1e-2] (学习率)
- batch_size: [16, 128] (批大小)
- num_epochs: [5, 20] (训练轮数)
- lstm_hidden_size: [64, 256] (LSTM 隐藏层大小)
- dropout: [0.3, 0.7] (Dropout 比例)
- weight_decay: [0, 0.001] (L2 正则化，可选)
- patience: [2, 10] (早停耐心值，可选)

## 优化目标
1. 提升 Balanced Accuracy (主要)
2. 提升 Macro F1
3. 降低 Val Loss

## 约束
- 每次只修改 1-2 个参数（避免同时改动过多）
- 优先调整 learning_rate 和 dropout（影响最大）
- 如果过拟合（train loss << val loss），增大 dropout 或 weight_decay
- 如果欠拟合（train loss 和 val loss 都高），增大 lstm_hidden_size 或降低 dropout

返回 JSON（不要修改的参数可保持原值）：
{{
  "learning_rate": 数值,
  "batch_size": 整数,
  "num_epochs": 整数,
  "lstm_hidden_size": 整数,
  "dropout": 数值,
  "weight_decay": 数值（可选）,
  "patience": 整数（可选）,
  "reasoning": "为什么这样调整会改善指标"
}}"""

    # 重试机制：最多 3 次
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(
                "https://www.sui-xiang.net/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": "claude-opus-5",
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            # 安全提取 content
            try:
                response_text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                raise ValueError(f"API 返回格式错误，找不到内容。原始返回：{data}")

            # 提取并校验 JSON
            result = _extract_json_safely(response_text)
            if result is None:
                raise ValueError(f"无法解析 JSON，原始返回：{response_text[:200]}")

            validated = _validate_hyperparams(result)
            return validated

        except (requests.RequestException, ValueError, KeyError) as e:
            last_error = e
            if attempt < 2:  # 前 2 次失败后重试
                time.sleep(2 ** attempt)  # 指数退避
                continue

    # 3 次均失败
    raise ValueError(f"API 调用失败（3 次重试后）: {last_error}")
