"""LLM Agent 模块 - Phase 2 重构版

功能：
1. 调用 Claude API 生成参数优化策略
2. 解析并严格校验 LLM 返回的参数
3. 读取历史日志和策略文档，避免重复错误
4. 仅返回验证后的策略字典，不直接修改文件
"""

import os
import re
import json
import math
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
                    f"轮次 {entry['round']}: MSE={entry['score']:.4f}, "
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


def _validate_params(result: dict) -> dict:
    """严格校验参数安全性

    Args:
        result: 解析后的字典

    Returns:
        校验后的字典

    Raises:
        ValueError: 参数不合法
    """
    required_keys = {"a", "b", "c", "reasoning", "modified_param"}
    missing = required_keys - result.keys()
    if missing:
        raise ValueError(f"缺少必需字段: {missing}")

    # 校验数值类型和有限性（防注入）
    for key in ("a", "b", "c"):
        val = result[key]
        if not isinstance(val, (int, float)):
            raise ValueError(f"参数 {key}={val} 不是数值类型")
        if not math.isfinite(val):
            raise ValueError(f"参数 {key}={val} 不是有限数值（inf/nan）")

    # 校验 modified_param
    if result["modified_param"] not in ("a", "b", "c"):
        raise ValueError(f"modified_param 必须是 a/b/c，实际: {result['modified_param']}")

    return result


def generate_strategy(
    current_params: dict,
    current_score: float,
    best_score: float,
    round_num: int,
    target_function: str,
    project_root: Path
) -> dict:
    """调用 Claude API 生成优化策略（带重试和历史联动）

    Args:
        current_params: 当前参数 {"a": float, "b": float, "c": float}
        current_score: 当前 MSE 分数
        best_score: 历史最佳分数
        round_num: 当前轮次
        target_function: 目标函数描述
        project_root: 项目根目录

    Returns:
        dict: {
            "a": float,
            "b": float,
            "c": float,
            "reasoning": str,
            "modified_param": "a" | "b" | "c"
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

    # 构造 Prompt（单变量约束）
    prompt = f"""你是数学优化专家。任务：优化二次函数参数逼近目标函数。

## 当前状态
目标函数: {target_function}
当前参数: a={current_params['a']}, b={current_params['b']}, c={current_params['c']}
当前 MSE: {current_score:.6f}
历史最佳 MSE: {best_score:.6f}
轮次: {round_num}/10

## 历史记录（最近 3 轮）
{history}

## 策略文档
{strategy_doc if strategy_doc else "无"}

## 核心约束
**每轮只能修改 a, b, c 中的一个参数**，其他两个必须保持不变。

## 任务
1. 分析当前参数与目标的差距
2. 选择修改 a/b/c 中的哪一个
3. 给出新值（保留一位小数）
4. 说明为什么这个修改会降低 MSE

返回 JSON：
{{
  "a": 数值（如果不修改 a，保持原值 {current_params['a']}）,
  "b": 数值（如果不修改 b，保持原值 {current_params['b']}）,
  "c": 数值（如果不修改 c，保持原值 {current_params['c']}）,
  "modified_param": "a" 或 "b" 或 "c"（本轮修改的参数）,
  "reasoning": "推理过程"
}}"""

    # 重试机制：最多 3 次
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(
                "https://www.sui-xiang.vip/v1/messages",
                headers={
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "x-api-key": api_key
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            # 安全提取 content
            content = data.get("content")
            if not content or not isinstance(content, list):
                raise ValueError("API 返回格式错误：缺少 content 字段")

            response_text = content[0].get("text", "")
            if not response_text:
                raise ValueError("API 返回为空")

            # 提取并校验 JSON
            result = _extract_json_safely(response_text)
            if result is None:
                raise ValueError(f"无法解析 JSON，原始返回：{response_text[:200]}")

            validated = _validate_params(result)
            return validated

        except (requests.RequestException, ValueError, KeyError) as e:
            last_error = e
            if attempt < 2:  # 前 2 次失败后重试
                time.sleep(2 ** attempt)  # 指数退避
                continue

    # 3 次均失败
    raise ValueError(f"API 调用失败（3 次重试后）: {last_error}")
