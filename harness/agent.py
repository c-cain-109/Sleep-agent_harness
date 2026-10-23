"""LLM Agent 模块 - Phase 2

功能：
1. 调用 Claude API 生成参数优化策略
2. 解析 LLM 返回的参数建议
3. 自动修改 model.py
"""

import os
import json
import anthropic
from pathlib import Path


def generate_strategy(current_params: dict, current_score: float, best_score: float, round_num: int, target_function: str) -> dict:
    """调用 Claude API 生成优化策略

    Args:
        current_params: 当前参数 {"a": float, "b": float, "c": float}
        current_score: 当前 MSE 分数
        best_score: 历史最佳分数
        round_num: 当前轮次
        target_function: 目标函数描述

    Returns:
        dict: {"a": float, "b": float, "c": float, "reasoning": str}
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("未设置 ANTHROPIC_API_KEY 环境变量")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""你是一个数学优化专家。任务是优化二次函数的参数使其逼近目标函数。

目标函数: {target_function}
当前参数: a={current_params['a']}, b={current_params['b']}, c={current_params['c']}
当前 MSE: {current_score:.6f}
历史最佳 MSE: {best_score:.6f}
当前轮次: {round_num}/10

请分析当前参数与目标函数的差距，提出下一组参数。要求：
1. 分析当前参数的问题
2. 给出新的 a, b, c 参数（保留一位小数）
3. 解释为什么这组参数会更接近目标

返回 JSON 格式：
{{
  "a": 数值,
  "b": 数值,
  "c": 数值,
  "reasoning": "推理过程"
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text

    # 提取 JSON
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0].strip()
    else:
        json_str = response_text.strip()

    result = json.loads(json_str)

    # 验证返回格式
    required_keys = {"a", "b", "c", "reasoning"}
    if not required_keys.issubset(result.keys()):
        raise ValueError(f"API 返回格式错误，缺少必需字段: {required_keys - result.keys()}")

    return result


def apply_strategy(strategy: dict, model_file: Path):
    """将策略应用到 model.py

    Args:
        strategy: generate_strategy 返回的策略
        model_file: model.py 文件路径
    """
    a, b, c = strategy["a"], strategy["b"], strategy["c"]

    new_content = f"""def model_function(x):
    \"\"\"待优化的模型函数\"\"\"
    a, b, c = {a}, {b}, {c}  # Agent 自动优化
    return a * x**2 + b * x + c
"""

    with open(model_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"✏️  Agent 建议: a={a}, b={b}, c={c}")
    print(f"   推理: {strategy['reasoning'][:100]}...")
