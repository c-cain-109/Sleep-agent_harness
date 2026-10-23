"""沙箱执行模块

功能：
1. 在受限环境中执行评估器
2. 限制执行时间和内存
3. 捕获异常并返回结构化结果
"""

import subprocess
import time
import json
import sys
from pathlib import Path


def run_evaluation(timeout: int = 300) -> dict:
    """在沙箱中运行评估

    Args:
        timeout: 超时时间（秒）

    Returns:
        dict: {
            "score": float or None,
            "duration": float,
            "success": bool,
            "error": str or None
        }
    """
    project_root = Path(__file__).parent.parent
    evaluate_script = project_root / "harness" / "evaluate.py"

    start_time = time.time()

    try:
        # 使用 subprocess 执行评估器
        # macOS 上 resource.setrlimit 对内存限制支持有限，暂时只用超时
        proc = subprocess.Popen(
            [sys.executable, str(evaluate_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_root)
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            duration = time.time() - start_time

            if proc.returncode != 0:
                return {
                    "score": None,
                    "duration": duration,
                    "success": False,
                    "error": f"评估器退出异常 (退出码 {proc.returncode}): {stderr}"
                }

            # 解析 JSON 输出
            result = json.loads(stdout)
            result["duration"] = duration
            return result

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return {
                "score": None,
                "duration": timeout,
                "success": False,
                "error": f"执行超时（超过 {timeout} 秒）"
            }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "score": None,
            "duration": duration,
            "success": False,
            "error": f"沙箱执行失败: {str(e)}"
        }
