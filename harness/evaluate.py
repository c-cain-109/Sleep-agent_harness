"""评估器模块 - 防篡改核心

功能：
1. SHA256 哈希验证自身完整性
2. 计算模型函数与目标函数的 MSE
3. 返回评估分数
"""

import hashlib
import sys
from pathlib import Path

# 期望哈希值（首次运行后手动填入）
EXPECTED_HASH = "31170b44a1d60d154f2939412ffa827d2ed815a9bd96a0a5e024cd5e0d9fdf75"


class SecurityError(Exception):
    """安全验证失败"""
    pass


def verify_integrity():
    """验证评估器自身完整性"""
    if not EXPECTED_HASH:
        # 首次运行，生成哈希值
        with open(__file__, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"首次运行，请将以下哈希值复制到 EXPECTED_HASH 常量:")
        print(f'EXPECTED_HASH = "{actual_hash}"')
        sys.exit(0)

    with open(__file__, 'rb') as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    if actual_hash != EXPECTED_HASH:
        raise SecurityError(f"评估器已被篡改! 期望: {EXPECTED_HASH}, 实际: {actual_hash}")


def evaluate():
    """评估模型函数性能

    返回:
        dict: {"score": float, "success": bool, "error": str or None}
    """
    try:
        # 验证自身完整性
        verify_integrity()

        # 导入模型和目标函数
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from toy_task.model import model_function
        from toy_task.target_function import target_function

        # 生成测试点 x ∈ [-10, 10]
        import numpy as np
        x_values = np.linspace(-10, 10, 100)

        # 计算 MSE
        errors = []
        for x in x_values:
            model_y = model_function(x)
            target_y = target_function(x)
            errors.append((model_y - target_y) ** 2)

        mse = sum(errors) / len(errors)

        return {
            "score": mse,
            "success": True,
            "error": None
        }

    except SecurityError as e:
        return {
            "score": None,
            "success": False,
            "error": f"安全验证失败: {str(e)}"
        }

    except Exception as e:
        return {
            "score": None,
            "success": False,
            "error": f"评估失败: {str(e)}"
        }


if __name__ == "__main__":
    import json
    result = evaluate()
    print(json.dumps(result, ensure_ascii=False))
