"""白名单验证模块

功能：
1. 加载白名单配置
2. 验证文件修改是否在白名单范围内
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict


def load_whitelist(whitelist_path: Path) -> Dict:
    """加载白名单配置

    Args:
        whitelist_path: whitelist.json 文件路径

    Returns:
        dict: 白名单配置
    """
    with open(whitelist_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_changed_files() -> List[str]:
    """获取当前修改的文件列表

    Returns:
        list: 修改的文件路径列表
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=True
    )

    changed_files = result.stdout.strip().split('\n')
    return [f for f in changed_files if f]  # 过滤空字符串


def validate_changes(whitelist: Dict) -> tuple[bool, List[str]]:
    """验证修改的文件是否都在白名单中

    Args:
        whitelist: 白名单配置

    Returns:
        tuple: (是否通过验证, 非法文件列表)
    """
    allowed_files = set(whitelist["allowed_files"])
    changed_files = get_changed_files()

    # 检查每个修改的文件是否在白名单中
    illegal_files = []
    for file in changed_files:
        if file not in allowed_files:
            illegal_files.append(file)

    return len(illegal_files) == 0, illegal_files
