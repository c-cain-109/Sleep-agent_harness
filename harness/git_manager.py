"""Git 管理模块

功能：
1. Accept changes: git add + commit
2. Rollback: git reset --hard HEAD~1
3. Get changed files: git diff --name-only HEAD
"""

import subprocess
from typing import List


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
    return [f for f in changed_files if f]


def accept_changes(commit_msg: str):
    """接受当前更改并提交

    Args:
        commit_msg: 提交信息
    """
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    print(f"✅ 已接受更改: {commit_msg}")


def rollback():
    """回滚到上一个提交"""
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
    print("⚠️  已回滚到上一个提交")


def get_last_commit_hash() -> str:
    """获取最后一次提交的哈希值

    Returns:
        str: 提交哈希（短格式）
    """
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()
