"""Git 管理模块 - 类封装版本

功能：
1. Accept changes: git add + commit
2. Rollback: git reset --hard HEAD
3. Get changed files: git diff --name-only
"""

import subprocess
from pathlib import Path
from typing import List


class GitManager:
    """Git 操作封装类"""

    def __init__(self, repo_path: Path):
        """初始化

        Args:
            repo_path: Git 仓库根目录
        """
        self.repo_path = Path(repo_path).resolve()

        # 确认是 Git 仓库
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            raise ValueError(f"{self.repo_path} 不是一个 Git 仓库")

    def has_changes(self, paths: List[str] | None = None) -> bool:
        """检查是否有未提交的修改

        Returns:
            bool: True 表示有修改
        """
        command = ["git", "status", "--porcelain", "--"]
        command.extend(paths or [])
        result = subprocess.run(
            command,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())

    def get_changed_files(self, paths: List[str] | None = None) -> List[str]:
        """获取当前修改的文件列表

        Returns:
            list: 修改的文件路径列表
        """
        command = ["git", "status", "--porcelain", "--"]
        command.extend(paths or [])
        result = subprocess.run(
            command,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        changed_files = result.stdout.splitlines()
        return [f[3:] for f in changed_files if len(f) > 3]

    def commit_changes(self, commit_msg: str, paths: List[str] | None = None):
        """接受当前更改并提交

        Args:
            commit_msg: 提交信息
        """
        selected = paths or ["."]
        subprocess.run(
            ["git", "add", "--", *selected],
            cwd=self.repo_path,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", commit_msg, "--", *selected],
            cwd=self.repo_path,
            check=True
        )

    def rollback_changes(self):
        """回滚工作区修改（不删除 commit）"""
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=self.repo_path,
            check=True
        )

    def get_last_commit_hash(self) -> str:
        """获取最后一次提交的哈希值

        Returns:
            str: 提交哈希（短格式）
        """
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
