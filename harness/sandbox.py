"""Subprocess helpers for task scripts.

This is process isolation with a timeout, not an OS-level security sandbox.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run_python_script(
    script: Path,
    args: list[str] | None = None,
    cwd: Path | None = None,
    timeout: int = 600,
) -> dict:
    """Run a project Python script using the current interpreter."""
    script = Path(script).resolve()
    command = [sys.executable, str(script), *(args or [])]
    start_time = time.time()

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd or script.parent),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return {
                "returncode": None,
                "duration": time.time() - start_time,
                "success": False,
                "stdout": stdout,
                "stderr": stderr,
                "error": f"执行超时（超过 {timeout} 秒）",
            }

        return {
            "returncode": proc.returncode,
            "duration": time.time() - start_time,
            "success": proc.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "error": None if proc.returncode == 0 else stderr.strip(),
        }
    except Exception as exc:
        return {
            "returncode": None,
            "duration": time.time() - start_time,
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": f"脚本启动失败: {exc}",
        }
