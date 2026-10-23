"""Sleep-Harness Phase 1: 玩具任务主控循环

流程：
1. 初始化：加载白名单，获取基线分数
2. 循环 10 轮：
   - 提示用户修改 test_task/model.py
   - 白名单验证 → 沙箱评估 → accept/rollback 决策
   - 记录 JSONL 日志
"""

import json
import time
from pathlib import Path
from datetime import datetime

from harness.whitelist import load_whitelist, validate_changes
from harness.sandbox import run_evaluation
from harness.git_manager import accept_changes, rollback, get_last_commit_hash, get_changed_files


def init_logs():
    """初始化日志目录"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir / "run_history.jsonl"


def log_round(log_file: Path, round_num: int, score: float, action: str, duration: float, commit_hash: str = None, error: str = None):
    """记录一轮评估结果"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "round": round_num,
        "score": score,
        "action": action,
        "duration": duration,
        "commit_hash": commit_hash,
        "error": error
    }

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    print("=" * 60)
    print("Sleep-Harness Phase 1: 玩具任务验证")
    print("=" * 60)

    # 初始化
    project_root = Path(__file__).parent
    whitelist_path = project_root / "whitelist.json"
    log_file = init_logs()

    whitelist = load_whitelist(whitelist_path)
    print(f"\n✅ 白名单已加载，允许修改文件:")
    for file in whitelist["allowed_files"]:
        print(f"   - {file}")

    # 获取基线分数（第 0 轮）
    print("\n🎯 第 0 轮: 获取基线分数...")
    baseline_result = run_evaluation()

    if not baseline_result["success"]:
        print(f"❌ 基线评估失败: {baseline_result['error']}")
        return

    baseline_score = baseline_result["score"]
    print(f"   基线 MSE: {baseline_score:.6f}")
    log_round(log_file, 0, baseline_score, "baseline", baseline_result["duration"])

    best_score = baseline_score

    # 主循环：10 轮
    for round_num in range(1, 11):
        print(f"\n{'=' * 60}")
        print(f"第 {round_num}/10 轮")
        print(f"{'=' * 60}")
        print(f"当前最佳 MSE: {best_score:.6f}")
        print(f"\n📝 请修改 test_task/model.py 中的参数 (a, b, c)")
        print(f"   目标函数: 2.5*x² - 3.0*x + 1.5")
        print(f"   等待文件修改...")

        # 轮询检测文件修改
        while True:
            changed_files = get_changed_files()
            if changed_files:
                print(f"✓ 检测到修改: {', '.join(changed_files)}")
                break
            time.sleep(1)  # 每秒检查一次

        # 白名单验证
        is_valid, illegal_files = validate_changes(whitelist)
        if not is_valid:
            print(f"❌ 白名单验证失败！非法修改文件:")
            for file in illegal_files:
                print(f"   - {file}")
            log_round(log_file, round_num, None, "reject", 0, error="whitelist_violation")
            print("   已拒绝评估，请撤销非法修改")
            continue

        print("✅ 白名单验证通过")

        # 沙箱评估
        print("🔍 正在评估...")
        result = run_evaluation()

        if not result["success"]:
            print(f"❌ 评估失败: {result['error']}")
            log_round(log_file, round_num, None, "error", result["duration"], error=result["error"])
            continue

        score = result["score"]
        print(f"   MSE: {score:.6f} (用时 {result['duration']:.2f}s)")

        # Accept/Rollback 决策
        if score < best_score:
            print(f"✅ 性能提升！({best_score:.6f} → {score:.6f})")
            commit_msg = f"Round {round_num}: 优化参数，MSE={score:.6f}"
            accept_changes(commit_msg)
            commit_hash = get_last_commit_hash()
            log_round(log_file, round_num, score, "accept", result["duration"], commit_hash)
            best_score = score
        else:
            print(f"⚠️  性能未提升 ({score:.6f} >= {best_score:.6f})")
            rollback()
            log_round(log_file, round_num, score, "rollback", result["duration"])

    # 总结
    print(f"\n{'=' * 60}")
    print("🎉 Phase 1 验证完成！")
    print(f"{'=' * 60}")
    print(f"初始 MSE: {baseline_score:.6f}")
    print(f"最终 MSE: {best_score:.6f}")
    print(f"改进幅度: {((baseline_score - best_score) / baseline_score * 100):.2f}%")
    print(f"\n📊 日志文件: {log_file}")
    print(f"📜 Git 历史: git log --oneline")


if __name__ == "__main__":
    main()
