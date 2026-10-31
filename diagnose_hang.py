"""诊断 main.py 卡住的位置"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

TASK_DIR = PROJECT_ROOT / "tasks" / "sleep_edf"
DATA_DIR = TASK_DIR / "data" / "processed" / "sleep-edf-sc-full"
SPLIT_FILE = DATA_DIR / "subject_split.json"

print("=" * 70)
print("诊断 main.py 卡住位置")
print("=" * 70)

# Step 1: validate_dataset_inputs
print("\n[1/4] 执行 validate_dataset_inputs()...")
start = time.time()
try:
    from main import validate_dataset_inputs
    validate_dataset_inputs()
    print(f"  ✓ 完成 ({time.time() - start:.2f}s)")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

# Step 2: 读取初始配置
print("\n[2/4] 读取 model_config.py...")
start = time.time()
try:
    CONFIG_FILE = TASK_DIR / "model_config.py"
    config_content = CONFIG_FILE.read_text(encoding="utf-8")
    print(f"  ✓ 完成 ({time.time() - start:.2f}s)")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

# Step 3: 模拟调用 run_experiment (但不真正训练)
print("\n[3/4] 测试 train.py 启动...")
start = time.time()
try:
    from harness.sandbox import run_python_script
    output_dir = TASK_DIR / "outputs" / "diagnostic_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只运行 smoke test 模式，不需要真实数据
    result = run_python_script(
        TASK_DIR / "train.py",
        ["--smoke-test", "--epochs", "1", "--output-dir", str(output_dir)],
        cwd=TASK_DIR,
        timeout=60,
    )

    if result["success"]:
        print(f"  ✓ train.py 可以正常启动和完成 ({result['duration']:.2f}s)")
        print(f"    返回码: {result['returncode']}")
    else:
        print(f"  ✗ train.py 执行失败 ({result['duration']:.2f}s)")
        print(f"    错误: {result['error']}")
        if result['stderr']:
            print("    stderr:", result['stderr'][:500])
except Exception as e:
    print(f"  ✗ 启动失败: {e}")
    import traceback
    traceback.print_exc()

# Step 4: 测试 API 调用
print("\n[4/4] 测试 generate_sleep_strategy()...")
start = time.time()
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    from harness.agent import generate_sleep_strategy

    test_metrics = {
        "balanced_accuracy": 0.7,
        "macro_f1": 0.65,
        "val_loss": 1.2,
        "checkpoint_epoch": 3,
    }

    print("  发送 API 请求...")
    proposal = generate_sleep_strategy(
        current_metrics=test_metrics,
        best_metrics=test_metrics,
        round_num=1,
        project_root=PROJECT_ROOT,
    )

    print(f"  ✓ API 调用成功 ({time.time() - start:.2f}s)")
    print(f"    返回配置: learning_rate={proposal.get('learning_rate')}, "
          f"batch_size={proposal.get('batch_size')}")

except Exception as e:
    print(f"  ✗ API 调用失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
print("\n如果以上步骤都通过，main.py 应该可以正常运行。")
print("如果某一步卡住，说明问题在该步骤。")
