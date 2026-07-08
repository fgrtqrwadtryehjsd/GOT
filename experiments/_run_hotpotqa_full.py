"""
批量串行运行：HotpotQA 500 条全量（5 方法 + cot_sc_gers）
用断点续跑机制，可中断后重新启动
"""
import os
import sys
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATASETS_AND_METHODS = [
    # HotpotQA 500 条（已有 100 条，从 101 续跑）
    ("hotpotqa", "gers_adaptive", 500),
    ("hotpotqa", "zero_shot", 500),
    ("hotpotqa", "standard_cot", 500),
    ("hotpotqa", "cot_sc", 500),
    ("hotpotqa", "cot_sc_gers", 500),
    ("hotpotqa", "tot", 500),
]

if __name__ == "__main__":
    print("=" * 70)
    print("HotpotQA 500 条全量重跑（5 方法 + cot_sc_gers）")
    print("=" * 70)
    for dataset, method, n in DATASETS_AND_METHODS:
        out = ROOT / "experiments" / "results" / f"{dataset}_{method}_results.json"
        if out.exists():
            data = json.load(open(out, encoding="utf-8"))
            existing_n = len(data.get("results", []))
            if existing_n >= n:
                print(f"\n[SKIP] {method} @ {dataset}: 已完成 {existing_n}/{n}")
                continue
            print(f"\n[RESUME] {method} @ {dataset}: 已完成 {existing_n}/{n}")
        else:
            print(f"\n[NEW] {method} @ {dataset}: 0/{n}")

        cmd = [
            sys.executable, "-X", "utf8",
            "experiments/run_quick_exp.py",
            "--dataset", dataset,
            "--method", method,
            "--num_samples", str(n),
            "--model", "qwen3-8b",
        ]
        ret = subprocess.run(cmd, capture_output=False)
        if ret.returncode != 0:
            print(f"[ERROR] {method} @ {dataset} 退出码 {ret.returncode}")
    print("\n" + "=" * 70)
    print("HotpotQA 500 条全量重跑完成（或部分完成）")
    print("=" * 70)
