"""批量运行并汇总结果"""
import subprocess, sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TASKS = [
    ("gsm8k",    "zero_shot",    50),
    ("gsm8k",    "standard_cot", 50),
    ("hotpotqa", "zero_shot",    50),
    ("hotpotqa", "standard_cot", 50),
    ("clutrr",   "zero_shot",    30),
    ("clutrr",   "standard_cot", 30),
]

for dataset, method, n in TASKS:
    log = f"experiments/results/log_{dataset}_{method}.txt"
    print(f"\n{'='*50}")
    print(f"运行: {method} @ {dataset} ({n}条)")
    result_file = Path(f"experiments/results/{dataset}_{method}_results.json")
    if result_file.exists():
        data = json.load(open(result_file, encoding="utf-8"))
        s = data["summary"]
        print(f"  [已存在] EM={s['avg_em']:.4f} F1={s['avg_f1']:.4f} n={s['num_samples']}")
        continue
    ret = subprocess.run(
        [sys.executable, "-X", "utf8", "experiments/run_quick_exp.py",
         "--dataset", dataset, "--method", method,
         "--num_samples", str(n), "--model", "qwen3-8b"],
        capture_output=False
    )
    if result_file.exists():
        data = json.load(open(result_file, encoding="utf-8"))
        s = data["summary"]
        print(f"  EM={s['avg_em']:.4f} F1={s['avg_f1']:.4f}")

print("\n\n=== 汇总 ===")
for dataset, method, _ in TASKS:
    f = Path(f"experiments/results/{dataset}_{method}_results.json")
    if f.exists():
        s = json.load(open(f, encoding="utf-8"))["summary"]
        print(f"{dataset:12} {method:15} EM={s['avg_em']:.4f} F1={s['avg_f1']:.4f} n={s['num_samples']}")
