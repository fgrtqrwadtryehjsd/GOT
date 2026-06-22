"""批量补全所有实验到目标样本量（断点续跑）"""
import subprocess, sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 目标样本量
TARGETS = [
    ("gsm8k",    "zero_shot",    500),
    ("gsm8k",    "standard_cot", 500),
    ("gsm8k",    "gers",         300),
    ("hotpotqa", "zero_shot",    500),
    ("hotpotqa", "standard_cot", 500),
    ("hotpotqa", "gers",         300),
    ("clutrr",   "zero_shot",    300),
    ("clutrr",   "standard_cot", 300),
    ("clutrr",   "gers",         200),
]

results_dir = Path("experiments/results")

for dataset, method, target in TARGETS:
    f = results_dir / f"{dataset}_{method}_results.json"
    done = 0
    if f.exists():
        try:
            done = json.load(open(f, encoding="utf-8"))["summary"]["num_samples"]
        except Exception:
            done = 0

    if done >= target:
        print(f"[跳过] {dataset:10} {method:15} 已完成 {done}/{target}")
        continue

    print(f"\n{'='*55}")
    print(f"运行: {method} @ {dataset}  ({done} → {target}条)")
    print(f"{'='*55}")

    ret = subprocess.run(
        [sys.executable, "-X", "utf8", "experiments/run_quick_exp.py",
         "--dataset", dataset, "--method", method,
         "--num_samples", str(target),
         "--model", "qwen3-8b",
         "--timeout", "120"],
        capture_output=False
    )

    if f.exists():
        s = json.load(open(f, encoding="utf-8"))["summary"]
        print(f"  完成: EM={s['avg_em']:.4f} F1={s['avg_f1']:.4f} n={s['num_samples']}")

print("\n\n=== 最终汇总 ===")
for dataset, method, target in TARGETS:
    f = results_dir / f"{dataset}_{method}_results.json"
    if f.exists():
        s = json.load(open(f, encoding="utf-8"))["summary"]
        cs_list = [r["metrics"].get("consistency_score", 0)
                   for r in json.load(open(f, encoding="utf-8"))["results"]
                   if r.get("error") is None]
        avg_cs = sum(cs_list)/len(cs_list) if cs_list and any(v > 0 for v in cs_list) else 0
        cs_str = f"{avg_cs:.4f}" if avg_cs > 0 else "   —  "
        print(f"{dataset:10} {method:15} EM={s['avg_em']:.4f} F1={s['avg_f1']:.4f} "
              f"CS={cs_str} n={s['num_samples']}/{target}")
