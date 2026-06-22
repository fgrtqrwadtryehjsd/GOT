"""汇总所有实验结果"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

results_dir = Path("experiments/results")
ALL = {
    "gsm8k":    ["zero_shot", "standard_cot", "gers"],
    "hotpotqa": ["zero_shot", "standard_cot", "gers"],
    "clutrr":   ["zero_shot", "standard_cot", "gers"],
}

print("\n" + "="*70)
print(f"{'数据集':<12} {'方法':<16} {'EM':>7} {'F1':>7} {'Consistency':>12} {'n':>5} {'耗时(s)':>9}")
print("-"*70)

for ds, methods in ALL.items():
    for m in methods:
        f = results_dir / f"{ds}_{m}_results.json"
        if not f.exists():
            print(f"{ds:<12} {m:<16} {'N/A':>7}")
            continue
        s = json.load(open(f, encoding="utf-8"))["summary"]
        cs = s.get("avg_consistency_score", "-")
        # 从 results 里算 consistency
        data = json.load(open(f, encoding="utf-8"))
        cs_vals = [r["metrics"].get("consistency_score", 0) for r in data["results"] if r.get("error") is None]
        avg_cs = sum(cs_vals) / len(cs_vals) if cs_vals and any(v > 0 for v in cs_vals) else "-"
        cs_str = f"{avg_cs:.4f}" if avg_cs != "-" else "-"
        print(f"{ds:<12} {m:<16} {s['avg_em']:>7.4f} {s['avg_f1']:>7.4f} {cs_str:>12} {s['num_samples']:>5} {s['avg_latency']:>9.2f}")
    print()

print("="*70)
