import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

results_dir = Path("experiments/results")
files = {
    "gsm8k_zero_shot":    "GSM8K     Zero-Shot",
    "gsm8k_standard_cot": "GSM8K     Standard CoT",
    "gsm8k_gers":         "GSM8K     GERS",
    "hotpotqa_zero_shot":    "HotpotQA  Zero-Shot",
    "hotpotqa_standard_cot": "HotpotQA  Standard CoT",
    "hotpotqa_gers":         "HotpotQA  GERS",
    "clutrr_zero_shot":    "CLUTRR    Zero-Shot",
    "clutrr_standard_cot": "CLUTRR    Standard CoT",
    "clutrr_gers":         "CLUTRR    GERS",
}

print(f"\n{'方法':<30} {'EM':>7} {'F1':>7} {'n':>5} {'目标':>6}")
print("-"*60)
targets = {"gsm8k": 500, "hotpotqa": 500, "clutrr": 300}
for key, label in files.items():
    f = results_dir / f"{key}_results.json"
    ds = key.split("_")[0]
    target = targets.get(ds, "?")
    if not f.exists():
        print(f"{label:<30} {'N/A':>7} {'N/A':>7} {'0':>5} {target:>6}")
        continue
    s = json.load(open(f, encoding="utf-8"))["summary"]
    print(f"{label:<30} {s['avg_em']:>7.4f} {s['avg_f1']:>7.4f} {s['num_samples']:>5} {target:>6}")
