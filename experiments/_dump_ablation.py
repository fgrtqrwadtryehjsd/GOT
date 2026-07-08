"""读取消融实验 summary。"""
import json
import os

ab_dir = r"d:\code\GOT\experiments\results\ablation"
for f in sorted(os.listdir(ab_dir)):
    if "summary" not in f:
        continue
    path = os.path.join(ab_dir, f)
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    print(f"--- {f} ---")
    print(json.dumps(d, indent=2, ensure_ascii=False)[:2000])
    print()
