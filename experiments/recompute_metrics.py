"""离线重算所有实验结果的 EM（使用修复后的 metrics，不重跑模型）

用法: python experiments/recompute_metrics.py
"""
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.metrics import Metrics

RESULTS_DIR = ROOT / "experiments" / "results"
ABLATION_DIR = RESULTS_DIR / "ablation"


def recompute_file(filepath: Path, dataset: str):
    """对单个结果文件重新计算 EM，返回新旧对比"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        return None

    old_em_sum = 0.0
    new_em_sum = 0.0
    n = len(results)
    changed = 0

    for item in results:
        pred = item.get("prediction", "")
        ref = item.get("reference", "")
        old_em = item.get("metrics", {}).get("em", 0.0)
        new_em = Metrics.exact_match(pred, ref, dataset=dataset)

        old_em_sum += old_em
        new_em_sum += new_em
        if abs(old_em - new_em) > 1e-6:
            changed += 1
            # 更新保存
            item["metrics"]["em"] = new_em
            item["metrics"]["em_old"] = old_em

    old_avg = old_em_sum / n if n else 0
    new_avg = new_em_sum / n if n else 0

    # 更新 summary
    if "summary" in data:
        data["summary"]["avg_em"] = new_avg
        data["summary"]["avg_em_old"] = old_avg

    # 写回文件
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "file": filepath.name,
        "dataset": dataset,
        "n": n,
        "old_em": round(old_avg, 4),
        "new_em": round(new_avg, 4),
        "delta": round(new_avg - old_avg, 4),
        "changed": changed,
    }


def main():
    print("=" * 80)
    print("离线重算所有实验结果 EM（修复双向子串匹配 bug）")
    print("=" * 80)

    # 主实验结果文件
    main_files = [
        ("gsm8k", "gsm8k_zero_shot_results.json"),
        ("gsm8k", "gsm8k_standard_cot_results.json"),
        ("gsm8k", "gsm8k_cot_sc_results.json"),
        ("gsm8k", "gsm8k_tot_results.json"),
        ("gsm8k", "gsm8k_gers_results.json"),
        ("gsm8k", "gsm8k_gers_adaptive_results.json"),
        ("hotpotqa", "hotpotqa_zero_shot_results.json"),
        ("hotpotqa", "hotpotqa_standard_cot_results.json"),
        ("hotpotqa", "hotpotqa_cot_sc_results.json"),
        ("hotpotqa", "hotpotqa_tot_results.json"),
        ("hotpotqa", "hotpotqa_gers_results.json"),
        ("hotpotqa", "hotpotqa_gers_adaptive_results.json"),
        ("hotpotqa", "hotpotqa_modegraph_results.json"),
        ("hotpotqa", "hotpotqa_gers_nli_results.json"),
        ("hotpotqa", "hotpotqa_gers_feedback_results.json"),
        ("clutrr", "clutrr_zero_shot_results.json"),
        ("clutrr", "clutrr_standard_cot_results.json"),
        ("clutrr", "clutrr_cot_sc_results.json"),
        ("clutrr", "clutrr_gers_results.json"),
    ]

    print("\n--- 主实验结果 ---")
    print(f"{'文件':<45} {'数据集':<10} {'N':<5} {'旧EM':<8} {'新EM':<8} {'Δ':<8} {'变化条数'}")
    print("-" * 100)

    all_results = []
    for dataset, fname in main_files:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            print(f"  [跳过] {fname} 不存在")
            continue
        r = recompute_file(fpath, dataset)
        if r:
            all_results.append(r)
            print(f"  {r['file']:<43} {r['dataset']:<10} {r['n']:<5} {r['old_em']:<8} {r['new_em']:<8} {r['delta']:<+8} {r['changed']}")

    # 消融实验
    print("\n--- 消融实验结果 ---")
    print(f"{'文件':<55} {'数据集':<10} {'N':<5} {'旧EM':<8} {'新EM':<8} {'Δ':<8}")
    print("-" * 100)

    ablation_files = sorted(ABLATION_DIR.glob("*_results.json"))
    for fpath in ablation_files:
        fname = fpath.name
        if fname.startswith("gsm8k"):
            dataset = "gsm8k"
        elif fname.startswith("hotpotqa"):
            dataset = "hotpotqa"
        else:
            dataset = "unknown"
        r = recompute_file(fpath, dataset)
        if r:
            all_results.append(r)
            print(f"  {r['file']:<53} {r['dataset']:<10} {r['n']:<5} {r['old_em']:<8} {r['new_em']:<8} {r['delta']:<+8}")

    # 汇总
    print("\n" + "=" * 80)
    print("汇总：EM 变化超过 1pt 的文件")
    print("=" * 80)
    big_changes = [r for r in all_results if abs(r["delta"]) >= 0.01]
    if big_changes:
        for r in sorted(big_changes, key=lambda x: x["delta"]):
            print(f"  {r['file']:<50} {r['old_em']:.3f} → {r['new_em']:.3f}  ({r['delta']:+.3f})")
    else:
        print("  无显著变化")

    print(f"\n共处理 {len(all_results)} 个文件")


if __name__ == "__main__":
    main()
