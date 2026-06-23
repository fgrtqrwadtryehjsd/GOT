"""
HotpotQA 分类型性能分析
分析 bridge（多跳桥接）和 comparison（对比）两种题型上各方法的表现
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_FILE = "data/processed/hotpotqa_test.json"
RESULTS_DIR = Path("experiments/results")


def load_hotpotqa():
    """加载 HotpotQA 数据，返回列表（按顺序）"""
    return json.load(open(DATA_FILE, encoding="utf-8"))


def analyze_method(method: str, data_list: list):
    """分析某方法在不同题型上的 EM（直接读已存储的 em 值）"""
    result_file = RESULTS_DIR / f"hotpotqa_{method}_results.json"
    if not result_file.exists():
        return None

    raw = json.load(open(result_file, encoding="utf-8"))
    results = raw.get("results", [])

    bridge_em, bridge_n = [], 0
    comparison_em, comparison_n = [], 0

    for i, r in enumerate(results):
        if i >= len(data_list):
            break
        item    = data_list[i]
        qtype   = item.get("type", "bridge")
        em      = r.get("metrics", {}).get("em", 0.0)

        if qtype == "comparison":
            comparison_em.append(em)
            comparison_n += 1
        else:
            bridge_em.append(em)
            bridge_n += 1

    bridge_avg     = sum(bridge_em)    / len(bridge_em)    if bridge_em    else 0.0
    comparison_avg = sum(comparison_em) / len(comparison_em) if comparison_em else 0.0
    total          = bridge_n + comparison_n
    total_avg      = (sum(bridge_em) + sum(comparison_em)) / total if total else 0.0

    return {
        "bridge":     {"em": bridge_avg,    "n": bridge_n},
        "comparison": {"em": comparison_avg, "n": comparison_n},
        "total":      {"em": total_avg,      "n": total},
    }


def main():
    data_list = load_hotpotqa()

    # 统计题型分布
    type_counter = {}
    for item in data_list:
        t = item.get("type", "bridge")
        type_counter[t] = type_counter.get(t, 0) + 1
    print(f"HotpotQA 题型分布（共{len(data_list)}条）: bridge={type_counter.get('bridge',0)}, comparison={type_counter.get('comparison',0)}\n")

    methods = ["zero_shot", "standard_cot", "gers"]
    method_labels = {
        "zero_shot":    "Zero-Shot",
        "standard_cot": "Standard CoT",
        "gers":         "GERS（本文）",
    }

    print(f"{'方法':<18} {'bridge EM':>10} {'bridge N':>8} {'comparison EM':>14} {'comparison N':>12} {'Total EM':>10}")
    print("-" * 78)

    rows = []
    for m in methods:
        stats = analyze_method(m, data_list)
        if stats is None:
            print(f"{method_labels[m]:<18}  (结果文件不存在)")
            continue
        b, c, t = stats["bridge"], stats["comparison"], stats["total"]
        print(f"{method_labels[m]:<18} {b['em']:>10.4f} {b['n']:>8d} {c['em']:>14.4f} {c['n']:>12d} {t['em']:>10.4f}")
        rows.append({"method": m, **{f"{k}_{kk}": v for k, s in stats.items() for kk, v in s.items()}})

    # 保存
    out_file = RESULTS_DIR / "hotpotqa_type_analysis.json"
    json.dump(rows, open(out_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n分析结果已保存: {out_file}")

    # GERS vs CoT 提升摘要
    gers_stats = analyze_method("gers", data_list)
    cot_stats  = analyze_method("standard_cot", data_list)
    if gers_stats and cot_stats:
        print(f"\nGERS vs Standard CoT 提升:")
        for qtype in ("bridge", "comparison", "total"):
            g = gers_stats[qtype]["em"]
            c = cot_stats[qtype]["em"]
            print(f"  {qtype:<12}: GERS {g:.4f} vs CoT {c:.4f}  ({(g-c)*100:+.1f}%)")


if __name__ == "__main__":
    main()

