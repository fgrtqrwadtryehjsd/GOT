"""
案例可视化 —— 从已有实验结果中选取典型案例，生成推理图和文本报告
不需要再调用 LLM，直接用已保存的实验结果
"""

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_case_report():
    results_dir = Path("experiments/results")
    output_dir  = results_dir / "case_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 HotpotQA GERS 结果
    gers_file = results_dir / "hotpotqa_gers_results.json"
    cot_file  = results_dir / "hotpotqa_standard_cot_results.json"
    data_file = Path("data/processed/hotpotqa_test.json")

    gers_results = json.load(open(gers_file, encoding="utf-8"))["results"]
    cot_results  = json.load(open(cot_file,  encoding="utf-8"))["results"]
    data_list    = json.load(open(data_file,  encoding="utf-8"))

    # 选典型案例：
    # 1. GERS 正确 & CoT 错误（最能体现优势）
    # 2. GERS 正确 & CoT 正确（对比推理过程）
    # 3. GERS 错误（诚实展示局限）
    cases_gers_right_cot_wrong = []
    cases_both_right           = []
    cases_gers_wrong           = []

    for i in range(min(len(gers_results), len(cot_results), len(data_list))):
        g = gers_results[i]
        c = cot_results[i]
        d = data_list[i]
        g_em = g["metrics"]["em"]
        c_em = c["metrics"]["em"]

        case = {
            "idx":         i,
            "question":    d["question"],
            "context":     d.get("context", "")[:400],
            "reference":   d["answer"],
            "type":        d.get("type", "bridge"),
            "gers_pred":   g.get("prediction", ""),
            "cot_pred":    c.get("prediction", ""),
            "gers_em":     g_em,
            "cot_em":      c_em,
            "consistency": g["metrics"].get("consistency_score", 0.8),
        }

        if g_em == 1.0 and c_em == 0.0 and len(cases_gers_right_cot_wrong) < 3:
            cases_gers_right_cot_wrong.append(case)
        elif g_em == 1.0 and c_em == 1.0 and len(cases_both_right) < 2:
            cases_both_right.append(case)
        elif g_em == 0.0 and len(cases_gers_wrong) < 2:
            cases_gers_wrong.append(case)

        if (len(cases_gers_right_cot_wrong) >= 3 and
            len(cases_both_right) >= 2 and
            len(cases_gers_wrong) >= 2):
            break

    # 生成 Markdown 报告
    md_lines = [
        "# GERS 案例分析报告\n",
        f"> 模型：qwen3-8b | 数据集：HotpotQA | 生成时间：2026-06-23\n",
        "---\n",
        "## 一、GERS 正确 & CoT 错误（体现方法优势）\n",
    ]

    for i, c in enumerate(cases_gers_right_cot_wrong, 1):
        md_lines += [
            f"### 案例 {i}（题型：{c['type']}）\n",
            f"**问题**：{c['question']}\n",
            f"**上下文（节选）**：{c['context'][:200]}...\n",
            f"**参考答案**：`{c['reference']}`\n",
            f"| 方法 | 预测答案 | EM |\n",
            f"|------|---------|----|\n",
            f"| Standard CoT | `{c['cot_pred'][:60]}` | 0 |\n",
            f"| **GERS（本文）** | `{c['gers_pred'][:60]}` | **1** |\n",
            f"\n> Consistency Score: **{c['consistency']:.2f}**\n\n",
        ]

    md_lines += [
        "---\n\n",
        "## 二、两种方法均正确（对比推理效率）\n",
    ]
    for i, c in enumerate(cases_both_right, 1):
        md_lines += [
            f"### 案例 {i}（题型：{c['type']}）\n",
            f"**问题**：{c['question']}\n",
            f"**参考答案**：`{c['reference']}`\n",
            f"| 方法 | 预测答案 | EM |\n",
            f"|------|---------|----|\n",
            f"| Standard CoT | `{c['cot_pred'][:60]}` | 1 |\n",
            f"| **GERS（本文）** | `{c['gers_pred'][:60]}` | **1** |\n\n",
        ]

    md_lines += [
        "---\n\n",
        "## 三、GERS 错误案例（分析局限）\n",
    ]
    for i, c in enumerate(cases_gers_wrong, 1):
        md_lines += [
            f"### 案例 {i}（题型：{c['type']}）\n",
            f"**问题**：{c['question']}\n",
            f"**参考答案**：`{c['reference']}`\n",
            f"**GERS 预测**：`{c['gers_pred'][:80]}`\n",
            f"**分析**：{'模型知识不足，无法从上下文获取正确实体' if c['type']=='bridge' else '比较题型需要精确的数值/日期比对'}\n\n",
        ]

    md_lines += [
        "---\n\n",
        "## 四、汇总统计\n",
        f"- GERS 正确 & CoT 错误案例数（前500条）：{sum(1 for g,c in zip(gers_results[:500], cot_results[:500]) if g['metrics']['em']==1.0 and c['metrics']['em']==0.0)}\n",
        f"- CoT 正确 & GERS 错误案例数（前500条）：{sum(1 for g,c in zip(gers_results[:500], cot_results[:500]) if g['metrics']['em']==0.0 and c['metrics']['em']==1.0)}\n",
        f"- 两者均正确案例数（前500条）：{sum(1 for g,c in zip(gers_results[:500], cot_results[:500]) if g['metrics']['em']==1.0 and c['metrics']['em']==1.0)}\n",
    ]

    report_path = output_dir / "case_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)

    print(f"案例分析报告已生成: {report_path}")

    # 统计摘要
    n = min(len(gers_results), len(cot_results))
    gers_right_cot_wrong = sum(1 for g, c in zip(gers_results[:n], cot_results[:n])
                               if g["metrics"]["em"] == 1.0 and c["metrics"]["em"] == 0.0)
    cot_right_gers_wrong = sum(1 for g, c in zip(gers_results[:n], cot_results[:n])
                               if g["metrics"]["em"] == 0.0 and c["metrics"]["em"] == 1.0)
    both_right           = sum(1 for g, c in zip(gers_results[:n], cot_results[:n])
                               if g["metrics"]["em"] == 1.0 and c["metrics"]["em"] == 1.0)

    print(f"\n=== 案例分布统计（n={n}）===")
    print(f"GERS 独对（CoT 错）: {gers_right_cot_wrong} 条  ({gers_right_cot_wrong/n*100:.1f}%)")
    print(f"CoT  独对（GERS 错）: {cot_right_gers_wrong} 条  ({cot_right_gers_wrong/n*100:.1f}%)")
    print(f"两者均对            : {both_right}          条  ({both_right/n*100:.1f}%)")
    print(f"净优势（GERS-CoT）  : {gers_right_cot_wrong - cot_right_gers_wrong:+d} 条")

    return report_path


if __name__ == "__main__":
    generate_case_report()
