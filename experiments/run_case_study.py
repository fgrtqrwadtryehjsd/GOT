"""
案例分析脚本 —— 可视化推理过程

对选定的典型案例，绘制推理图并生成可视化分析
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import GraphVisualizer


def run_case_study(args):
    """运行案例分析"""
    from experiments.run_comparison import load_dataset, create_model, create_method
    from src.chain_generation import GraphGuidedGenerator

    samples = load_dataset(args.dataset, num_samples=args.num_cases * 3)
    model = create_model(args.model)

    output_dir = Path(args.output_dir) / "case_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = GraphVisualizer(output_dir=str(output_dir))
    generator = GraphGuidedGenerator(model=model)

    # 选择典型案例（成功+失败各选几个）
    success_cases = []
    failure_cases = []

    for i, sample in enumerate(samples[:args.num_cases * 3]):
        result = generator.reason(sample["question"], sample.get("context", ""))

        case = {
            "sample_id": i,
            "question": sample["question"],
            "reference": sample["answer"],
            "prediction": result["answer"],
            "correct": result["answer"].strip().lower() == sample["answer"].strip().lower(),
            "consistency_score": result.get("consistency_score", 0),
            "iterations": result.get("iterations", 0),
            "graph_summary": result["graph"].summary() if result.get("graph") else {},
        }

        if case["correct"] and len(success_cases) < args.num_cases // 2 + 1:
            success_cases.append(case)
            # 绘制推理图
            if result.get("graph"):
                fig_path = str(output_dir / f"case_success_{len(success_cases)}.png")
                try:
                    visualizer.plot_networkx(
                        result["graph"],
                        title=f"成功案例 #{len(success_cases)}",
                        save_path=fig_path
                    )
                except Exception as e:
                    print(f"可视化失败: {e}")

        elif not case["correct"] and len(failure_cases) < args.num_cases // 2 + 1:
            failure_cases.append(case)
            if result.get("graph"):
                fig_path = str(output_dir / f"case_failure_{len(failure_cases)}.png")
                try:
                    visualizer.plot_networkx(
                        result["graph"],
                        title=f"失败案例 #{len(failure_cases)}",
                        save_path=fig_path
                    )
                except Exception as e:
                    print(f"可视化失败: {e}")

        if len(success_cases) + len(failure_cases) >= args.num_cases:
            break

    # 保存案例报告
    report = {
        "success_cases": success_cases,
        "failure_cases": failure_cases,
        "summary": {
            "total_success": len(success_cases),
            "total_failure": len(failure_cases),
            "avg_consistency_success": (
                sum(c["consistency_score"] for c in success_cases) / len(success_cases)
                if success_cases else 0
            ),
            "avg_consistency_failure": (
                sum(c["consistency_score"] for c in failure_cases) / len(failure_cases)
                if failure_cases else 0
            ),
        }
    }

    report_path = output_dir / "case_study_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n案例分析报告已保存至: {report_path}")
    print(f"成功案例: {len(success_cases)}, 失败案例: {len(failure_cases)}")
    if success_cases:
        print(f"成功案例平均一致性: {report['summary']['avg_consistency_success']:.4f}")
    if failure_cases:
        print(f"失败案例平均一致性: {report['summary']['avg_consistency_failure']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="案例分析")
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--model", type=str, default="qwen3-8b")
    parser.add_argument("--num_cases", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="experiments/results/")

    args = parser.parse_args()
    run_case_study(args)