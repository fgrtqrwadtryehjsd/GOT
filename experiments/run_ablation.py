"""
消融实验脚本

5种配置对比：
1. Full (GERS)         - 图表征 + 约束生成 + 一致性校验
2. w/o Graph           - 移除图表征（退化为CoT+校验）
3. w/o Constraint      - 移除约束生成（自由生成+校验）
4. w/o Check           - 移除校验（图约束生成无闭环）
5. w/o Feedback        - 校验但不回溯修正

使用方法：
python experiments/run_ablation.py \
    --dataset hotpotqa \
    --model qwen3-8b \
    --num_samples 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import Metrics


def run_ablation(args):
    """运行消融实验"""
    print(f"=== 消融实验 ===")

    # 加载数据集
    from experiments.run_comparison import load_dataset, create_model
    samples = load_dataset(args.dataset, num_samples=args.num_samples)
    model = create_model(args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 消融配置
    configs = {
        "full": {"enable_graph": True, "enable_constraint": True, 
                 "enable_check": True, "enable_feedback": True},
        "wo_graph": {"enable_graph": False, "enable_constraint": True, 
                     "enable_check": True, "enable_feedback": True},
        "wo_constraint": {"enable_graph": True, "enable_constraint": False, 
                        "enable_check": True, "enable_feedback": True},
        "wo_check": {"enable_graph": True, "enable_constraint": True, 
                    "enable_check": False, "enable_feedback": False},
        "wo_feedback": {"enable_graph": True, "enable_constraint": True, 
                       "enable_check": True, "enable_feedback": False},
    }

    all_summaries = {}

    for config_name, config in configs.items():
        print(f"\n--- 配置: {config_name} ---")
        print(f"  图表征: {config['enable_graph']}")
        print(f"  约束生成: {config['enable_constraint']}")
        print(f"  一致性校验: {config['enable_check']}")
        print(f"  闭环修正: {config['enable_feedback']}")

        from src.chain_generation import GraphGuidedGenerator

        generator = GraphGuidedGenerator(
            model=model,
            constraint_mode="soft" if config["enable_constraint"] else None,
            consistency_threshold=0.7 if config["enable_check"] else 0.0,
            max_iterations=3 if config["enable_feedback"] else 0,
            enable_nli=config["enable_check"],
        )

        results = []
        for i, sample in enumerate(samples):
            if not config["enable_graph"]:
                # 退化为Standard CoT
                from src.baselines import StandardCoT
                cot = StandardCoT(model=model)
                result = cot.reason(sample["question"], sample.get("context", ""))
            else:
                result = generator.reason(sample["question"], sample.get("context", ""))

            metrics = Metrics.compute_all(
                prediction=result["answer"],
                reference=sample["answer"],
            )
            results.append(metrics)

            if (i + 1) % 50 == 0:
                print(f"  已完成 {i+1}/{len(samples)}")

        # 汇总
        avg_em = sum(r["em"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        avg_consistency = 0.0
        if config["enable_check"] and results:
            consistency_scores = [r.get("consistency_score", 0) for r in results 
                                if "consistency_score" in r]
            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)

        summary = {
            "config": config_name,
            "avg_em": round(avg_em, 4),
            "avg_f1": round(avg_f1, 4),
            "avg_consistency": round(avg_consistency, 4),
        }
        all_summaries[config_name] = summary

        print(f"  EM: {avg_em:.4f}, F1: {avg_f1:.4f}, Consistency: {avg_consistency:.4f}")

    # 保存汇总
    summary_path = output_dir / f"{args.dataset}_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\n消融实验汇总已保存至: {summary_path}")

    # 打印对比表
    print("\n" + "="*70)
    print(f"{'配置':<20} {'EM':>10} {'F1':>10} {'一致性':>10}")
    print("-"*70)
    for name, summary in all_summaries.items():
        print(f"{name:<20} {summary['avg_em']:>10.4f} {summary['avg_f1']:>10.4f} "
              f"{summary['avg_consistency']:>10.4f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--model", type=str, default="qwen3-8b")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="experiments/results/")

    args = parser.parse_args()
    run_ablation(args)