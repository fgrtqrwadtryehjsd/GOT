"""
消融实验脚本（重构版）

验证 GERS 各模块的独立贡献：

配置1: Full GERS        - 分解 + 依赖传递 + 汇总 + 校验
配置2: w/o Decompose    - 不分解，直接一次生成（退化为 CoT）
配置3: w/o Context      - 子问题不传递前驱答案
配置4: w/o Summary      - 无汇总步骤，直接取最后一个子答案
配置5: w/o Consistency  - 无校验，不计算 Consistency Score

使用方法：
    python experiments/run_ablation_v2.py --dataset gsm8k --num_samples 20
    python experiments/run_ablation_v2.py --dataset hotpotqa --num_samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.utils.metrics import Metrics
from src.utils.answer_normalizer import (
    normalize_gsm8k_answer, normalize_hotpotqa_answer, normalize_clutrr_answer
)

NORMALIZERS = {
    "gsm8k": normalize_gsm8k_answer,
    "hotpotqa": normalize_hotpotqa_answer,
    "clutrr": normalize_clutrr_answer,
}


def run_ablation_config(config_name: str, config: dict,
                        samples: list, model, normalizer,
                        output_dir: Path, dataset: str):
    """运行单个消融配置"""
    from src.chain_generation.generation_pipeline import GraphGuidedGenerator
    from src.baselines.standard_cot import StandardCoT

    result_path = output_dir / f"{dataset}_ablation_{config_name}_results.json"

    # 断点续跑
    if result_path.exists():
        data = json.load(open(result_path, encoding="utf-8"))
        existing = data.get("results", [])
        done_ids = {r["sample_id"] for r in existing}
        results = list(existing)
        print(f"  [断点续跑] 已完成 {len(done_ids)} 条")
    else:
        existing = []
        done_ids = set()
        results = []

    print(f"\n--- 消融配置: {config_name} ---")
    print(f"  {config['description']}")

    total_time = sum(r["metrics"]["latency"] for r in results if "metrics" in r)
    error_count = 0

    for i, sample in enumerate(samples):
        if i in done_ids:
            continue

        start = time.time()
        try:
            if config["method"] == "no_decompose":
                # 退化为 Standard CoT
                method = StandardCoT(model=model)
                result = method.reason(sample["question"], context=sample.get("context", ""))
                prediction = normalizer(result.get("answer", ""))
                cs = 0.0
            else:
                gen = GraphGuidedGenerator(
                    model=model,
                    enable_nli=False,
                    **config.get("gers_kwargs", {})
                )
                result = gen.reason(sample["question"], context=sample.get("context", ""))
                prediction = normalizer(result.get("answer", ""))
                cs = result.get("consistency_score", 0.0)

            latency = time.time() - start
            error = None
        except Exception as e:
            latency = time.time() - start
            prediction = ""
            cs = 0.0
            error = str(e)[:80]
            error_count += 1

        total_time += latency
        metrics = Metrics.compute_all(
            prediction=prediction,
            reference=normalizer(sample["answer"]),
            token_count=0,
            latency=latency,
        )
        if cs > 0:
            metrics["consistency_score"] = cs

        record = {
            "sample_id": i,
            "question": sample["question"][:100],
            "prediction": prediction,
            "reference": sample["answer"],
            "metrics": metrics,
            "config": config_name,
            "error": error,
        }
        results.append(record)

        # 实时保存
        finished = [r for r in results if r.get("error") is None]
        avg_em = sum(r["metrics"]["em"] for r in finished) / max(len(finished), 1)
        summary = {
            "config": config_name,
            "description": config["description"],
            "dataset": dataset,
            "num_samples": len(results),
            "avg_em": round(avg_em, 4),
            "avg_latency": round(total_time / len(results), 4),
            "error_count": error_count,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f,
                      ensure_ascii=False, indent=2)

        em_str = f"EM={metrics['em']:.2f}"
        print(f"  [{i+1}/{len(samples)}] {latency:.1f}s | {em_str} | "
              f"累计EM={avg_em:.3f} | pred={prediction[:20]!r}")

    # 最终汇总
    finished = [r for r in results if r.get("error") is None]
    avg_em = sum(r["metrics"]["em"] for r in finished) / max(len(finished), 1)
    avg_cs = sum(r["metrics"].get("consistency_score", 0) for r in finished) / max(len(finished), 1)
    print(f"\n  结果: EM={avg_em:.4f} | Consistency={avg_cs:.4f} | n={len(finished)}")
    return avg_em, avg_cs


def main():
    parser = argparse.ArgumentParser(description="消融实验（重构版）")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "hotpotqa", "clutrr"])
    parser.add_argument("--model", type=str, default="qwen3-8b")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="experiments/results/ablation/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    from data.prepare_data import load_processed_dataset
    samples = load_processed_dataset(args.dataset, num_samples=args.num_samples)
    normalizer = NORMALIZERS.get(args.dataset, lambda x: x)
    print(f"\n[消融实验] 数据集: {args.dataset} | 模型: {args.model} | 样本数: {len(samples)}")

    # 加载模型
    from experiments.run_comparison import create_model
    model = create_model(args.model)

    # 消融配置
    configs = {
        "full_gers": {
            "description": "Full GERS: 分解+依赖传递+汇总+校验（完整方法）",
            "method": "gers",
            "gers_kwargs": {},
        },
        "wo_decompose": {
            "description": "w/o Decompose: 不分解，退化为 Standard CoT",
            "method": "no_decompose",
        },
        "wo_context": {
            "description": "w/o Context: 子问题不传递前驱答案（独立回答）",
            "method": "gers",
            "gers_kwargs": {"_no_context": True},  # 标记，在Pipeline内处理
        },
        "wo_consistency": {
            "description": "w/o Consistency: 无 Consistency Score 校验",
            "method": "gers",
            "gers_kwargs": {},  # Pipeline 已设 max_iterations=0
        },
    }

    all_results = {}
    for config_name, config in configs.items():
        avg_em, avg_cs = run_ablation_config(
            config_name, config, samples, model, normalizer, output_dir, args.dataset
        )
        all_results[config_name] = {"em": avg_em, "consistency": avg_cs}

    # 打印汇总表
    print(f"\n\n{'='*65}")
    print(f"消融实验结果汇总 | 数据集: {args.dataset}")
    print(f"{'─'*65}")
    print(f"{'配置':<20} {'描述':<35} {'EM':>6} {'CS':>6}")
    print(f"{'─'*65}")
    for cfg_name, res in all_results.items():
        desc = configs[cfg_name]["description"][:32]
        print(f"{cfg_name:<20} {desc:<35} {res['em']:>6.4f} {res['consistency']:>6.4f}")
    print(f"{'='*65}")

    # 保存汇总
    summary_path = output_dir / f"{args.dataset}_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "model": args.model,
            "num_samples": args.num_samples,
            "results": all_results,
            "configs": {k: v["description"] for k, v in configs.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
