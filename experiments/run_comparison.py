"""
对比实验主脚本

使用方法：
python experiments/run_comparison.py \
    --dataset hotpotqa \
    --methods gers,standard_cot,cot_sc,tot,zero_shot \
    --model qwen3-8b \
    --num_samples 500 \
    --output_dir experiments/results/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import Metrics
from src.utils.config import Config


def load_dataset(dataset_name: str, split: str = "test", num_samples: int = None):
    """加载基准数据集"""
    from datasets import load_dataset

    if dataset_name == "hotpotqa":
        ds = load_dataset("hotpot_qa", "distractor", split=f"{split}")
        samples = []
        for item in ds:
            question = item["question"]
            context = " ".join([" ".join(ctx) for ctx in item["context"]["sentences"]])
            answer = item["answer"]
            samples.append({"question": question, "context": context, "answer": answer})
    
    elif dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        samples = []
        for item in ds:
            question = item["question"]
            answer = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else item["answer"]
            samples.append({"question": question, "context": "", "answer": answer})
    
    elif dataset_name == "clutrr":
        # CLUTRR需要从GitHub下载
        samples = []  # TODO: 实现CLUTRR数据加载
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if num_samples and num_samples < len(samples):
        samples = samples[:num_samples]

    return samples


def create_model(model_name: str):
    """创建LLM模型实例"""
    from src.models import QwenModel

    model_map = {
        "qwen3-8b": "Qwen/Qwen3-8B",
        "qwen3-70b": "Qwen/Qwen3-70B-Instruct-GPTQ-Int4",
    }

    actual_name = model_map.get(model_name, model_name)
    return QwenModel(model_name=actual_name, load_method="transformers")


def create_method(method_name: str, model):
    """创建推理方法实例"""
    from src.chain_generation import GraphGuidedGenerator
    from src.baselines import StandardCoT, CoTSC, TreeOfThoughts, ZeroShot

    methods = {
        "gers": lambda: GraphGuidedGenerator(model=model),
        "standard_cot": lambda: StandardCoT(model=model),
        "cot_sc": lambda: CoTSC(model=model, num_samples=5),
        "tot": lambda: TreeOfThoughts(model=model),
        "zero_shot": lambda: ZeroShot(model=model),
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")

    return methods[method_name]()


def run_comparison(args):
    """运行对比实验"""
    print(f"=== 对比实验 ===")
    print(f"数据集: {args.dataset}")
    print(f"方法: {args.methods}")
    print(f"模型: {args.model}")
    print(f"样本数: {args.num_samples}")

    # 加载数据
    samples = load_dataset(args.dataset, num_samples=args.num_samples)
    print(f"已加载 {len(samples)} 个样本")

    # 创建模型
    model = create_model(args.model)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行各方法
    for method_name in args.methods.split(","):
        print(f"\n--- 运行方法: {method_name} ---")
        method = create_method(method_name, model)

        results = []
        total_time = 0

        for i, sample in enumerate(samples):
            start_time = time.time()

            result = method.reason(
                question=sample["question"],
                context=sample.get("context", "")
            )

            latency = time.time() - start_time
            total_time += latency

            # 计算指标
            metrics = Metrics.compute_all(
                prediction=result["answer"],
                reference=sample["answer"],
                token_count=model.count_tokens(result.get("reasoning_text", "")),
                latency=latency,
            )

            results.append({
                "sample_id": i,
                "question": sample["question"],
                "prediction": result["answer"],
                "reference": sample["answer"],
                "metrics": metrics,
                "method": method_name,
            })

            if (i + 1) % 50 == 0:
                print(f"  已完成 {i+1}/{len(samples)} 样本")

        # 计算汇总指标
        avg_em = sum(r["metrics"]["em"] for r in results) / len(results)
        avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results)
        avg_latency = total_time / len(results)

        summary = {
            "method": method_name,
            "dataset": args.dataset,
            "num_samples": len(results),
            "avg_em": round(avg_em, 4),
            "avg_f1": round(avg_f1, 4),
            "avg_latency": round(avg_latency, 4),
            "total_time": round(total_time, 2),
        }

        print(f"\n  结果汇总:")
        print(f"    EM: {avg_em:.4f}")
        print(f"    F1: {avg_f1:.4f}")
        print(f"    平均耗时: {avg_latency:.4f}s")

        # 保存结果
        result_path = output_dir / f"{args.dataset}_{method_name}_results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

        print(f"  结果已保存至: {result_path}")

    print("\n=== 对比实验完成 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比实验")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                       choices=["hotpotqa", "gsm8k", "clutrr"])
    parser.add_argument("--methods", type=str, default="gers,standard_cot,cot_sc,tot,zero_shot")
    parser.add_argument("--model", type=str, default="qwen3-8b")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="experiments/results/")

    args = parser.parse_args()
    run_comparison(args)