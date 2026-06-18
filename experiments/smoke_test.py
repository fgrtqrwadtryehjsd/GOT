"""
快速冒烟测试 —— 验证 DashScope 模型接入与 GERS Pipeline

测试内容：
1. DashScopeModel 单次调用
2. GERS 完整推理流水线（1条 CLUTRR 样本）
3. 打印推理图摘要与一致性得分

使用方法：
    python experiments/smoke_test.py
    python experiments/smoke_test.py --model qwen-max
    python experiments/smoke_test.py --dataset gsm8k
"""

import argparse
import sys
import time
from pathlib import Path

# Windows 终端强制 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def test_model_basic(model_name: str):
    """测试模型基本调用"""
    from src.models import DashScopeModel

    print(f"\n{'='*50}")
    print(f"[1] 基础调用测试: {model_name}")
    print(f"{'='*50}")

    model = DashScopeModel(model_name=model_name)
    prompt = "请用一句话介绍图神经网络。"

    start = time.time()
    response = model.generate(prompt, max_tokens=200, temperature=0.3)
    latency = time.time() - start

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"耗时: {latency:.2f}s")
    return model


def test_gers_pipeline(model, dataset: str = "clutrr"):
    """测试 GERS 完整推理流水线"""
    from src.chain_generation import GraphGuidedGenerator
    from data.prepare_data import load_processed_dataset

    print(f"\n{'='*50}")
    print(f"[2] GERS Pipeline 测试（数据集: {dataset}）")
    print(f"{'='*50}")

    # 加载1条测试样本
    samples = load_processed_dataset(dataset, num_samples=1)
    sample = samples[0]

    print(f"问题: {sample['question']}")
    print(f"参考答案: {sample['answer']}")
    if sample.get("context"):
        print(f"上下文: {sample['context'][:100]}...")

    gen = GraphGuidedGenerator(
        model=model,
        constraint_mode="soft",
        max_iterations=2,
        consistency_threshold=0.6,
        enable_nli=False,
    )

    print("\n推理中...")
    start = time.time()
    result = gen.reason(sample["question"], context=sample.get("context", ""))
    latency = time.time() - start

    print(f"\n{'─'*40}")
    print(f"预测答案: {result['answer']}")
    print(f"推理文本:\n{result['reasoning_text'][:300]}...")
    print(f"推理图摘要: {result['graph'].summary()}")
    print(f"一致性得分: {result['consistency_score']:.4f}")
    print(f"修正迭代: {result['iterations']} 次")
    print(f"总耗时: {latency:.2f}s")

    # 简单评估
    pred = result["answer"].strip().lower()
    ref = sample["answer"].strip().lower()
    is_correct = pred == ref or ref in pred or pred in ref
    print(f"\n答案评估: {'[正确]' if is_correct else '[不完全匹配]'}")

    return result


def test_baselines(model, dataset: str = "clutrr"):
    """对比 GERS vs Standard CoT"""
    from src.baselines import StandardCoT
    from data.prepare_data import load_processed_dataset

    print(f"\n{'='*50}")
    print(f"[3] 基线对比（Standard CoT vs GERS）")
    print(f"{'='*50}")

    samples = load_processed_dataset(dataset, num_samples=3)

    cot = StandardCoT(model=model)

    for i, sample in enumerate(samples):
        print(f"\n样本 {i+1}: {sample['question']}")
        print(f"参考答案: {sample['answer']}")

        start = time.time()
        result = cot.reason(sample["question"], context=sample.get("context", ""))
        latency = time.time() - start
        print(f"CoT 答案: {result['answer']} ({latency:.2f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GERS 冒烟测试")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="模型名称（默认 qwen-plus）")
    parser.add_argument("--dataset", type=str, default="clutrr",
                        choices=["clutrr", "gsm8k", "hotpotqa"])
    parser.add_argument("--skip_pipeline", action="store_true",
                        help="只测试模型调用，跳过 Pipeline 测试")
    args = parser.parse_args()

    try:
        # 1. 基础调用
        model = test_model_basic(args.model)

        if not args.skip_pipeline:
            # 2. GERS Pipeline
            test_gers_pipeline(model, args.dataset)

            # 3. 基线对比
            test_baselines(model, args.dataset)

        print(f"\n{'='*50}")
        print("冒烟测试完成！")
        print(f"{'='*50}")

    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
