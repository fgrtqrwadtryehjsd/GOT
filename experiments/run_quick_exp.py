"""
快速实验脚本 —— 带超时保护、实时进度、断点续跑

特点：
1. 每条样本有超时限制（默认 120s）
2. 每跑完一条立即保存，防止中途中断丢失
3. 支持断点续跑（跳过已完成样本）
4. 实时打印进度

使用方法：
    python experiments/run_quick_exp.py --dataset gsm8k --method gers --num_samples 50
    python experiments/run_quick_exp.py --dataset hotpotqa --method standard_cot --num_samples 50
"""

import argparse
import json
import sys
import time
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.utils.metrics import Metrics
from src.utils.answer_normalizer import (
    normalize_gsm8k_answer, normalize_hotpotqa_answer, normalize_2wikimultihopqa_answer, normalize_musique_answer
)

NORMALIZERS = {
    "gsm8k": normalize_gsm8k_answer,
    "hotpotqa": normalize_hotpotqa_answer,
    "2wikimultihopqa": normalize_2wikimultihopqa_answer,
    "musique": normalize_musique_answer,
    # LongBench 5 个英文 QA 子集：答案格式与 HotpotQA 类似（短实体/短语），复用其归一化
    "longbench_narrativeqa": normalize_hotpotqa_answer,
    "longbench_musique": normalize_hotpotqa_answer,
    "longbench_multifieldqa_en": normalize_hotpotqa_answer,
    "longbench_qasper": normalize_hotpotqa_answer,
    "longbench_2wikimqa": normalize_hotpotqa_answer,
}


def load_results(result_path: Path):
    """加载已有结果（断点续跑）"""
    if result_path.exists():
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", [])
    return []


def save_results(result_path: Path, results: list, summary: dict):
    """保存结果"""
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f,
                  ensure_ascii=False, indent=2)


def run_one_sample(method, sample, timeout_sec=120):
    """运行单条样本，带超时保护"""
    start = time.time()
    try:
        result = method.reason(
            question=sample["question"],
            context=sample.get("context", "")
        )
        latency = time.time() - start
        return result, latency, None
    except Exception as e:
        latency = time.time() - start
        return None, latency, str(e)


def create_method(method_name: str, model, dataset: str = None):
    from src.chain_generation import GraphGuidedGenerator
    from src.baselines import StandardCoT, CoTSC, CoTSCWithGERS, TreeOfThoughts, ZeroShot, MoDeGraphBaseline

    methods = {
        # GERS 核心配置：_no_constraint=True（消融证明约束解码负贡献，已砍掉）
        "gers":             lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=False, adaptive=False, consistency_threshold=0.75, _no_constraint=True, dataset=dataset),
        "gers_adaptive":    lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=False, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset),
        # 图级 Self-Consistency：生成K条DAG，用Consistency Score选优
        "gers_sc":          lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, self_consistency_k=3, dataset=dataset),
        # GERS-SC + 子答案双向交叉验证（方向1创新点：修复CS区分度）
        "gers_sc_cv":       lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, self_consistency_k=3, dataset=dataset, enable_backward_verify=True, enable_llm_match=False),
        # GERS-SC + 方向1(反向验证) + 方向2(置信度加权汇总) 叠加
        "gers_sc_cv2":      lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, self_consistency_k=3, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True),
        # GERS+自适应 + 反向验证（消融：单路DAG+交叉验证，隔离反向验证本身贡献）
        "gers_adaptive_cv": lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False),
        # GERS+自适应 + 方向1 + 方向2 叠加
        "gers_adaptive_cv2":lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True),
        # 证据约束反向验证：crossval 同时要求答案一致且证据落在上下文中
        "gers_grounded":    lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True, enable_evidence_grounding=True),
        # GF-GERS：前向证据落地 + grounding-as-routing（任一子答案未落地→回退free-form CoT-SC）
        "gf_gers":          lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True, enable_evidence_grounding=True, enable_forward_grounding=True, forward_grounding_threshold=0.5),
        # B：CV2 + 逐子问题 BM25 检索（每个子问题只看 top-k 相关段落，减少干扰）
        "gers_cv2_retr":    lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True, enable_subq_retrieval=True, subq_retrieval_k=2),
        # 公平对照：CV2 用全量 context（不截断1500，对齐 baselines）
        # cap=100000 字符(~25k tok)：对 HotpotQA(4.7k字符)无截断、对 MuSiQue(~45k字符)也不截断，
        # 跨数据集统一为"真正全 context"。原 8000 对 HotpotQA 够用但对 MuSiQue 会截断到~2k tok 不公平。
        "gers_cv2_fullctx": lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True, context_char_limit=100000),
        "gers_grounded_soft": lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True, enable_evidence_grounding=True, enable_soft_match=True),
        # 验证驱动局部修复：用 crossval mismatch 定位不可靠子问题，重答该节点及其下游后重新汇总
        "gers_repair":      lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True, enable_verification_repair=True),
        # 软匹配版本：用于验证 token-F1 部分匹配能否改善 CS 校准
        "gers_repair_soft": lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True, enable_verification_repair=True, enable_soft_match=True),
        # P2.3 消融：crossval 权重均匀(uniform) vs 默认下游加权
        "gers_cv2_uniform": lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True, uniform_crossval_weight=True),
        # P2.4 控制：反向验证仅用context(不用最终答案A)
        "gers_cv2_ctxonly": lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=True, consistency_threshold=0.75, _no_constraint=True, dataset=dataset, enable_backward_verify=True, enable_llm_match=False, enable_confidence_weighting=True, backward_anchor_mode="context_only"),
        "gers_nli":         lambda: GraphGuidedGenerator(model=model, max_iterations=1, enable_nli=True, adaptive=False, consistency_threshold=0.75, _no_constraint=True, dataset=dataset),
        "gers_feedback":    lambda: GraphGuidedGenerator(model=model, max_iterations=2, enable_nli=False, adaptive=False, consistency_threshold=0.75, _no_constraint=True, dataset=dataset),
        # CoT 系基线：透传 dataset 给答案提取（公平性，修复空答案/长句问题）
        "standard_cot":     lambda: StandardCoT(model=model, dataset=dataset),
        "cot_sc":           lambda: CoTSC(model=model, num_samples=3, dataset=dataset),
        # CoT-SC + GERS 加权重排：N次CoT采样 → 归一化投票 → GERS质量分加权 → 选优
        "cot_sc_gers":      lambda: CoTSCWithGERS(model=model, num_samples=3, gers_lambda=1.0, enable_gers_rerank=True, dataset=dataset),
        "tot":              lambda: TreeOfThoughts(model=model, max_depth=3, beam_width=2),
        "zero_shot":        lambda: ZeroShot(model=model, dataset=dataset),
        "modegraph":        lambda: MoDeGraphBaseline(model=model, dataset=dataset),
    }
    return methods[method_name]()


def main():
    parser = argparse.ArgumentParser(description="快速实验（带超时保护）")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "hotpotqa", "2wikimultihopqa", "musique",
                                 "longbench_narrativeqa", "longbench_musique",
                                 "longbench_multifieldqa_en", "longbench_qasper",
                                 "longbench_2wikimqa"])
    parser.add_argument("--method", type=str, default="gers",
                        choices=["gers", "gers_adaptive", "gers_sc", "gers_sc_cv", "gers_sc_cv2", "gers_adaptive_cv", "gers_adaptive_cv2", "gers_grounded", "gers_grounded_soft", "gf_gers", "gers_cv2_retr", "gers_cv2_fullctx", "gers_repair", "gers_repair_soft", "gers_cv2_uniform", "gers_cv2_ctxonly", "gers_nli", "gers_feedback", "standard_cot", "cot_sc", "cot_sc_gers", "tot", "zero_shot", "modegraph"])
    parser.add_argument("--model", type=str, default="qwen3-8b")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120,
                        help="单条样本超时秒数")
    parser.add_argument("--output_dir", type=str, default="experiments/results/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.dataset}_{args.method}_results.json"

    # 加载数据
    from data.prepare_data import load_processed_dataset
    samples = load_processed_dataset(args.dataset, num_samples=args.num_samples)
    print(f"\n[{args.method.upper()} @ {args.dataset}] 共 {len(samples)} 条样本")

    # 断点续跑
    existing = load_results(result_path)
    # 兼容旧格式（无 error 字段）
    for r in existing:
        r.setdefault("error", None)
    done_ids = {r["sample_id"] for r in existing}
    results = list(existing)
    print(f"已完成: {len(done_ids)} 条，剩余: {len(samples) - len(done_ids)} 条")

    # 创建模型和方法
    from experiments.run_comparison import create_model
    model = create_model(args.model)
    method = create_method(args.method, model)

    # 答案标准化器
    normalizer = NORMALIZERS.get(args.dataset, lambda x: x)
    total_time = sum(r["metrics"]["latency"] for r in results)
    timeout_count = 0
    error_count = 0

    for i, sample in enumerate(samples):
        if i in done_ids:
            continue

        result, latency, error = run_one_sample(method, sample, args.timeout)
        total_time += latency

        if error:
            error_count += 1
            prediction = ""
            print(f"  [{i+1}/{len(samples)}] 错误({latency:.1f}s): {error[:60]}")
        else:
            prediction = normalizer(result.get("answer", ""))
            # 获取 consistency_score（GERS专有）
            cs = result.get("consistency_score", 0)
            if isinstance(cs, dict):
                cs = cs.get("consistency_score", 0)

        metrics = Metrics.compute_all(
            prediction=prediction,
            reference=normalizer(sample["answer"]),
            token_count=model.count_tokens(
                result.get("reasoning_text", "") if result else ""
            ),
            latency=latency,
            dataset=args.dataset,
        )
        if result and "consistency_score" in result:
            cs = result["consistency_score"]
            metrics["consistency_score"] = cs if isinstance(cs, float) else cs.get("consistency_score", 0)

        record = {
            "sample_id": i,
            "question": sample["question"][:100],
            "prediction": prediction,
            "reference": sample["answer"],
            "metrics": metrics,
            "method": args.method,
            "error": error,
        }
        # 记录 GERS 专有指标
        if result and args.method == "gers":
            record["iterations"] = result.get("iterations", 0)
            record["consistency_detail"] = result.get("consistency_detail")
            record["token_count"] = result.get("token_count", 0)
            record["num_sub_questions"] = len(result.get("sub_qa_chain", []))
        results.append(record)

        # 实时保存
        finished = [r for r in results if r["error"] is None]
        avg_em = sum(r["metrics"]["em"] for r in finished) / max(len(finished), 1)
        avg_f1 = sum(r["metrics"]["f1"] for r in finished) / max(len(finished), 1)
        summary = {
            "method": args.method,
            "dataset": args.dataset,
            "model": args.model,
            "num_samples": len(results),
            "avg_em": round(avg_em, 4),
            "avg_f1": round(avg_f1, 4),
            "avg_latency": round(total_time / len(results), 4),
            "timeout_count": timeout_count,
            "error_count": error_count,
        }
        save_results(result_path, results, summary)

        em_str = f"EM={metrics['em']:.2f}"
        print(f"  [{i+1}/{len(samples)}] {latency:.1f}s | {em_str} | 累计EM={avg_em:.3f} | pred={prediction[:20]!r}")

    # 最终汇总
    finished = [r for r in results if r["error"] is None]
    avg_em = sum(r["metrics"]["em"] for r in finished) / max(len(finished), 1)
    avg_f1 = sum(r["metrics"]["f1"] for r in finished) / max(len(finished), 1)
    avg_cs = sum(r["metrics"].get("consistency_score", 0) for r in finished) / max(len(finished), 1)

    print(f"\n{'='*55}")
    print(f"方法: {args.method} | 数据集: {args.dataset} | 模型: {args.model}")
    print(f"{'─'*55}")
    print(f"样本数: {len(finished)} (有效) / {len(results)} (总计)")
    print(f"EM:            {avg_em:.4f}")
    print(f"F1:            {avg_f1:.4f}")
    if avg_cs > 0:
        print(f"Consistency:   {avg_cs:.4f}")
    print(f"平均耗时:      {total_time/max(len(results),1):.2f}s/条")
    print(f"错误数:        {error_count}")
    print(f"结果已保存:    {result_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
