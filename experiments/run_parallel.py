"""
多线程并行实验 runner
======================

解决单进程逐条跑 LLM 推理太慢的问题。瓶颈是 DashScope API 网络等待（I/O 密集），
多线程在 I/O 等待时释放 GIL，加速比 ≈ worker 数。

特点：
1. ThreadPoolExecutor，--workers 控制并发（默认 8）
2. 共享一个 DashScopeModel（缓存 client + 重试），每样本新建 method 实例
   （避免 GERS-SC 临时属性的多线程竞态）
3. 断点续跑：跳过已完成 sample_id
4. 增量落盘：threading.Lock 保护，每完成 5 条写一次
5. 软超时：--timeout 控制单样本最大等待（依赖 client 60s 超时 + 重试兜底）
6. 跨方法汇总表

使用：
    # smoke 20 条
    python experiments/run_parallel.py --dataset hotpotqa \
        --methods cot_sc_gers,cot_sc,gers_sc,gers_adaptive,standard_cot,zero_shot \
        --num_samples 20 --workers 8

    # 全量 100
    python experiments/run_parallel.py --dataset hotpotqa \
        --methods cot_sc_gers,cot_sc,gers_adaptive,standard_cot,zero_shot \
        --num_samples 100 --workers 8
"""

import argparse
import json
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.utils.metrics import Metrics
from src.utils.answer_normalizer import (
    normalize_gsm8k_answer, normalize_hotpotqa_answer, normalize_2wikimultihopqa_answer
)
from experiments.run_comparison import create_model
from experiments.run_quick_exp import create_method

NORMALIZERS = {
    "gsm8k": normalize_gsm8k_answer,
    "hotpotqa": normalize_hotpotqa_answer,
    "2wikimultihopqa": normalize_2wikimultihopqa_answer,
}


def load_samples(dataset_name: str, num_samples: int):
    from data.prepare_data import load_processed_dataset
    return load_processed_dataset(dataset_name, num_samples=num_samples)


def load_existing(result_path: Path):
    if result_path.exists():
        try:
            with open(result_path, encoding="utf-8") as f:
                return json.load(f).get("results", [])
        except Exception:
            return []
    return []


def save_results(result_path: Path, results: list, summary: dict):
    tmp = result_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f,
                  ensure_ascii=False, indent=2)
    tmp.replace(result_path)  # 原子写，避免中途被打断导致文件损坏


def process_one(method_name: str, model, sample: dict, idx: int,
                dataset: str, normalizer):
    """运行单条样本（在工作线程内执行）。每样本新建 method 实例，避免共享可变状态。"""
    method = create_method(method_name, model, dataset=dataset)
    start = time.time()
    try:
        result = method.reason(
            question=sample["question"],
            context=sample.get("context", "")
        )
        latency = time.time() - start
        error = None
    except Exception as e:
        latency = time.time() - start
        result = None
        error = str(e)[:120]

    if error or result is None:
        prediction = ""
        cs = 0.0
    else:
        prediction = normalizer(result.get("answer", ""))
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
        dataset=dataset,
    )
    if result and "consistency_score" in result:
        cs_raw = result["consistency_score"]
        metrics["consistency_score"] = cs_raw if isinstance(cs_raw, float) \
            else cs_raw.get("consistency_score", 0)

    record = {
        "sample_id": idx,
        "question": sample["question"][:100],
        "prediction": prediction,
        "reference": sample["answer"],
        "metrics": metrics,
        "method": method_name,
        "error": error,
    }
    # 保存推理文本（截断）便于离线复检答案提取/重排，无需重跑 LLM
    if result:
        rt = result.get("reasoning_text", "")
        if rt:
            record["reasoning_text"] = rt[:600]
    # GERS 专有字段
    if result and method_name in ("gers", "gers_adaptive", "gers_sc", "gers_nli",
                                  "gers_feedback", "gers_sc_cv", "gers_sc_cv2",
                                  "gers_adaptive_cv", "gers_adaptive_cv2",
                                  "gers_grounded", "gers_grounded_soft",
                                  "gers_repair", "gers_repair_soft",
                                  "gers_cv2_uniform", "gers_cv2_ctxonly"):
        record["iterations"] = result.get("iterations", 0)
        record["consistency_detail"] = result.get("consistency_detail")
        record["token_count"] = result.get("token_count", 0)
        record["num_sub_questions"] = len(result.get("sub_qa_chain", []))
        # 方向1：单独存 crossval_score 便于区分度分析
        cd = result.get("consistency_detail") or {}
        if isinstance(cd, dict) and "crossval_score" in cd:
            record["crossval_score"] = cd["crossval_score"]
            record["struct_score"] = cd.get("struct_score")
        if isinstance(cd, dict) and "repair_detail" in cd:
            record["repair_detail"] = cd["repair_detail"]
    if result and method_name == "cot_sc_gers":
        record["gers_rerank_triggered"] = result.get("gers_rerank_triggered", False)
        record["gers_scores"] = result.get("gers_scores")
        record["majority_answer"] = result.get("majority_answer")
    return record, latency


def run_method(method_name: str, samples: list, model, dataset: str,
               normalizer, output_dir: Path, workers: int, timeout: int,
               resume: bool):
    result_path = output_dir / f"{dataset}_{method_name}_results.json"

    # 断点续跑
    existing = load_existing(result_path) if resume else []
    for r in existing:
        r.setdefault("error", None)
    done_ids = {r["sample_id"] for r in existing}
    results = list(existing)
    pending = [(i, s) for i, s in enumerate(samples) if i not in done_ids]

    print(f"\n=== [{method_name} @ {dataset}] 共 {len(samples)} 条 | "
          f"已完成 {len(done_ids)} | 待跑 {len(pending)} | workers={workers} ===")

    if not pending:
        print("  无待跑样本，跳过。")

    lock = threading.Lock()
    total_time = sum(r["metrics"]["latency"] for r in results if "metrics" in r)
    error_count = sum(1 for r in results if r.get("error"))
    completed = 0
    t0 = time.time()

    def _save_and_progress(records_to_add):
        nonlocal total_time, error_count, completed
        with lock:
            results.extend(records_to_add)
            completed += len(records_to_add)
            # 排序保持文件有序
            results_sorted = sorted(results, key=lambda r: r["sample_id"])
            finished = [r for r in results_sorted if r.get("error") is None]
            avg_em = sum(r["metrics"]["em"] for r in finished) / max(len(finished), 1)
            avg_f1 = sum(r["metrics"]["f1"] for r in finished) / max(len(finished), 1)
            summary = {
                "method": method_name,
                "dataset": dataset,
                "num_samples": len(results_sorted),
                "avg_em": round(avg_em, 4),
                "avg_f1": round(avg_f1, 4),
                "avg_latency": round(total_time / max(len(results_sorted), 1), 4),
                "error_count": error_count,
                "workers": workers,
            }
            save_results(result_path, results_sorted, summary)
            return avg_em, avg_f1, len(finished)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_one, method_name, model, s, i, dataset, normalizer): i
            for i, s in pending
        }
        batch = []
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                record, latency = fut.result(timeout=timeout)
            except Exception as e:
                latency = timeout
                record = {
                    "sample_id": idx,
                    "question": samples[idx]["question"][:100],
                    "prediction": "",
                    "reference": samples[idx]["answer"],
                    "metrics": Metrics.compute_all(
                        prediction="", reference=normalizer(samples[idx]["answer"]),
                        token_count=0, latency=latency, dataset=dataset),
                    "method": method_name,
                    "error": f"timeout/error: {str(e)[:100]}",
                }
            with lock:
                total_time += latency
                if record.get("error"):
                    error_count += 1
            batch.append(record)

            # 每 5 条落盘一次
            if len(batch) >= 5:
                avg_em, avg_f1, n_fin = _save_and_progress(batch)
                batch = []
                elapsed = time.time() - t0
                print(f"  [{completed}/{len(samples)}] 已用 {elapsed:.0f}s | "
                      f"累计 EM={avg_em:.3f} F1={avg_f1:.3f} (n={n_fin}) | "
                      f"err={error_count}")

        # 收尾
        if batch:
            avg_em, avg_f1, n_fin = _save_and_progress(batch)

    # 最终汇总
    finished = [r for r in results if r.get("error") is None]
    avg_em = sum(r["metrics"]["em"] for r in finished) / max(len(finished), 1)
    avg_f1 = sum(r["metrics"]["f1"] for r in finished) / max(len(finished), 1)
    avg_cs = sum(r["metrics"].get("consistency_score", 0) for r in finished) / max(len(finished), 1)
    elapsed = time.time() - t0
    print(f"\n  [{method_name}] 完成: EM={avg_em:.4f} F1={avg_f1:.4f} "
          f"CS={avg_cs:.4f} | 有效 {len(finished)}/{len(results)} | "
          f"错误 {error_count} | 耗时 {elapsed:.0f}s | 保存 {result_path}")
    return {
        "method": method_name, "em": avg_em, "f1": avg_f1, "cs": avg_cs,
        "n": len(finished), "errors": error_count, "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="多线程并行实验 runner")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["gsm8k", "hotpotqa", "2wikimultihopqa"])
    parser.add_argument("--methods", type=str, default="gers_adaptive,standard_cot,cot_sc,cot_sc_gers,zero_shot",
                        help="逗号分隔的方法名（见 run_quick_exp.create_method）")
    parser.add_argument("--model", type=str, default="qwen3-8b")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8, help="并发线程数")
    parser.add_argument("--timeout", type=int, default=300, help="单样本最大等待秒数")
    parser.add_argument("--output_dir", type=str, default="experiments/results/")
    parser.add_argument("--context_field", type=str, default="context", help="样本中用作 context 的字段(context/context_full)")
    parser.add_argument("--no_resume", action="store_true", help="不读取已有结果，从头跑")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.dataset, args.num_samples)
    if args.context_field != "context":
        for s in samples:
            if args.context_field in s:
                s["context"] = s[args.context_field]
        print(f"[数据] context 字段映射为 {args.context_field}")
    normalizer = NORMALIZERS.get(args.dataset, lambda x: x)
    print(f"[数据] {args.dataset}: {len(samples)} 条样本")

    model = create_model(args.model)
    print(f"[模型] {args.model}（client 已缓存，{args.workers} 线程共享）")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_summary = []
    for m in methods:
        try:
            s = run_method(m, samples, model, args.dataset, normalizer,
                           output_dir, args.workers, args.timeout,
                           resume=not args.no_resume)
            all_summary.append(s)
        except Exception as e:
            print(f"\n  [{m}] 运行出错: {e}")

    # 跨方法汇总表
    print(f"\n\n{'='*72}")
    print(f"并行实验汇总 | {args.dataset} | model={args.model} | workers={args.workers}")
    print(f"{'-'*72}")
    print(f"{'方法':<16} {'EM':>8} {'F1':>8} {'CS':>8} {'有效n':>7} {'错误':>5} {'耗时s':>8}")
    print(f"{'-'*72}")
    for s in all_summary:
        print(f"{s['method']:<16} {s['em']:>8.4f} {s['f1']:>8.4f} "
              f"{s['cs']:>8.4f} {s['n']:>7} {s['errors']:>5} {s['elapsed']:>8.0f}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
