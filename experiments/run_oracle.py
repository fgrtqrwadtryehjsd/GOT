"""SOP Stage-2 Oracle decomposition runner — localize the bottleneck.

Injects gold annotations from MuSiQue's question_decomposition into the GERS
pipeline to quantify each module's "罪值" (how much F1 each perfect substitution
recovers). Five configurations, run on the same samples:

  Baseline (0 Oracle) : model self-decomposes, self-retrieves, self-answers
  Oracle-1 (DAG)      : gold sub-questions (skip _decompose)
  Oracle-1+2 (DAG+Ret): gold sub-questions + gold paragraph per sub-question
  Oracle-1+3 (DAG+Ans): gold sub-questions + gold intermediate answers (skip reasoner)
  Oracle-1+3+4 (...)  : = Oracle-1+3; Oracle-4 is "given gold intermediates, can
                        the aggregator produce the gold final answer" — measured
                        directly by the O1+3 F1 ceiling (aggregator is the only
                        non-oracle module left).

The waterfall: Baseline → O1 (Δ = graph-generation罪值) → O1+2 (Δ = retriever罪值)
→ O1+3 (Δ = reasoner/error-propagation罪值). The largest Δ localizes the bottleneck.
Per SOP "No Oracle, No Design": the Minimal Fix targets the largest-Δ module ONLY.

Run on MuSiQue 4-hop (where Wall-1 fails hardest, CoT-SC F1=0.370, room is largest):
    python experiments/run_oracle.py --dataset musique --hop 4 --num_samples 85
Then 2-hop / 3-hop for the full picture:
    python experiments/run_oracle.py --dataset musique --hop 2 --num_samples 263
    python experiments/run_oracle.py --dataset musique --hop 3 --num_samples 152
"""
import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.utils.metrics import Metrics
from src.utils.answer_normalizer import normalize_musique_answer, normalize_hotpotqa_answer
from src.chain_generation import GraphGuidedGenerator
from experiments.run_comparison import create_model

NORMALIZERS = {"musique": normalize_musique_answer, "hotpotqa": normalize_hotpotqa_answer}


def load_samples(dataset: str, hop: int, num_samples: int):
    from data.prepare_data import load_processed_dataset
    samples = load_processed_dataset(dataset, num_samples=10000)  # load all, filter by hop
    if hop > 0:
        samples = [s for s in samples if s.get("hop_count") == hop]
    return samples[:num_samples]


def build_oracle_context_paragraphs(sample):
    """Oracle-2: extract gold paragraph text per sub-question from MuSiQue.

    MuSiQue's question_decomposition[i].paragraph_support_idx points into the
    paragraphs list. We return the gold paragraph text per sub-question so the
    pipeline sees ONLY the supporting paragraph (zero distractor).
    """
    decomp = sample.get("question_decomposition", [])
    # paragraphs are not kept in processed json (only supporting_paragraphs idx list).
    # We rebuild paragraph→text from context_full by [Title] segmentation.
    # But paragraph_support_idx indexes the ORIGINAL paragraphs order, which matches
    # the join order in prepare_musique. We segment context_full to recover them.
    import re
    ctx = sample.get("context_full", "")
    paras = re.split(r"\s*\|\s*(?=\[)", ctx)
    paras = [p.strip() for p in paras if p.strip()]
    gold_paras = []
    for step in decomp:
        idx = step.get("paragraph_support_idx", -1)
        if 0 <= idx < len(paras):
            gold_paras.append(paras[idx])
        else:
            gold_paras.append("")  # fallback: no gold para for this step
    return gold_paras


def make_method(model, dataset, config, sample):
    """Build a GraphGuidedGenerator with the right oracle injection for this config."""
    decomp = sample.get("question_decomposition", [])
    base = dict(
        model=model, max_iterations=1, enable_nli=True, adaptive=False,
        consistency_threshold=0.75, _no_constraint=True, dataset=dataset,
        enable_backward_verify=True, enable_llm_match=False,
        enable_confidence_weighting=True, context_char_limit=100000,
    )
    if config == "baseline":
        return GraphGuidedGenerator(**base)
    if config == "o1_dag":
        return GraphGuidedGenerator(oracle_decomposition=decomp, **base)
    if config == "o12_dag_ret":
        gold_paras = build_oracle_context_paragraphs(sample)
        return GraphGuidedGenerator(oracle_decomposition=decomp,
                                    oracle_context_paragraphs=gold_paras, **base)
    if config == "o13_dag_ans":
        return GraphGuidedGenerator(oracle_decomposition=decomp,
                                    oracle_subanswers=True, **base)
    raise ValueError(f"unknown config {config}")


def run_one(config, model, sample, idx, dataset, normalizer):
    method = make_method(model, dataset, config, sample)
    start = time.time()
    try:
        result = method.reason(question=sample["question"],
                               context=sample.get("context_full", sample.get("context", "")))
        latency = time.time() - start
        error = None
    except Exception as e:
        latency = time.time() - start
        result = None
        error = str(e)[:120]
    pred = "" if (error or result is None) else normalizer(result.get("answer", ""))
    metrics = Metrics.compute_all(
        prediction=pred, reference=normalizer(sample["answer"]),
        token_count=0, latency=latency, dataset=dataset)
    return {"sample_id": idx, "prediction": pred, "reference": sample["answer"],
            "metrics": metrics, "config": config, "error": error}


def run_config(config, samples, model, dataset, normalizer, workers, out_dir):
    out_path = out_dir / f"{dataset}_oracle_{config}_results.json"
    # resume
    existing = []
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8")).get("results", [])
    done = {r["sample_id"] for r in existing}
    pending = [(i, s) for i, s in enumerate(samples) if i not in done]
    results = list(existing)
    print(f"\n=== [Oracle {config}] {dataset} | 共 {len(samples)} | 待跑 {len(pending)} ===")
    if not pending:
        print("  全部已完成（resume）。")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, config, model, s, i, dataset, normalizer): i
                for i, s in pending}
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: r["sample_id"])
    finished = [r for r in results if not r.get("error")]
    avg_f1 = sum(r["metrics"]["f1"] for r in finished) / max(len(finished), 1)
    avg_em = sum(r["metrics"]["em"] for r in finished) / max(len(finished), 1)
    summary = {"config": config, "dataset": dataset, "hop": samples[0].get("hop_count"),
               "num_samples": len(results), "avg_em": round(avg_em, 4),
               "avg_f1": round(avg_f1, 4), "error_count": sum(1 for r in results if r.get("error"))}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"  [Oracle {config}] EM={avg_em:.4f} F1={avg_f1:.4f} | n={len(finished)}/{len(results)} | "
          f"err={summary['error_count']} | {time.time()-t0:.0f}s | → {out_path}")
    return summary


def main():
    p = argparse.ArgumentParser(description="SOP Stage-2 Oracle 解剖")
    p.add_argument("--dataset", default="musique", choices=["musique", "hotpotqa"])
    p.add_argument("--hop", type=int, default=4, help="跳数过滤(0=全部)")
    p.add_argument("--num_samples", type=int, default=85)
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--configs", default="baseline,o1_dag,o12_dag_ret,o13_dag_ans",
                   help="逗号分隔的 Oracle 配置")
    p.add_argument("--output_dir", default="experiments/results/oracle_musique_8b")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalizer = NORMALIZERS[args.dataset]

    samples = load_samples(args.dataset, args.hop, args.num_samples)
    print(f"[数据] {args.dataset} hop={args.hop}: {len(samples)} 条样本")
    # 关键检查：每条都要有 question_decomposition（Oracle 注入前提）
    with_decomp = sum(1 for s in samples if s.get("question_decomposition"))
    print(f"[数据] 含 question_decomposition 标注: {with_decomp}/{len(samples)} "
          f"({'OK' if with_decomp == len(samples) else '⚠️部分缺失'})")

    model = create_model(args.model)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    summaries = []
    for cfg in configs:
        summaries.append(run_config(cfg, samples, model, args.dataset,
                                    normalizer, args.workers, out_dir))

    # 瀑布图
    print("\n" + "=" * 70)
    print(f"ORACLE 瀑布图 (waterfall) | {args.dataset} hop={args.hop} | {args.model}")
    print("=" * 70)
    print(f"{'配置':<22} {'EM':>8} {'F1':>8} {'ΔF1(累积)':>12} {'ΔF1(本步)':>12}")
    print("-" * 70)
    prev_f1 = None
    cum_prev = None
    for s in summaries:
        f1 = s["avg_f1"]
        cum_delta = f"{f1 - cum_prev:+.4f}" if cum_prev is not None else "-"
        step_delta = f"{f1 - prev_f1:+.4f}" if prev_f1 is not None else "-"
        print(f"{s['config']:<22} {s['avg_em']:>8.4f} {f1:>8.4f} {cum_delta:>12} {step_delta:>12}")
        prev_f1 = f1
        if cum_prev is None:
            cum_prev = f1
    print("=" * 70)
    print("解读: ΔF1(本步) 最大者 = 瓶颈模块。Minimal Fix 只打这个模块。")


if __name__ == "__main__":
    main()
