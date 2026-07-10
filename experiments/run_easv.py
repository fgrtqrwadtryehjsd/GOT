"""EASV prototype runner — Evidence-Anchored Stepwise Verification (SOP Stage-3).

Minimal Fix targeting the reasoner-error-propagation bottleneck (Oracle Stage-2:
71% of recoverable F1 at 4-hop). For each sub-question: generate sub-answer,
NLI-verify (supporting paragraph ⊨ sub-answer?), re-answer the node if not entail.

This runner tests the MECHANISM in isolation (per user decision): Oracle-1 gold
decomposition injected, so decomposition quality is held fixed and only the
verify+re-answer mechanism's contribution is measured. Target: push decomposition
above CoT-SC's 0.370 at MuSiQue 4-hop.

Configs:
  baseline   : Oracle-1 gold decomposition, model answers each hop, NO verification
  easv_o1    : Oracle-1 + stepwise NLI verify + re-answer (the Minimal Fix)
  easv_o1_r2 : same with 2 retries (re-answer budget ablation)

Run:
  python experiments/run_easv.py --dataset musique --hop 4 --num_samples 85
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
from src.utils.answer_normalizer import normalize_musique_answer
from src.chain_generation import GraphGuidedGenerator
from experiments.run_comparison import create_model


def load_samples(dataset, hop, num_samples):
    from data.prepare_data import load_processed_dataset
    samples = load_processed_dataset(dataset, num_samples=10000)
    if hop > 0:
        samples = [s for s in samples if s.get("hop_count") == hop]
    return samples[:num_samples]


def make_method(model, config, sample):
    decomp = sample.get("question_decomposition", [])
    base = dict(
        model=model, max_iterations=1, enable_nli=True, adaptive=False,
        consistency_threshold=0.75, _no_constraint=True, dataset="musique",
        enable_backward_verify=False, enable_llm_match=False,
        enable_confidence_weighting=False, context_char_limit=100000,
        oracle_decomposition=decomp,
    )
    if config == "baseline":
        return GraphGuidedGenerator(**base)
    if config == "easv_o1":
        return GraphGuidedGenerator(enable_stepwise_verify=True,
                                    stepwise_verify_retries=1, **base)
    if config == "easv_o1_r2":
        return GraphGuidedGenerator(enable_stepwise_verify=True,
                                    stepwise_verify_retries=2, **base)
    raise ValueError(config)


def run_one(config, model, sample, idx, normalizer):
    method = make_method(model, config, sample)
    start = time.time()
    try:
        result = method.reason(question=sample["question"],
                               context=sample.get("context_full", ""))
        latency = time.time() - start
        error = None
    except Exception as e:
        latency = time.time() - start
        result = None
        error = str(e)[:120]
    pred = "" if (error or result is None) else normalizer(result.get("answer", ""))
    metrics = Metrics.compute_all(
        prediction=pred, reference=normalizer(sample["answer"]),
        token_count=0, latency=latency, dataset="musique")
    return {"sample_id": idx, "prediction": pred, "reference": sample["answer"],
            "metrics": metrics, "config": config, "error": error}


def run_config(config, samples, model, normalizer, workers, out_dir):
    out_path = out_dir / f"musique_easv_{config}_results.json"
    existing = []
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8")).get("results", [])
    done = {r["sample_id"] for r in existing}
    pending = [(i, s) for i, s in enumerate(samples) if i not in done]
    results = list(existing)
    print(f"\n=== [{config}] 共 {len(samples)} | 待跑 {len(pending)} ===")
    if not pending:
        print("  全部完成（resume）。")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, config, model, s, i, normalizer): i
                for i, s in pending}
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: r["sample_id"])
    finished = [r for r in results if not r.get("error")]
    avg_f1 = sum(r["metrics"]["f1"] for r in finished) / max(len(finished), 1)
    avg_em = sum(r["metrics"]["em"] for r in finished) / max(len(finished), 1)
    summary = {"config": config, "hop": samples[0].get("hop_count"),
               "num_samples": len(results), "avg_em": round(avg_em, 4),
               "avg_f1": round(avg_f1, 4),
               "error_count": sum(1 for r in results if r.get("error"))}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"  [{config}] EM={avg_em:.4f} F1={avg_f1:.4f} | n={len(finished)}/{len(results)} | "
          f"err={summary['error_count']} | {time.time()-t0:.0f}s | → {out_path}")
    return summary


def main():
    p = argparse.ArgumentParser(description="EASV 雏形 runner (SOP Stage-3)")
    p.add_argument("--dataset", default="musique")
    p.add_argument("--hop", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=85)
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--configs", default="baseline,easv_o1,easv_o1_r2")
    p.add_argument("--output_dir", default="experiments/results/easv_musique_8b")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalizer = normalize_musique_answer

    samples = load_samples(args.dataset, args.hop, args.num_samples)
    print(f"[数据] {args.dataset} hop={args.hop}: {len(samples)} 条")
    model = create_model(args.model)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    summaries = []
    for cfg in configs:
        summaries.append(run_config(cfg, samples, model, normalizer,
                                    args.workers, out_dir))

    # 对比表（含 CoT-SC 参照 0.370）
    print("\n" + "=" * 70)
    print(f"EASV 机制验证 (Oracle-1 隔离) | MuSiQue hop={args.hop} | {args.model}")
    print("=" * 70)
    print(f"{'配置':<16} {'EM':>8} {'F1':>8} {'vs CoT-SC 0.370':>16}")
    print("-" * 70)
    for s in summaries:
        delta = s["avg_f1"] - 0.370
        print(f"{s['config']:<16} {s['avg_em']:>8.4f} {s['avg_f1']:>8.4f} {delta:>+16.4f}")
    print("-" * 70)
    print("判读: easv_o1 F1 > baseline F1 = 验证+重答机制有效;")
    print("      easv_o1 F1 > 0.370 = 机制足以让分解超过 CoT-SC（端到端有希望）。")


if __name__ == "__main__":
    main()
