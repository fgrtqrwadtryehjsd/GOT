"""Diagnose model decomposition failure modes vs gold (MuSiQue 4-hop).

Before designing a decomposition-quality fix, confirm WHERE model decomposition
fails. Hypothesis: 4-hop gets compressed to 2-hop (depth loss). Alternatives:
dependency structure wrong, sub-question semantics drift, etc.

Run: python experiments/_decomp_diagnosis.py --num_samples 30 --hop 4
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.chain_generation import GraphGuidedGenerator
from experiments.run_comparison import create_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--hop", type=int, default=4)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--idd", action="store_true",
                    help="启用 IDD 迭代深化分解(修深度压缩)")
    args = ap.parse_args()

    samples = json.loads(Path("data/processed/musique_test.json").read_text(encoding="utf-8"))
    samples = [s for s in samples if s.get("hop_count") == args.hop][:args.num_samples]
    mode = "IDD-v2(复合度判据)" if args.idd else "baseline(模型自主)"
    print(f"[数据] MuSiQue {args.hop}-hop: {len(samples)} 条 | 分解模式: {mode}")

    model = create_model(args.model)
    # baseline 分解（模型自主，不开 oracle/验证）；--idd 启用迭代深化
    gen = GraphGuidedGenerator(
        model=model, max_iterations=1, enable_nli=True, adaptive=False,
        consistency_threshold=0.75, _no_constraint=True, dataset="musique",
        enable_backward_verify=False, enable_llm_match=False,
        enable_confidence_weighting=False, context_char_limit=100000,
        enable_iterative_deepening=args.idd, idd_max_depth=4)

    depth_dist = Counter()
    detail = []
    for si, s in enumerate(samples):
        gold_n = len(s.get("question_decomposition", []))
        try:
            r = gen.reason(question=s["question"], context=s.get("context_full", ""))
            model_n = len(r.get("sub_qa_chain", []))
        except Exception as e:
            model_n = -1
            print(f"  [{si+1}] error: {e}")
            continue
        depth_dist[(gold_n, model_n)] += 1
        detail.append({"si": si, "gold_n": gold_n, "model_n": model_n,
                       "model_qs": [it.get("sub_question", "")[:50] for it in r.get("sub_qa_chain", [])],
                       "gold_qs": [d.get("question", "")[:50] for d in s.get("question_decomposition", [])]})
        tag = "✓深度对" if model_n == gold_n else (f"✗压缩({model_n}<{gold_n})" if model_n < gold_n else f"?膨胀({model_n}>{gold_n})")
        print(f"  [{si+1}] gold={gold_n}模型={model_n} {tag}")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print(f"分解深度分布 (gold跳数, 模型跳数) → 样本数 | {args.hop}-hop n={len(samples)}")
    print("=" * 60)
    for (g, m), c in sorted(depth_dist.items()):
        tag = "深度对" if m == g else (f"压缩{m}<{g}" if m < g else f"膨胀{m}>{g}")
        print(f"  gold={g} 模型={m}: {c:>3}  ({tag})")

    exact = sum(c for (g, m), c in depth_dist.items() if m == g)
    compressed = sum(c for (g, m), c in depth_dist.items() if 0 < m < g)
    expanded = sum(c for (g, m), c in depth_dist.items() if m > g)
    zero = sum(c for (g, m), c in depth_dist.items() if m <= 0)
    print(f"\n深度精确匹配: {exact}/{len(samples)} = {exact/len(samples):.2f}")
    print(f"压缩(模型<gold): {compressed}/{len(samples)} = {compressed/len(samples):.2f}")
    print(f"膨胀(模型>gold): {expanded}/{len(samples)} = {expanded/len(samples):.2f}")
    print(f"零/失败: {zero}")

    # 抽 2 个压缩案例细看
    print("\n=== 压缩案例细看（模型跳数 < gold）===")
    shown = 0
    for d in detail:
        if 0 < d["model_n"] < d["gold_n"] and shown < 2:
            print(f"\n样本{d['si']+1}: gold {d['gold_n']}跳 → 模型 {d['model_n']}跳")
            print("  gold 分解:")
            for i, q in enumerate(d["gold_qs"]):
                print(f"    {i+1}. {q}")
            print("  模型分解:")
            for i, q in enumerate(d["model_qs"]):
                print(f"    {i+1}. {q}")
            shown += 1

    Path("experiments/results/verify_probe").mkdir(parents=True, exist_ok=True)
    suffix = "_idd" if args.idd else ""
    out = Path(f"experiments/results/verify_probe/decomp_diagnosis{suffix}.json")
    out.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已存: {out}")


if __name__ == "__main__":
    main()
