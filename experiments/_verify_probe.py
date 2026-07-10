"""Pre-validate the stepwise verifier (LLM-as-NLI probe) before building EASV.

Question this answers: does a per-sub-step entailment check (gold-paragraph ⊨
sub-answer) have any DISCRIMINATIVE power between correct and incorrect sub-answers?
If correct sub-answers entail at ~same rate as wrong ones, the verifier is no
better than the self-agreement signal it would replace (H3: fluency-indexed) —
and EASV's foundation is dead. If there's a clear gap, the verifier has legs and
we invest in an independent NLI model next.

This is a PROBE using LLM-as-verifier (qwen3-8b judging entailment), explicitly
NOT EASV's final form — that needs an independent small NLI model to avoid the
self-bias the literature flags. The probe just checks whether the *signal* exists
at all before spending on the independent model.

Pipeline:
  1. Run baseline GERS on N MuSiQue 4-hop samples, KEEP the sub_qa_chain.
  2. For each (sub-question, model sub-answer): retrieve the gold supporting
     paragraph (paragraph_support_idx), ask qwen3-8b to judge
     entail/contradict/neutral of (paragraph ⊨ sub-answer).
  3. Ground truth: sub-answer correct? = normalize(sub_answer) matches
     normalize(gold_sub_answer) (substring either direction).
  4. Report: entail rate among correct vs incorrect sub-answers; AUROC of
     "entail => correct" as a correctness signal.

Run: python experiments/_verify_probe.py --num_samples 30
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.chain_generation import GraphGuidedGenerator
from src.utils.answer_normalizer import normalize_musique_answer
from experiments.run_comparison import create_model

NLI_PROMPT = """You are a strict entailment judge. Given a CONTEXT paragraph and a CLAIM (a sub-answer), decide whether the context supports the claim.

Context: {context}

Claim: {claim}

Answer with exactly one label on the first line:
- entail  : the context explicitly supports the claim
- contradict : the context explicitly contradicts the claim
- neutral : the context does not state the claim (insufficient info)

Then on the next line give a one-sentence reason. Label:"""


def segment_paras(ctx):
    return [p.strip() for p in re.split(r"\s*\|\s*(?=\[)", ctx) if p.strip()]


def judge_entailment(model, context, claim):
    prompt = NLI_PROMPT.format(context=context[:2500], claim=claim[:300])
    resp = model.generate(prompt, max_tokens=60, temperature=0.0)
    first = resp.strip().split("\n")[0].strip().lower()
    for label in ("entail", "contradict", "neutral"):
        if label in first:
            return label
    return "neutral"


def is_correct(pred, gold):
    p = normalize_musique_answer(pred).lower().strip()
    g = normalize_musique_answer(gold).lower().strip()
    if not p or not g:
        return False
    return g in p or p in g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--hop", type=int, default=4)
    ap.add_argument("--model", default="qwen3-8b")
    args = ap.parse_args()

    samples = json.loads(Path("data/processed/musique_test.json").read_text(encoding="utf-8"))
    samples = [s for s in samples if s.get("hop_count") == args.hop][:args.num_samples]
    print(f"[数据] MuSiQue {args.hop}-hop: {len(samples)} 条")

    model = create_model(args.model)
    # 用 Oracle-1 配置（gold 分解注入）：模型只负责回答每跳子问题，
    # 这样子答案链长度 = gold 跳数，position 对齐成立（避免模型自主分解
    # 出 2 跳而 gold 是 4 跳导致的 position 错位）。
    gen = GraphGuidedGenerator(
        model=model, max_iterations=1, enable_nli=True, adaptive=False,
        consistency_threshold=0.75, _no_constraint=True, dataset="musique",
        enable_backward_verify=True, enable_llm_match=False,
        enable_confidence_weighting=True, context_char_limit=100000)

    records = []
    for si, s in enumerate(samples):
        decomp = s.get("question_decomposition", [])
        paras = segment_paras(s.get("context_full", ""))
        # Oracle-1: 注入 gold 分解，模型只回答每跳
        gen_oracle = GraphGuidedGenerator(
            model=model, max_iterations=1, enable_nli=True, adaptive=False,
            consistency_threshold=0.75, _no_constraint=True, dataset="musique",
            enable_backward_verify=False, enable_llm_match=False,
            enable_confidence_weighting=False, context_char_limit=100000,
            oracle_decomposition=decomp)
        try:
            r = gen_oracle.reason(question=s["question"], context=s.get("context_full", ""))
        except Exception as e:
            print(f"  [{si+1}] reason error: {e}")
            continue
        chain = r.get("sub_qa_chain", [])
        for ci, item in enumerate(chain):
            if ci >= len(decomp):
                break
            gold_ans = decomp[ci].get("answer", "")
            pidx = decomp[ci].get("paragraph_support_idx", -1)
            gold_para = paras[pidx] if 0 <= pidx < len(paras) else ""
            sub_ans = item.get("sub_answer", "")
            correct = is_correct(sub_ans, gold_ans)
            label = judge_entailment(model, gold_para, sub_ans) if gold_para else "neutral"
            records.append({
                "sample": si, "step": ci, "sub_answer": sub_ans[:60],
                "gold_answer": gold_ans[:60], "correct": correct, "nli_label": label,
            })
            print(f"  [{si+1} 跳{ci+1}] correct={correct} nli={label} | ans={sub_ans[:25]!r} gold={gold_ans[:25]!r}")

    # ── 区分度统计 ──
    print("\n" + "=" * 65)
    print("VERIFIER 区分度 (LLM-as-NLI probe)")
    print("=" * 65)
    n = len(records)
    correct = [r for r in records if r["correct"]]
    wrong = [r for r in records if not r["correct"]]
    ent_correct = sum(1 for r in correct if r["nli_label"] == "entail")
    ent_wrong = sum(1 for r in wrong if r["nli_label"] == "entail")
    print(f"总子答案数: {n} | 正确: {len(correct)} | 错误: {len(wrong)}")
    print(f"正确子答案 entail 率: {ent_correct}/{len(correct)} = {ent_correct/max(len(correct),1):.3f}")
    print(f"错误子答案 entail 率: {ent_wrong}/{len(wrong)} = {ent_wrong/max(len(wrong),1):.3f}")
    gap = ent_correct/max(len(correct),1) - ent_wrong/max(len(wrong),1)
    print(f"区分度 gap (correct - wrong entail率): {gap:+.3f}")
    print(f"\n判读: gap>0.2 = verifier有区分度(EASV有腿); gap<0.1 = verifier无区分度(self-agreement同病)")

    # AUROC of "entail => correct" (treat entail=1, neutral/contradict=0)
    import numpy as np
    if records:
        scores = np.array([1.0 if r["nli_label"] == "entail" else 0.0 for r in records])
        labels = np.array([1 if r["correct"] else 0 for r in records])
        if labels.sum() > 0 and labels.sum() < len(labels):
            try:
                from sklearn.metrics import roc_auc_score
                auroc = float(roc_auc_score(labels, scores))
            except Exception:
                # fallback: Mann-Whitney U
                pos = scores[labels == 1]
                neg = scores[labels == 0]
                u = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(1 for a in pos for b in neg if a == b)
                auroc = u / (len(pos) * len(neg)) if len(pos) * len(neg) else 0.5
            print(f"verifier AUROC (entail=>correct): {auroc:.3f}  (0.5=随机, 对比self-agreement CS AUROC≈0.58)")

    Path("experiments/results/verify_probe").mkdir(parents=True, exist_ok=True)
    out = Path("experiments/results/verify_probe/nli_probe_results.json")
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已存: {out}")


if __name__ == "__main__":
    main()
