"""Hop-axis analysis: does a CoT-SC/GERS-CV2 crossover exist on reasoning depth?

Reads MuSiQue n=500 results (experiments/results/musique_n500_8b/) and the
hop_count labels from data/processed/musique_test.json. Splits by 2/3/4-hop.

This is the H2 / SOP Stage-1 "reasoning-depth curve". Two decisive questions:
  (Wall-1 precondition) Does flat CoT-SC F1 DEGRADE from 2-hop to 4-hop?
      If not, decomposition/verification has no room (model can single-pass 4-hop).
  (H2 crossover)        Does the CV2-CoT gap narrow/reverse as hops increase?
      H2 predicts CV2 catches up or wins at 4-hop (error propagation hurts CoT
      more than decomposition).

Run: python experiments/_hop_curve.py
"""
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import chi2
except ImportError:  # pragma: no cover
    chi2 = None

MUSIQUE_PROCESSED = Path("data/processed/musique_test.json")
R = Path("experiments/results/musique_n500_8b")
FILES = {
    "cot_sc": R / "musique_cot_sc_results.json",
    "cv2": R / "musique_gers_cv2_fullctx_results.json",
}
B = 10000
SEED = 42


def load_hops():
    """Return {sample_id(int): hop_count}.

    run_parallel assigns sample_id = enumerate index over the *prepared* samples
    (which prepare_musique already shuffled with seed=42). So sample_id i maps to
    the i-th row of musique_test.json — align by position, not by the string id.
    """
    data = json.loads(MUSIQUE_PROCESSED.read_text(encoding="utf-8"))
    return {i: int(s.get("hop_count", 0)) for i, s in enumerate(data)}


def load_results(path):
    if not path.exists():
        print(f"  [warn] missing {path}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["sample_id"]: (float(r["metrics"]["em"]), float(r["metrics"]["f1"]))
            for r in data["results"]}


def paired(a_by, b_by):
    ids = sorted(set(a_by) & set(b_by))
    if not ids:
        return None
    em_a = np.array([a_by[i][0] for i in ids])
    em_b = np.array([b_by[i][0] for i in ids])
    f1_a = np.array([a_by[i][1] for i in ids])
    f1_b = np.array([b_by[i][1] for i in ids])
    n = len(ids)
    em_diff = float(em_a.mean() - em_b.mean())
    f1_diff = float(f1_a.mean() - f1_b.mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    em_ci = (float(np.percentile((em_a - em_b)[idx].mean(1), 2.5)),
             float(np.percentile((em_a - em_b)[idx].mean(1), 97.5)))
    f1_ci = (float(np.percentile((f1_a - f1_b)[idx].mean(1), 2.5)),
             float(np.percentile((f1_a - f1_b)[idx].mean(1), 97.5)))
    b = int(((em_a == 1) & (em_b == 0)).sum())
    c = int(((em_a == 0) & (em_b == 1)).sum())
    p_cc = float("nan")
    if (b + c) > 0:
        chi2_cc = (abs(b - c) - 1) ** 2 / (b + c)
        if chi2 is not None:
            p_cc = float(chi2.sf(chi2_cc, 1))
        else:
            from math import erf, sqrt
            p_cc = 2 * (1 - 0.5 * (1 + erf(sqrt(chi2_cc) / sqrt(2))))
    return dict(n=n, em_a=float(em_a.mean()), em_b=float(em_b.mean()),
                f1_a=float(f1_a.mean()), f1_b=float(f1_b.mean()),
                em_diff=em_diff, em_ci=em_ci, f1_diff=f1_diff, f1_ci=f1_ci,
                p_cc=p_cc)


def main():
    hops = load_hops()
    cot = load_results(FILES["cot_sc"])
    cv2 = load_results(FILES["cv2"])
    if not cot or not cv2:
        print("结果文件不完整，先跑 musique n=500。")
        return

    common = sorted(set(cot) & set(cv2) & set(hops))
    print(f"三表交集样本数: {len(common)} (cot={len(cot)} cv2={len(cv2)} hops={len(hops)})")

    # 按 hop 分组
    by_hop = {2: [], 3: [], 4: []}
    for sid in common:
        h = hops.get(sid)
        if h in by_hop:
            by_hop[h].append(sid)

    print("\n" + "=" * 95)
    print("HOP-AXIS CURVE  (diff = GERS-CV2 minus CoT-SC; positive => CV2 wins)")
    print("MuSiQue, qwen3-8b, full context (context_full).")
    print("=" * 95)
    print(f"{'hop':>4} {'n':>5} {'CoT EM':>8} {'CV2 EM':>8} {'dEM':>7} {'EM CI':>20} "
          f"{'CoT F1':>8} {'CV2 F1':>8} {'dF1':>7} {'McNemar p':>10}")
    print("-" * 95)

    hop_rows = []
    for h in (2, 3, 4):
        ids = by_hop[h]
        if not ids:
            continue
        a_by = {i: cv2[i] for i in ids}  # CV2 = a
        b_by = {i: cot[i] for i in ids}  # CoT = b
        st = paired(a_by, b_by)
        st["hop"] = h
        hop_rows.append(st)
        print(f"{h:>4} {st['n']:>5} {st['em_b']:>8.3f} {st['em_a']:>8.3f} "
              f"{st['em_diff']:>+7.3f} [{st['em_ci'][0]:+.3f},{st['em_ci'][1]:+.3f}] "
              f"{st['f1_b']:>8.3f} {st['f1_a']:>8.3f} {st['f1_diff']:>+7.3f} "
              f"{st['p_cc']:>10.3f}")
    print("=" * 95)

    if len(hop_rows) < 2:
        print("跳数点不足，无法判 crossover。")
        return

    # ── 判据 1: Wall-1 前提 —— CoT-SC 是否随跳数 degrade ──
    print("\n判据 1 (Wall-1 前提): flat CoT-SC 是否随跳数 degrade?")
    cot_f1_by_hop = {r["hop"]: r["f1_b"] for r in hop_rows}
    for h in sorted(cot_f1_by_hop):
        print(f"   {h}-hop: CoT-SC F1 = {cot_f1_by_hop[h]:.3f}")
    drops = [(h1, h2, cot_f1_by_hop[h2] - cot_f1_by_hop[h1])
             for h1, h2 in zip(sorted(cot_f1_by_hop)[:-1], sorted(cot_f1_by_hop)[1:])]
    for h1, h2, d in drops:
        print(f"   {h1}-hop → {h2}-hop: ΔF1 = {d:+.3f} {'(掉)' if d < 0 else '(没掉/上升)'}")
    cot_degrades = all(d < 0 for _, _, d in drops)
    print(f"   => CoT-SC 随跳数单调下降? {'YES (Wall-1 失效, 分解有空间)' if cot_degrades else 'NO (模型单 pass 仍能处理深跳, 分解无空间)'}")

    # ── 判据 2: H2 crossover —— CV2-CoT 差距是否随跳数增大(往CV2有利方向) ──
    print("\n判据 2 (H2 crossover): CV2-CoT 差距是否随跳数往 CV2 方向变化?")
    for r in hop_rows:
        win = "CV2 wins" if r["f1_diff"] > 0 else "CoT-SC wins"
        sig = " (显著)" if (r["f1_diff"] > 0 and r["f1_ci"][0] > 0) or \
                         (r["f1_diff"] < 0 and r["f1_ci"][1] < 0) else ""
        print(f"   {r['hop']}-hop: dF1(CV2-CoT) = {r['f1_diff']:+.3f}  {win}{sig}")
    diffs = [r["f1_diff"] for r in hop_rows]
    increasing = all(d2 >= d1 for d1, d2 in zip(diffs[:-1], diffs[1:]))
    sign_flip = any(d > 0 for d in diffs) and any(d < 0 for d in diffs)
    low = hop_rows[0]
    high = hop_rows[-1]
    print(f"\n   差距随跳数单调增大(往CV2有利)? {'YES' if increasing else 'NO'}")
    print(f"   符号翻转(CV2某跳反超)? {'YES' if sign_flip else 'NO'}")
    if high["f1_diff"] > low["f1_diff"]:
        print(f"   高跳比低跳更利于 CV2: {low['f1_diff']:+.3f} → {high['f1_diff']:+.3f}")
    verdict = (increasing or sign_flip) and cot_degrades
    print(f"\n=> {'H2 CROSSOVER 成立 (分解在深跳显价值)' if verdict else 'H2 crossover 不成立'}")


if __name__ == "__main__":
    main()
