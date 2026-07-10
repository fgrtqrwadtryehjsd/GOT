"""Budget-curve analysis: does a stable CoT-SC/GERS-CV2 crossover exist?

Reads the symmetric budget sweep under experiments/results/budget_curve_8b/ctx{TAG}/
(where TAG in {800,1500,2000,2500,3000,4000,full}). At each budget BOTH methods saw
the SAME first-N chars (CoT-SC via --context_budget, CV2 via gers_cv2_fullctx with
context_char_limit=8000 so it does not re-truncate). This is the fair test of whether
decomposition's apparent advantage under context scarcity is a real, stable crossover
or an artifact of asymmetric truncation.

Outputs:
  - Curve table: budget | CoT-SC EM/F1 | CV2 EM/F1 | diff(CV2-CoT) | paired CI | McNemar p
  - Crossover verdict: sign-change in diff, and whether the low-budget CV2 lead and
    high-budget CoT lead are each individually significant.

Run: python experiments/_budget_curve.py
"""
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import binomtest
except ImportError:  # pragma: no cover
    binomtest = None

BASE = Path("experiments/results/budget_curve_8b")
B = 10000
SEED = 42

# (tag, numeric budget for sorting; "full" sorts last)
TAGS = [("800", 800), ("1500", 1500), ("2000", 2000),
        ("2500", 2500), ("3000", 3000), ("4000", 4000), ("full", 10_000)]


def load(tag):
    by_method = {}
    for method, fname in [("cot_sc", "hotpotqa_cot_sc_results.json"),
                          ("cv2", "hotpotqa_gers_cv2_fullctx_results.json")]:
        p = BASE / f"ctx{tag}" / fname
        if not p.exists():
            print(f"  [warn] missing {p}")
            by_method[method] = {}
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        by_id = {r["sample_id"]: (float(r["metrics"]["em"]), float(r["metrics"]["f1"]))
                 for r in data["results"]}
        by_method[method] = by_id
    return by_method


def paired(a_by, b_by, name_a, name_b):
    """a vs b, returns stats. diff = a - b (so CV2-CoT when a=cv2)."""
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
    em_d = em_a - em_b
    f1_d = f1_a - f1_b
    idx = rng.integers(0, n, size=(B, n))
    em_ci = (float(np.percentile(em_d[idx].mean(1), 2.5)),
             float(np.percentile(em_d[idx].mean(1), 97.5)))
    f1_ci = (float(np.percentile(f1_d[idx].mean(1), 2.5)),
             float(np.percentile(f1_d[idx].mean(1), 97.5)))

    b = int(((em_a == 1) & (em_b == 0)).sum())  # a correct, b wrong
    c = int(((em_a == 0) & (em_b == 1)).sum())  # a wrong, b correct
    p_cc = float("nan")
    if (b + c) > 0:
        chi2_cc = (abs(b - c) - 1) ** 2 / (b + c)
        try:
            from scipy.stats import chi2
            p_cc = float(chi2.sf(chi2_cc, 1))
        except ImportError:
            from math import erf, sqrt
            p_cc = 2 * (1 - 0.5 * (1 + erf(sqrt(chi2_cc) / sqrt(2))))

    return dict(n=n, em_a=float(em_a.mean()), em_b=float(em_b.mean()),
                f1_a=float(f1_a.mean()), f1_b=float(f1_b.mean()),
                em_diff=em_diff, em_ci=em_ci, f1_diff=f1_diff, f1_ci=f1_ci,
                p_cc=p_cc, b=b, c=c)


def main():
    rows = []
    for tag, _ in TAGS:
        m = load(tag)
        if not m.get("cot_sc") or not m.get("cv2"):
            print(f"\n[ctx{tag}] incomplete, skipping")
            continue
        # diff = CV2 - CoT-SC (positive => CV2 wins)
        st = paired(m["cv2"], m["cot_sc"], "CV2", "CoT-SC")
        if st is None:
            continue
        st["tag"] = tag
        rows.append(st)

    if not rows:
        print("No complete budget points found. Run the sweep first.")
        return

    print("\n" + "=" * 92)
    print("BUDGET CURVE  (diff = GERS-CV2 minus CoT-SC; positive => CV2 wins)")
    print("Both methods see the SAME first-N chars at each budget (symmetric).")
    print("=" * 92)
    print(f"{'budget':>8} {'CoT EM':>8} {'CV2 EM':>8} {'dEM':>7} {'EM CI':>20} "
          f"{'CoT F1':>8} {'CV2 F1':>8} {'dF1':>7} {'McNemar p':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r['tag']:>8} {r['em_b']:>8.3f} {r['em_a']:>8.3f} "
              f"{r['em_diff']:>+7.3f} [{r['em_ci'][0]:+.3f},{r['em_ci'][1]:+.3f}] "
              f"{r['f1_b']:>8.3f} {r['f1_a']:>8.3f} {r['f1_diff']:>+7.3f} "
              f"{r['p_cc']:>10.3f}")
    print("=" * 92)

    # Crossover verdict
    print("\nCROSSOVER VERDICT")
    print("-" * 60)
    f1_diffs = [(r["tag"], r["f1_diff"], r["f1_ci"]) for r in rows]
    em_diffs = [(r["tag"], r["em_diff"], r["em_ci"]) for r in rows]

    # sign of F1 diff across budgets (CV2 - CoT)
    signs = [(t, d) for t, d, _ in f1_diffs]
    print("F1 diff (CV2-CoT) by budget:")
    for t, d in signs:
        print(f"   ctx{t:>6}: {d:+.3f}  {'CV2 wins' if d > 0 else 'CoT-SC wins'}")

    # does sign flip?
    pos = [t for t, d in signs if d > 0]
    neg = [t for t, d in signs if d < 0]
    flips = bool(pos) and bool(neg)

    # low-budget CV2 lead significant? high-budget CoT lead significant?
    low = rows[0]
    high = rows[-1]
    low_cv2_sig = low["f1_diff"] > 0 and low["f1_ci"][0] > 0
    high_cot_sig = high["f1_diff"] < 0 and high["f1_ci"][1] < 0

    print(f"\nLowest budget (ctx{low['tag']}): CV2-CoT F1 diff = {low['f1_diff']:+.3f}, "
          f"CI [{low['f1_ci'][0]:+.3f},{low['f1_ci'][1]:+.3f}]  "
          f"=> {'CV2 SIGNIFICANTLY leads' if low_cv2_sig else 'NOT significant CV2 lead'}")
    print(f"Highest budget (ctx{high['tag']}): CV2-CoT F1 diff = {high['f1_diff']:+.3f}, "
          f"CI [{high['f1_ci'][0]:+.3f},{high['f1_ci'][1]:+.3f}]  "
          f"=> {'CoT-SC SIGNIFICANTLY leads' if high_cot_sig else 'NOT significant CoT lead'}")

    print("\n=> STABLE CROSSOVER" if (flips and low_cv2_sig and high_cot_sig)
          else "\n=> NO stable crossover (sign does not robustly flip with significance)")


if __name__ == "__main__":
    main()
