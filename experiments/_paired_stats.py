"""Paired significance tests on existing per-sample HotpotQA results.

Self-audit + new comparison:
  - Reproduce the three rows already in the paper (CV2 vs CoT-SC, CV2 vs StdCoT,
    GERS-SC vs CoT-SC) to confirm methodology consistency.
  - Add the missing GERS-CV2 vs MoDeGraph-style (v4) paired test (review risk #4).

Methodology:
  - Align samples by sample_id.
  - Paired bootstrap 95% CI: B=10000 resamples of per-sample differences
    (seed=42), percentile method, on EM (0/1) and F1 (continuous).
  - McNemar on EM correctness: exact two-sided binomial test on discordant pairs
    (scipy.stats.binomtest(b, b+c, 0.5)).

Run: python experiments/_paired_stats.py
"""
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import binomtest
except ImportError:  # pragma: no cover
    binomtest = None

R = Path("experiments/results")
B = 10000
SEED = 42

FILES = {
    "cv2": R / "hotpotqa_gers_adaptive_cv2_results.json",
    "gers_sc": R / "hotpotqa_gers_sc_results.json",
    "cot_sc": R / "hotpotqa_cot_sc_results.json",
    "std_cot": R / "hotpotqa_standard_cot_results.json",
    "modegraph": R / "graph_baseline_v4" / "hotpotqa_modegraph_results.json",
}


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {}
    for r in data["results"]:
        by_id[r["sample_id"]] = (float(r["metrics"]["em"]), float(r["metrics"]["f1"]))
    return by_id


def paired_test(a_by, b_by, name_a, name_b):
    ids = sorted(set(a_by) & set(b_by))
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
    em_boot = em_d[idx].mean(axis=1)
    f1_boot = f1_d[idx].mean(axis=1)
    em_ci = (float(np.percentile(em_boot, 2.5)), float(np.percentile(em_boot, 97.5)))
    f1_ci = (float(np.percentile(f1_boot, 2.5)), float(np.percentile(f1_boot, 97.5)))

    # McNemar on EM correctness discordance
    both_correct = int(((em_a == 1) & (em_b == 1)).sum())
    both_wrong = int(((em_a == 0) & (em_b == 0)).sum())
    b = int(((em_a == 1) & (em_b == 0)).sum())  # a correct, b wrong
    c = int(((em_a == 0) & (em_b == 1)).sum())  # a wrong, b correct
    p_exact = p_cc = float("nan")
    if binomtest is not None and (b + c) > 0:
        p_exact = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    if (b + c) > 0:
        # chi-square with Edwards' continuity correction -- matches the values
        # already used in the paper (0.029 / 0.040 / 0.275).
        chi2_cc = (abs(b - c) - 1) ** 2 / (b + c)
        try:
            from scipy.stats import chi2
            p_cc = float(chi2.sf(chi2_cc, 1))
        except ImportError:
            from math import erf, sqrt
            p_cc = 2 * (1 - 0.5 * (1 + erf(sqrt(chi2_cc) / sqrt(2))))

    print(f"\n{name_a} vs {name_b}  (n={n})")
    print(f"  EM: {em_a.mean():.3f} vs {em_b.mean():.3f}  diff={em_diff:+.3f}  "
          f"95% CI [{em_ci[0]:+.3f}, {em_ci[1]:+.3f}]  excludes_zero={em_ci[0] > 0}")
    print(f"  F1: {f1_a.mean():.3f} vs {f1_b.mean():.3f}  diff={f1_diff:+.3f}  "
          f"95% CI [{f1_ci[0]:+.3f}, {f1_ci[1]:+.3f}]  excludes_zero={f1_ci[0] > 0}")
    print(f"  McNemar (EM): b={b} c={c}  p(exact binom)={p_exact:.3f}  p(chi2-cc)={p_cc:.3f}")
    return dict(n=n, em_diff=em_diff, em_ci=em_ci, em_excludes_zero=em_ci[0] > 0,
                f1_diff=f1_diff, f1_ci=f1_ci, f1_excludes_zero=f1_ci[0] > 0,
                mcnemar_p_cc=p_cc, mcnemar_p_exact=p_exact, b=b, c=c)


def main():
    data = {k: load(v) for k, v in FILES.items()}
    print("Loaded:", {k: len(v) for k, v in data.items()})

    print("\n=== Self-audit: reproduce existing paper rows (PAIRED bootstrap) ===")
    paired_test(data["cv2"], data["cot_sc"], "GERS-CV2", "CoT-SC")
    paired_test(data["cv2"], data["std_cot"], "GERS-CV2", "StdCoT")
    paired_test(data["gers_sc"], data["cot_sc"], "GERS-SC", "CoT-SC")

    print("\n=== NEW: GERS-CV2 vs MoDeGraph-style (v4) ===")
    paired_test(data["cv2"], data["modegraph"], "GERS-CV2", "MoDeGraph")

    # ---- Diagnose the CI discrepancy: was the paper's CI unpaired? ----
    print("\n=== Diagnosis: UNPAIRED bootstrap (resample each method independently) ===")
    print("(paper reports 'paired' CI [-0.014,+0.096] for CV2 vs CoT-SC EM)")
    ids = sorted(set(data["cv2"]) & set(data["cot_sc"]))
    em_a = np.array([data["cv2"][i][0] for i in ids])
    em_b = np.array([data["cot_sc"][i][0] for i in ids])
    n = len(ids)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(B):
        ia = rng.integers(0, n, size=n)
        ib = rng.integers(0, n, size=n)
        boot.append(em_a[ia].mean() - em_b[ib].mean())
    boot = np.array(boot)
    print(f"  UNPAIRED EM CI: [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}]")

    print("\n=== Paired bootstrap robustness across seeds (CV2 vs CoT-SC EM) ===")
    em_d = em_a - em_b
    for s in (0, 1, 42, 123, 2024):
        r = np.random.default_rng(s)
        idx = r.integers(0, n, size=(B, n))
        ci = (np.percentile(em_d[idx].mean(1), 2.5), np.percentile(em_d[idx].mean(1), 97.5))
        print(f"  seed={s:5d}: paired EM CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  excludes_zero={ci[0] > 0}")


if __name__ == "__main__":
    main()
