"""Paired significance tests on LongBench multifieldqa_en / musique / narrativeqa.

Tests CV2-fullctx vs CoT-SC on the 3 LongBench subsets to determine whether
the +0.070 / +0.081 F1 gains observed in §1.12 are statistically significant.

Also reruns the §1.10 UPDATE Oracle-1 vs CoT-SC 4-hop paired test at n=200
(previously n=85 was inconclusive with CI [-0.024, +0.188]).

Methodology (matches _paired_stats.py):
  - Align samples by sample_id.
  - Paired bootstrap 95% CI: B=10000 resamples of per-sample differences (seed=42).
  - McNemar with Edwards' continuity correction on EM correctness discordance.

Run: python experiments/_paired_stats_longbench.py
"""
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import binomtest, chi2
except ImportError:
    binomtest = None
    chi2 = None

R = Path("experiments/results")
B = 10000
SEED = 42


def load(path):
    p = Path(path)
    if not p.exists():
        print(f"  [warn] missing: {p}")
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    results = data.get("results", data if isinstance(data, list) else [])
    by_id = {}
    for r in results:
        # skip samples with errors
        if r.get("error"):
            continue
        by_id[r["sample_id"]] = (float(r["metrics"]["em"]), float(r["metrics"]["f1"]))
    return by_id


def paired_test(a_by, b_by, name_a, name_b):
    ids = sorted(set(a_by) & set(b_by))
    if not ids:
        print(f"\n{name_a} vs {name_b}: NO OVERLAPPING SAMPLES")
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
    em_boot = em_d[idx].mean(axis=1)
    f1_boot = f1_d[idx].mean(axis=1)
    em_ci = (float(np.percentile(em_boot, 2.5)), float(np.percentile(em_boot, 97.5)))
    f1_ci = (float(np.percentile(f1_boot, 2.5)), float(np.percentile(f1_boot, 97.5)))
    prob_positive_em = float((em_boot > 0).mean())
    prob_positive_f1 = float((f1_boot > 0).mean())

    # McNemar on EM correctness discordance
    b_wins = int(((em_a == 1) & (em_b == 0)).sum())
    c_wins = int(((em_a == 0) & (em_b == 1)).sum())
    p_exact = p_cc = float("nan")
    if binomtest is not None and (b_wins + c_wins) > 0:
        p_exact = float(binomtest(b_wins, b_wins + c_wins, 0.5, alternative="two-sided").pvalue)
    if (b_wins + c_wins) > 0:
        chi2_cc = (abs(b_wins - c_wins) - 1) ** 2 / (b_wins + c_wins)
        if chi2 is not None:
            p_cc = float(chi2.sf(chi2_cc, 1))
        else:
            from math import erf, sqrt
            p_cc = 2 * (1 - 0.5 * (1 + erf(sqrt(chi2_cc) / sqrt(2))))

    print(f"\n{name_a} vs {name_b}  (n={n} paired)")
    print(f"  EM: {em_a.mean():.4f} vs {em_b.mean():.4f}  diff={em_diff:+.4f}")
    print(f"      95% CI [{em_ci[0]:+.4f}, {em_ci[1]:+.4f}]  "
          f"P(diff>0)={prob_positive_em:.3f}  excludes_zero={em_ci[0] > 0}")
    print(f"  F1: {f1_a.mean():.4f} vs {f1_b.mean():.4f}  diff={f1_diff:+.4f}")
    print(f"      95% CI [{f1_ci[0]:+.4f}, {f1_ci[1]:+.4f}]  "
          f"P(diff>0)={prob_positive_f1:.3f}  excludes_zero={f1_ci[0] > 0}")
    print(f"  McNemar (EM): a-wins={b_wins}  b-wins={c_wins}  "
          f"p(exact)={p_exact:.3f}  p(chi2-cc)={p_cc:.3f}")

    verdict = []
    if f1_ci[0] > 0:
        verdict.append("F1 SIGNIFICANT")
    elif f1_ci[1] < 0:
        verdict.append("F1 SIGNIFICANT (reverse)")
    else:
        verdict.append("F1 n.s.")
    if em_ci[0] > 0:
        verdict.append("EM SIGNIFICANT")
    elif em_ci[1] < 0:
        verdict.append("EM SIGNIFICANT (reverse)")
    else:
        verdict.append("EM n.s.")
    print(f"  → {' | '.join(verdict)}")

    return dict(n=n, em_diff=em_diff, em_ci=em_ci, f1_diff=f1_diff, f1_ci=f1_ci,
                mcnemar_p_cc=p_cc, b=b_wins, c=c_wins,
                prob_positive_em=prob_positive_em, prob_positive_f1=prob_positive_f1)


def main():
    print("=" * 78)
    print("SOP §1.12 LongBench H1 verification — PAIRED SIGNIFICANCE")
    print("Method: paired bootstrap 95% CI (B=10000, seed=42), McNemar chi2-cc")
    print("=" * 78)

    # LongBench 3 subsets
    subsets = ["multifieldqa_en", "musique", "narrativeqa"]
    all_summaries = {}
    for subset in subsets:
        cot_path = R / f"longbench_{subset}_8b" / f"longbench_{subset}_cot_sc_results.json"
        cv2_path = R / f"longbench_{subset}_8b" / f"longbench_{subset}_gers_cv2_fullctx_results.json"
        cot_data = load(cot_path)
        cv2_data = load(cv2_path)
        print(f"\n\n>> LongBench-{subset}")
        print(f"  CoT-SC:  {len(cot_data)} valid samples ({cot_path.name})")
        print(f"  CV2-fullctx: {len(cv2_data)} valid samples ({cv2_path.name})")
        if cot_data and cv2_data:
            summary = paired_test(cv2_data, cot_data,
                                  f"CV2-fullctx({subset})",
                                  f"CoT-SC({subset})")
            all_summaries[subset] = summary

    # Oracle §1.13 — Oracle-1 vs CoT-SC on MuSiQue 4-hop n=200
    # This is the KEY test: does gold-decomposition ceiling now beat CoT-SC significantly at n=200?
    print("\n\n" + "=" * 78)
    print("SOP §1.13 Oracle-1 vs CoT-SC on MuSiQue 4-hop — EXTENDED TO n=200")
    print("(previous §1.10 UPDATE was n=85, CI [-0.024, +0.188] crossed zero)")
    print("=" * 78)

    oracle_dir = R / "oracle_musique_8b"
    o1_path = oracle_dir / "musique_oracle_o1_dag_results.json"
    baseline_path = oracle_dir / "musique_oracle_baseline_results.json"

    o1_data = load(o1_path)
    baseline_data = load(baseline_path)
    print(f"\n  Oracle-1 (gold DAG): {len(o1_data)} valid samples")
    print(f"  Baseline (model self-decompose): {len(baseline_data)} valid samples")

    if o1_data and baseline_data:
        print("\n>> Oracle-1 vs Baseline (measures graph-generation gain)")
        paired_test(o1_data, baseline_data, "Oracle-1", "Baseline")

    # For Oracle-1 vs CoT-SC we need CoT-SC results on the SAME 4-hop samples.
    # As of 2026-07-10 the musique_cot_sc_results.json has been extended to cover
    # the first 200 4-hop samples (§1.13 extension via _extend_musique_cot_sc_hop4.py).
    cot_hop_path = R / "musique_n500_8b" / "musique_cot_sc_results.json"
    if cot_hop_path.exists():
        cot_hop_data_raw = load(cot_hop_path)
        # Filter to 4-hop: read musique_test.json to know which sample_ids are 4-hop
        musique_json = Path("data/processed/musique_test.json")
        if musique_json.exists():
            all_musique = json.loads(musique_json.read_text(encoding="utf-8"))
            # Get global indices of first N 4-hop samples (aligned with Oracle local idx)
            hop4_global_ids = [i for i, s in enumerate(all_musique) if s.get("hop_count") == 4]
            # Oracle local idx k -> 4-hop global idx hop4_global_ids[k]
            # We map Oracle results (local idx) to CoT-SC results (global idx)

            # n=85 comparison (compatible with §1.10 UPDATE)
            first_n_gids_85 = hop4_global_ids[:85]
            o1_first85 = {i: o1_data[i] for i in range(85) if i in o1_data}
            cot_first85 = {}
            for k in range(85):
                gid = hop4_global_ids[k] if k < len(hop4_global_ids) else None
                if gid is not None and gid in cot_hop_data_raw:
                    cot_first85[k] = cot_hop_data_raw[gid]

            print(f"\n  CoT-SC on 4-hop first 85: {len(cot_first85)} valid samples")
            if o1_first85 and cot_first85:
                print("\n>> Oracle-1 vs CoT-SC on 4-hop first 85 samples (comparable to §1.10 UPDATE)")
                paired_test(o1_first85, cot_first85, "Oracle-1(n=85)", "CoT-SC(n=85 hop4)")

            # NEW: n=200 comparison (the whole point of §1.13 extension)
            o1_first200 = {i: o1_data[i] for i in range(200) if i in o1_data}
            cot_first200 = {}
            for k in range(200):
                gid = hop4_global_ids[k] if k < len(hop4_global_ids) else None
                if gid is not None and gid in cot_hop_data_raw:
                    cot_first200[k] = cot_hop_data_raw[gid]

            print(f"\n  CoT-SC on 4-hop first 200: {len(cot_first200)} valid samples")
            if o1_first200 and cot_first200:
                print("\n>> Oracle-1 vs CoT-SC on 4-hop first 200 samples (DEFINITIVE §1.13 test)")
                paired_test(o1_first200, cot_first200, "Oracle-1(n=200)", "CoT-SC(n=200 hop4)")

            # Also: CV2 vs CoT-SC on 4-hop first 200 (for the CV2 direct comparison)
            # We need CV2 4-hop results at the same 200 global idx
            cv2_hop_path = R / "musique_n500_8b" / "musique_gers_cv2_fullctx_results.json"
            if cv2_hop_path.exists():
                cv2_hop_data_raw = load(cv2_hop_path)
                cv2_first200 = {}
                for k in range(200):
                    gid = hop4_global_ids[k] if k < len(hop4_global_ids) else None
                    if gid is not None and gid in cv2_hop_data_raw:
                        cv2_first200[k] = cv2_hop_data_raw[gid]
                print(f"\n  CV2 on 4-hop first 200: {len(cv2_first200)} valid samples (using existing n=500 run)")
                if cv2_first200 and cot_first200:
                    print("\n>> CV2 vs CoT-SC on 4-hop first 200 samples (H2 crossover test extended)")
                    paired_test(cv2_first200, cot_first200, "CV2(n=200 hop4)", "CoT-SC(n=200 hop4)")
        else:
            print("  (skipping Oracle-1 vs CoT-SC — musique_test.json not found)")
    else:
        print(f"  (skipping — {cot_hop_path} not found; run hop_curve first)")

    # Print final summary table
    print("\n\n" + "=" * 78)
    print("SUMMARY — LongBench H1 verification")
    print("=" * 78)
    print(f"{'Subset':<20} {'n':>5} {'CoT F1':>8} {'CV2 F1':>8} {'dF1':>8} "
          f"{'F1 95% CI':>22} {'McN p':>8} {'Verdict':>18}")
    print("-" * 118)
    for subset, s in all_summaries.items():
        if s is None:
            continue
        f1_ci_str = f"[{s['f1_ci'][0]:+.3f},{s['f1_ci'][1]:+.3f}]"
        sig = "F1 SIG" if s['f1_ci'][0] > 0 else ("F1 SIG (rev)" if s['f1_ci'][1] < 0 else "F1 n.s.")
        print(f"{subset:<20} {s['n']:>5} "
              f"{-s['f1_diff']+0:>8.4f}"  # placeholder for CoT F1 (paired data doesn't give it directly)
              f" ------ "
              f"{s['f1_diff']:>+8.4f} {f1_ci_str:>22} {s['mcnemar_p_cc']:>8.4f} {sig:>18}")


if __name__ == "__main__":
    main()
