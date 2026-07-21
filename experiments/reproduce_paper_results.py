"""Reproduce the principal numerical claims from released per-sample results.

This script does not call an LLM. It reads the result JSON files included in the
AAAI code-and-data supplement and recomputes paired means, percentile bootstrap
confidence intervals, McNemar p-values, and BiCheck diagnostic statistics.

Run from the archive root:

    python experiments/reproduce_paper_results.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
BOOTSTRAP_SAMPLES = 10_000
SEED = 42


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(path: Path) -> Dict[int, dict]:
    payload = read_json(path)
    rows = payload.get("results", payload if isinstance(payload, list) else [])
    return {
        int(row["sample_id"]): row
        for row in rows
        if not row.get("error") and "metrics" in row
    }


def metric_arrays(rows: Mapping[int, dict], ids: Iterable[int], metric: str) -> np.ndarray:
    return np.asarray([float(rows[i]["metrics"][metric]) for i in ids], dtype=float)


def mcnemar_cc_pvalue(em_a: np.ndarray, em_b: np.ndarray) -> float:
    a_wins = int(((em_a == 1) & (em_b == 0)).sum())
    b_wins = int(((em_a == 0) & (em_b == 1)).sum())
    discordant = a_wins + b_wins
    if discordant == 0:
        return 1.0
    chi_square = (abs(a_wins - b_wins) - 1) ** 2 / discordant
    return math.erfc(math.sqrt(chi_square / 2.0))


def paired_statistics(a: Mapping[int, dict], b: Mapping[int, dict]) -> dict:
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("No valid paired samples")

    em_a = metric_arrays(a, ids, "em")
    em_b = metric_arrays(b, ids, "em")
    f1_a = metric_arrays(a, ids, "f1")
    f1_b = metric_arrays(b, ids, "f1")

    rng = np.random.default_rng(SEED)
    bootstrap_ids = rng.integers(0, len(ids), size=(BOOTSTRAP_SAMPLES, len(ids)))

    def summarize(x: np.ndarray, y: np.ndarray) -> tuple[float, tuple[float, float]]:
        differences = x - y
        bootstrap = differences[bootstrap_ids].mean(axis=1)
        ci = tuple(float(v) for v in np.percentile(bootstrap, [2.5, 97.5]))
        return float(differences.mean()), ci

    em_diff, em_ci = summarize(em_a, em_b)
    f1_diff, f1_ci = summarize(f1_a, f1_b)
    return {
        "n": len(ids),
        "em_a": float(em_a.mean()),
        "em_b": float(em_b.mean()),
        "em_diff": em_diff,
        "em_ci": em_ci,
        "f1_a": float(f1_a.mean()),
        "f1_b": float(f1_b.mean()),
        "f1_diff": f1_diff,
        "f1_ci": f1_ci,
        "mcnemar_p": mcnemar_cc_pvalue(em_a, em_b),
    }


def print_pair(label: str, stats: dict, a_name: str, b_name: str) -> None:
    print(f"{label} (n={stats['n']})")
    print(
        f"  EM  {a_name}={stats['em_a']:.3f}, {b_name}={stats['em_b']:.3f}, "
        f"delta={stats['em_diff']:+.3f}, "
        f"95% CI=[{stats['em_ci'][0]:+.3f},{stats['em_ci'][1]:+.3f}], "
        f"McNemar p={stats['mcnemar_p']:.3f}"
    )
    print(
        f"  F1  {a_name}={stats['f1_a']:.3f}, {b_name}={stats['f1_b']:.3f}, "
        f"delta={stats['f1_diff']:+.3f}, "
        f"95% CI=[{stats['f1_ci'][0]:+.3f},{stats['f1_ci'][1]:+.3f}]"
    )


def longbench_results() -> None:
    print("\n=== Five LongBench subsets: GERS-DAG minus CoT-SC ===")
    for subset in ("multifieldqa_en", "musique", "narrativeqa", "qasper", "2wikimqa"):
        directory = RESULTS / f"longbench_{subset}_8b"
        gers = load_results(directory / f"longbench_{subset}_gers_cv2_fullctx_results.json")
        cot = load_results(directory / f"longbench_{subset}_cot_sc_results.json")
        print_pair(subset, paired_statistics(gers, cot), "GERS", "CoT-SC")


def hotpot_boundary() -> None:
    print("\n=== Full-context HotpotQA boundary: GERS-DAG minus CoT-SC ===")
    directory = RESULTS / "n500_fullctx_8b"
    gers = load_results(directory / "hotpotqa_gers_cv2_fullctx_results.json")
    cot = load_results(directory / "hotpotqa_cot_sc_results.json")
    print_pair("HotpotQA", paired_statistics(gers, cot), "GERS", "CoT-SC")


def oracle_results() -> None:
    print("\n=== MuSiQue 4-hop Oracle interventions ===")
    directory = RESULTS / "oracle_musique_8b"
    files = {
        "model decomposition": "musique_oracle_baseline_results.json",
        "gold decomposition": "musique_oracle_o1_dag_results.json",
        "gold decomposition + retrieval": "musique_oracle_o12_dag_ret_results.json",
        "gold decomposition + sub-answers": "musique_oracle_o13_dag_ans_results.json",
    }
    rows = {name: load_results(directory / filename) for name, filename in files.items()}
    for name, result_rows in rows.items():
        ids = sorted(result_rows)
        em = metric_arrays(result_rows, ids, "em").mean()
        f1 = metric_arrays(result_rows, ids, "f1").mean()
        print(f"{name:<36} n={len(ids):>3}  EM={em:.3f}  F1={f1:.3f}")

    stats = paired_statistics(rows["gold decomposition"], rows["model decomposition"])
    print_pair("gold decomposition minus model decomposition", stats, "gold", "model")

    musique = read_json(ROOT / "data" / "processed" / "musique_test.json")
    hop4_global_ids = [i for i, row in enumerate(musique) if int(row.get("hop_count", 0)) == 4]
    cot_global = load_results(RESULTS / "musique_n500_8b" / "musique_cot_sc_results.json")
    cot_local = {
        local_id: cot_global[global_id]
        for local_id, global_id in enumerate(hop4_global_ids[:200])
        if global_id in cot_global
    }
    print_pair(
        "gold decomposition minus CoT-SC",
        paired_statistics(rows["gold decomposition"], cot_local),
        "gold",
        "CoT-SC",
    )


def empirical_auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def diagnostic_summary(path: Path) -> dict:
    rows = load_results(path)
    ordered = [rows[i] for i in sorted(rows)]
    labels = [int(float(row["metrics"]["em"]) == 1.0) for row in ordered]
    scores = [float(row["metrics"]["consistency_score"]) for row in ordered]
    correct = [score for score, label in zip(scores, labels) if label]
    wrong = [score for score, label in zip(scores, labels) if not label]
    correct_mean = float(np.mean(correct))
    wrong_mean = float(np.mean(wrong))
    return {
        "n": len(rows),
        "correct_mean": correct_mean,
        "wrong_mean": wrong_mean,
        "separation": correct_mean - wrong_mean,
        "auroc": empirical_auc(labels, scores),
        "perfect": sum(score == 1.0 for score in scores),
        "wrong_perfect": sum(score == 1.0 and label == 0 for score, label in zip(scores, labels)),
    }


def bicheck_results() -> None:
    print("\n=== HotpotQA consistency diagnostics ===")
    variants = {
        "structural CS": RESULTS / "hotpotqa_gers_sc_results.json",
        "BiCheck CS": RESULTS / "hotpotqa_gers_adaptive_cv2_results.json",
    }
    for name, path in variants.items():
        stats = diagnostic_summary(path)
        print(
            f"{name:<14} n={stats['n']}  correct={stats['correct_mean']:.4f}  "
            f"wrong={stats['wrong_mean']:.4f}  separation={stats['separation']:+.4f}  "
            f"AUROC={stats['auroc']:.3f}"
        )
        if name == "BiCheck CS":
            print(
                f"  perfect-score errors: {stats['wrong_perfect']} wrong among "
                f"{stats['perfect']} samples with CS=1.0"
            )


def qwen_plus_results() -> None:
    print("\n=== Limited Qwen-Plus evidence: GERS-DAG minus CoT-SC ===")
    directory = RESULTS / "p2_qwenplus"
    gers = load_results(directory / "hotpotqa_gers_adaptive_cv2_results.json")
    cot = load_results(directory / "hotpotqa_cot_sc_results.json")
    print_pair("HotpotQA / Qwen-Plus", paired_statistics(gers, cot), "GERS", "CoT-SC")


def main() -> None:
    print(
        f"Paired percentile bootstrap: B={BOOTSTRAP_SAMPLES}, seed={SEED}; "
        "McNemar uses Edwards' continuity correction."
    )
    longbench_results()
    hotpot_boundary()
    oracle_results()
    bicheck_results()
    qwen_plus_results()


if __name__ == "__main__":
    main()
