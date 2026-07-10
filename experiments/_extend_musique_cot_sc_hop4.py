"""Extend musique CoT-SC results to cover first 200 4-hop samples.

Currently `experiments/results/musique_n500_8b/musique_cot_sc_results.json` covers
sample_id 0..499 (which contains 85 4-hop samples). To do the paired test with
Oracle-1 at n=200 (§1.13 needs this), we need CoT-SC on the *next 115* 4-hop
samples (global sample_ids >= 500 in musique_test.json).

Strategy: run CoT-SC only on the specific 115 sample_ids (avoids running ~1000
irrelevant non-4-hop samples). Append to the existing results file preserving
sample_id alignment.

Usage: python experiments/_extend_musique_cot_sc_hop4.py
"""
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
from src.utils.answer_normalizer import normalize_musique_answer
from experiments.run_comparison import create_model
from experiments.run_quick_exp import create_method


RESULT_FILE = Path("experiments/results/musique_n500_8b/musique_cot_sc_results.json")
MUSIQUE_JSON = Path("data/processed/musique_test.json")
DATASET = "musique"
METHOD = "cot_sc"
WORKERS = 4
TIMEOUT = 300


def run_one(model, sample, sid, method_name):
    method = create_method(method_name, model, dataset=DATASET)
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

    pred = "" if (error or result is None) else normalize_musique_answer(result.get("answer", ""))
    metrics = Metrics.compute_all(
        prediction=pred, reference=normalize_musique_answer(sample["answer"]),
        token_count=model.count_tokens(result.get("reasoning_text", "") if result else ""),
        latency=latency, dataset=DATASET,
    )
    return {
        "sample_id": sid,
        "question": sample["question"][:100],
        "prediction": pred,
        "reference": sample["answer"],
        "metrics": metrics,
        "method": method_name,
        "error": error,
        "reasoning_text": (result or {}).get("reasoning_text", "")[:800],
    }


def main():
    # Load all musique samples
    all_samples = json.loads(MUSIQUE_JSON.read_text(encoding="utf-8"))
    print(f"[data] total musique samples: {len(all_samples)}")

    # Find the first 200 4-hop global indices
    hop4_ids = [i for i, s in enumerate(all_samples) if s.get("hop_count") == 4]
    target_hop4 = hop4_ids[:200]
    print(f"[data] first 200 4-hop global indices: min={target_hop4[0]}, max={target_hop4[-1]}")

    # Load existing results
    existing_data = {"results": []}
    if RESULT_FILE.exists():
        existing_data = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    existing_ids = {r["sample_id"] for r in existing_data["results"]}
    print(f"[data] existing CoT-SC results: {len(existing_ids)} sample_ids "
          f"(range {min(existing_ids) if existing_ids else '-'}-{max(existing_ids) if existing_ids else '-'})")

    # Filter to only the 4-hop samples NOT yet computed
    todo = [(sid, all_samples[sid]) for sid in target_hop4 if sid not in existing_ids]
    print(f"[plan] need to compute {len(todo)} new 4-hop samples")
    if not todo:
        print("  nothing to do (all target 4-hop samples already in results).")
        return

    # Run
    model = create_model("qwen3-8b")
    print(f"[model] qwen3-8b, workers={WORKERS}, timeout={TIMEOUT}s")

    lock = threading.Lock()
    results = list(existing_data["results"])
    completed = [0]
    t0 = time.time()
    todo_sids = {sid for sid, _ in todo}

    def save_progress_nolock():
        """Save without acquiring lock (caller must already hold it)."""
        out = dict(existing_data)
        out["results"] = sorted(results, key=lambda r: r["sample_id"])
        # Write atomically to avoid corruption on crash
        tmp_path = RESULT_FILE.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(RESULT_FILE)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, model, s, sid, METHOD): sid for sid, s in todo}
        for fut in as_completed(futs):
            try:
                r = fut.result(timeout=TIMEOUT)
            except Exception as e:
                sid = futs[fut]
                r = {"sample_id": sid, "prediction": "", "reference": "",
                     "metrics": {"em": 0, "f1": 0, "latency": 0}, "method": METHOD,
                     "error": f"future timeout: {str(e)[:80]}", "reasoning_text": ""}
            with lock:
                results.append(r)
                completed[0] += 1
                # Save every 5 iterations (frequent snapshotting to avoid data loss)
                if completed[0] % 5 == 0 or completed[0] == len(todo):
                    finished = [x for x in results if x["sample_id"] in todo_sids
                                and not x.get("error")]
                    if finished:
                        avg_em = sum(x["metrics"]["em"] for x in finished) / len(finished)
                        avg_f1 = sum(x["metrics"]["f1"] for x in finished) / len(finished)
                    else:
                        avg_em = avg_f1 = 0
                    err = sum(1 for x in results if x["sample_id"] in todo_sids
                              and x.get("error"))
                    print(f"  [{completed[0]}/{len(todo)}] elapsed {time.time()-t0:.0f}s | "
                          f"cumulative new-4hop EM={avg_em:.3f} F1={avg_f1:.3f} | err={err}",
                          flush=True)
                    save_progress_nolock()

    with lock:
        save_progress_nolock()

    # Report new 4-hop metrics
    new_ids = {sid for sid, _ in todo}
    new_results = [r for r in results if r["sample_id"] in new_ids and not r.get("error")]
    all_hop4_results = [r for r in results if r["sample_id"] in set(target_hop4) and not r.get("error")]
    if new_results:
        avg_em = sum(r["metrics"]["em"] for r in new_results) / len(new_results)
        avg_f1 = sum(r["metrics"]["f1"] for r in new_results) / len(new_results)
        print(f"\n[NEW 4-hop {len(new_results)} samples] EM={avg_em:.4f} F1={avg_f1:.4f}")
    if all_hop4_results:
        avg_em = sum(r["metrics"]["em"] for r in all_hop4_results) / len(all_hop4_results)
        avg_f1 = sum(r["metrics"]["f1"] for r in all_hop4_results) / len(all_hop4_results)
        print(f"[ALL 4-hop first 200: {len(all_hop4_results)} valid] EM={avg_em:.4f} F1={avg_f1:.4f}")
    print(f"\nTotal file entries: {len(results)}")
    print(f"Saved: {RESULT_FILE}")


if __name__ == "__main__":
    main()
