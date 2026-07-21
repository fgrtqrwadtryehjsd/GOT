# Anonymous Code and Data Supplement

This archive accompanies the anonymous submission **“When Does Graph
Decomposition Help Multi-Hop QA? Cross-Benchmark Evidence and Sub-Answer
Diagnostics.”** It contains the implementation of GERS-DAG and BiCheck,
experiment runners, selected processed evaluation data, per-sample outputs, and
scripts for recomputing the principal statistics reported in the paper.

The archive contains no API keys, repository history, author metadata, local
absolute paths, or external code/data repository links.

## Fastest verification (no API access required)

Use Python 3.9 or newer. Install the analysis dependencies and recompute the
main numerical claims from the included per-sample outputs:

```bash
python -m pip install numpy scipy
python experiments/reproduce_paper_results.py
```

The script reports:

- paired results for all five LongBench subsets;
- the full-context HotpotQA boundary result;
- the MuSiQue 4-hop Oracle interventions and paired comparisons;
- structural-CS and BiCheck correct/wrong separation and AUROC;
- the limited Qwen-Plus comparison.

Bootstrap confidence intervals use 10,000 paired resamples with seed 42.
McNemar tests use the chi-square statistic with Edwards' continuity correction.

## Environment for rerunning model inference

Install the full dependencies with:

```bash
python -m pip install -r requirements.txt
```

The reported Qwen3-8B and Qwen-Plus experiments used the DashScope
OpenAI-compatible endpoint with `enable_thinking=False`. Set the credential only
in the process environment; do not write it into the archive:

```bash
export DASHSCOPE_API_KEY=<your-key>
```

On PowerShell:

```powershell
$env:DASHSCOPE_API_KEY = "<your-key>"
```

API inference is nondeterministic and incurs provider cost. The released
per-sample outputs are therefore included so that all reported statistics can be
checked without repeating paid inference.

## Principal rerun commands

LongBench uses the full `context_full` field. The stable API configuration used
four workers.

```bash
python experiments/run_parallel.py --dataset longbench_multifieldqa_en --methods cot_sc,gers_cv2_fullctx --model qwen3-8b --num_samples 100 --workers 4 --context_field context_full --output_dir experiments/results/longbench_multifieldqa_en_8b

python experiments/run_parallel.py --dataset longbench_musique --methods cot_sc,gers_cv2_fullctx --model qwen3-8b --num_samples 200 --workers 4 --context_field context_full --output_dir experiments/results/longbench_musique_8b

python experiments/run_parallel.py --dataset hotpotqa --methods cot_sc,gers_cv2_fullctx --model qwen3-8b --num_samples 500 --workers 4 --context_field context_full --output_dir experiments/results/n500_fullctx_8b

python experiments/run_oracle.py --dataset musique --hop 4 --num_samples 200 --model qwen3-8b --workers 4
```

Equivalent LongBench commands can be run for `longbench_narrativeqa`,
`longbench_qasper`, and `longbench_2wikimqa` with their included evaluation
sizes. Existing result files are resumable by default.

## Tests

The core graph implementation can be checked without an API key:

```bash
python -m pytest tests/test_graph.py -q
```

## Archive layout

```text
src/                         GERS-DAG, BiCheck, model adapters, baselines, metrics
data/prepare_data.py         benchmark preparation and validation
data/download_longbench.py   LongBench preparation helper
data/processed/              processed evaluation subsets used in the paper
experiments/run_parallel.py  paired/resumable experiment runner
experiments/run_oracle.py    MuSiQue privileged-information interventions
experiments/reproduce_paper_results.py
                             no-API statistics reproducer
experiments/results/         selected per-sample outputs behind paper claims
tests/                       core implementation tests
```

The processed benchmark examples remain subject to the terms of their original
datasets. They are included only to make the anonymous review artifact
self-contained and to preserve the exact evaluated sample order.

## Method/configuration mapping

The publication-facing name **GERS-DAG** corresponds to the released
`gers_cv2_fullctx` configuration for the main Qwen3-8B comparisons. In that
configuration, GERS-DAG produces the answer through topological sub-question
execution and confidence-aware aggregation. BiCheck then computes a post-hoc
forward/re-derived sub-answer agreement score; it does not change the already
fixed answer in the evaluated single-path protocol.

The result files retain development-time configuration names to preserve exact
provenance. The paper reports BiCheck diagnostic statistics from
`hotpotqa_gers_adaptive_cv2_results.json` and the structural-score baseline from
`hotpotqa_gers_sc_results.json`.
