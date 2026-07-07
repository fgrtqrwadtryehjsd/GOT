# AGENTS.md

## Repo Shape
- Python package `gers` is installed from `setup.py`; source lives under `src/`, while `data/` and `experiments/` are scripts, not packaged modules.
- Core GERS flow is `src/chain_generation/generation_pipeline.py`: decompose question -> build DAG -> topological execution -> final answer -> consistency scoring/optional feedback.
- Main experiment datasets in executable scripts are `hotpotqa`, `gsm8k`, and `2wikimultihopqa`; CLUTRR is intentionally removed from active experiments.

## Setup And Verification
- Install with dev test deps: `pip install -e ".[dev]"`. `requirements.txt` is runtime-oriented and omits `pytest`.
- Run tests with `python -m pytest tests/ -v`; focused tests use `python -m pytest tests/test_graph.py -v` or another single test file.
- Current caveat: `tests/test_data_prepare.py` imports removed `_clutrr_builtin_samples`, so the full suite currently fails during collection unless that stale test is fixed or skipped.
- No lint, formatter, typecheck, pre-commit, or CI config is present; do not invent those verification steps.

## Data Commands And Gotchas
- Prepare all active datasets with `python data/prepare_data.py --dataset all --num_samples 500 --validate`.
- 2WikiMultiHopQA requires a local official file at `data/2wiki_raw/data/dev.json` or `data/2wiki_raw/data/train.json`; `prepare_2wikimultihopqa` raises `FileNotFoundError` instead of downloading from HF.
- HotpotQA prefers local `data/hotpotqa/hotpot_dev_distractor_v1.json` or `data/hotpotqa_raw/raw/hotpot_dev_distractor_v1.json`, then falls back to HuggingFace.
- CLUTRR helpers now raise `NotImplementedError`; see `docs/clutrr_changelog.md`. Do not re-add CLUTRR to new active experiment configs unless explicitly asked.

## Experiment Commands
- Quick/resumable single-method run: `python experiments/run_quick_exp.py --dataset hotpotqa --method gers_adaptive --num_samples 50`.
- Parallel runner with resume and atomic saves: `python experiments/run_parallel.py --dataset hotpotqa --methods gers_adaptive,standard_cot,cot_sc,cot_sc_gers,zero_shot --num_samples 100 --workers 4`.
- `experiments/run_parallel.py` defaults to `--workers 8`, but repo notes in `docs/experiment_records.md` say 4 workers was the stable DashScope QPM sweet spot; 6+ workers can trigger frequent 429s.
- Older `experiments/run_comparison.py` only knows `gers,standard_cot,cot_sc,tot,zero_shot`; newer methods such as `gers_adaptive`, `gers_sc`, and `cot_sc_gers` are created in `experiments/run_quick_exp.py`.
- `experiments/smoke_test.py` still defaults to removed `clutrr`; pass `--dataset gsm8k` or `--dataset hotpotqa` if using it.

## Model And Env
- Default `qwen3-*` models route through `DashScopeModel` and require `DASHSCOPE_API_KEY` in the environment or root `.env`.
- `DashScopeModel` uses OpenAI-compatible base URL `https://dashscope.aliyuncs.com/compatible-mode/v1`, disables `enable_thinking` for qwen3 models, caches one client, and has its own retry/backoff.
- GPT models require `OPENAI_API_KEY`; LLaMA API defaults to `LLAMA_API_BASE=http://localhost:11434/v1`.

## Output Hygiene
- Experiment results are written under `experiments/results/`; smoke logs and raw dataset directories are ignored by `.gitignore`, but many JSON results may already be tracked or unignored, so check `git status` before committing.
- Keep `.env` and API keys out of commits; model wrappers explicitly expect environment variables.
