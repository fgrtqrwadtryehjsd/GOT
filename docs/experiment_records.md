# GERS Experiment Records

> Last updated: 2026-07-08  
> Current paper target: AAAI-style submission draft  
> Active datasets: HotpotQA, 2WikiMultiHopQA, GSM8K diagnostics  
> Deprecated: CLUTRR and all pre-fix HotpotQA numbers are historical only and must not be used as paper claims.

> ⚠️ **CRITICAL (2026-07-08): the headline HotpotQA result in §1.1 is a context-truncation artifact.** Under a fair full-context comparison (n=500, qwen3-8b), plain CoT-SC beats every GERS variant; the "GERS-CV2 > CoT-SC, +0.041 F1, p=0.029" claim reverses sign. See §1.6 for the definitive numbers and §4 (updated) for the re-positioned paper framing. §1.1 is retained as the *truncated-context* record for the confound audit only.

## 1. Current Trustworthy Results

These are the results aligned with the current implementation and paper draft. Earlier records that used the old EM substring bug, unstable baseline extraction, or CLUTRR synthetic data are intentionally excluded from the main narrative.

### 1.1 HotpotQA Main Results (n=500, qwen3-8b) — TRUNCATED-CONTEXT (artifact-prone)

> ⚠️ These numbers use the standard preprocessed `context` field (truncated to 2000 chars in `prepare_data.py`) and the GERS pipeline's internal 1500-char cap (`generation_pipeline.py` reason()). Both methods are therefore context-starved, and CoT-SC (2000 chars) is starved less than GERS-CV2 (1500 chars). The apparent GERS-CV2 gain **does not survive** a fair full-context comparison (§1.6). Treat §1.1 as the truncated-context arm of the confound audit, NOT as a paper claim.

| Method | EM | F1 | CS | Notes |
|---|---:|---:|---:|---|
| Zero-Shot | 0.276 | 0.389 | - | unified extraction |
| Standard CoT | 0.264 | 0.368 | - | unified extraction |
| CoT-SC (N=3) | 0.262 | 0.373 | - | majority voting |
| CoT-SC+GERS rerank | 0.264 | 0.372 | - | post-hoc graph rerank only |
| MoDeGraph-style prompt | 0.252 | 0.366 | - | graph-prompt baseline, v4 |
| GERS-Adaptive | 0.284 | 0.395 | 0.996 | single-path DAG execution |
| GERS-SC (K=3) | 0.282 | 0.398 | 0.662 | graph-level self-consistency with original structural CS |
| GERS-CV | 0.298 | 0.409 | 0.782 | bidirectional sub-answer cross-checking |
| GERS-CV2 | 0.302 | 0.413 | 0.777 | confidence-weighted aggregation |

Statistical interpretation:

| Comparison | EM diff | EM 95% CI | F1 diff | F1 95% CI | McNemar p |
|---|---:|---|---:|---|---:|
| GERS-CV2 vs CoT-SC | +0.040 | [+0.006, +0.074] | +0.041 | [+0.006, +0.075] | 0.029 |
| GERS-CV2 vs Standard CoT | +0.038 | [+0.004, +0.072] | +0.045 | [+0.010, +0.081] | 0.040 |
| GERS-CV2 vs MoDeGraph-style | +0.050 | [+0.014, +0.086] | +0.047 | [+0.012, +0.083] | 0.010 |
| GERS-SC vs CoT-SC | +0.020 | [-0.012, +0.052] | +0.025 | [-0.009, +0.060] | 0.275 |

Conclusion: bidirectional cross-checking improves the graph consistency signal and yields a small but statistically significant HotpotQA gain. GERS-CV2 is paired-significant over CoT-SC, Standard CoT, and the MoDeGraph-style baseline (McNemar p <= 0.040; paired bootstrap CIs exclude zero), while GERS-SC without cross-checking is not significant (p=0.275, CI crosses zero) --- corroborating that the gain stems from cross-checking.

Stats correction (2026-07-08): an earlier version of these CIs was computed with an unpaired bootstrap (resampling each method independently) yet labeled "paired", producing wider zero-crossing intervals (e.g., CV2 vs CoT-SC EM [-0.014,+0.096]). The correct paired bootstrap (resampling per-sample differences, 10000 resamples, seed=42) gives [+0.006,+0.074] and is what the paper now reports. Reproducer: `experiments/_paired_stats.py`. McNemar uses chi-square with Edwards' continuity correction (matches prior 0.029/0.040/0.275); the MoDeGraph row is new (p=0.010).

Graph-prompt baseline status:

| Item | Status | Path | Notes |
|---|---|---|---|
| MoDeGraph-style v1 | deprecated | `experiments/results/graph_baseline/` | completed n=500, but the final-answer prompt did not receive the graph/reasoning/context in the stateless API call, so it underestimates the baseline and must not be cited |
| MoDeGraph-style v2 | superseded intermediate | `experiments/results/graph_baseline_v2/` | completed n=500, EM 0.216 / F1 0.328; final-answer prompt receives context, graph, and reasoning, but current code has a further concise-output prompt fix |
| MoDeGraph-style v3 | superseded | `experiments/results/graph_baseline_v3/` | same as v2 plus stricter concise final-answer instruction, but still used truncated context |
| MoDeGraph-style v4 | current | `experiments/results/graph_baseline_v4/` | completed n=500, EM 0.252 / F1 0.366; full context, graph/reasoning carried into final answer, concise final-answer instruction |

### 1.2 Consistency Score Diagnostics (HotpotQA, n=500)

| CS scheme | Correct mean | Wrong mean | Discrimination | AUROC |
|---|---:|---:|---:|---:|
| Old structural CS (GERS-SC) | 0.6592 | 0.6628 | -0.0035 | 0.498 |
| New cross-validated CS (GERS-CV) | 0.7888 | 0.7042 | +0.0847 | 0.581 |
| New cross-validated CS + confidence (GERS-CV2) | - | - | - | 0.589 |

High-score caveat for GERS-CV2:

| Bucket | Samples | EM accuracy | Mean F1 |
|---|---:|---:|---:|
| CS < 0.5 | 75 | 0.200 | 0.278 |
| 0.5 <= CS < 0.7 | 86 | 0.244 | 0.333 |
| 0.7 <= CS < 0.9 | 85 | 0.247 | 0.375 |
| CS = 1.0 | 250 | 0.376 | 0.499 |

Important limitation: CS=1.0 is not correctness. In GERS-CV2, 250/500 samples receive CS=1.0, but 156 of those are EM-wrong. The score is a weak ranking/diagnostic signal, not a verifier of factual correctness.

### 1.3 HotpotQA Type Analysis (n=500)

| Method | bridge EM (n=404) | comparison EM (n=96) |
|---|---:|---:|
| Zero-Shot | 0.218 | 0.521 |
| CoT-SC | 0.218 | 0.448 |
| GERS-CV2 | 0.240 | 0.562 |

Interpretation: the gain is concentrated on comparison/branch-merge questions. HotpotQA bridge questions often contain extractable evidence in context, so strong zero-shot reading already performs close to structured methods.

### 1.4 2WikiMultiHopQA Boundary Analysis

Small diagnostic split (n=100):

| Method | comparison | bridge_comp | compositional | inference |
|---|---:|---:|---:|---:|
| Standard CoT | 0.815 | 0.905 | 0.284 | 0.361 |
| Zero-Shot | 0.800 | 0.587 | 0.366 | 0.330 |
| CoT-SC | 0.720 | 0.864 | 0.268 | 0.343 |
| GERS-CV2 | 0.775 | 0.524 | 0.297 | 0.288 |

Larger diagnostic run (n=300): GERS-CV2 EM 0.390 / F1 0.462, below Standard CoT EM 0.443 / F1 0.521. This is a real generalization boundary, not a result to hide.

Interpretation: GERS is close on comparison-style decomposition but suffers on deep bridge-comparison questions because early entity errors propagate through the graph.

### 1.5 Second-Model Check

Qwen-Plus on HotpotQA n=300: GERS-CV2 EM 0.363 / F1 0.496; CoT-SC EM 0.367 / F1 0.494; paired difference is effectively zero. The method's advantage is therefore concentrated on medium-capability settings where explicit decomposition helps.

### 1.6 Full-Context Fair Comparison (n=500, qwen3-8b) — DEFINITIVE, overturns §1.1

Setup: regenerated `hotpotqa_test.json` with an untruncated `context_full` field (mean ~4675 chars, 10 paragraphs vs the 2000-char `context`). Ran `run_parallel.py --context_field context_full` so CoT-SC and all GERS variants see the *same* full context. Added a `context_char_limit` param (default 1500) so GERS's internal cap can be relaxed; `gers_cv2_fullctx` sets it to 8000 (effectively no cap).

| Method | EM | F1 | CS | n | context seen |
|---|---:|---:|---:|---:|---|
| CoT-SC (N=3) | **0.5560** | **0.7155** | - | 500 | full (~4675) |
| GERS-CV2-fullctx | 0.5360 | 0.6833 | 0.664 | 500 | full (~4675) |
| GERS-CV2-retr (BM25 top-2/sub-q) | 0.4228 | 0.5629 | 0.691 | 499 | 2 paras/sub-q |

Paired statistics (paired bootstrap, B=10000, seed=42; McNemar χ²-cc; reproducer `experiments/_paired_stats.py`):

| Comparison | EM diff | EM 95% CI | F1 diff | F1 95% CI | McNemar p |
|---|---:|---|---:|---|---:|
| CoT-SC vs GERS-CV2-fullctx | +0.020 | [-0.016, +0.056] (n.s.) | +0.032 | [+0.002, +0.063] | 0.314 |
| CoT-SC vs GERS-CV2-retr | +0.134 | [+0.092, +0.178] | +0.154 | [+0.113, +0.196] | <0.001 |
| GERS-CV2-fullctx vs GERS-CV2-retr | +0.114 | [+0.074, +0.152] | +0.121 | [+0.083, +0.159] | <0.001 |

Interpretation:
1. **The §1.1 headline gain is a truncation artifact.** The +0.041 F1 / +0.040 EM advantage of GERS-CV2 over CoT-SC (n=500, truncated) reverses sign under fair full context: CoT-SC is now +0.032 F1 (barely significant, CI [+0.002, +0.063]) and +0.020 EM (not significant, p=0.314). Both methods gain a lot from full context, but CoT-SC gains more (F1 0.373→0.715, +0.342) than CV2 (0.413→0.683, +0.270) — when the full evidence is available, plain reading beats decompose-then-cross-check.
2. **Honest nuance: not a collapse.** Under fair conditions CV2 is *statistically tied* with CoT-SC on EM (p=0.314) and only marginally behind on F1, at higher cost. So the defensible claim is "the gain vanishes and slightly reverses," not "the method is broken."
3. **Direction B (per-sub-question BM25 retrieval) is refuted.** `gers_cv2_retr` is the *worst* variant — 0.121 F1 below CV2-fullctx (p<0.001). Top-2 retrieval per sub-question is too aggressive and drops gold paragraphs. Retrieve-then-reason does not rescue decomposition here.

Why the artifact exists: in the truncated regime, CoT-SC sees 2000 chars and CV2 sees 1500 (its internal cap), so both are starved but CoT-SC less so; decomposition's value is largest exactly when the context is too short to read directly. This is a methodological caution for the graph-decomposition-for-multi-hop-QA subfield: apparent gains over CoT-SC can be an artifact of unequal/greater context truncation rather than of the decomposition itself.

Note: GERS-SC (graph-level self-consistency, K=3 DAGs) has *not* yet been run under full-context. Under truncated context it was already non-significant vs CoT-SC (+0.020 EM, CI [-0.012, +0.052], p=0.275), so it is unlikely to recover under fair conditions, but the ablation is not yet closed.

### 1.7 Budget Curve — Symmetric Crossover Test (n=200, qwen3-8b) — NO STABLE CROSSOVER

Setup: tests whether the §1.1 truncated-context "CV2 lead" is a real, stable crossover (decomposition wins under context scarcity, loses under abundance) or an asymmetric artifact. **Symmetric**: at each budget N, both CoT-SC (`--context_budget N`) and GERS-CV2-fullctx (internal cap relaxed to 8000) see the *same* first-N chars of `context_full`. 7 budgets × 2 methods × n=200.

| budget | CoT-SC EM/F1 | CV2 EM/F1 | dF1 (CV2−CoT) | F1 95% CI | McNemar p |
|---:|---|---|---:|---|---:|
| 800 | 0.230 / 0.310 | 0.230 / 0.323 | +0.013 | [-0.033, +0.061] | 0.831 |
| 1500 | 0.285 / 0.383 | 0.285 / 0.401 | +0.018 | crosses zero | 0.823 |
| 2000 | 0.260 / 0.366 | 0.315 / 0.426 | **+0.060** | **[+0.005, +0.105]** | 0.054 |
| 2500 | 0.325 / 0.433 | 0.335 / 0.470 | +0.037 | crosses zero | 0.838 |
| 3000 | 0.365 / 0.475 | 0.350 / 0.458 | −0.017 | crosses zero | 0.677 |
| 4000 | 0.430 / 0.561 | 0.385 / 0.523 | −0.038 | crosses zero | 0.137 |
| full | 0.540 / 0.687 | 0.540 / 0.678 | −0.009 | crosses zero | 0.860 |

Verdict: **a crossover *shape* exists (sign flips between 2500 and 3000: CV2 leads at 800–2500, CoT-SC leads at 3000–full) but it is not statistically stable.**
- Of 7 points, only ONE (budget 2000) has an F1 CI excluding zero (+0.060), and its McNemar p=0.054 sits right on the 0.05 line. EM is significant at 2000 (+0.055, CI [+0.005,+0.105]) but nowhere else.
- The CV2 lead is **non-monotone**: +0.013 → +0.018 → +0.060 (peak) → +0.037 → crosses zero. A stable crossover would be monotone-then-flip; this is a single-point spike at 2000.
- The CV2 low-budget lead (+0.013~+0.060) is far smaller than the old *asymmetric* truncated lead (+0.041) — confirming that asymmetric truncation (CoT@2000 vs CV2@1500) manufactured most of the apparent advantage.

Conclusion: on HotpotQA with **artificial** truncation, decomposition shows a weak, non-monotone, mostly non-significant tendency to help under scarcity. This does not meet the "stable crossover" bar and does **not** justify a Figure-1 claim on its own. Per the SOP, artificial truncation is a toy setting anyway; H1 (decomposition = noise-isolation) must be re-tested on **natural** long-context data (LongBench 4k–64k) where the crossover, if real, should be far stronger and monotone.

Reproducer: `experiments/_budget_curve.py` (paired bootstrap B=10000, seed=42; McNemar χ²-cc). Results: `experiments/results/budget_curve_8b/ctx{800,1500,2000,2500,3000,4000,full}/`.

## 2. Negative and Diagnostic Experiments

### 2.1 Repair Variants Are Not Main Methods

HotpotQA first 100 samples:

| Method | EM | F1 | CS | Repair triggers |
|---|---:|---:|---:|---:|
| GERS-CV2 | 0.330 | 0.451 | 0.781 | 0/100 |
| gers_repair | 0.300 | 0.401 | 0.847 | 51/100 |
| gers_repair_soft | 0.320 | 0.411 | 0.874 | 47/100 |

2Wiki first 100 samples:

| Method | EM | F1 | CS | Repair triggers |
|---|---:|---:|---:|---:|
| GERS-CV2 | 0.400 | 0.469 | 0.745 | 0/100 |
| gers_repair_soft | 0.340 | 0.403 | 0.851 | 53/100 |

Conclusion: repair increases CS while reducing EM/F1. It should remain a negative result and must not be included as a main method.

### 2.2 Evidence-Grounded Checking Is Diagnostic

HotpotQA first 100 samples:

| Method | EM | F1 | CS | Paired change vs GERS-CV2 |
|---|---:|---:|---:|---|
| GERS-CV2 | 0.330 | 0.451 | 0.781 | - |
| gers_grounded | 0.330 | 0.455 | 0.730 | wc=0 / cw=0 / dF1=+0.004 |
| gers_grounded_soft | 0.330 | 0.452 | 0.735 | wc=0 / cw=0 / dF1=+0.001 |

2Wiki first 100 samples:

| Method | EM | F1 | CS | Paired change vs GERS-CV2 |
|---|---:|---:|---:|---|
| GERS-CV2 | 0.400 | 0.469 | 0.745 | - |
| gers_grounded | 0.400 | 0.463 | 0.766 | wc=1 / cw=1 / dF1=-0.007 |
| gers_grounded_soft | 0.400 | 0.467 | 0.767 | wc=0 / cw=0 / dF1=-0.003 |

Per-type 2Wiki first 100 samples from the grounded diagnostic rerun:

This is a separate diagnostic rerun. Compare `gers_grounded` and `gers_grounded_soft` against the same-table GERS-CV2 row only; do not mix these per-type numbers with the earlier n=100 boundary table, which came from the main run.

| Method | comparison | bridge_comp | compositional | inference |
|---|---:|---:|---:|---:|
| GERS-CV2 | 0.775 | 0.524 | 0.286 | 0.360 |
| gers_grounded | 0.735 | 0.571 | 0.297 | 0.288 |
| gers_grounded_soft | 0.775 | 0.524 | 0.286 | 0.341 |

Conclusion: evidence grounding suppresses inflated CS and locally helps bridge_comparison, but it is not yet a robust overall improvement.

### 2.3 GF-GERS (Grounded-Forward + CoT-SC Fallback) — REJECTED

Motivation: force each sub-answer to be *grounded* in the context before it is trusted, and fall back to free-form CoT-SC if any sub-answer is ungrounded (routing around unreliable decomposition). Two versions: v1 checks evidence-grounding (lexical, bypassable); v2 checks the sub-answer directly.

HotpotQA smoke (qwen3-8b, n=20):

| Method | EM | F1 | context |
|---|---:|---:|---|
| GERS-CV2 (reference, truncated) | 0.250 | 0.395 | 1500-cap |
| GF-GERS v1 (evidence-grounding check) | 0.250 | 0.348 | 1500-cap |
| GF-GERS v2 (sub-answer check) | 0.250 | 0.318 | 1500-cap |
| GF-GERS v2 (qwen3-14b) | 0.200 | 0.346 | 1500-cap |

Conclusion: both GF-GERS versions underperform GERS-CV2 at n=20. Root causes: (1) lexical grounding is bypassable (model rephrases evidence); (2) the fallback-to-CoT-SC path is structurally weaker than CV2 on 8b, so routing *away* from decomposition hurts. Grounded-forward gating does not improve over plain CV2. Not pursued to n=500.

### 2.4 Cross-Model Inversion (qwen3-14b) and Retrieval Smoke — DIAGNOSTIC

qwen3-14b smoke (n=20), full context via `context_full`:

| Method | EM | F1 |
|---|---:|---:|
| CoT-SC | 0.500 | 0.731 |
| GERS-CV2-retr | 0.400 | 0.657 |
| GERS-CV2 (1500-cap, truncated) | 0.250 | 0.446 |

qwen3-8b smoke (n=20), full context:

| Method | EM | F1 |
|---|---:|---:|
| CoT-SC | 0.500 | 0.697 |
| GERS-CV2-retr | 0.450 | 0.640 |
| GERS-CV2-fullctx | 0.400 | 0.599 |
| GERS-CV2 (1500-cap, truncated) | 0.250 | 0.395 |

Diagnosis (code-verified at `generation_pipeline.py:944-945`): the cross-validation consistency score is pure *self-agreement* — when `enable_evidence_grounding=False` (the default), `grounding_score=1.0` unconditionally, so the score measures whether K decompositions agree with each other, not whether they are correct. On a fluent model (14b) the decompositions agree fluently whether or not they are right, so the signal inverts: CoT-SC (0.731 F1) >> CV2 (0.446) on 14b, whereas on 8b the truncated-context CV2 "win" was driven by the same fluency producing coincidentally-correct agreement. The CV2 advantage is fluency-indexed, not correctness-indexed, and does not transfer to stronger models or fair context.

qwen2.5-7b-instruct smoke (n=20): all 20 samples returned HTTP 403 (model not activated on the DashScope account). This model is unavailable; the second-model generalization check remains Qwen-Plus only.

## 3. Deprecated Results and Non-Claims

The following must not be used as current paper evidence:

| Item | Status | Reason |
|---|---|---|
| HotpotQA GERS EM=0.404 | deprecated | old EM bidirectional-substring bug and older extraction pipeline inflated results |
| HotpotQA "GERS-CV2 > CoT-SC" (§1.1, truncated) | artifact | context-truncation confound: the +0.041 F1 gain reverses sign under fair full context (§1.6); retain only as the truncated arm of the confound audit |
| CLUTRR synthetic dataset | removed | synthetic Chinese kinship labels produced low, noisy, non-standard results; active code raises NotImplementedError |
| Early ToT/CoT-SC comparisons | deprecated | generated before answer extraction and metric fixes |
| MoDeGraph-style v1 HotpotQA | deprecated | final-answer prompt omitted graph/reasoning/context under stateless model calls; v4 supersedes it |
| Repair as improvement | rejected | raises CS but hurts EM/F1 |
| GF-GERS (grounded-forward) | rejected | both v1/v2 underperform CV2 at n=20 (§2.3); lexical grounding bypassable + fallback structurally weaker |
| Per-sub-question BM25 retrieval (CV2-retr) | rejected | worst variant under full context, -0.121 F1 vs CV2-fullctx (§1.6); top-2 drops gold paragraphs |

If these numbers appear in older logs or draft files, treat them as historical debugging records only.

## 4. Current Paper Positioning (updated 2026-07-08)

> The previous framing (small-but-significant HotpotQA gain, p=0.029) is **retired**: it does not survive the fair full-context comparison in §1.6. Positioning is under review. The honest, defensible assets currently on hand are:

1. **Confound audit (primary candidate contribution).** Graph-decomposition methods' apparent gains over CoT-SC on HotpotQA can be entirely explained by unequal context access: GERS caps context at 1500 chars while baselines read more, so the decomposition advantage appears only under context starvation and reverses under fair full context (§1.1 vs §1.6). This is a methodological caution for the subfield, not a method win.
2. **Mechanism diagnosis.** The bidirectional cross-validation consistency score is *self-agreement*, not correctness: with `enable_evidence_grounding=False` (default), `grounding_score=1.0` unconditionally (`generation_pipeline.py:944-945`). It raises CS AUROC from ~0.498 (random) to ~0.58-0.59, but the signal is *fluency-indexed* and inverts on the stronger qwen3-14b (CoT-SC 0.731 F1 >> CV2 0.446; §2.4). The 8b "gain" was fluency-accidental.
3. **CS is not correctness.** 250/500 samples receive CS=1.0 yet 156 are EM-wrong (§1.2). High-CS-wrong answers remain common.
4. **Negative results on two natural fixes.** Grounded-forward gating (GF-GERS, §2.3) and per-sub-question BM25 retrieval (§2.4/§1.6) both fail — retrieval is in fact the *worst* variant (drops gold paragraphs).
5. **Honest boundary.** 2Wiki bridge-comparison and stronger models (Qwen-Plus n=300: paired diff ≈ 0) are real generalization boundaries.

What is NOT defensible anymore: any claim that GERS-CV2 *beats* CoT-SC on HotpotQA. Under fair full context, CoT-SC is at least tied (EM p=0.314) and marginally ahead (F1 +0.032, CI [+0.002, +0.063]) at lower cost. The MoDeGraph comparison (p=0.010) is also under the truncated regime and needs re-running under full context before it can be cited.

Open ablation: GERS-SC (graph-level self-consistency) and GERS-SC-CV2 have not been run under full context. Under truncated context GERS-SC was already non-significant vs CoT-SC (p=0.275), so recovery is unlikely, but the "all variants lose under fair context" picture is not yet closed.
