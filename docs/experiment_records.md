# GERS Experiment Records

> Last updated: 2026-07-07  
> Current paper target: AAAI-style submission draft  
> Active datasets: HotpotQA, 2WikiMultiHopQA, GSM8K diagnostics  
> Deprecated: CLUTRR and all pre-fix HotpotQA numbers are historical only and must not be used as paper claims.

## 1. Current Trustworthy Results

These are the results aligned with the current implementation and paper draft. Earlier records that used the old EM substring bug, unstable baseline extraction, or CLUTRR synthetic data are intentionally excluded from the main narrative.

### 1.1 HotpotQA Main Results (n=500, qwen3-8b)

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

## 3. Deprecated Results and Non-Claims

The following must not be used as current paper evidence:

| Item | Status | Reason |
|---|---|---|
| HotpotQA GERS EM=0.404 | deprecated | old EM bidirectional-substring bug and older extraction pipeline inflated results |
| CLUTRR synthetic dataset | removed | synthetic Chinese kinship labels produced low, noisy, non-standard results; active code raises NotImplementedError |
| Early ToT/CoT-SC comparisons | deprecated | generated before answer extraction and metric fixes |
| MoDeGraph-style v1 HotpotQA | deprecated | final-answer prompt omitted graph/reasoning/context under stateless model calls; v4 supersedes it |
| Repair as improvement | rejected | raises CS but hurts EM/F1 |

If these numbers appear in older logs or draft files, treat them as historical debugging records only.

## 4. Current Paper Positioning

Best defensible framing:

1. Structural graph validity is almost useless as a reasoning-quality signal because generated DAGs are usually legal.
2. Bidirectional sub-answer cross-checking turns CS into a more informative content-consistency signal, improving AUROC from roughly random (0.498) to weakly useful (0.58-0.59).
3. The resulting HotpotQA improvement is small but statistically significant: paired McNemar-significant on EM correctness (p <= 0.040) with paired bootstrap CIs excluding zero, including over the MoDeGraph-style baseline (p=0.010).
4. CS is not correctness. High CS wrong answers remain common, so evidence grounding and conservative repair are future work.
5. The method has a real boundary on 2Wiki bridge-comparison and stronger models.
6. The fixed MoDeGraph-style v4 graph-prompt baseline is below GERS-CV2 on HotpotQA (F1 0.366 vs 0.413; paired-significant p=0.010), but this covers one near-neighbor prompt baseline rather than all graph-reasoning systems.
