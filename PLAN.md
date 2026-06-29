# 方案:多线程加速 + CoT-SC+GERS 重排完善 + 实验

> 决策已定:并发用**多线程**;方法改进聚焦**完善 CoT-SC+GERS 重排**。
> 实验范围(未指定,取合理默认):先 smoke 20 验证管线 → HotpotQA 全量 100 对比。

## 背景与关键发现(读码确认)

1. **当前实验串行**:`run_quick_exp.py` / `run_ablation_v2.py` 单进程逐条跑,每条 GERS 串行 3-5 次 LLM 调用,HotpotQA ~8.8s/条,100 条 ~15min;cot_sc / gers_sc 多采样更慢。瓶颈是 DashScope API 网络等待(I/O 密集)。
2. **DashScopeModel 每次调用新建 OpenAI client**(`dashscope_model.py:72`),并发下连接池反复建,开销大;且**无 429/超时重试**,8 worker 并发易触发限流报错。
3. **投票碎片化 bug**:`cot_sc.py:34` 与 `cot_sc_gers.py:117` 直接 `Counter(原始答案)`,未归一化 → "yes"/"Yes"/"Yes." 不聚拢,削弱 CoT-SC(HotpotQA 多 yes/no 题)。两个变体都要修,否则对比不公平。
4. **GERS-SC 温度多样性是空操作**(`generation_pipeline.py:394`):`self.model.temperature = temp` 但 `generate()` 用的是参数 `temperature`(硬编码 0.2/0.3/0.1),**不读 `self.model.temperature`** → K 条 DAG 实际无温度差异,自一致性失效。若实验纳入 `gers_sc` 必须修。
5. **StandardCoT prompt 仍含中文** `参考信息：`(`standard_cot.py:23`),HotpotQA/2Wiki 是英文数据集,上次 commit 声称改英文但未落地。

---

## Track A:多线程并行基建(解决"太慢")

### A1. 新建 `experiments/run_parallel.py`
- `ThreadPoolExecutor(max_workers=--workers, 默认 8)`,I/O 密集 → 线程释放 GIL,加速 ≈ worker 数。
- **共享一个 DashScopeModel 实例**(缓存 client,见 A2);**每样本新建 method 实例**(`create_method` 已是 lambda 工厂,廉价,且避免 GERS-SC 临时改 `self.model.temperature` 的线程竞态)。
- **断点续跑**:读已有结果 JSON,跳过已完成 `sample_id`。
- **增量落盘**:`threading.Lock` 保护,每完成 5 条写一次;最终按 `sample_id` 排序输出。
- **超时**:OpenAI client 设 `timeout=60`(A2) + `future.result(timeout=--timeout)` 软上限。
- **进度**:每 10 条打印累计 EM / 已用 worker 利用率。
- 复用 `run_comparison.create_model` + `run_quick_exp.create_method`(已含全部方法名)。
- CLI:`--dataset --methods(a,b,c) --model --num_samples --workers --timeout --output_dir --resume`。

### A2. 改 `src/models/dashscope_model.py`
- `__init__` 时创建**单例 OpenAI client** 并缓存(`self._client`),`generate` 复用 → 并发下省去反复建连接池。
- client 设 `timeout=60.0`。
- `generate` 加**指数退避重试**(3 次):捕获 `RateLimitError` / `APITimeoutError` / `APIConnectionError`,sleep 1→2→4s。并发稳定性关键。

### A3. 改 `src/baselines/standard_cot.py`(小修,实验公平)
- `参考信息：` → `Context:`。

---

## Track B:CoT-SC+GERS 重排完善(你选的方法改进)

### B1. 改 `src/baselines/cot_sc.py`(投票归一化,两个变体共用)
- 投票前对每个候选答案做归一化:yes/no 归一 + lower + 去标点(复用 `answer_normalizer`)。
- 用归一化答案投票,返回归一化答案。保证 CoT-SC 与 CoT-SC+GERS 用同一归一化, isolating 重排贡献。

### B2. 改 `src/baselines/cot_sc_gers.py`(核心:加权投票升级)
现状:仅"票数并列时"用 GERS lite 打分重排,gap 小则退回多数票 → GERS 分大多数情况不影响结果。
升级为**GERS 加权投票**(对应文献支撑版第五节思路):
1. N 次 CoT 采样 → 归一化答案。
2. **去重**:对 unique 答案集合打分(而非 N 个样本全打),省 LLM 调用。
3. 每个 unique 答案取一条代表 reasoning,用现有 `_score_by_gers_lite`(实体覆盖/推理连贯/答案落地)打分。
4. **加权得分**:`score(a) = count(a) + λ · Σ gers_score(样本属于 a)`(λ 默认 1.0,可调)。count 主导,GERS 分在票数接近时翻盘。
5. 选 `score` 最高的答案;同时记录 majority 答案作对照,输出 `rerank_triggered / weighted_scores / majority_answer` 诊断字段。
6. 保留 `enable_gers_rerank=False` 退化为(归一化后的)标准 CoT-SC,便于消融。

### B3.(实验有效性,可选)修 GERS-SC 温度 bug
- `generation_pipeline.py:_reason_with_self_consistency`:把 decompose 温度作为参数真正传进 `_decompose` → `generate(..., temperature=temp)`,使 K 条 DAG 有真实多样性。
- 仅当实验纳入 `gers_sc` 时需要;不影响 cot_sc_gers 本身。

---

## Track C:实验(用新 runner)

### C1. smoke 20 条(HotpotQA)——验证管线
方法:`cot_sc_gers, cot_sc, gers_sc, gers_adaptive, standard_cot, zero_shot`
确认:并行 runner 正常、重试生效、归一化/加权投票输出正确、gers_sc 温度修复后 CS 有区分度。
→ 跑通后再决定全量。

### C2. 全量 100 条(HotpotQA)——主对比
同上方法。核心问题:**CoT-SC+GERS 加权重排是否打败 CoT-SC?** 是否优于 gers_adaptive?

### C3.(可选)2WikiMultiHopQA 100 条
数据已备好(`data/processed/2wikimultihopqa_test.json`)。验证文献支撑版核心论点:GERS 系方法在对比型/依赖密集推理上优于 CoT/CoT-SC。视 C1/C2 结果再定。

### C4. 汇总
跑完用统一口径刷新 `docs/experiment_records.md` 对比表(EM/F1/CS/平均耗时),标注多线程 worker 数。

---

## 明确不做(边界)
- 不改 `metrics.py`(已修)。
- 不动 ConstrainedDecoder(已禁用)。
- 不做子答案置信度 / 分解自审(你未选;留下一轮)。
- 不直接跑 500 条全量(先 100 确认趋势,省 API 费,你说停就停)。
- 不提交 git(除非你要求)。

## 交付物清单
- `experiments/run_parallel.py`(新)
- `src/models/dashscope_model.py`(缓存 client + 重试)
- `src/baselines/standard_cot.py`(英文 prompt)
- `src/baselines/cot_sc.py`(投票归一化)
- `src/baselines/cot_sc_gers.py`(加权投票)
- `src/chain_generation/generation_pipeline.py`(GERS-SC 温度修复,可选)
- `experiments/results/*.json`(新结果)
- `docs/experiment_records.md`(更新对比表)

## 风险
- DashScope QPM 限流:8 worker 可能触发 429 → 靠 A2 重试 + 可降 `--workers`。
- 多线程下 GERS-SC 临时属性:靠"每样本新建 method"规避(已在 A1 设计)。
- API 费用:smoke 20 先验,全量前征求确认。
