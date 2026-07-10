# Project Boundary & Minimal Fix — 可证伪假设 (Stage 0)

> SOP 阶段 0 交付物。论文 Intro 和 Conclusion 完全围绕这些假设写。
> 每条假设必须能通过实验给出 **Yes/No** 回答，不模棱两可。
> 起草日期: 2026-07-08。状态: **草案待用户确认**。

---

## 背景动机（从已 code-verified 的发现推导）

我们在 HotpotQA 上发现三个互相关联的事实，它们共同指向一个核心问题——**图分解（GERS-CV2）的收益到底从哪里来**：

1. **截断-confound**（n=500 full-ctx 实锤）：公平全 context 下 CoT-SC 0.716 F1 > CV2 0.683，旧"CV2 +0.041 F1 sig 领先"在公平条件下符号反转。但截断条件下 CV2 确实领先（0.413 > 0.373）。
2. **流利度反转**（code-verified @ generation_pipeline.py:944-945）：crossval consistency score 是 self-agreement 非 correctness——`enable_evidence_grounding=False` 时 `grounding_score=1.0` 无条件成立。信号在 qwen3-14b 上反转（CoT-SC 0.731 >> CV2 0.446）。
3. **深跳失败**（2Wiki bridge_comp）：复合题上分解语义错，GERS-CV2 0.524 << Standard CoT 0.905。

这三条放在一起的自然解释是：**分解的收益不是"增强逻辑推演"，而是某种 regime 依赖的鲁棒性**——而这个 regime 在 HotpotQA 全 context 下不存在。下面 2-3 个假设就是要把这个直觉钉成可证伪的命题。

---

## 假设 H1（信息轴）—— 分解收益 = 隔离干扰，非增强逻辑

**陈述**：图分解（Graph Decomposition）的收益主要来源于隔离长上下文中的自然信息干扰（distractor suppression），而非增强大模型的逻辑推演能力。

**可证伪的 Yes/No 回答**：
- 在 **真实长文档**（LongBench，4k→64k 真实干扰随长度增长）上，Graph 分解的 F1 优势随 context 长度/干扰增加而**单调增大**（Yes），还是**减小或不变**（No）？
- 在 **干扰固定、只增逻辑深度**的对照下（同 context 长度，2-hop vs 4-hop），Graph 优势**不增大**（Yes：说明增益来自干扰轴而非逻辑轴）还是**增大**（No）？

**现有证据**：
- ✅ 支持（间接）：HotpotQA 截断（高干扰/稀缺）下 CV2 赢，全 context（低干扰）下输——增益与"模型读取干扰程度"正相关，与"逻辑难度"非简单正相关。
- ⚠️ 未闭合：HotpotQA 是人为截断（SOP 视为 toy），需 LongBench 真实长文复现这个 crossover。
- 🎯 正在跑的 budget 曲线：若有稳定 crossover（人为截断版），是 H1 的 cheap 初步信号。

**若 H1=Yes（增益来自隔离干扰）**：论文主张 = "分解是 context-noise 鲁棒器，不是逻辑放大器"。Minimal Fix 应针对**干扰隔离/证据选择**（Retriever 模块），而非推理本身。
**若 H1=No（增益不随干扰增长）**：分解收益另有来源（或纯 artifact），需重新定位。

---

## 假设 H2（逻辑轴）—— 深跳瓶颈 = 误差传播，非图结构生成错误

**陈述**：在深层多跳（>3 hops）场景中，图推理系统的绝对性能瓶颈是单一节点的误差向下游的指数级传播（Error Propagation），而非图结构（DAG）的生成错误。

**可证伪的 Yes/No 回答（依赖 SOP 阶段 2 的 Oracle 解剖）**：
- Oracle 实验：替换完美 DAG 结构（Oracle 1）的 F1 增幅 **小于** 替换完美中间子答案（Oracle 3）的增幅（Yes：瓶颈在下游误差传播而非结构生成）？
- 具体数字判据：Oracle 1 增益 < 总可恢复增益的 30%，Oracle 3 增益 > 50%（Yes）。

**现有证据**：
- ✅ 支持（间接）：2Wiki bridge_comp（深组合）上 GERS-CV2 0.524 << Standard CoT 0.905——分解反而更差，符合"误差传播雪崩"而非"结构错误"（若是结构错误，CoT 也该差，但 CoT 0.905 很高）。
- ✅ 支持（间接）：crossval self-agreement 无法检出错误（流利度索引）——下游误差因为无 grounded 校验而畅通传播。
- ⚠️ 未闭合：需 Oracle 解剖（阶段 2）直接量化各模块"罪恶值"。现在只是推测。

**若 H2=Yes（瓶颈是误差传播）**：Minimal Fix 应针对**下游节点对上游错误的鲁棒性**（多假设保留 / 基于上下文片段而非实体传递 / 局部一致性卡点），而非图生成或检索。
**若 H2=No（瓶颈是结构生成）**：Minimal Fix 应针对 DAG 分解语义（如 bridge_comparison 的分解错法）。

---

## 假设 H3（我们独有，机制诊断）—— 一致性信号是流利度索引，非 correctness

**陈述**：基于 self-agreement 的图一致性打分（如 GERS-CV2 的 crossval CS）度量的是生成的"流利一致性"（fluency-indexed agreement），而非答案正确性；该信号在更强/更流利的模型上**反转**（失去甚至反向预测正确性的能力）。

**可证伪的 Yes/No 回答**：
- 同一方法、同一数据，qwen3-8b 上 CS 与 EM 正相关（AUROC > 0.55），qwen3-14b 上 AUROC **≤ 0.55 或反转**（Yes）？
- 引入 grounded 信号（evidence grounding / 外部 verifier）后 AUROC **显著上升**且不再随模型流利度反转（Yes：证明原信号缺的是 grounding 而非统计量本身）？

**现有证据**：
- ✅ 支持（已 code-verified）：`generation_pipeline.py:944-945`，`enable_evidence_grounding=False` 时 `grounding_score=1.0` 无条件。
- ✅ 支持（数据）：8b 上 CV2-CS AUROC≈0.58（弱有用），14b 上 CoT-SC 0.731 >> CV2 0.446（信号失效）。
- ✅ 支持（证据 grounding 实验）：`gers_grounded` 把 CS 从 0.781 降到 0.730（抑制虚高），但 EM 不涨——说明 grounding 能修信号校准但修不了正确性。

**若 H3=Yes**：这是一个**方法学警示**贡献——子领域里 self-agreement 类一致性打分（self-consistency / graph-SC）普遍假设"一致=正确"，但 fluent 模型上不成立。Minimal Fix 若走此轴，应针对"如何给 self-agreement 补 grounding 而不引入循环依赖"。

---

## 假设之间的优先级与依赖

- **H1 是入口**（阶段 1 验证）：决定方法有没有"自然 regime 优势"。若 H1=No 且 HotpotQA budget 也无 crossover → 分解在这类任务上无救，转向纯诊断论文或换赛道。
- **H2 是核心机制**（阶段 2 Oracle 验证）：决定 Minimal Fix 打哪个靶点。**SOP 箴言"No Oracle, No Design"——H2 不验证就不写新算法。**
- **H3 是我们的独有诊断增量**：H1/H2 是 SOP 模板，H3 来自我们的 code-verified 发现，可作为论文的差异化贡献（其他图推理论文没讲过 self-agreement 的流利度反转）。

## 待用户确认

1. 这 3 条假设是否成立为论文灵魂？要加/减/改哪条？
2. H3（流利度反转）要不要升格为独立主轴，还是作为 H1/H2 的支撑诊断？
3. Minimal Fix 的靶点预判：若 H1=Yes 且 H2=Yes，靶点在"误差传播鲁棒性"——你倾向哪个机制方向（多假设保留 / 上下文片段传递 / 局部一致性卡点）？

---

## 与 budget 曲线（task #43）的关系

正在跑的 HotpotQA budget 曲线是 H1 的 **cheap 初步信号**（人为截断版）。判定规则：
- **有稳定 crossover** → H1 在 toy 版成立 → 进 SOP 阶段 1 用 LongBench 真实长文复现（task #45）。
- **无稳定 crossover** → H1 在 toy 版都不成立 → 真实长文几乎不可能成立 → H1 作废，重新定位假设（可能转纯 H3 诊断论文，或换"分解=逻辑放大器"的 H1' 变体去 MuSiQue 深跳验证 H2）。
