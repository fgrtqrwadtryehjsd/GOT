# GERS PDF 严格学术评审意见

评审对象：`docs/图约束推理链与子答案双向交叉验证_面向多跳推理的大模型方法.pdf`

对应源码：`docs/paper_en.tex`、`docs/paper_cn.tex`、`docs/references.bib`

远端最新提交：`d5d4361 docs: 添加英文版与最终PDF文档`

## 总体判断

这篇稿件已经具备一篇完整方法论文的基本形态：问题动机明确，方法模块完整，实验数据较充分，且能诚实报告负面发现。当前最有价值的贡献不再是“GERS 全面超越 CoT-SC”，而是“子答案双向交叉验证使原本无区分度的结构 Consistency Score 获得内容级区分能力，并在 HotpotQA 500 条样本上带来显著但中等幅度的提升”。

但是，以严格学术评审标准看，稿件仍存在几类会影响录用概率的问题：

- 核心贡献与实验收益之间存在张力：CS 区分度提升明显，但最终任务性能提升只有 4pt F1，且 bootstrap CI 仍跨 0，需要更谨慎地表述贡献。
- 方法描述还不够可复现：反向验证 prompt、match 函数、权重设计、置信度启发式没有足够细节。
- 部分论断过强，例如“unique to DAGs”“infeasible for linear CoT/ToT”，容易被审稿人挑战。
- 2WikiMultiHopQA 结果是负面边界，但论文中与主贡献的关系还没有完全理顺。
- 图表和引用格式仍偏草稿化，部分图缺少真实说明力，参考文献存在未来年份/不完整作者信息等风险。

以下按审稿维度给出详细问题与可操作修改建议。

## 一、研究逻辑与结构连贯性

### 问题 1：论文主线在“方法有效”与“边界分析”之间摇摆

当前摘要和引言强调 GERS-CV2 相比 CoT-SC 提升 4pt F1，并将其作为主结果；但讨论部分又强调 effect-size marginal、graph-level self-consistency 无增益、2Wiki bridge-comparison 上失败。这是诚实的，但当前组织方式会让读者觉得主张不够聚焦：到底论文要证明一个强方法，还是要证明一种诊断机制？

建议：把主线收束为：

- 第一层贡献：双向交叉验证修复了 Consistency Score 的区分度，这是机制贡献。
- 第二层贡献：在 HotpotQA 上带来显著但中等幅度的性能提升，这是经验贡献。
- 第三层贡献：系统分析了图结构推理在不同问题类型上的适用边界，这是诊断贡献。

摘要和引言中不要把“性能提升”写成唯一主线，而应把“CS 从结构合法性指标变成内容自洽指标”放在首位。

可改写贡献句：

> Rather than claiming uniform superiority over all prompting baselines, this work identifies and fixes a key bottleneck of graph-based reasoning: structural graph validity is not equivalent to reasoning correctness. We show that bidirectional sub-answer validation makes the consistency signal discriminative and yields statistically significant but moderate gains on HotpotQA, while also revealing clear failure modes on deeper bridge-comparison questions.

### 问题 2：“GERS-SC”与“GERS-CV/GERS-CV2”的关系需要重命名或重新解释

论文中出现 GERS+adaptive、GERS-SC、GERS-CV、GERS-CV2。当前结论显示：

- GERS-SC 多路径选择无额外增益。
- GERS-CV/CV2 才是最终有效方法。

但标题和多处文字仍强调“图级自一致性”。这会造成概念错位：读者会以为核心方法是 GERS-SC，但实验又说 GERS-SC 不显著。

建议：

- 将最终方法命名为 `GERS-CV` 或 `GERS-BCV`（Bidirectional Cross-Validation）。
- 将 GERS-SC 降级为一个被验证无明显增益的扩展/消融项。
- 标题可以改为：
  - `Graph-Constrained Reasoning with Bidirectional Sub-Answer Verification for Multi-Hop QA`
  - 或中文：`基于子答案双向验证的图约束多跳推理方法`

### 问题 3：2Wiki 负面结果的位置应提前进入“边界分析”，不要只放在实验后半段

目前 2Wiki 的失败/边界主要在 4.4 与讨论里出现。读者读到方法和主实验时，可能先形成“方法普遍有效”的预期，然后在 2Wiki 处突然发现失效，落差较大。

建议：在引言贡献第 3 点中明确预告：

> We further provide a boundary analysis on 2WikiMultiHopQA, showing that the method is more suitable for comparison-style decomposition than for deep bridge-comparison questions where upstream entity errors propagate.

这样审稿人会把 2Wiki 视为诚实边界分析，而不是主结果失败。

## 二、论点支撑与证据充分性

### 问题 4：性能收益“显著但中等”的表述还不够稳健

HotpotQA 500 条上，GERS-CV2 对 CoT-SC 的 EM 差异 +0.040，McNemar p=0.029，但 bootstrap 95% CI 为 [−0.014, +0.096]，跨 0。这说明基于成对对错方向的检验显著，但效应大小的不确定性仍较大。

当前论文已经写了“paired-significant, effect-size marginal”，这是好的。但摘要中“outperforming CoT-SC by 4pt F1”仍略显强。建议摘要中加限定：

> ... improves over CoT-SC by 4 F1 points, with a significant paired McNemar test but a marginal bootstrap effect-size interval.

同时在结论避免使用 “significantly outperforming” 这种单一表述，改为 “achieves a paired-significant improvement”。

### 问题 5：需要补充“为什么 Zero-Shot 很强”的解释与控制实验

Zero-Shot F1=0.389，接近 GERS-CV2 的 0.413，超过 CoT/CoT-SC。这是一个会被审稿人追问的异常现象：如果 Zero-Shot 已很强，结构化推理方法的实际必要性是什么？

建议补充一段解释：

- HotpotQA context 已包含较直接证据，Qwen3-8B 在短上下文抽取型多跳问题上无需显式推理即可回答部分样本。
- 结构方法的优势主要体现在 comparison 子集与需要分支合流的样本，而非所有样本。

更好的是加一个控制分析：按题型或证据距离报告 Zero-Shot 与 GERS-CV2 的差异。如果没有证据距离字段，至少报告 bridge/comparison 上 Zero-Shot 的 F1。

### 问题 6：Case Study 太抽象，不足以支撑机制主张

4.7 的两个 case 都是概括性描述，没有真实问题、上下文片段、子问题、正向答案、反向答案、最终 CS 变化。这不符合顶会论文对 qualitative analysis 的期待。

建议替换为真实案例，至少包含：

- 原问题。
- 关键上下文证据。
- 分解得到的 q1/q2/q3。
- 正向答案 a_i。
- 反向答案 a'_i。
- match 结果。
- 旧 CS 与新 CS。
- 最终是否被 GERS-CV 选择。

如果篇幅不足，可放一个简化图，详细案例放 appendix。

### 问题 7：对 CoT-SC+GERS rerank 的负面结论没有充分解释

表 1 中 CoT-SC+GERS rerank F1=0.372，几乎等于 CoT-SC 0.373。这说明“用 GERS 重排 CoT-SC 候选”无效。但论文只是把它列为 ablation，没有解释为什么。

建议增加一两句：

> This suggests that applying graph scores post hoc to linear CoT samples is insufficient; the graph must participate in the generation and verification process rather than merely reranking already-generated linear answers.

这句话能强化你的方法逻辑：图不是后处理器，而是执行与验证结构。

## 三、数据准确性与分析方法严谨性

### 问题 8：CS 区分度数值在不同位置需要统一说明变体来源

摘要写“from −0.0035 to +0.0847”。表 3 中 New CS correct mean=0.7888, wrong mean=0.7042, discrimination=+0.0847。但根据结果文件，不同配置的区分度不同：

- GERS-SC-CV2 的区分度约 +0.0559。
- GERS-adaptive-CV2 的区分度约 +0.0826。
- 表 3 的 +0.0847 更像某个特定配置/统计口径。

问题不是数值一定错，而是论文没有明确“New CS”对应哪个方法配置。审稿人复现实验时会困惑。

建议：表 3 标题或表内注明：

> New CS (+cross-validation, measured on GERS-adaptive-CV / GERS-CV variant)

并在实验设置中说明 CS 区分度统计基于哪个结果文件、哪个字段、是否使用 EM 正确性划分样本。

### 问题 9：F1 显著性没有单独检验，但正文强调 F1 +4pt

表 2 是 EM diff 的显著性检验，摘要和主文又强调 F1 +4pt。EM 的 McNemar 检验不能直接证明 F1 差异显著。

建议补充 F1 的 bootstrap paired CI：

- 对每个样本计算 `F1_GERS-CV2 - F1_CoTSC`。
- bootstrap 10000 次报告均值差和 95% CI。

如果 CI 跨 0，则写“F1 improves by 4 points, while EM paired test is significant”。如果不跨 0，则这会显著增强论文。

### 问题 10：2Wiki 只用 100 条，且作为边界分析可接受，但不要与 HotpotQA 主表同等权重

2WikiMultiHopQA 只有 100 条，且分题型后每类样本数更小。当前表 4 中 bridge comp 等细分结论如果用作强论断，统计风险较高。

建议：

- 明确标注 2Wiki 是 boundary/probing analysis，不是主实验。
- 加一句：due to the limited sample size of per-type subsets, these results should be interpreted as diagnostic rather than definitive.
- 若要投主会，建议将 2Wiki 扩到至少 300 条，或者只保留为补充实验。

### 问题 11：计算成本表缺少 LLM 调用次数，只有 latency

4.5 说“GERS-CV2 at 6.6s, about 80% of CoT-SC’s cost”，但只给了延迟。延迟受并发、API 波动、缓存影响大；调用次数更可复现。

建议表 5 增加一列：平均 LLM calls per sample。

例如：

- Zero-Shot：1。
- Standard CoT：1。
- CoT-SC：3。
- GERS+adaptive：根据分解步数，约 1 decomposition + n substeps + 1 aggregation。
- GERS-CV2：再加 backward verification calls。
- GERS-SC：K 倍。

如果确切 calls 统计已在日志里，直接报均值；没有则报理论估计。

### 问题 12：指标修复与答案提取公平性需要更可复现

4.1 说修复 EM bidirectional-substring bug、统一 concise extraction，但没有给出精确定义。评估管线本身影响很大，必须可复现。

建议在方法或实验设置中增加：

- EM normalize 规则。
- F1 tokenization 规则。
- concise extraction 的正则/优先级。
- HotpotQA 与 GSM8K 是否使用不同 EM 策略。

至少给出 GitHub 文件路径，例如：`src/utils/metrics.py` 与 `src/utils/answer_extractor.py`。

## 四、方法描述严谨性

### 问题 13：match 函数“graded semantic match”定义不清

公式 (2) 中 `match(ai, a'_i)` 是核心，但正文只说 graded semantic match。审稿人需要知道它到底是：字符串 exact match、token F1、LLM judge、embedding similarity，还是启发式。

建议明确写成：

> We implement match as a three-level heuristic: exact normalized match = 1, token-F1 above threshold = 0.5/0.7, otherwise 0; for numeric answers we use numeric equality.

如果是 LLM judge，也必须说明 prompt、temperature、是否用于所有方法、公平性与成本。

### 问题 14：wi “downstream nodes higher” 需要具体定义

公式 (2) 中 wi 权重很重要，但没有定义。读者无法复现，也无法判断是否合理。

建议给出明确公式：

> wi = 1 + out_degree(vi) / max_j out_degree(vj)

或：

> wi = 1 + number of descendants of vi.

并做一个消融：uniform weight vs downstream weight。若没有消融，至少说明选择该权重的直觉。

### 问题 15：结构分 Sstruct 权重前后不一致或解释不足

3.4 写 `w1 = w3 = 0.35, w2 = 0.30`，但早期版本/描述可能有 0.4/0.3/0.3。当前 PDF 内部是自洽的，但需要说明这些权重是否调参得到，还是经验设定。

建议：

- 说明权重固定于所有实验，没有按测试集调参。
- 若有验证集，说明在验证集上设定。
- 若无，写为 heuristic 并在局限中承认。

### 问题 16：反向验证存在 confirmation bias，需要更强控制

你已经写“do not copy the final answer”，这是好的。但用最终答案 A 作为 anchor 反向重答子问题，本质上会把模型引向与 A 一致的解释，可能造成伪一致。

建议增加控制实验或讨论：

- Control 1：只用 context 不用 final answer 做 backward verification。
- Control 2：用错误 final answer perturbation 测试 crossval 是否下降。
- Control 3：随机打乱子答案和最终答案，看 crossval 是否显著下降。

至少在局限中更具体地写：reverse verification may still be biased toward the final answer; future work will use adversarial anchors or external verifiers.

### 问题 17：“unique to DAGs; infeasible for linear CoT/ToT”表述过强

线性 CoT 也可以通过句子级步骤抽取做反向验证，ToT 也有中间状态。因此“infeasible”容易被反驳。

建议改成更稳妥：

> naturally supported by explicit DAG decomposition and less straightforward for unstructured linear CoT outputs.

或：

> DAGs provide explicit sub-question units and dependency edges, making such verification more direct and interpretable than applying it to unstructured CoT text.

## 五、图表与格式规范性

### 问题 18：PDF 中表格显示过于拥挤，部分表头和数字连在一起

从 PDF 文本抽取看，表 1、表 2、表 3 的列标题与数值严重挤压，如 “MethodEMF1CS”、“Correct meanWrong mean”。这说明当前表格排版在 PDF 中可读性不足。

建议：

- 使用 `booktabs` + `tabularx` 或 `resizebox{\linewidth}{!}`。
- 缩短方法名，例如 `GERS-Adap.`, `GERS-CV2`。
- 表 1 可以拆成两个表：主性能表与 CS 表。
- 确保表格在 PDF 中不是靠文本阅读，而是真正视觉可读。

### 问题 19：图题太短，缺少自解释信息

例如 Figure 1 caption 只有 “GERS reasoning framework.”。顶会论文中 caption 应该能让读者不看正文也知道图表达什么。

建议改为：

> Figure 1: Overview of GERS. The model decomposes a question into a dependency DAG, executes sub-questions in topological order, aggregates a final answer, and performs backward sub-answer verification anchored by the final answer and context.

Figure 2/4/5 也应补充一句主要结论。

### 问题 20：图像是否为位图生成，需要确认清晰度与可编辑性

当前 `docs/figures/*.png` 是 PNG，尺寸尚可，但投稿时更推荐 PDF/SVG 矢量图，尤其流程图和曲线图。

建议：

- 方法图用 TikZ、PDF 或 SVG。
- 数据图用 matplotlib 输出 PDF。
- 避免 AI 生成图中出现不可控文字错误。
- 所有图中文字统一英文、字体与论文一致。

### 问题 21：参考文献存在未来年份和不完整作者信息风险

references.bib 中多条为 2025/2026 或 arXiv 未来编号，如 `arXiv:2510.02827`、`arXiv:2510.19842`。当前日期是 2026-07-03，未来 arXiv 编号 2510 在时间上可存在，但如果论文未正式发表、作者名为 “DAG-Math Authors” 或 “and others”，会显得不严谨。

建议：

- 所有参考文献使用完整作者、标题、会议/期刊、年份、URL/arXiv ID。
- 对未正式发表的工作标注为 arXiv preprint，不要写成会议论文。
- 删除或替换无法核实的引用。
- 使用 BibTeX 自动生成引用，不要手写 References 列表。

### 问题 22：中文/英文版本并存，投稿版本应统一为英文且符合模板

当前 PDF 是英文内容但用 ctexart 结构生成，可能不是目标会议模板。AAAI/ACL 都要求特定模板、匿名格式、页数限制。

建议：

- 若投 AAAI：转成 AAAI 2027 LaTeX 模板，双栏，匿名，页数限制。
- 若投 ACL/ARR：转成 ACL Rolling Review 模板，匿名，包含 limitations/ethics/reproducibility checklist。
- 目前版本可以作为技术报告/arXiv 初稿，但不是可直接投稿模板。

## 六、学术语言问题

### 问题 23：部分措辞过强或口语化

例如：

- “hardest evidence”
- “equivalent to random”
- “honest negative finding”
- “superficial failures”
- “true reasoning gap”

这些表达在中文讨论中没问题，但英文论文中略显口语或主观。

建议替换为：

- “strongest evidence” → “direct empirical evidence”
- “equivalent to random” → “provides little discriminative signal”
- “honest negative finding” → “negative result” 或 “limited gain”
- “superficial failures” → “format-induced errors”
- “true reasoning gap” → “remaining reasoning errors”

### 问题 24：一些句子太长且信息密度过高

摘要第一句和贡献部分较长，包含方法、指标、结果、边界多层信息。建议拆分，降低审稿人阅读负担。

建议摘要结构：

- 背景问题：结构合法性不等于推理正确性。
- 方法：GERS-CV 通过 DAG 正向执行和反向子答案验证构建内容自洽分数。
- 结果：CS 区分度提升，HotpotQA 性能提升。
- 边界：多路径选择无明显收益，深桥接仍困难。

## 七、建议的优先级修改清单

### P0：投稿前必须修

1. 明确最终方法命名：建议以 GERS-CV2 或 GERS-BCV 为主，GERS-SC 降为消融。
2. 统一并精确定义 `match(ai, a'_i)` 和 `wi`。
3. 为 F1 差异补 paired bootstrap CI，因为正文主要强调 F1 +4pt。
4. 将“unique/infeasible”弱化为 “naturally supported / less straightforward”。
5. 修表格排版，避免 PDF 中列名数字粘连。
6. 用真实案例替换抽象 Case Study。
7. 检查并补全参考文献，删除无法核实的未来/不完整引用。

### P1：强烈建议修

1. 图 caption 改成长描述，图像尽量转 PDF/SVG。
2. 增加 LLM calls per sample 到成本表。
3. 给 2Wiki 结果加“diagnostic only”说明。
4. 增加 reverse verification 的 confirmation bias 控制讨论。
5. 将 Zero-Shot 异常强作为单独讨论点。

### P2：若冲主会建议补

1. 将 2Wiki 扩到 300 条以上，或将其放到 appendix。
2. 加 1 个额外模型（如 Qwen2.5-7B 或 Llama-3-8B）验证泛化。
3. 加 uniform vs downstream-weighted crossval 的小消融。
4. 加 final-answer-anchor vs context-only backward verification 控制实验。

## 八、建议改写后的论文定位

当前论文最稳的定位是：

> 本文不是声称图约束推理在所有多跳任务上全面超越 CoT，而是指出现有图推理方法中一个被忽视的问题：图结构合法性并不等于推理内容正确性。GERS-CV 通过子答案双向交叉验证，为 DAG 推理引入内容级自洽信号，使 Consistency Score 获得可测的对错区分能力，并在 HotpotQA 上带来成对显著的中等幅度提升。同时，本文系统分析其在深桥接问题上的失败边界。

这个定位严谨、可信，也能把负面结果转化为论文价值。

## 九、结论性评审意见

如果按顶会审稿口径，我会给当前稿件一个“有潜力但需要修改”的评价。优点是问题真实、方法有明确机制、实验足够诚实，且 500 条 HotpotQA 上有显著提升；缺点是方法细节复现性不足、部分主张过强、图表和参考文献尚未达到正式投稿规范。

建议先完成 P0 修改，再考虑投稿。如果目标是 AAAI/ACL 主会，还应补充至少一个控制实验或额外模型；如果目标是 Findings/Workshop 或技术报告，完成 P0 后基本可以进入排版投稿阶段。
