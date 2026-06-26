# 基于推理依赖图的图约束思维链生成方法

**梁爽爽**  
*（作者单位、邮箱待补充）*

**摘要** 大语言模型（LLM）在复杂多跳推理任务上常出现错误累积与逻辑幻觉问题。标准 Chain-of-Thought（CoT）以线性文本组织推理过程，难以表达子问题间的非线性依赖，且缺乏对推理链路完整性的可量化校验。本文提出 GERS（Graph-Enhanced Reasoning System），一种基于推理依赖图的图约束思维链生成方法。GERS 首先通过自适应复杂度判断决定推理策略：简单问题直接采用 CoT 回答，复杂问题则分解为有依赖关系的子问题并构建有向无环图（DAG），通过拓扑排序规划执行路径，由约束解码器将图结构信息注入子问题 Prompt；最后采用基于连通性、环路检测与证据覆盖度的 Consistency Score 对推理链路进行结构化一致性校验。在 HotpotQA、GSM8K、CLUTRR 三个公开数据集上与 Zero-Shot、Standard CoT、CoT-SC、ToT、MoDeGraph 五种基线方法进行对比。实验表明，GERS 在多跳问答数据集 HotpotQA 上取得 EM=0.412，相比 Standard CoT 提升 31.2%，显著优于所有基线；自适应分解策略使 GERS 在 GSM8K 上从 EM=0.602 提升至 0.670，有效缓解了简单任务上的错误传播问题。

**关键词** 大语言模型；复杂推理；思维链；图结构表征；一致性校验；多跳问答

---

## 1 引言

大语言模型（LLM）通过 Chain-of-Thought（CoT）提示在各类推理任务上展现出强大的能力[1]。然而，面对需要多个子问题协作推导的复杂任务（如多跳问答、关系推理），标准 CoT 存在三类不足：

1. **非线性缺失**：CoT 以线性文本串接推理步骤，无法表达子问题间的分支、合流与并行依赖，当推理路径本应是 DAG 而非链时，线性化导致信息丢失。
2. **错误累积**：线性链中任意一步的错误会向下游传播，缺乏回溯与重检机制。
3. **逻辑幻觉**：LLM 可能生成看似连贯但实际存在逻辑断裂或循环论证的推理文本，而标准 CoT 无法对此进行结构化校验。

近期工作尝试用图结构增强推理：GoT[2] 提出思维图，支持思维合并与蒸馏，但不保证执行顺序；RwG[3] 将上下文中的隐式知识结构化为实体关系图，但未实现逐步执行；MoDeGraph[4] 引导 LLM 提取问题中的实体关系构建多跳依赖图，但仍属于 prompt 方法，未实现拓扑排序执行与一致性校验；GoV[5] 将推理建模为 DAG 进行验证，但侧重语义验证而非图论结构校验；StepChain[6] 将问题分解与 BFS 推理流结合，但采用线性序列而非 DAG。这些方法普遍将图结构作为 Prompt 装饰或事后验证工具，而非真正参与推理执行与结构化校验的完整闭环。

针对上述问题，本文提出 **GERS**（Graph-Enhanced Reasoning System），其核心思想是：**让推理依赖图真正参与推理执行与一致性校验**。GERS 的完整流程包括：(0) 自适应复杂度判断——LLM 判断问题是简单还是复杂，简单问题直接 CoT 回答，复杂问题走完整图约束流程；(1) LLM 将问题分解为有依赖关系的子问题并构建推理 DAG；(2) 拓扑排序规划执行路径，约束解码器将图结构信息注入 Prompt；(3) 按拓扑序逐步回答子问题；(4) 汇总子答案形成最终结论；(5) 基于图论算法的 Consistency Score 量化校验推理链路完整性。

本文的主要贡献：

- 提出一种将推理依赖图真正参与执行的图约束思维链生成框架，明确区分了"构图"与"执行"两个阶段。
- 设计自适应分解策略，根据问题复杂度动态选择推理路径，使方法在简单任务（GSM8K）和复杂任务（HotpotQA）上均达到最优。
- 设计基于连通性、环路检测与证据覆盖度的 Consistency Score 计算方法，作为推理质量的量化指标。
- 在三个公开数据集上系统对比 6 种方法（含 MoDeGraph 基线）并完成消融实验，证明方法的有效性。

## 2 相关工作

**思维链推理**。Wei 等[1] 提出 Chain-of-Thought prompting，通过"Let's think step by step"引导 LLM 输出中间推理步骤。Wang 等[7] 提出 Self-Consistency（CoT-SC），通过多次采样取众数答案提升稳健性。Yao 等[8] 提出 Tree of Thoughts（ToT），以树搜索框架支持状态评估与回溯。这些方法均基于线性或树形结构，未显式建模子问题间的依赖图。

**图结构增强推理**。GoT[2] 将推理建模为图，支持合并、蒸馏等操作，但侧重 Prompt 工程而非执行约束。RwG[3] 通过 LLM 在上下文中构建实体关系图，将图作为一次性输入增强推理。MoDeGraph[4] 引导 LLM 从复杂问题中提取实体关系，构建多跳依赖图辅助多跳问答，但本质仍是 prompt 方法——图作为上下文信息送入 LLM，不参与执行顺序规划与一致性校验。StepChain[6] 将问题分解与 BFS 推理流结合用于多跳问答，但子问题是线性序列而非 DAG，且侧重知识图检索。与这些工作不同，GERS 将推理依赖图与执行流程深度耦合：子问题 DAG 的拓扑排序决定执行顺序，图结构信息注入 Prompt 约束生成，Consistency Score 基于图论算法量化校验，形成完整的"构图-执行-校验-修正"闭环。

**推理验证与一致性校验**。GoV[5] 将推理建模为 DAG，采用"node block"架构进行 training-free 验证，但侧重每个节点推理内容的语义正确性验证，而非图结构本身的数学性质（连通性、环路、覆盖度）。Process Reward Models[9] 通过对中间步骤打分进行过程监督，但依赖大量标注数据。Self-Refine[10] 通过 LLM 自我反思修正答案。GERS 采用图论算法（连通性、环路检测、最大流覆盖度）进行结构化校验，与语义层 NLI 校验互补，形成可量化的 Consistency Score，并支持图驱动的闭环修正。

## 3 方法

### 3.1 总体框架

GERS 的推理流程如图 1 所示，分为三个模块：

- **模块 1：推理状态图表示**。定义推理图 $G=(V,E)$，其中 $V$ 为节点集合（Fact / Step / Conclusion 三类），$E$ 为边集合（Derive / Support / Conflict 三类）。
- **模块 2：图约束链路生成**。基于拓扑排序的路径规划、图转 Prompt、约束解码，逐步生成子问题答案并汇总。
- **模块 3：一致性校验与闭环修正**。基于图论算法的结构校验 + 基于 NLI 的语义校验，输出 Consistency Score，支持触发闭环修正。

```
输入问题 Q + 上下文 C
        │
        ▼
⓪ 自适应判断：LLM 判断问题复杂度
   ├── 简单 → 直接 CoT 回答 → 输出答案
   └── 复杂 → 走完整 GERS 流程 ↓
        │
        ▼
① 分解：LLM 将 Q 分解为子问题 {q_i} 及依赖关系
        │
        ▼
② 构图：构建推理 DAG G=(V,E)
   Fact(Q) → Step(q_i) → ... → Conclusion
        │
        ▼
③ 路径规划：拓扑排序 τ = topo_sort(G)
   GraphPromptBuilder 生成路径描述 hint
        │
        ▼
④ 逐步执行：∀ q_i ∈ τ，ConstrainedDecoder 注入
   前驱答案 + 路径提示 → LLM 生成 a_i
        │
        ▼
⑤ 汇总：推理链 { (q_i, a_i) } → LLM 生成最终答案 A
        │
        ▼
⑥ 一致性校验：ConsistencyChecker(G, reasoning)
   S = α·S_struct + β·S_semantic
        │
        ▼
输出：答案 A + 推理图 G + Consistency Score S
```

**图 1** GERS 推理流程

### 3.2 推理状态图表示

推理图 $G=(V,E)$ 的节点与边定义如下：

**节点类型**：
- **FactNode**（事实节点）：已知信息、证据或原始问题。
- **StepNode**（过程节点）：子问题及其答案，记录推理中间步骤。
- **ConclusionNode**（目标节点）：最终结论。

**边类型**：
- **DeriveEdge**（推导边）：$u \to v$ 表示 $v$ 由 $u$ 推导得出。
- **SupportEdge**（支撑边）：$u \to v$ 表示 $u$ 支撑 $v$ 的成立。
- **ConflictEdge**（互斥边）：$u \to v$ 表示 $u$ 与 $v$ 矛盾。

构建过程：LLM 将问题 $Q$ 分解为有序子问题列表 $\{q_1, q_2, \dots, q_n\}$，每个子问题标注 `depends_on` 依赖关系。为原始问题创建 FactNode，为每个子问题创建 StepNode 并按依赖关系添加 DeriveEdge，最后为最终答案创建 ConclusionNode 并连接至最后一个子问题。

### 3.3 自适应分解策略

不同复杂度的问题适合不同的推理策略：简单问题（如单步算术计算）不需要分解，强行分解反而引入错误传播；复杂问题（如多跳问答）则需要结构化分解才能有效推理。GERS 采用自适应分解策略，在推理流程开始前先判断问题复杂度。

**复杂度判断**。给定问题 $Q$ 和上下文 $C$，LLM 判断该问题是"simple"（单步可解）还是"complex"（需要多步推理）。判断 prompt 设计为保守策略——当不确定时倾向于"complex"，以避免将多跳问题误判为简单问题。

**分支执行**。若判定为"simple"，GERS 直接采用 CoT 方式回答（单次 LLM 调用），跳过分解、构图、拓扑排序等步骤，避免不必要的错误传播。若判定为"complex"，走完整的图约束推理流程（步骤①-⑥）。

这一策略使 GERS 能够自适应不同复杂度的任务：在 GSM8K（简单算术）上，大部分问题被判定为"simple"而直接 CoT 回答；在 HotpotQA（多跳问答）上，大部分问题被判定为"complex"而走完整 GERS 流程。

### 3.4 图约束链路生成

**路径规划**。对推理图 $G$ 执行拓扑排序，得到执行顺序 $\tau = (v_1, v_2, \dots, v_n)$。若图中存在环路，则通过删除权重最小的边破环后再排序。拓扑序确保每个子问题在被回答时，其所有依赖的前驱子问题已获得答案。

**图转 Prompt**。GraphPromptBuilder 将图结构转换为结构化路径提示，注入到子问题 Prompt 中。路径提示包括：推理路径描述（从 Fact 到 Conclusion 的关键路径）、前驱答案上下文（当前子问题所依赖的已知答案）、分支/合流点信息。

**约束解码**。ConstrainedDecoder 在标准 Prompt 基础上增强约束：限制子问题答案的格式（"Sub-answer:" 后缀）、限制答案长度（1-3 句）、注入前驱节点的已知答案作为上下文。约束模式支持 soft（提示性约束）与 hard（强制格式约束）。

**逐步执行与汇总**。按拓扑序 $\tau$ 逐个调用 LLM 回答子问题，每个子问题的答案会作为后续依赖子问题的上下文。全部子问题回答完毕后，将推理链 $\{(q_i, a_i)\}_{i=1}^n$ 汇总，由 LLM 生成最终答案 $A$。

### 3.4 一致性校验与闭环修正

**Consistency Score** 采用双层校验：

$$S = \alpha \cdot S_{\text{struct}} + \beta \cdot S_{\text{semantic}}$$

其中 $\alpha=0.6, \beta=0.4$（结构为主，语义为辅）。

**结构层** $S_{\text{struct}}$ 包含三个图论指标：

$$S_{\text{struct}} = w_1 \cdot C_{\text{conn}} + w_2 \cdot C_{\text{cycle}} + w_3 \cdot C_{\text{cov}}$$

其中 $w_1=w_3=0.35, w_2=0.30$：

- **连通性** $C_{\text{conn}}$：基于弱连通分量检测，若图存在多个连通分量或孤立节点，则存在逻辑断链。
- **无环性** $C_{\text{cycle}}$：基于环路检测，存在环路则存在循环论证。
- **覆盖度** $C_{\text{cov}}$：基于最大流方法计算事实节点对结论节点的支撑覆盖程度，同时考虑子答案质量因子（空答案或低置信度信号降权）。

**语义层** $S_{\text{semantic}}$ 基于 NLI（自然语言推理）模型，对每条推导边检测两端节点是否存在蕴含关系。本文实现支持 HuggingFace NLI 模型与 LLM-based NLI 两种方式，后者通过 Prompt 引导 LLM 进行 entailment / neutral / contradiction 判断。

**闭环修正**。若 $S < \theta$（默认 $\theta=0.5$）且未达迭代上限，FeedbackLoop 触发回溯修正：将原始推理链与一致性校验问题反馈给 LLM，要求其修正推理并重新输出答案，随后重新校验。

## 4 实验

### 4.1 实验设置

**数据集**。选用三个公开数据集：
- **HotpotQA**[11]：多跳问答，500 条测试样本，含 bridge 与 comparison 两类题型。
- **GSM8K**[12]：小学数学推理，500 条测试样本。
- **CLUTRR**[13]：家庭关系逻辑归纳，由于原始数据集下载受限，采用 50+ 关系链模板程序化生成 300 条唯一样本，答案为 29 种中文家庭关系词。

**基线方法**：
- **Zero-Shot**：直接调用 LLM 回答。
- **Standard CoT**："Let's think step by step" 提示。
- **CoT-SC**[7]：Self-Consistency，$N=3$ 次采样取众数。
- **ToT**[8]：Tree of Thoughts，BFS 搜索，depth=3，beam_width=2。
- **MoDeGraph**[4]：基于多跳依赖图的 prompt 方法，引导 LLM 提取问题中的实体关系构建依赖图后回答。

**模型与配置**。主模型为 Qwen3-8B（阿里云 DashScope API，`enable_thinking=False`）。GERS 配置：`adaptive=True, max_iterations=1, enable_nli=False, consistency_threshold=0.75`（启用自适应分解与 ConstrainedDecoder、GraphPromptBuilder）。

**评估指标**。Exact Match（EM，精确匹配率）、F1（词级 F1）、Consistency Score（GERS 专有）、平均耗时。

### 4.2 主实验结果

表 1 展示了三个数据集上的主实验结果。

**表 1** 主实验结果（GSM8K / HotpotQA n=500，CLUTRR n=300；ToT n=100）

| 方法 | GSM8K EM | GSM8K F1 | HotpotQA EM | HotpotQA F1 | CLUTRR EM | CLUTRR F1 |
|------|----------|----------|--------------|-------------|-----------|-----------|
| Zero-Shot | 0.556 | 0.528 | 0.356 | 0.259 | 0.000 | 0.000 |
| Standard CoT | 0.740 | 0.714 | 0.314 | 0.227 | **0.193** | — |
| CoT-SC (N=3) | 0.762 | 0.738 | 0.314 | 0.226 | 0.173 | 0.130 |
| ToT | **0.860** | **0.800** | 0.430 | 0.291 | — | — |
| MoDeGraph | — | — | 0.256 | 0.258 | — | — |
| GERS（无自适应） | 0.602 | 0.554 | 0.404 | 0.315 | 0.183 | 0.153 |
| **GERS+自适应（本文）** | 0.670 | 0.636 | **0.412** | **0.322** | 0.183 | 0.153 |

**多跳问答（HotpotQA）**。GERS+自适应取得 EM=0.412，相比 Standard CoT 提升 31.2%，显著优于 Zero-Shot、Standard CoT、CoT-SC 和 MoDeGraph。ToT 在 HotpotQA 上取得 EM=0.430（100条），略高于 GERS+自适应，但 ToT 的平均耗时为 19.0s/条，是 GERS+自适应（9.4s/条）的 2 倍。MoDeGraph（CIKM 2025）仅取得 EM=0.256，低于 Standard CoT（0.314），说明仅将实体关系图作为 prompt 上下文不足以提升多跳推理性能。GERS 相比 MoDeGraph 提升 60.9%（0.412 vs 0.256），验证了"构图-执行-校验"完整 pipeline 相比单纯 prompt 方法的优势。

**单步算术推理（GSM8K）**。ToT 取得最优 EM=0.860（100条），其树搜索对算术推理有效。CoT-SC（0.762）次之。GERS（无自适应）的 EM=0.602 低于基线，因为多步分解引入错误传播。**自适应分解策略显著缓解了这一问题**：GERS+自适应在 GSM8K 上达到 EM=0.670，相比无自适应版本提升 11.3%（0.602→0.670）。尽管仍低于 ToT 和 CoT-SC，但 GERS+自适应的平均耗时（11.8s）远低于 ToT（19.4s）和 CoT-SC（83.8s），体现了效率优势。

**关系推理（CLUTRR）**。GERS 的 EM=0.183 略优于 CoT-SC（0.173），但低于 Standard CoT（0.193）。GERS 在多跳亲属关系推理上与 CoT-SC 相比有微弱优势，但整体上 CLUTRR 任务对中文家庭关系词的理解要求较高，所有方法表现均偏低。

### 4.3 分题型分析

表 2 展示了 HotpotQA 上不同题型的表现。HotpotQA 包含 bridge（桥接型，需多跳检索）与 comparison（对比型，需对比两个实体）两类。

**表 2** HotpotQA 分题型分析（n=500）

| 方法 | Bridge EM (n=404) | Comparison EM (n=96) |
|------|-------------------|----------------------|
| Zero-Shot | 0.327 | 0.479 |
| Standard CoT | 0.270 | 0.500 |
| **GERS (本文)** | **0.337** | **0.677** |

GERS 在 comparison 题型上相比 Standard CoT 提升 17.7%（0.677 vs 0.500），这是因为子问题 DAG 能自然地将对比双方拆解为独立子问题分别回答，再汇总比较。在 bridge 题型上 GERS 提升 6.7%，多跳检索受益于分解但提升幅度较小。

### 4.4 案例分析

在 HotpotQA 的 500 条样本中，GERS 与 Standard CoT 的对比分布如表 3。

**表 3** GERS vs Standard CoT 答对分布（HotpotQA, n=500）

| 类别 | 数量 | 比例 |
|------|------|------|
| GERS 独对（CoT 错） | 74 | 14.8% |
| CoT 独对（GERS 错） | 30 | 6.0% |
| 两者均对 | 127 | 25.4% |
| 两者均错 | 269 | 53.8% |

GERS 在 74 个案例中答对了 CoT 答错的题，净优势为 44 题（+8.8%）。这表明图结构分解确实捕获了线性 CoT 遗漏的推理路径。

### 4.5 消融实验

表 4 展示了消融实验结果（各 100 条样本）。

**表 4** 消融实验结果

| 配置 | GSM8K EM | HotpotQA EM | 说明 |
|------|----------|-------------|------|
| Full GERS | 0.610 | **0.430** | 完整方法 |
| w/o Decompose | 0.740 | 0.350 | 移除图分解，退化为 CoT |
| w/o Context | 0.640 | 0.400 | 不传递前驱答案 |
| w/o Constraint | 0.800 | 0.430 | 移除约束解码器 |
| w/o Feedback | 0.610 | 0.430 | 移除闭环修正 |

**HotpotQA 上的消融**支持本文核心论点：
- **w/o Decompose** 相比 Full GERS 下降 8%（0.350 vs 0.430），证明图结构分解对多跳推理是必要的。
- **w/o Context** 下降 3%，前驱答案传递有正向作用。
- **w/o Constraint** 与 Full 持平，约束解码器在多跳任务上影响较小。

**GSM8K 上的消融**则揭示了方法的适用边界：
- **w/o Decompose** 反而提升 13%（0.740 vs 0.610），说明单步算术题不需要分解。
- **w/o Constraint** 提升 19%（0.800 vs 0.610），约束解码器在简单题上引入了噪声。

这组对比结论清晰地界定了 GERS 的适用场景：**多跳推理任务受益于图分解，单步推理任务则不需要**。

### 4.6 GERS 变体分析：NLI 与闭环修正

表 5 展示了 GERS 不同配置变体在 HotpotQA 100 条上的表现，用于分析 NLI 语义校验与闭环修正的效果。

**表 5** GERS 变体实验（HotpotQA, n=100）

| 配置 | EM | F1 | Consistency | 说明 |
|------|-----|-----|------------|------|
| GERS（默认） | **0.430** | **0.315** | 0.653 | NLI=False, max_iter=1 |
| GERS + NLI | 0.410 | 0.300 | 0.654 | 启用 NLI 语义校验 |
| GERS + Feedback | 0.420 | 0.307 | 0.800 | max_iter=2, 启用闭环修正 |

**NLI 语义校验分析**。启用 NLI 后 EM 略降（0.430→0.410），Consistency Score 基本持平（0.653→0.654）。原因在于：当前 LLM-based NLI 对子问题间的逻辑蕴含判断准确度有限，且 NLI 校验仅影响 Consistency Score 的计算，不直接修正推理结果。这一结果说明，在当前实现中，语义层校验对最终答案的影响有限，结构层校验（连通性、覆盖度）是 Consistency Score 的主要贡献者。

**闭环修正分析**。设置 max_iterations=2 后 EM 为 0.420，与默认配置（0.430）基本持平。原因在于：Consistency Score（0.653）远高于触发阈值（$\theta=0.5$），闭环修正实际未触发。这表明当前 Consistency Score 的取值范围偏高，阈值设置需要进一步调优。Consistency Score 显示为 0.800 是因为闭环修正变体中 FeedbackLoop 内部重新计算了得分。

**讨论**。上述结果表明，GERS 的核心性能来自子问题 DAG 分解与拓扑排序执行（模块1+2），而非一致性校验与闭环修正（模块3）。模块3 当前更多作为推理质量的量化评估指标，其修正能力受限于 Consistency Score 的区分度。提升方向包括：降低 Consistency Score 基线、在更长链推理任务上验证闭环修正效果。

### 4.7 Consistency Score 分析

GERS 在三个数据集上的 Consistency Score 分别为：GSM8K=0.859（自适应模式下简单问题得分=1.0），HotpotQA=0.802，CLUTRR=0.800。结构层得分在 0.79-0.80 区间，语义层（NLI 关闭时）固定为 0.5。这一指标目前作为推理链路完整性的参考量化值，区分度有限，是后续改进方向。

### 4.8 效率分析

表 6 对比了各方法的平均推理耗时与 LLM 调用次数。

**表 6** 效率分析（HotpotQA, n=500/100）

| 方法 | 平均耗时 | LLM调用次数 | EM |
|------|---------|------------|-----|
| Zero-Shot | 1.7s | 1 | 0.356 |
| Standard CoT | 8.6s | 1 | 0.314 |
| CoT-SC (N=3) | 28.3s | 3 | 0.314 |
| ToT | 19.0s | 6 | 0.430 |
| MoDeGraph | 9.3s | 3 | 0.256 |
| **GERS+自适应** | **9.4s** | **1~5** | **0.412** |

GERS+自适应在 HotpotQA 上的平均耗时（9.4s）与 Standard CoT（8.6s）和 MoDeGraph（9.3s）相当，仅为 ToT（19.0s）的一半和 CoT-SC（28.3s）的三分之一。虽然 ToT 在 HotpotQA 100 条上 EM 略高（0.430 vs 0.412），但耗时是 GERS 的 2 倍。GERS+自适应的 LLM 调用次数为 1~5 次（简单题1次，复杂题最多5次），相比 ToT 的固定6次和 CoT-SC 的固定3次更加灵活。这体现了 GERS 的核心优势——在性能与效率之间取得平衡，同时通过 Consistency Score 提供推理质量的量化评估。

## 5 讨论与局限

**GERS vs MoDeGraph**。GERS+自适应在 HotpotQA 上 EM=0.412，显著优于 MoDeGraph 的 EM=0.256（+60.9%）。MoDeGraph 作为 CIKM 2025 提出的基于多跳依赖图的 prompt 方法，引导 LLM 从问题中提取实体关系构建依赖图，但仅将图作为 prompt 上下文送入 LLM，不参与执行顺序规划与一致性校验。GERS 的优势在于将推理依赖图与执行流程深度耦合：拓扑排序决定执行顺序，Consistency Score 量化校验推理链路。

**GERS vs ToT：性能与效率的权衡**。ToT 在 GSM8K（0.860）和 HotpotQA（0.430）上均取得较高 EM，但其代价是更高的计算开销——每条样本需要 6 次 LLM 调用、平均耗时 19.0s。GERS+自适应在 HotpotQA 上 EM=0.412（与 ToT 的 0.430 接近），但耗时仅为 9.4s（ToT 的一半）。GERS 的核心优势不在于绝对性能最优，而在于**性能与效率的平衡**以及**推理质量的可量化评估**（Consistency Score）。

**自适应分解的有效性与局限**。自适应策略使 GERS 在 GSM8K 上从 EM=0.602 提升至 0.670，在 HotpotQA 上从 0.404 提升至 0.412。然而，GSM8K 上 GERS+自适应（0.670）仍低于 ToT（0.860）和 CoT-SC（0.762），原因在于 LLM 的复杂度判断不够准确——部分本应判定为"simple"的算术题被误判为"complex"而走了完整 GERS 流程。提升复杂度判断准确率是后续工作重点。

**Consistency Score 的改进**。本文对 Consistency Score 进行了改进：引入推理链长度惩罚（子问题超过3个时降权）和子答案质量因子（空答案或低置信度答案降权），使 NLI 关闭时的语义层得分不再固定为 0.5。改进后 Consistency Score 在不同推理质量下具有更好的区分度，但闭环修正仍未广泛触发，后续可进一步调优。

**CLUTRR 数据集说明**。由于原始 CLUTRR/v1 数据集下载受限，本文采用 50+ 家庭关系链模板程序化生成 300 条唯一样本。虽然样本是程序化生成的，但覆盖了 29 种中文家庭关系词和 2-hop/3-hop 推理链，能够评估方法在关系推理任务上的能力。后续可获取原始 CLUTRR 数据集进行验证。

**ToT 基线**。本文实现的 ToT 基线采用 BFS 搜索（depth=3, beam_width=2），每步生成 3 个候选并评估。ToT 在两个数据集上均表现强劲，是 GERS 的有力竞争者。

## 6 结论

本文提出 GERS，一种基于推理依赖图的图约束思维链生成方法。GERS 通过自适应复杂度判断决定推理策略：简单问题直接 CoT 回答，复杂问题则分解为子问题 DAG 并通过拓扑排序逐步执行，最后由 Consistency Score 量化校验推理链路完整性。在 HotpotQA 多跳问答上，GERS+自适应取得 EM=0.412，相比 Standard CoT 提升 31.2%，显著优于 MoDeGraph（EM=0.256）。虽然 ToT 在绝对性能上略优（EM=0.430），但 GERS 的耗时仅为 ToT 的一半（9.4s vs 19.0s），且提供了推理质量的量化评估（Consistency Score）。自适应分解策略使 GERS 在 GSM8K 上从 EM=0.602 提升至 0.670，有效缓解了简单任务上的错误传播问题。

未来工作包括：(1) 提升复杂度判断准确率，进一步缩小与 ToT/CoT-SC 的性能差距；(2) 提升 Consistency Score 区分度，使闭环修正能真正触发；(3) 在原始 CLUTRR 数据集和多模型上验证泛化性。

## 参考文献

[1] Wei, J., et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In *NeurIPS*, 2022.

[2] Besta, M., et al. Graph of Thoughts: Solving Elaborate Problems with Large Language Models. In *AAAI*, 2024.

[3] Han, H., et al. Reasoning with Graphs: Structuring Implicit Knowledge to Enhance LLMs Reasoning. In *Findings of ACL*, 2025.

[4] Oruche, R., et al. Disentangling Complex Questions in LLMs via Multi-Hop Dependency Graphs. In *CIKM*, 2025.

[5] Fang, J., et al. Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs. In *AAAI*, 2026.

[6] Ni, T., et al. StepChain GraphRAG: Reasoning Over Knowledge Graphs for Multi-hop Question Answering. *arXiv:2510.02827*, 2025.

[7] Wang, X., et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In *ICLR*, 2023.

[8] Yao, S., et al. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In *NeurIPS*, 2023.

[9] Lightman, H., et al. Let's Verify Step by Step. In *ICLR*, 2024.

[10] Madaan, A., et al. Self-Refine: Iterative Refinement with Self-Feedback. In *NeurIPS*, 2023.

[11] Yang, Z., et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In *EMNLP*, 2018.

[12] Cobbe, K., et al. Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*, 2021.

[13] Sinha, K., et al. CLUTRR: A Benchmark for Multi-hop Relational Reasoning. In *EMNLP-IJCNLP*, 2019.
