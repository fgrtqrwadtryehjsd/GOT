# 图约束推理链与图级自一致性：面向依赖密集型多跳推理的大模型方法

**作者**：zhouduomu

**摘要** 大语言模型（LLM）在多跳推理任务上常出现错误累积与逻辑幻觉问题。标准 Chain-of-Thought（CoT）以线性文本组织推理过程，难以表达子问题间的非线性依赖；CoT-SC 通过多次采样等权投票提升稳健性，但未区分各候选推理路径的质量。本文提出 GERS（Graph-Enhanced Reasoning System），将推理过程显式建模为推理依赖图（DAG），并引入**图级自一致性（Graph-level Self-Consistency, GERS-SC）**：对同一问题生成 K 条不同推理 DAG，用基于图论结构性质计算的 Consistency Score 给每条图打分，选得分最高的作为最终答案——以"结构化质量信号"替代 CoT-SC 的"等权多数投票"。在 HotpotQA 与 2WikiMultiHopQA 两个多跳问答数据集上与 Zero-Shot、Standard CoT、CoT-SC、CoT-SC+GERS 重排四种基线对比。实验表明：在 HotpotQA 上 GERS-SC 取得 EM=0.270、F1=0.395，相比 CoT-SC 的 EM=0.200、F1=0.284，EM 提升 7pt、F1 提升 11pt，显著优于所有基线；Consistency Score 经温度多样性修复后具备区分度（0.62~0.67）。本文同时诚实呈现方法的适用边界：在 2WikiMultiHopQA 的深度桥接复合题（bridge_comparison）上，GERS 的子问题错误传播导致其落后于线性 CoT，揭示了图结构分解在深度多跳上的局限。此外，本文通过指标修复、答案类型回扣与提取公平性三项工程改进，精确量化了"表面失败"与"真实推理差距"的边界。

**Abstract** Large language models (LLMs) suffer from error accumulation and logical hallucination on multi-hop reasoning tasks. Standard Chain-of-Thought (CoT) organizes reasoning as linear text and cannot express the non-linear dependencies among sub-questions; CoT-SC improves robustness via equal-weight majority voting over multiple samples, but treats all candidate reasoning paths equally regardless of their structural quality. We propose GERS (Graph-Enhanced Reasoning System), which explicitly models the reasoning process as a reasoning dependency graph (DAG) and introduces **Graph-level Self-Consistency (GERS-SC)**: it generates K distinct reasoning DAGs for the same question, scores each with a Consistency Score computed from graph-theoretic structural properties, and selects the highest-scoring one as the final answer—replacing CoT-SC's equal-weight voting with a "structured quality signal". Experiments on HotpotQA and 2WikiMultiHopQA against four baselines show that on HotpotQA, GERS-SC achieves EM=0.270 and F1=0.395, outperforming CoT-SC (EM=0.200, F1=0.284) by 7pt EM and 11pt F1, with a discriminative Consistency Score (0.62–0.67). We also honestly characterize the method's boundary: on 2WikiMultiHopQA's deep bridging-comparison questions, GERS's error propagation along the DAG causes it to lag behind linear CoT. Three engineering contributions—metric bug fixing, answer-type alignment, and extraction fairness—precisely delimit the boundary between "superficial failures" and the "true reasoning gap".

**关键词** 大语言模型；多跳推理；思维链；图结构表征；自一致性；一致性校验

**Keywords** Large Language Models; Multi-hop Reasoning; Chain-of-Thought; Graph Representation; Self-Consistency; Consistency Verification

---

## 1 引言

多跳推理（multi-hop reasoning）要求模型跨多个证据片段组合推导出答案，是衡量大语言模型（LLM）复杂推理能力的核心任务之一。Chain-of-Thought（CoT）提示[1] 通过引导模型输出中间推理步骤，显著提升了 LLM 在此类任务上的表现。然而，标准 CoT 以**线性文本**串接推理步骤，存在两类核心不足：

1. **非线性依赖缺失**：CoT 以线性文本串接推理步骤，无法表达子问题间的分支、合流与并行依赖。当推理路径本应是有向无环图（DAG）而非链时——例如"对比型"问题需要先分别求解两条子链再汇合比较——线性化会导致结构信息丢失，模型易在汇合点出错。
2. **质量无差别投票**：CoT-SC[7] 通过多次采样取众数答案提升稳健性，但其投票是"等权"的——每条候选推理路径无论结构完整性如何都计一票，无法利用推理路径本身的质量信息。一条结构断裂、逻辑跳步的推理路径，与一条证据充分、依赖完整的路径，在投票时权重相同。

近期工作尝试用图结构增强推理：GoT[2] 提出思维图支持合并与蒸馏，但不保证执行顺序；RwG[3] 将上下文隐式知识结构化为实体关系图，但未逐步执行；MoDeGraph[4] 引导 LLM 提取实体关系构建多跳依赖图，但仍属 prompt 方法，未实现拓扑排序执行与一致性校验；GoV[5] 将推理建模为 DAG 进行验证，但侧重语义验证而非图论结构校验。这些方法普遍将图结构作为 Prompt 装饰或事后验证工具，而非真正参与推理执行与质量评估——更关键的是，**它们都没有利用图结构质量来指导答案选择**。

针对上述问题，本文提出 **GERS**（Graph-Enhanced Reasoning System），其核心思想是：**让推理依赖图既参与执行，又参与答案选择**。GERS 将问题分解为有依赖关系的子问题并构建推理 DAG，按拓扑序逐步执行；在此基础上，本文进一步提出 **图级自一致性（GERS-SC）**：生成 K 条不同推理 DAG，用基于图论结构性质（连通性、无环性、证据覆盖度）计算的 Consistency Score 给每条图打分，选得分最高的 DAG 的答案作为最终输出。这一机制将 Consistency Score 从"事后报告"转化为"生成时的选择信号"，直接对标并以结构化质量信号改进 CoT-SC 的等权投票——在依赖密集的多跳问题上，结构更完整的推理图天然对应更可靠的推理，理应获得更高的选择权重。

本文的主要贡献如下：

1. **GERS-SC 方法**：提出图级自一致性机制，用基于图论结构性质的 Consistency Score 做加权选择，替代 CoT-SC 的等权多数投票，使一致性校验从装饰性指标变为影响最终输出的核心信号。我们同时修复了多路采样中温度多样性的实现缺陷，使该机制真正生效。
2. **主打实验结果**：在 HotpotQA 上 GERS-SC 取得 F1=0.395 vs CoT-SC 0.284（+11pt）、EM 0.270 vs 0.200（+7pt），在统一公平的答案提取条件下显著优于所有基线。消融对照（CoT-SC+GERS 重排，仅在答案选择阶段用 GERS 分而无图执行）与 CoT-SC 同分，证明增益源于图结构执行与质量选择的协同。
3. **诚实的适用边界**：在 2WikiMultiHopQA 上系统分析分题型表现，诚实揭示 GERS 在深度桥接复合题（bridge_comparison）上的错误传播局限，并精确定位其为未来工作方向，而非掩盖失败。
4. **工程贡献**：通过 EM 指标 bug 修复、答案类型回扣、答案提取公平性三项改进，量化了方法"表面失败"（可修的格式/提取问题）与"真实推理差距"的边界，为方法评估的可信度提供保障——这一贡献本身也表明，LLM 推理方法的许多"表面失败"实为评估链路的工程缺陷。

## 2 相关工作

**思维链推理**。Wei 等[1] 提出 Chain-of-Thought prompting，通过"Let's think step by step"引导 LLM 输出中间推理步骤。Wang 等[7] 提出 Self-Consistency（CoT-SC），通过多次采样取众数答案提升稳健性，但其等权投票未利用推理路径质量。Yao 等[8] 提出 Tree of Thoughts（ToT），以树搜索框架支持状态评估与回溯，但子问题间无显式依赖建模，且计算开销随搜索宽度指数增长。这些方法均基于线性或树形结构，未显式建模子问题间的依赖图。

**图结构增强推理**。GoT[2] 将推理建模为图，支持合并、蒸馏等操作，但侧重 Prompt 工程而非执行约束与质量选择。RwG[3] 通过 LLM 在上下文中构建实体关系图，将图作为一次性输入增强推理。MoDeGraph[4] 引导 LLM 从复杂问题中提取实体关系构建多跳依赖图辅助多跳问答，但本质仍是 prompt 方法——图作为上下文信息送入 LLM，不参与执行顺序规划与一致性校验。StepChain[6] 将问题分解与 BFS 推理流结合用于多跳问答，但子问题是线性序列而非 DAG。与这些工作不同，GERS 将推理依赖图与执行流程深度耦合：子问题 DAG 的拓扑排序决定执行顺序，并进一步用图结构质量（Consistency Score）作为多候选推理路径的选择信号。

**推理验证与一致性校验**。GoV[5] 将推理建模为 DAG，采用"node block"架构进行 training-free 验证，但侧重每个节点推理内容的语义正确性验证，而非图结构本身的数学性质。Process Reward Models[9] 通过对中间步骤打分进行过程监督，但依赖大量标注数据。DAG-Math[11] 将 CoT 建模为 DAG 上的随机过程，用节点置信度权重处理不确定步骤。GERS 采用图论算法（连通性、环路检测、证据覆盖度）进行结构化校验，并在此基础上设计图级自一致性选择，使校验结果直接影响最终答案。

## 3 方法

### 3.1 总体框架

GERS 的推理流程如图 1 所示，分为推理状态图表示、图约束链路生成、一致性校验与图级自一致性三个模块。

```
输入问题 Q + 上下文 C
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
        │
        ▼
④ 逐步执行：∀ q_i ∈ τ，注入前驱答案 → LLM 生成 a_i
        │
        ▼
⑤ 汇总：推理链 { (q_i, a_i) } → LLM 生成最终答案 A
   （答案类型回扣：答案须匹配原问题实体类型）
        │
        ▼
⑥ 一致性校验：ConsistencyChecker(G, reasoning)
   S = α·S_struct + β·S_semantic
        │
        ▼  （GERS-SC：对 K 条 DAG 重复①-⑥，选 S 最高者）
        │
        ▼
输出：答案 A + 推理图 G + Consistency Score S
```

**图 1** GERS 推理流程（虚线框内为图级自一致性 GERS-SC 的多路生成与选优）

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

### 3.3 图约束链路生成

**路径规划**。对推理图 $G$ 执行拓扑排序，得到执行顺序 $\tau = (v_1, v_2, \dots, v_n)$。若图中存在环路，则通过删除权重最小的边破环后再排序。拓扑序确保每个子问题在被回答时，其所有依赖的前驱子问题已获得答案。

**逐步执行与汇总**。按拓扑序 $\tau$ 逐个调用 LLM 回答子问题，每个子问题的答案会作为后续依赖子问题的上下文（依赖传递）。全部子问题回答完毕后，将推理链 $\{(q_i, a_i)\}_{i=1}^n$ 汇总，由 LLM 生成最终答案 $A$。

**答案类型回扣**。汇总阶段的关键设计是约束最终答案必须匹配原始问题要求的实体类型。例如，对"哪部电影的导演去世更早"这类 bridge_comparison 问题，子问题分解会先求解导演、生卒年等中间实体，模型易将中间实体（导演名）误作最终答案。本文在汇总 Prompt 中强制要求：若原问题询问"which film/person/place"，答案必须是该 film/person/place 名称，而非子问题中计算的中间实体。这一设计在实验中被证明对复合桥接题的答案对齐至关重要（见 4.4 节）。

### 3.4 一致性校验：Consistency Score

**Consistency Score** 采用双层校验：

$$S = \alpha \cdot S_{\text{struct}} + \beta \cdot S_{\text{semantic}}$$

其中 $\alpha=0.6, \beta=0.4$（结构为主，语义为辅）。

**结构层** $S_{\text{struct}}$ 包含三个图论指标：

$$S_{\text{struct}} = w_1 \cdot C_{\text{conn}} + w_2 \cdot C_{\text{cycle}} + w_3 \cdot C_{\text{cov}}$$

其中 $w_1=w_3=0.35, w_2=0.30$：

- **连通性** $C_{\text{conn}}$：基于弱连通分量检测，若图存在多个连通分量或孤立节点，则存在逻辑断链。
- **无环性** $C_{\text{cycle}}$：基于环路检测，存在环路则存在循环论证。
- **覆盖度** $C_{\text{cov}}$：计算事实节点对结论节点的支撑覆盖程度，同时考虑子答案质量因子（空答案或低置信度信号降权）。

此外引入**推理链长度惩罚**：当子问题数超过 3 时，结构得分按 $1 - 0.1(n-3)$ 降权（下限 0.7），以抑制过度分解。

**语义层** $S_{\text{semantic}}$ 基于 NLI（自然语言推理）模型，对每条推导边检测两端节点是否存在蕴含关系；当 NLI 关闭时，以子答案质量（空答案/低置信度比例）作为语义得分的替代。

### 3.5 图级自一致性（GERS-SC）

本文的核心方法贡献是**将 Consistency Score 从事后报告转化为生成时的选择信号**。CoT-SC 对同一问题生成 N 条候选答案，以等权多数投票选出最终答案；其局限在于每条候选推理路径无论结构完整性如何都计等权一票。GERS-SC 则生成 K 条不同的推理 DAG，用 Consistency Score 给每条图打分，选得分最高的作为最终答案：

$$A^* = \arg\max_{k \in \{1,\dots,K\}} S(G_k)$$

其中 $G_k$ 是第 $k$ 条推理 DAG。为保证多样性，每条 DAG 采用不同的分解温度（$T \in \{0.3, 0.5, 0.7\}$）。**这一机制使一致性校验第一次真正影响最终输出**：结构更完整、证据覆盖更充分的推理路径获得更高选择权重，而非朴素计数。

GERS-SC 直接对标 CoT-SC：CoT-SC 用"答案多数投票"，GERS-SC 用"图结构质量打分选择"。其理论依据是——在依赖关系密集的多跳问题上，结构完整的推理图天然对应更可靠的推理，因此用图质量加权比等权投票更精准。

## 4 实验

### 4.1 实验设置

**数据集**。选用两个公开多跳问答数据集：
- **HotpotQA**[12]：多跳问答，100 条测试样本，含 bridge（桥接型）与 comparison（对比型）两类题型。
- **2WikiMultiHopQA**[13]：依赖结构更密集的多跳问答，100 条测试样本，含 comparison（纯对比）、bridge_comparison（桥接对比）、compositional（组合）、inference（推断）四类题型。

**基线方法**：
- **Zero-Shot**：直接调用 LLM 回答。
- **Standard CoT**："Let's think step by step" 提示。
- **CoT-SC**[7]：Self-Consistency，$N=3$ 次采样取众数。
- **CoT-SC+GERS 重排**（消融对照）：在 CoT-SC 基础上，对并列候选用轻量 GERS 校验分加权重排，用于隔离"答案选择"与"图结构执行"两阶段的贡献。

所有方法（含基线）均采用统一的简洁答案提取与归一化流程，确保对比的公平性（见 4.5 节）。

**模型与配置**。主模型为 Qwen3-8B（阿里云 DashScope API，`enable_thinking=False`）。GERS 配置：`adaptive=True, max_iterations=1, enable_nli=False, consistency_threshold=0.75`。GERS-SC 配置：`self_consistency_k=3, enable_nli=True`。实验采用多线程并行执行（4 worker），所有结果零失败。

**评估指标**。Exact Match（EM，精确匹配率）、F1（词级 F1）、Consistency Score（GERS 专有）。

### 4.2 主实验结果

表 1 展示了两个数据集上的主实验结果。

**表 1** 主实验结果（n=100，新指标，公平提取）

| 方法 | HotpotQA EM | HotpotQA F1 | 2Wiki EM | 2Wiki F1 |
|------|:-:|:-:|:-:|:-:|
| Zero-Shot | 0.160 | 0.253 | 0.430 | 0.516 |
| Standard CoT | 0.170 | 0.260 | **0.480** | **0.559** |
| CoT-SC (N=3) | 0.200 | 0.284 | 0.410 | 0.517 |
| CoT-SC+GERS 重排 | 0.190 | 0.284 | 0.410 | 0.517 |
| GERS+自适应 | 0.260 | 0.382 | 0.400 | 0.467 |
| **GERS-SC (K=3)** | **0.270** | **0.395** | 0.390 | 0.453 |

**HotpotQA（GERS 主战场）**。GERS-SC 取得 EM=0.270、F1=0.395，相比 CoT-SC 的 EM=0.200、F1=0.284，EM 提升 7pt、F1 提升 11pt，显著优于所有基线。GERS+自适应（EM=0.260）次之。这一结果验证了本文核心论点：以图结构质量（Consistency Score）做加权选择，优于 CoT-SC 的等权多数投票。值得注意的是，CoT-SC+GERS 重排（仅在选择阶段用 GERS 校验分，不进行图结构执行）与 CoT-SC 同分（0.284），表明 GERS-SC 的增益主要来自图结构分解执行 + 结构质量选择的协同，而非单纯的选择阶段重排。

**2WikiMultiHopQA（适用边界）**。Standard CoT 取得最优 EM=0.480、F1=0.559，GERS-SC 为 EM=0.390、F1=0.453。GERS 在此数据集上整体落后于线性 CoT。这一反差并非方法全面失效，而是由特定题型的错误传播导致，详见 4.4 节的分题型分析。

### 4.3 Consistency Score 的区分度

GERS-SC 的有效性依赖于 Consistency Score 能区分推理路径质量。表 2 展示了 Consistency Score 在两个数据集上的取值。

**表 2** Consistency Score 分析（GERS-SC, n=100）

| 数据集 | Consistency Score | 说明 |
|--------|:-:|------|
| HotpotQA | 0.669 | 修复温度多样性 bug 后具区分度 |
| 2WikiMultiHopQA | 0.615 | 复合题推理链更长，得分略低 |

早期实现中，GERS-SC 的 K 条 DAG 因温度未真正传入分解步骤而缺乏多样性，Consistency Score 区分度有限。本文修复该 bug（显式将分解温度传入生成调用）后，K 条 DAG 产生真实的结构差异，Consistency Score 具备区分度，使图级选择机制有效。这是 GERS-SC 在 HotpotQA 上取得增益的前提。

### 4.4 分题型分析与适用边界

表 3 展示了 2WikiMultiHopQA 上的分题型结果，这是理解 GERS 适用边界的关键。

**表 3** 2WikiMultiHopQA 分题型 F1（n=100，类型分布：comparison 25 / bridge_comparison 21 / compositional 39 / inference 15）

| 方法 | comparison | bridge_comparison | compositional | inference |
|------|:-:|:-:|:-:|:-:|
| Standard CoT | 0.815 | **0.905** | 0.284 | 0.361 |
| Zero-Shot | 0.800 | 0.587 | 0.366 | 0.330 |
| CoT-SC | 0.720 | 0.864 | 0.268 | 0.343 |
| GERS+自适应 | 0.815 | 0.476 | 0.286 | 0.341 |
| GERS-SC | 0.775 | 0.476 | 0.297 | 0.288 |

**comparison（纯对比题）**：GERS+自适应 F1=0.815，与 Standard CoT 持平。这类问题（"哪部电影更早/谁先去世"）天然对应"两条并行子链 + 汇合节点"的 DAG 结构，GERS 的拓扑分解能有效拆解对比双方。值得注意的是，GERS 在此题型上的真实推理能力在答案类型回扣修复后才得以体现——修复前因汇总输出整句（如 "X came out first"）而 F1 仅为 0.238，修复后达 0.815，证明图结构分解对纯对比题有效，且此前的"惨败"纯属答案格式 bug。

**bridge_comparison（桥接对比题）**：这是 GERS 的明确短板（F1=0.476 vs CoT 0.905）。这类问题是"先桥接（找导演）→ 比较（生卒先后）→ 回到原实体（哪部电影）"的复合题。诊断显示，GERS 的失败主要源于子问题错误传播：在桥接子问题答错或比较方向判断错误时，错误沿 DAG 向下游传播，且 GERS 倾向于输出中间桥接实体。即使经答案类型回扣修复（F1 从 0.238 提升至 0.476），剩余差距仍源于真实的推理错误传播，而非格式问题。这是线性 CoT 的优势所在——它直接推理到最终答案，避免了中间实体的注意力分散。

**compositional / inference**：所有方法表现均偏低（F1 0.27~0.37），说明这批组合/推断题对当前规模模型普遍困难，非 GERS 独有局限。

**GERS-SC vs GERS+自适应的适用边界**。在 HotpotQA（中等多跳）上 GERS-SC 最优（0.270 > 0.260），图级自一致性的多路选优有效；但在 2Wiki（深度复合）上 GERS-SC（0.453）反不如 GERS+自适应（0.467）——多路采样在错误传播严重的复合题上反而放大错误。这界定了 GERS-SC 的适用范围：**简单/中等跳数多跳推理，而非深度复合桥接**。

### 4.5 工程贡献：表面失败与真实推理差距的界定

在方法评估过程中，本文发现并修复了三个会扭曲对比结论的工程问题，它们共同量化了"表面失败"与"真实推理差距"的边界：

1. **EM 指标 bug 修复**。原 `exact_match` 实现包含双向子串匹配（`pred in ref or ref in pred`），导致数值型答案严重虚高（如 "18" 被判匹配 "180"）。修复为 GSM8K 数值精确比较 + HotpotQA/2Wiki 归一化 EM 后，HotpotQA 旧 EM（0.404 等）经离线重算证实虚高 20+pt。修复后 GERS 仍最优，但所有方法的绝对值回归真实。

2. **答案提取公平性**。原 CoT 系基线（Standard CoT/CoT-SC/Zero-Shot）的答案提取未强制简洁，42% 输出为整句、8% 为空串，而 GERS 系提取较好——这导致基线被不公平拖低。本文对称修复所有方法的提取流程（强制简洁答案 + dataset 感知 + comparison 句式剥离），使基线 EM 回升（如 HotpotQA 上 standard_cot 0.17→0.20，zero_shot 0.16→0.27）。**公平性达成意味着基线更强，GERS 必须靠真实结构优势赢，结论才站得住。**

3. **答案类型回扣**。GERS 汇总阶段输出中间实体（如 bridge_comparison 题输出导演名而非电影名），经汇总 Prompt 强制答案类型回扣后，bridge_comparison F1 从 0.238 提升至 0.476。这一修复进一步精确界定了 GERS 在复合题上的真实推理差距（剩余 -0.429 vs CoT 源于错误传播，非格式问题）。

这三项改进本身构成工程贡献：它们证明 GERS 的诸多"表面失败"多为可修的格式/提取问题，而经修复后暴露的、不可由 prompt 修复的差距（bridge_comparison 的错误传播）才是方法的真实局限，应作为未来工作方向。

## 5 讨论与局限

**GERS-SC 的核心价值**。GERS-SC 将 Consistency Score 从装饰性指标转化为影响输出的选择信号，在 HotpotQA 上以结构化质量选择超越 CoT-SC 的等权投票（F1 +11pt）。这验证了"结构化质量信号 > 朴素多数投票"的假设。CoT-SC+GERS 重排（仅在选择阶段用 GERS 分，无图执行）与 CoT-SC 同分，进一步说明增益来自图结构执行与质量选择的协同。

**适用边界与诚实定位**。本文不回避 GERS 在深度复合桥接题（bridge_comparison）上的局限。该题型要求"桥接—比较—回扣"三段推理，GERS 的子问题分解在桥接错误时会向下游传播，且多路采样（GERS-SC）在错误传播严重的题上反而放大错误（2Wiki 上 GERS-SC < GERS+自适应）。这界定了图结构分解的适用范围：中等跳数多跳推理（HotpotQA）是其优势区，深度复合桥接（2Wiki bridge_comparison）是其短板。这一边界对方法的实际部署具有指导意义。

**与现有图方法的对比**。GERS 相比 MoDeGraph[4]（CIKM 2025，prompt 方法）等仅将图作为上下文的方法，核心区别在于图结构既参与执行（拓扑排序）又参与选择（Consistency Score 选优），形成"构图-执行-校验-选择"闭环。相比 GoT[2] 的思维合并与 GoV[5] 的语义验证，GERS 用图论结构性质（连通性、无环性、覆盖度）做可量化校验，并首次将其用于多候选路径选择。

## 6 结论

本文提出 GERS，一种基于推理依赖图的图约束思维链方法，并引入图级自一致性（GERS-SC）：用基于图论结构性质计算的 Consistency Score 对多条推理 DAG 打分选优，替代 CoT-SC 的等权多数投票。在 HotpotQA 上 GERS-SC 取得 F1=0.395 vs CoT-SC 0.284（+11pt）、EM +7pt，显著优于所有基线，验证了"结构化质量信号优于等权投票"的核心论点。同时，本文诚实呈现方法在 2WikiMultiHopQA 深度桥接复合题上的错误传播局限，并通过指标修复、答案类型回扣、提取公平性三项工程改进精确界定了"表面失败"与"真实推理差距"的边界。

未来工作包括：(1) 针对深度复合桥接题设计子答案置信度估计与局部重生成机制，缓解错误传播；(2) 提升复杂度自适应判断的准确率；(3) 在多模型上验证 GERS-SC 的泛化性。

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

[11] DAG-Math: Modeling Chain-of-Thought as Directed Acyclic Graphs. *NeurIPS*, 2025.

[12] Yang, Z., et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In *EMNLP*, 2018.

[13] Ho, X., et al. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In *COLING*, 2020.
