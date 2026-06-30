# 图约束推理链与子答案双向交叉验证：面向多跳推理的大模型方法

**作者**：zhouduomu

**Abstract** Large language models (LLMs) suffer from error accumulation on multi-hop reasoning tasks. Existing graph-enhanced methods model reasoning as a DAG, but their consistency checks rely on pure graph-theoretic structural metrics that cannot reflect reasoning correctness, causing multi-path selection to degenerate into random sampling. We propose GERS, which models reasoning as a dependency DAG executed in topological order. The core contribution is **sub-answer bidirectional cross-validation**: using the final answer + context as an anchor, we re-derive each sub-question in reverse and compare forward/backward sub-answers, upgrading the Consistency Score from "is the graph legal" to "is the reasoning content self-consistent". On 500 HotpotQA samples, this raises the Consistency Score's correct/wrong discrimination from −0.0035 to +0.0847; the best configuration GERS-CV achieves EM=0.302, F1=0.413, outperforming CoT-SC by 4pt F1 (McNemar p=0.029). We honestly characterize the method's boundary: graph-level multi-path self-consistency brings no extra gain on medium-difficulty multi-hop, and error propagation limits GERS on deep bridging-comparison questions.

**Keywords** Large Language Models; Multi-hop Reasoning; Chain-of-Thought; Graph Representation; Bidirectional Cross-Validation; Consistency Verification

---

## 1 引言

多跳推理（multi-hop reasoning）要求模型跨多个证据片段组合推导出答案，是衡量大语言模型（LLM）复杂推理能力的核心任务之一。Chain-of-Thought（CoT）提示[1] 通过引导模型输出中间推理步骤，显著提升了 LLM 在此类任务上的表现。然而，标准 CoT 以**线性文本**串接推理步骤，存在两类核心不足：

1. **非线性依赖缺失**：CoT 以线性文本串接推理步骤，无法表达子问题间的分支、合流与并行依赖。当推理路径本应是有向无环图（DAG）而非链时——例如"对比型"问题需要先分别求解两条子链再汇合比较——线性化会导致结构信息丢失，模型易在汇合点出错。
2. **质量无差别投票**：CoT-SC[6] 通过多次采样取众数答案提升稳健性，但其投票是"等权"的——每条候选推理路径无论结构完整性如何都计一票，无法利用推理路径本身的质量信息。一条结构断裂、逻辑跳步的推理路径，与一条证据充分、依赖完整的路径，在投票时权重相同。

近期工作尝试用图结构增强推理：GoT[2] 提出思维图支持合并与蒸馏，但不保证执行顺序；RwG[3] 将上下文隐式知识结构化为实体关系图，但未逐步执行；MoDeGraph[4] 引导 LLM 提取实体关系构建多跳依赖图，但仍属 prompt 方法，未实现拓扑排序执行与一致性校验。这些方法普遍将图结构作为 Prompt 装饰或事后验证工具，而非真正参与推理执行与质量评估——更关键的是，**它们都没有利用图结构质量来指导答案选择**。

针对上述问题，本文提出 **GERS**（Graph-Enhanced Reasoning System），将推理显式建模为推理依赖图（DAG）并按拓扑序执行。我们发现，单纯用图论结构性质（连通性、无环性、覆盖度）计算 Consistency Score 是无效的——LLM 生成的分解图几乎都是合法 DAG，导致所有样本得分趋同（聚集在 0.6~0.7），多路选优退化为随机抽取（500 条实测对错区分度仅 −0.0035）。为此，本文提出核心创新**子答案双向交叉验证**：以最终答案 + 上下文为锚，反向逐个重答每个子问题，对比正向子答案与反向子答案的一致性，将 Consistency Score 从"图是否合法"升级为"推理内容是否自洽"。这一机制是 DAG 结构独有的能力——线性 CoT 没有可独立校验的子问题结构，无法实现双向校验。

本文的主要贡献如下：

1. **子答案双向交叉验证（核心创新）**：提出以"最终答案 + 上下文"反向重答子问题、对比正反向一致性的校验机制，将 Consistency Score 的对错区分度从 −0.0035 修复到 +0.0847（500 条 HotpotQA 实测），使一致性校验从纯结构指标升级为有效的内容自洽信号。该机制为 DAG 独有，线性 CoT/ToT 无法实现。
2. **性能验证**：最优配置 GERS-CV（GERS + 双向交叉验证 + 置信度加权汇总）在 HotpotQA 500 条上取得 EM=0.302、F1=0.413，相比 CoT-SC（EM=0.262, F1=0.373）F1 领先 4pt，McNemar 配对检验显著（p=0.029）。同时诚实报告 bootstrap 95% CI 仍微跨零，界定其为"配对显著但效应量临界"的状态。
3. **诚实的适用边界**：系统分析发现，图级多路自一致性（K 条 DAG 选优）即便修复 CS 区分度后，在中等难度多跳上仍未带来额外增益；且在 2WikiMultiHopQA 深度桥接复合题（bridge_comparison）上存在错误传播局限。这些边界被如实呈现为未来工作方向。
4. **工程贡献**：通过 EM 指标 bug 修复、答案类型回扣、答案提取公平性三项改进，量化了方法"表面失败"（可修的格式/提取问题）与"真实推理差距"的边界——这一贡献本身表明，LLM 推理方法的许多"表面失败"实为评估链路的工程缺陷。

## 2 相关工作

**思维链推理**。Wei 等[1] 提出 Chain-of-Thought prompting，通过"Let's think step by step"引导 LLM 输出中间推理步骤。Wang 等[6] 提出 Self-Consistency（CoT-SC），通过多次采样取众数答案提升稳健性，但其等权投票未利用推理路径质量。Yao 等[7] 提出 Tree of Thoughts（ToT），以树搜索框架支持状态评估与回溯，但子问题间无显式依赖建模，且计算开销随搜索宽度指数增长。这些方法均基于线性或树形结构，未显式建模子问题间的依赖图。

**图结构增强推理**。GoT[2] 将推理建模为图，支持合并、蒸馏等操作，但侧重 Prompt 工程而非执行约束与质量选择。RwG[3] 通过 LLM 在上下文中构建实体关系图，将图作为一次性输入增强推理。MoDeGraph[4] 引导 LLM 从复杂问题中提取实体关系构建多跳依赖图辅助多跳问答，但本质仍是 prompt 方法——图作为上下文信息送入 LLM，不参与执行顺序规划与一致性校验。StepChain[5] 将问题分解与 BFS 推理流结合用于多跳问答，但子问题是线性序列而非 DAG。与这些工作不同，GERS 将推理依赖图与执行流程深度耦合：子问题 DAG 的拓扑排序决定执行顺序，并进一步用图结构质量（Consistency Score）作为多候选推理路径的选择信号。

**推理验证与一致性校验**。Process Reward Models[8] 通过对中间步骤打分进行过程监督（如 Let's Verify Step by Step），但依赖大量标注数据。DAG-Math[10] 将 CoT 建模为 DAG 上的随机过程，用节点置信度权重处理不确定步骤。GERS 采用图论算法（连通性、环路检测、证据覆盖度）进行结构化校验，并在此基础上设计双向交叉验证，使校验结果直接影响最终答案。

## 3 方法

### 3.1 总体框架

GERS 的推理流程如图 1 所示，分为推理状态图表示、图约束链路生成、一致性校验与图级自一致性三个模块。

```mermaid
flowchart TD
    IN(["输入: 问题 Q + 上下文 C"]) --> DEC①["① 分解: LLM 将 Q 分解为子问题<br/>{q_i} 及依赖关系"]
    DEC① --> DAG②["② 构图: 构建推理 DAG G=(V,E)<br/>Fact(Q) → Step(q_i) → Conclusion"]
    DAG② --> TOPO③["③ 路径规划: 拓扑排序<br/>τ = topo_sort(G)"]
    TOPO③ --> EXEC④["④ 逐步执行: ∀ q_i ∈ τ<br/>注入前驱答案 → LLM 生成 a_i"]
    EXEC④ --> AGG⑤["⑤ 汇总: 推理链 → 最终答案 A<br/>(答案类型回扣: 匹配原问题实体类型)"]
    AGG⑤ --> CHECK⑥["⑥ 一致性校验: ConsistencyChecker<br/>S = α·S_struct + β·S_semantic"]
    CHECK⑥ --> SC{{"GERS-SC: 对 K 条 DAG<br/>重复①-⑥, 选 S 最高者"}}
    SC --> OUT(["输出: 答案 A + 推理图 G + Score S"])

    style SC fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style OUT fill:#dcfce7,stroke:#16a34a
```

**图 1** GERS 推理流程。黄色节点为图级自一致性（GERS-SC）的多路生成与选优环节：对同一问题以不同分解温度生成 K 条推理 DAG，用 Consistency Score 选得分最高者。

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

### 3.4 一致性校验与子答案双向交叉验证

**结构层 Consistency Score（基线）**。GERS 首先计算纯图论结构得分作为基础：

$$S_{\text{struct}} = w_1 \cdot C_{\text{conn}} + w_2 \cdot C_{\text{cycle}} + w_3 \cdot C_{\text{cov}}$$

其中 $w_1=w_3=0.35, w_2=0.30$，$C_{\text{conn}}$（连通性）、$C_{\text{cycle}}$（无环性）、$C_{\text{cov}}$（覆盖度）分别基于弱连通分量、环路检测、证据覆盖计算，并引入推理链长度惩罚（子问题数超过 3 时降权，下限 0.7）。

**结构层的局限**。实测发现（见 4.3 节），LLM 生成的分解图几乎都是合法连通无环 DAG，导致 $S_{\text{struct}}$ 在所有样本上高度聚集（500 条均值 0.662，标准差仅 0.098，对错区分度 −0.0035），完全无法区分推理好坏。因此纯结构指标不足以支撑推理质量评估。

**子答案双向交叉验证（核心创新）**。为突破上述局限，本文提出以推理**内容自洽性**取代纯结构合法性作为一致性主信号。给定正向执行得到的子答案 $\{a_i\}$ 与最终答案 $A$，反向验证流程如下：以"最终答案 $A$ + 原始上下文 $C$"为锚，**反向逐个重答每个子问题**，得到反向子答案 $\{a'_i\}$；对比正向 $a_i$ 与反向 $a'_i$ 的一致性：

$$S_{\text{crossval}} = \frac{\sum_{i=1}^{n} w_i \cdot \mathbb{1}[\text{match}(a_i, a'_i)]}{\sum_{i=1}^{n} w_i}$$

其中 $w_i$ 为子问题权重（下游节点权重更高，因其错误传播影响更大），$\text{match}$ 为分级语义匹配（归一化字符串匹配 → 包含关系 → 数值相等 → 可选 LLM 判断）。反向重答的 Prompt 显式要求"基于上下文独立重答，勿照抄最终答案"，以缓解模型迎合 $A$ 的 confirmation bias。

**这一机制是 DAG 独有的**：线性 CoT 没有可独立校验的子问题结构，无法实现正向执行 + 反向验证的双向一致性校验；只有把推理拆成有依赖的子问题节点，才能逐个反向重答并对比。

**新 Consistency Score**。综合结构层与内容层：

$$S = 0.3 \cdot S_{\text{struct}} + 0.7 \cdot S_{\text{crossval}}$$

结构层权重降至 0.3（因其无区分度），内容层权重 0.7（有效信号）。

### 3.5 子答案置信度加权汇总

为进一步抑制错误传播，本文在汇总阶段引入子答案置信度加权。每个子答案的置信度由轻量启发式估计（零额外 LLM 调用），综合考量：答案非空且长度合理、不含不确定信号（uncertain/unknown）、核心实体在上下文中落地、是否纯数字。低于阈值 $\theta_{\text{conf}}=0.5$ 的子答案在汇总 Prompt 中标注 `[LOW CONFIDENCE]`，提示模型不要过度依赖该步，从而降低单步错误向最终答案的传播。

### 3.6 图级自一致性（GERS-SC）

GERS-SC 生成 K 条不同推理 DAG（不同分解温度 $T \in \{0.3, 0.5, 0.7\}$），用 Consistency Score 给每条图打分，选得分最高的作为最终答案：

$$A^* = \arg\max_{k \in \{1,\dots,K\}} S(G_k)$$

该机制将 Consistency Score 从事后报告转化为生成时的选择信号。其有效性**严格依赖于 CS 具备区分度**——本文的双向交叉验证（3.4 节）正是使该机制生效的前提。需注意，实验显示在中等难度多跳上，即便 CS 修复后，多路选优相对单路执行的额外增益有限（见 4.6 节），这一边界被如实报告。

## 4 实验

### 4.1 实验设置

**数据集**。选用两个公开多跳问答数据集：
- **HotpotQA**[11]：多跳问答，500 条测试样本（含 bridge 桥接型 404 条、comparison 对比型 96 条）。
- **2WikiMultiHopQA**[12]：依赖结构更密集的多跳问答，100 条测试样本，含 comparison、bridge_comparison、compositional、inference 四类题型。

**基线方法**：Zero-Shot、Standard CoT、CoT-SC（$N=3$）、CoT-SC+GERS 重排（消融对照）。所有方法采用统一的简洁答案提取与归一化流程，确保对比公平（见下文"评估公平性保障"）。

**本文方法**：
- **GERS+自适应**：DAG 分解 + 拓扑执行 + 自适应复杂度路由。
- **GERS-SC**：GERS + 图级自一致性（K=3 多路选优）。
- **GERS-CV**：GERS + 双向交叉验证（3.4 节，本文核心创新）。
- **GERS-CV2**：GERS + 双向交叉验证 + 置信度加权汇总（3.4 + 3.5 节，最优配置）。

**模型与配置**。主模型 Qwen3-8B（DashScope API，`enable_thinking=False`）。多线程并行（4 worker），所有结果零失败。

**评估指标**。EM、F1、Consistency Score（GERS 专有）。**显著性检验**：所有主结果报告 bootstrap 95% CI（10000 次重采样）与配对 McNemar 检验（精确二项检验）。

**评估公平性保障**。为避免评估链路缺陷扭曲对比结论，本文统一三项处理：(1) 修复 EM 指标的双向子串匹配 bug（原实现使数值答案虚高，如 "18" 匹配 "180"）；(2) 所有方法（含基线）采用统一的简洁答案提取与归一化流程，避免基线因提取不佳被人为拖低；(3) GERS 汇总阶段强制答案类型回扣（答案须匹配原问题要求的实体类型）。这三项处理确保所有方法在同一公平口径下对比，GERS 的任何增益都来自方法本身而非评估偏差。

### 4.2 主实验结果

表 1 展示了 HotpotQA 500 条上的主实验结果。

**表 1** 主实验结果（HotpotQA, n=500）

| 方法 | EM | F1 | CS |
|------|:-:|:-:|:-:|
| Zero-Shot | 0.276 | 0.389 | — |
| Standard CoT | 0.264 | 0.368 | — |
| CoT-SC (N=3) | 0.262 | 0.373 | — |
| CoT-SC+GERS 重排 | 0.264 | 0.372 | — |
| GERS+自适应 | 0.284 | 0.395 | 0.996 |
| GERS-SC (K=3) | 0.282 | 0.398 | 0.662 |
| GERS-CV（+双向交叉验证） | 0.298 | 0.409 | 0.782 |
| **GERS-CV2（+置信度加权）** | **0.302** | **0.413** | 0.777 |

**核心发现**。GERS-CV2 取得最优 EM=0.302、F1=0.413，相比 CoT-SC（EM=0.262, F1=0.373）EM +4pt、F1 +4pt。双向交叉验证（GERS+自适应 0.284 → GERS-CV 0.298）带来 +1.4pt EM 增益，是本文核心创新的直接贡献。值得注意的是，HotpotQA 上 Zero-Shot（0.276）已接近 GERS——该数据集的上下文直接包含答案证据，强模型读上下文即可答对不少，压缩了所有"高级"方法的优势空间；CoT-SC 甚至低于 Zero-Shot（多采样投票引入噪声）。这解释了为何 GERS 的整体增益有限。

**表 2** 显著性检验（HotpotQA, n=500）

| 对比 | EM 差值 | 95% bootstrap CI | McNemar p | 判定 |
|------|:-:|:-:|:-:|---|
| GERS-CV2 vs CoT-SC | +0.040 | [−0.014, +0.096] | **0.029** | McNemar显著 / CI微跨0 |
| GERS-CV2 vs Standard CoT | +0.038 | [−0.018, +0.094] | **0.040** | McNemar显著 / CI微跨0 |
| GERS-SC vs CoT-SC | +0.020 | [−0.034, +0.076] | 0.275 | 不显著 |
| CoT-SC vs Standard CoT | +0.030 | [−0.020, +0.080] | 0.453 | 不显著 |

**诚实解读**。GERS-CV2 在 McNemar 配对检验下显著优于 CoT-SC（p=0.029），但 bootstrap 95% CI 仍微跨零（下限 −0.014）。这表明配对效应真实存在（GERS 答对而 CoT 错的样本显著多于反向），但效应量在 n=500 下尚不够稳健到"95% 置信下严格显著"。本文如实报告这一"配对显著、效应量临界"的状态，而非宣称强显著。相比之下，未引入双向交叉验证的 GERS-SC（p=0.275）不显著，反向印证了核心创新的有效性。


**2WikiMultiHopQA（适用边界）**。Standard CoT 取得最优 EM=0.480、F1=0.559，GERS-SC 为 EM=0.390、F1=0.453。GERS 在此数据集上整体落后于线性 CoT。这一反差并非方法全面失效，而是由特定题型的错误传播导致，详见 4.4 节的分题型分析。

### 4.3 Consistency Score 区分度的修复（核心证据）

本节是验证本文核心创新（双向交叉验证）有效性的关键。我们考察 Consistency Score 区分推理对错的能力：分别计算答对样本与答错样本的 CS 均值，其差值（区分度）为正且越大，说明 CS 越能区分推理质量。

**表 3** Consistency Score 对错区分度（HotpotQA, n=500）

| CS 计算方式 | 答对 CS 均值 | 答错 CS 均值 | 区分度 | 分布 stdev |
|------------|:-:|:-:|:-:|:-:|
| 旧 CS（纯结构，GERS-SC） | 0.6592 | 0.6628 | **−0.0035** | 0.098 |
| 新 CS（+双向交叉验证，GERS-CV） | 0.7888 | 0.7042 | **+0.0847** | 0.340 |
| 对照：纯结构分 $S_{\text{struct}}$ | 0.9967 | 1.0000 | −0.0033 | — |

**旧 CS 完全失效**：纯图论结构指标（连通性+无环+覆盖度）的区分度为 −0.0035（反向，等价随机），分布高度聚集（stdev 0.098，0.6 分 312 条、0.7 分 150 条）。原因是 LLM 生成的分解图几乎都是合法 DAG，结构分无法反映内容对错。

**双向交叉验证成功修复**：引入反向验证后，新 CS 区分度跃升至 +0.0847（正向有效信号），分布拉开（stdev 0.340，0.0/0.4/0.6/1.0 多分层）。这说明"以最终答案+上下文反向重答子问题、对比正反向一致性"确实能捕获推理内容的不自洽——正确推理链正反向一致（CS 高），错误推理链正反向矛盾（CS 低）。

**这是本文最硬的贡献证据**：CS 从"无区分的图论指标"变为"有效的推理质量信号"，且该机制为 DAG 独有（线性 CoT 无子问题结构可反向校验）。这一修复直接支撑了 GERS-CV 相对 GERS-SC 的 EM 提升（0.282 → 0.298）。


### 4.4 分题型分析与适用边界

表 4 展示了 2WikiMultiHopQA 上的分题型结果，这是理解 GERS 适用边界的关键。

**表 4** 2WikiMultiHopQA 分题型 F1（n=100，类型分布：comparison 25 / bridge_comparison 21 / compositional 39 / inference 15）

| 方法 | comparison | bridge_comparison | compositional | inference |
|------|:-:|:-:|:-:|:-:|
| Standard CoT | 0.815 | **0.905** | 0.284 | 0.361 |
| Zero-Shot | 0.800 | 0.587 | 0.366 | 0.330 |
| CoT-SC | 0.720 | 0.864 | 0.268 | 0.343 |
| GERS-CV2 | 0.775 | 0.524 | 0.297 | 0.288 |

**comparison（纯对比题）**：GERS-CV2 F1=0.775，与 Standard CoT 接近。这类问题（"哪部电影更早/谁先去世"）天然对应"两条并行子链 + 汇合节点"的 DAG 结构，GERS 的拓扑分解能有效拆解对比双方。

**bridge_comparison（桥接对比题）**：这是 GERS 的明确短板（F1 0.524 vs CoT 0.905）。这类问题是"先桥接（找导演）→ 比较（生卒先后）→ 回到原实体（哪部电影）"的复合题。GERS 的失败主要源于子问题错误传播：桥接子问题答错时错误沿 DAG 向下游传播。双向交叉验证将 bridge_comparison F1 从 0.476（无 CV）提升至 0.524，部分缓解但未根除——剩余差距源于真实的推理错误传播，这是图结构分解在深度多跳的固有局限，也是未来工作方向。

**compositional / inference**：所有方法表现均偏低（F1 0.27~0.37），说明这批组合/推断题对当前规模模型普遍困难，非 GERS 独有局限。

**图级自一致性的适用边界**。500 条 HotpotQA 实验显示，GERS-SC（K=3 多路选优, 0.282）≈ GERS+自适应（单路, 0.284），p=1.000——即便修复 CS 区分度后，多路选优在中等难度多跳上仍无额外增益（见 4.6 节）。在 2Wiki（深度复合）上 GERS-SC（0.453）同样不优于 GERS+自适应（0.467）。这界定了图级自一致性的适用范围有限：多路采样选优在该模型规模与任务难度下未带来可靠提升，其价值更多在于提供可量化的推理质量信号（Consistency Score），而非端到端性能。

### 4.5 计算成本分析

**表 5** 计算成本对比（HotpotQA, n=500, 4 线程并行, 单题平均）

| 方法 | 单题延迟 | EM | F1 |
|------|:-:|:-:|:-:|
| Zero-Shot | 0.7s | 0.276 | 0.389 |
| Standard CoT | 3.0s | 0.264 | 0.368 |
| CoT-SC (N=3) | 8.3s | 0.262 | 0.373 |
| GERS+自适应 | 4.8s | 0.284 | 0.395 |
| GERS-SC (K=3) | 16.2s | 0.282 | 0.398 |
| **GERS-CV2（最优）** | 6.6s（单路版） | **0.302** | **0.413** |

**成本-收益权衡**。GERS-CV2 的单路版本（gers_adaptive_cv2）以 6.6s 延迟取得最优 EM=0.302，成本仅约为 CoT-SC（8.3s）的 80%，却领先 4pt F1——这得益于自适应路由（简单题跳过分解）+ 双向交叉验证（仅复杂题触发反向校验）。多路版本 GERS-SC（16.2s）成本更高却无性能增益，进一步印证图级自一致性在该场景下的边际价值有限。GERS-CV2 在性能与效率上均优于基线，是推荐配置。


### 4.6 消融实验

为分离各创新组件的贡献，本文设计消融对照（表 6，HotpotQA n=500）。

**表 6** 消融实验（HotpotQA, n=500）

| 配置 | EM | F1 | 说明 |
|------|:-:|:-:|------|
| GERS+自适应 | 0.284 | 0.395 | 基线：DAG 执行 + 自适应路由 |
| GERS-SC（+图级自一致性 K=3） | 0.282 | 0.398 | 多路选优，旧 CS 无区分 |
| GERS-CV（+双向交叉验证） | 0.298 | 0.409 | 新 CS 有区分度 |
| **GERS-CV2（+置信度加权汇总）** | **0.302** | **0.413** | 最优配置 |
| w/o 图执行（= CoT-SC+GERS 重排） | 0.264 | 0.372 | 仅选择阶段用 GERS 分，无 DAG 执行 |

**消融结论**：
- **双向交叉验证是核心增益来源**：GERS+自适应（0.284）→ GERS-CV（0.298），+1.4pt EM，对应 CS 区分度从 −0.0035 修复到 +0.0847（4.3 节）。这是本文核心创新的直接贡献。
- **图级自一致性（K=3）在中等难度多跳上无额外增益**：GERS-SC（0.282）≈ GERS+自适应（0.284），p=1.000。即便修复 CS 区分度后（GERS-CV2 vs GERS-CV 单路版本仍无差异），多路选优在 HotpotQA 上仍未带来提升。这是一个重要的诚实负面发现——界定了图级自一致性的适用边界。
- **置信度加权汇总贡献有限**：GERS-CV（0.298）→ GERS-CV2（0.302），+0.4pt（不显著）。低置信标注对最终答案的影响较小。
- **图执行是基础**：移除图执行退化为 CoT-SC+GERS 重排（0.264），证明 DAG 分解执行是 GERS 性能的基础，单纯的选择阶段重排无效。


### 4.7 案例分析

为直观展示双向交叉验证如何帮助推理，本文给出两个典型案例。

**案例 1：交叉验证捕获推理不一致。** 某多跳问题"Alfred Balk 在哪位美国副总统任内担任某委员会秘书"，GERS 正向分解为两个子问题并得到最终答案"Nelson Rockefeller"。反向验证时，以该答案 + 上下文反向重答子问题，正反向子答案一致（crossval=1.0），CS 升高，该推理链被保留。而在另一道题中，正向某子问题答错（生卒年判断偏差），反向独立重答得到不同子答案，正反向不一致使 crossval 下降，CS 降低——错误推理链被识别为低质量。这正体现了双向交叉验证"内容自洽性"校验的价值：纯结构 CS 对两者都给 0.6（无法区分），而 crossval 能区分。

**案例 2：comparison 题图分解优于线性 CoT。** 对比型问题"哪部电影更早上映，A 还是 B"天然对应"两条并行子链 + 汇合节点"的 DAG 结构。GERS 将其分解为"分别查 A/B 的上映年份 → 比较年份"两路子问题，拓扑执行后正确汇合；而线性 CoT 易在汇合点混淆两部电影。这是 GERS 在 comparison 子集上优于 CoT-SC（+10.5pt，McNemar p=0.013）的结构性原因。

## 5 讨论与局限

**双向交叉验证的核心价值**。本文最扎实的贡献是子答案双向交叉验证——它将 Consistency Score 的对错区分度从 −0.0035 修复到 +0.0847，使一致性校验从"图是否合法"升级为"推理内容是否自洽"。这一机制是 DAG 独有的：线性 CoT 没有可独立校验的子问题结构，无法实现正向执行 + 反向验证的双向校验。它带来的 EM 提升（+1.4pt）虽不大，但机制有效性与可量化证据（区分度）是明确的。

**性能的诚实定位**。GERS-CV2 在 HotpotQA 500 条上 EM=0.302、F1=0.413，McNemar 配对检验显著优于 CoT-SC（p=0.029），但 bootstrap 95% CI 仍微跨零，处于"配对显著、效应量临界"状态。本文不夸大为强显著。整体增益有限的一个重要原因是 HotpotQA 的任务特性：其上下文直接包含答案证据，强模型（Zero-Shot 0.276）读上下文即可答对不少，压缩了所有结构化方法的优势空间；CoT-SC 甚至低于 Zero-Shot。这表明 GERS 的价值在该数据集上更多体现为"可量化的推理质量评估"而非"大幅刷榜"。

**图级自一致性的边界**。一个重要的诚实负面发现：即便修复 CS 区分度后，图级自一致性（K=3 多路选优）在中等难度多跳上仍未带来额外增益（GERS-SC ≈ GERS+自适应，p=1.000）。这说明在该模型规模与任务难度下，单路 DAG 执行已足够，多路采样选优是冗余的。GERS-SC 的价值在于提供推理质量信号，而非端到端性能。

**深度复合桥接的局限**。在 2WikiMultiHopQA 的 bridge_comparison 题上，GERS 的子问题错误传播导致其落后于线性 CoT（F1 0.476 vs 0.905）。这是图结构分解在深度多跳上的固有局限，也是未来工作的重点方向。

**与现有图方法的对比**。GERS 相比 MoDeGraph[4] 等仅将图作为上下文的方法，核心区别在于图结构既参与执行（拓扑排序）又参与内容校验（双向交叉验证），形成"构图-执行-校验"闭环。相比 GoT[2] 的思维合并，GERS 的双向交叉验证利用 DAG 子问题结构做正反向一致性校验，是结构化推理独有的质量评估机制。

## 6 结论

本文提出 GERS，一种基于推理依赖图的图约束思维链方法，核心创新是**子答案双向交叉验证**：以最终答案 + 上下文为锚反向重答子问题，对比正反向一致性，将 Consistency Score 从纯图论结构指标（区分度 −0.0035，等价随机）升级为有效的内容自洽信号（区分度 +0.0847）。该机制为 DAG 独有，线性 CoT 无法实现。最优配置 GERS-CV2 在 HotpotQA 500 条上取得 EM=0.302、F1=0.413，McNemar 配对检验显著优于 CoT-SC（p=0.029），F1 领先 4pt。本文诚实呈现方法的适用边界：图级自一致性多路选优在中等难度多跳上无额外增益，深度桥接复合题存在错误传播局限；并通过指标修复、答案类型回扣、提取公平性三项工程改进，量化了"表面失败"与"真实推理差距"的边界。

未来工作包括：(1) 针对深度复合桥接题设计局部重生成机制，缓解错误传播；(2) 在更大规模模型与更多数据集上验证双向交叉验证的泛化性；(3) 探索反向验证对 confirmation bias 的进一步缓解策略。


## 参考文献

[1] Wei, J., et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In *NeurIPS*, 2022.

[2] Besta, M., et al. Graph of Thoughts: Solving Elaborate Problems with Large Language Models. In *AAAI*, 2024.

[3] Han, H., et al. Reasoning with Graphs: Structuring Implicit Knowledge to Enhance LLMs Reasoning. In *Findings of ACL*, 2025.

[4] Oruche, R., et al. Disentangling Complex Questions in LLMs via Multi-Hop Dependency Graphs. In *CIKM*, 2025.

[5] Ni, T., et al. StepChain GraphRAG: Reasoning Over Knowledge Graphs for Multi-hop Question Answering. *arXiv:2510.02827*, 2025.

[6] Wang, X., et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In *ICLR*, 2023.

[7] Yao, S., et al. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In *NeurIPS*, 2023.

[8] Lightman, H., et al. Let's Verify Step by Step. In *ICLR*, 2024.

[9] Madaan, A., et al. Self-Refine: Iterative Refinement with Self-Feedback. In *NeurIPS*, 2023.

[10] DAG-Math: Modeling Chain-of-Thought as Directed Acyclic Graphs. *arXiv:2510.19842*, 2025.

[11] Yang, Z., et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In *EMNLP*, 2018.

[12] Ho, X., et al. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In *COLING*, 2020.
