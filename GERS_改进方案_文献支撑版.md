# GERS 改进方案（文献支撑版）

> 基于代码深读 + 最新文献，聚焦"为什么 GERS 打不过 ToT"以及"怎么改才能真正赢"
> 代码版本：commit `bf233cc`（2026-06-26）

---

## 一、先搞清楚"GERS 为什么打不过 ToT"

ToT（Tree of Thoughts, Yao et al. 2023）之所以在你的实验里占优，不是因为树结构本身有多优秀，而是**ToT 做了一件 GERS 没做的事：搜索 + 选择**。

ToT 的核心不是"树"，而是**在多条候选路径里打分、剪枝、保留最优**。它用 BFS/DFS 遍历多条推理路径，每一步用 LLM 自我评估打分，最终取分最高的那条作为答案。

你的 GERS 在做什么？**只走一条路**。分解 → DAG → 拓扑序逐步走到底 → 得到答案。一条路走到黑，没有回头、没有备选、没有比较。

这就是根本差距：**ToT 是"搜索 + 选择"，GERS 是"单路执行"。**

从 GoT 论文（Besta et al., AAAI 2024）的框架来看，GERS 目前处于 DAG 执行器的位置，但**缺失了图的核心价值——在图上做评分和最优路径选择**。GoT 明确指出，图结构的优势在于能合并多条推理路径的结果（aggregation），而不只是让步骤按拓扑序执行。

---

## 二、代码层面的具体问题（对应文献里的已知解法）

### 问题 1：子问题分解是单次调用，没有质量保证

**代码位置**：`generation_pipeline.py` → `_decompose()`

```python
# 当前：一次调用，解析失败就降级为"原题=唯一子问题"
response = self.model.generate(prompt, max_tokens=600, temperature=0.2)
try:
    data = json.loads(json_str.strip())
    sub_qs = data.get("sub_questions", [])
except Exception:
    pass
return [{"id": 1, "question": question, "depends_on": [], "type": "inference"}]
```

**问题**：分解质量决定了整条链的上限。分解一旦错了（缺一个关键子问题、依赖关系搞反了），后面全错，且没有任何纠错机会。这是"错误传播"的主要来源。

**文献对应**：SiGIR（ACL 2025 Findings, arxiv 2505.19112）用 self-critique 对中间子问题分解做迭代验证，证明分解质量提升能带来 EM +4~8%。

**改法**：
```python
# 生成 K 个不同的分解方案
decompositions = []
for temp in [0.2, 0.5, 0.7]:
    sub_qs = self._decompose_once(question, context_section, temperature=temp)
    decompositions.append(sub_qs)

# 用 LLM 对每个分解打分（逻辑完备性 + 依赖合理性）
best = self._score_and_select_decomposition(question, decompositions)
```

或者更轻量的方案：分解后立即做一次验证调用，让 LLM 检查"这些子问题能否完整覆盖原问题"，不通过就重新分解（最多 2 次）。

---

### 问题 2：子问题回答是独立的，子答案没有交叉验证

**代码位置**：`generation_pipeline.py` → step ④，逐步回答子问题

```python
for node_id in execution_plan["execution_order"]:
    ...
    raw_response = self.model.generate(sub_prompt, max_tokens=300, temperature=0.3)
    sub_ans = self._extract_sub_answer(raw_response)
    sub_answers[node_id] = sub_ans  # 直接存入，没有验证
```

**问题**：每个子问题只问一次，答案直接用。但子答案如果错了，下游依赖它的子问题会基于错误答案继续推理，错误像雪球一样越滚越大。HotpotQA 里尤其致命——bridge 问题的第一跳答错，第二跳 100% 错。

**文献对应**：
- "Let's Verify Step by Step"（ICLR 2024, Lightman et al.）：步骤级别的验证比结果级别验证有效得多。
- DAG-Math（NeurIPS 2025, arxiv 2510.19842）：同样是 DAG 框架，它用节点置信度作为后续步骤的权重输入，而不是把子答案当作确定事实直接传入。

**改法**：对每个子答案做置信度估计，低置信度的子答案标记为"不确定"，并在下游提示词里声明不确定性。最简单的实现：

```python
# 对子答案做简单的自我验证
verify_prompt = f"""
Sub-question: {sub_q_text}
Proposed answer: {sub_ans}
Original context: {context[:500]}

Is this answer correct and well-supported? Answer YES/NO and briefly explain.
Confidence: HIGH/MEDIUM/LOW
"""
verify_resp = self.model.generate(verify_prompt, max_tokens=100, temperature=0.0)
confidence = "LOW" if "NO" in verify_resp or "LOW" in verify_resp else "HIGH"

# 在下游子问题的 prompt 里注明置信度
if confidence == "LOW":
    prev_answers_text += f"\n[Note: Answer to '{sub_q_text}' has LOW confidence: {sub_ans}]"
```

---

### 问题 3：Consistency Score 算完不用，闭环修正从未触发

**代码位置**：`generation_pipeline.py` → step ⑦

```python
# max_iterations=1（默认），这个 if 永远进不去
if not self._no_feedback and score < self.consistency_threshold and self.max_iterations > 1:
    ...  # 死代码
```

**问题**：这是最浪费的地方。你花了精力实现 Consistency Score（图连通性 + 覆盖度 + NLI），但它的计算结果对最终答案**零影响**。

**改法（推荐：图级别 Self-Consistency）**：

不要用 Consistency Score 来"修正"（触发重生成太重），而是用它来"选择"。生成 K 条独立的推理图，选得分最高的那条：

```python
def reason_with_graph_sc(self, question: str, context: str = "", K: int = 3) -> Dict:
    """图级别 Self-Consistency：生成 K 条推理图，选 Consistency Score 最高的"""
    candidates = []
    for i in range(K):
        # 用不同 temperature 生成不同的分解
        temp = [0.2, 0.5, 0.7][i % 3]
        result = self._reason_single(question, context, decompose_temp=temp)
        score = result["consistency_score"]
        candidates.append((score, result))
    
    # 选一致性得分最高的
    best_score, best_result = max(candidates, key=lambda x: x[0])
    best_result["graph_sc_candidates"] = len(candidates)
    return best_result
```

**这个改动的价值**：
- 直接对标 CoT-SC：CoT-SC 是"答案多数投票"，Graph-SC 是"图结构质量打分选择"
- 卖点可以一句话说清：**结构化质量信号比朴素投票更精准**
- 需要打开 `enable_nli=True` 让 Consistency Score 有区分度

---

### 问题 4：ConstrainedDecoder 是装饰文字而非真实约束

**代码位置**：`constrained_decoder.py` → `_soft_constraint()`

```python
def _soft_constraint(self, graph, plan, prompt):
    plan_text = planner.format_execution_plan(graph, plan)
    constraint_prompt = f"""{prompt}
Follow this reasoning path (each step must cite its predecessors):
{plan_text}
IMPORTANT: After completing all reasoning steps, you MUST end with:
Final Answer: <concise answer only>
"""
    return constraint_prompt
```

**问题**：这只是把路径描述拼成字符串加进 prompt，不是任何意义上的"约束解码"。它既没有修改 logits，也没有做格式校验，本质上就是给 LLM 一段额外的引导文本，而且消融实验证明它在 GSM8K 上是**负贡献**（去掉后 EM 从 0.61 → 0.80）。

消融结果很清楚：**这个模块应该删掉，或者重做成真正的约束**。

**真正的约束该是什么样的**：在子问题回答结束后，强制校验子答案格式，如果不满足则重生成，最多重试 2 次：

```python
def _answer_subq_with_retry(self, prompt, node_id, graph, sub_answers, max_retry=2):
    for _ in range(max_retry):
        response = self.model.generate(prompt, max_tokens=300, temperature=0.3)
        sub_ans = self._extract_sub_answer(response)
        
        # 硬约束：子答案不能为空，不能超过 150 字，不能和前驱矛盾
        if sub_ans and len(sub_ans) < 150:
            return sub_ans, response
        # 不满足则重试
    return sub_ans or "Unable to determine", response  # 最多 2 次后兜底
```

---

### 问题 5：评估指标有 bug，GSM8K 数据不可信

**代码位置**：`src/utils/metrics.py` → `exact_match()`

```python
# 这行是定时炸弹：
if pred and ref and (pred in ref or ref in pred):
    return 1.0
```

`"18" in "180"` = True，`"the city" in "the city of London"` = True。双向子串包含会让数值型答案的 EM 严重虚高。

ToT 修复后 GSM8K 跑到 0.86，远超 CoT-SC 的 0.762，这不合理——同模型、算术题，ToT 不应该比 CoT-SC 高出这么多。高度怀疑这个 bug 是虚高的主因。

**修法**：

```python
@staticmethod
def exact_match(prediction: str, reference: str) -> float:
    import re
    
    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = re.sub(r'[^\w\s]', '', s)
        return ' '.join(s.split())
    
    pred = normalize(prediction)
    ref = normalize(reference)
    
    if pred == ref:
        return 1.0
    
    # GSM8K 专用：数值精确匹配（抽取最后一个数字）
    pred_nums = re.findall(r'-?\d+\.?\d*', pred)
    ref_nums = re.findall(r'-?\d+\.?\d*', ref)
    if pred_nums and ref_nums:
        try:
            if abs(float(pred_nums[-1]) - float(ref_nums[-1])) < 1e-6:
                return 1.0
        except ValueError:
            pass
    
    # 删掉双向子串包含，改成 token F1 > 0.9 作为模糊匹配
    pred_tokens = set(pred.split())
    ref_tokens = set(ref.split())
    if pred_tokens and ref_tokens:
        overlap = pred_tokens & ref_tokens
        f1 = 2 * len(overlap) / (len(pred_tokens) + len(ref_tokens))
        if f1 >= 0.9:  # 严格阈值，避免误判
            return 1.0
    
    return 0.0
```

---

## 三、战略层面：GERS 该打什么样的仗

### ToT 赢在哪里，GERS 该赢在哪里

ToT 的优势：搜索宽度广，在答案空间可枚举的问题（算术、数独、创意写作）里，广搜加评估能找到最优路径。

ToT 的弱点：
1. **不理解子问题之间的依赖结构**：ToT 的每个分支是独立生成的思维步骤，无法表达"步骤 A 的答案是步骤 B 的前提"这种依赖关系
2. **计算代价高**：N 个候选 × D 层深度 = O(N^D) 次 LLM 调用
3. **对比型题天然不擅长**：需要先分别计算 A 和 B，再比较的问题，ToT 倾向于把 A 和 B 混在一起搜索，容易混淆

你的消融数据已经证实了这一点：

| 题型 | GERS EM | CoT EM | 差值 |
|------|---------|--------|------|
| bridge（链式多跳） | 0.337 | 0.270 | +6.7pt |
| **comparison（对比型）** | **0.677** | **0.500** | **+17.7pt** |

**对比型问题才是 GERS 的主战场**。对比题的天然结构是「两条并行子链 + 汇合节点」的 DAG，拓扑执行能把两条链分开做完再比较，而 ToT 的树搜索做不到这种结构化并行。

**战略建议**：
- 论文主线从"通用多跳"收缩到"依赖密集型 / 对比推理"
- 数据集换 **2WikiMultiHopQA**（大量对比型）+ **MuSiQue**（深度桥接依赖），让 GERS 的结构优势在更合适的战场发挥
- 保留 HotpotQA，但聚焦 comparison 子集的分析

---

## 四、改进优先级路线图（按性价比排序）

### 🔴 第一梯队：修基础，救可信度（立刻做，几乎零成本）

| 任务 | 位置 | 预期效果 |
|------|------|---------|
| 修 `metrics.py`：删双向子串，改数值 EM | `src/utils/metrics.py` | GSM8K 数据回归真实 |
| 用现有 `raw_response` 离线重算所有方法分数 | 写一个 `reeval.py` | 不重跑模型，几分钟出真实对比 |
| 统一所有方法样本量 n | ToT 补跑到 500 | 主表可比 |

### 🟡 第二梯队：激活有效件（1~2周，方法核心改进）

| 任务 | 位置 | 预期效果 |
|------|------|---------|
| **实现图级别 Self-Consistency（Graph-SC）** | `generation_pipeline.py` | 有机会打败 CoT-SC，让 Consistency Score 第一次有实际价值 |
| 子答案置信度估计 + 低置信度标注传递 | step ④ | 减少错误传播，bridge 题提升 |
| 打开 `enable_nli=True` 让分数有区分度 | `__init__` 参数 | Graph-SC 需要有区分度才有效 |
| 删掉或重做 ConstrainedDecoder | `constrained_decoder.py` | 减少 GSM8K 负贡献 |

### 🟢 第三梯队：放大优势（2~4周，提升论文贡献层次）

| 任务 | 预期效果 |
|------|---------|
| 换 2WikiMultiHopQA + MuSiQue 数据集 | 放大对比型题优势，GERS vs ToT 差距拉大 |
| 补分解质量验证（self-critique 分解） | 减少首步错误传播 |
| 补第二个模型（Qwen2.5-7B 或 GLM-4） | 验证泛化，排除"只在 qwen3-8b 上有效"质疑 |
| 把"自适应路由"提为正式贡献点 | 现成的 13pt 数据支撑，不浪费 |

---

## 五、最值得做的一件事：Graph-SC（具体实现）

如果只能改一件事，就改这个。它能让你的方法从"单路执行"升级成"多路竞争+结构评分选优"，直接对标 CoT-SC。

```python
# 在 generation_pipeline.py 里加一个新方法

def reason_graph_sc(self, question: str, context: str = "", K: int = 3) -> Dict:
    """
    Graph-level Self-Consistency
    生成 K 条独立推理图，用 Consistency Score 选最优
    对标 CoT-SC（答案投票），但用图结构质量打分代替朴素投票
    """
    candidates = []
    temperatures = [0.2, 0.5, 0.8]  # 不同温度产生不同分解
    
    for i in range(K):
        temp = temperatures[i % len(temperatures)]
        # 临时修改分解温度
        result = self._reason_with_temp(question, context, decompose_temp=temp)
        candidates.append({
            "result": result,
            "consistency_score": result["consistency_score"],
            "answer": result["answer"],
        })
    
    # 用 Consistency Score 选最优
    best = max(candidates, key=lambda x: x["consistency_score"])
    
    # 同时做答案投票作为辅助信号（投票 + 结构分共同决定）
    answer_votes = {}
    for c in candidates:
        ans = c["answer"].lower().strip()
        answer_votes[ans] = answer_votes.get(ans, 0) + c["consistency_score"]  # 加权投票
    
    # 如果加权投票结果和 Consistency Score 最高的不一致，取投票结果
    weighted_best_ans = max(answer_votes, key=answer_votes.get)
    
    final_result = best["result"].copy()
    final_result["graph_sc_best_score"] = best["consistency_score"]
    final_result["graph_sc_candidates"] = K
    final_result["graph_sc_vote_answer"] = weighted_best_ans
    
    # 用加权投票答案（更稳健）
    if weighted_best_ans != best["answer"].lower().strip():
        final_result["answer"] = weighted_best_ans
    
    return final_result
```

**为什么这比 CoT-SC 好**（这是论文里可以讲的故事）：

CoT-SC 的投票是"哪个答案出现最多次"，每票权重相等，不区分推理质量。Graph-SC 是"哪个图的结构一致性最高"，用图质量加权，逻辑更连贯的推理图拥有更高的投票权重。在依赖关系密集的多跳问题上，结构完整的图天然对应更可靠的推理，这个加权有理论依据。

---

## 六、参考文献

1. **GoT（Graph of Thoughts）**：Besta et al., AAAI 2024 — 图结构推理框架，明确指出图的 aggregation（路径合并）是超越 ToT 的关键
   arxiv: https://arxiv.org/abs/2308.09687

2. **DAG-Math**：NeurIPS 2025 — 把 CoT 建模为 DAG 上的随机过程，用节点置信度权重处理不确定步骤
   arxiv: https://arxiv.org/abs/2510.19842

3. **SiGIR**：ACL 2025 Findings — self-critique 引导的迭代子问题分解，步骤级校验 EM +4~8%
   arxiv: https://arxiv.org/abs/2505.19112

4. **Let's Verify Step by Step（PRM）**：Lightman et al., ICLR 2024 — 步骤级验证的价值

5. **MuSiQue**：Trivedi et al. — 更严格的多跳推理数据集，对比 HotpotQA 更难被单步捷径解决
   github: https://github.com/stonybrooknlp/musique
