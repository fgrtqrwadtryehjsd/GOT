"""
图约束推理生成流水线 —— GERS核心Pipeline（重构版）

核心思想（修正后）：
=====================================
原版问题：图结构只是装饰性的Prompt，LLM并不真正按图执行。

重构后的正确做法：
1. 用 LLM 将问题分解为有依赖关系的子问题（构图）
2. 按拓扑排序逐个调用 LLM 回答每个子问题（真正的图约束执行）
3. 将所有子答案作为上下文，汇总得出最终答案
4. 一致性校验：检测推理链中的矛盾和断链

这样图结构真正参与了推理过程，而不只是Prompt中的装饰文字。
"""

from typing import Dict, List, Optional
from ..graph_representation.reasoning_graph import ReasoningGraph
from ..graph_representation.dynamic_builder import DynamicGraphBuilder
from ..graph_representation.extractor import EntityRelationExtractor
from .path_planner import PathPlanner
from ..consistency_check.consistency_score import ConsistencyChecker


# ─── 子问题分解 Prompt ────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """You are an expert at breaking down complex questions into step-by-step sub-questions.

Question: {question}
{context_section}

Decompose this into ordered sub-questions where each builds on the previous.
Output ONLY valid JSON:

```json
{{
  "sub_questions": [
    {{
      "id": 1,
      "question": "first sub-question to answer",
      "depends_on": [],
      "type": "fact_lookup/comparison/calculation/inference"
    }},
    {{
      "id": 2,
      "question": "second sub-question (may reference answer to #1)",
      "depends_on": [1],
      "type": "fact_lookup/comparison/calculation/inference"
    }}
  ]
}}
```

Rules:
- Keep sub-questions atomic (one fact per question)
- Order them so each can be answered using only previous answers
- For simple questions, 1-2 sub-questions is enough
- For multi-hop questions, 2-4 sub-questions
- Output ONLY JSON, no other text."""


# ─── 子问题回答 Prompt ────────────────────────────────────────────────────────

ANSWER_SUBQ_PROMPT = """Answer this specific sub-question concisely.

Original question: {original_question}
{context_section}

{previous_answers}

Current sub-question: {sub_question}

Answer in 1-3 sentences. End with:
Sub-answer: <concise answer>"""


# ─── 最终汇总 Prompt ──────────────────────────────────────────────────────────

FINAL_ANSWER_PROMPT = """Based on the step-by-step reasoning below, give the final answer.

Original question: {original_question}

Step-by-step reasoning:
{reasoning_chain}

Now answer the original question directly and concisely.
Final Answer: <your answer>"""


class GraphGuidedGenerator:
    """
    图约束推理生成器 —— GERS系统核心（重构版）

    真正的图约束推理流程：
    ① 分解：将问题分解为有依赖关系的子问题（LLM调用×1）
    ② 构图：依赖关系形成有向图（DAG）
    ③ 按拓扑序逐步回答每个子问题（LLM调用×N）
    ④ 汇总：将子答案链汇总为最终答案（LLM调用×1）
    ⑤ 校验：Consistency Score 评估推理链路完整性
    """

    def __init__(self,
                 model=None,
                 constraint_mode: str = "soft",
                 max_iterations: int = 1,
                 consistency_threshold: float = 0.6,
                 enable_nli: bool = False,
                 _no_context: bool = False):
        self.model = model
        self.path_planner = PathPlanner()
        self.consistency_checker = ConsistencyChecker(
            enable_nli=enable_nli,
            nli_model=None
        )
        self.max_iterations = max_iterations
        self.consistency_threshold = consistency_threshold
        self._no_context = _no_context  # 消融：不传递前驱答案

    def reason(self, question: str, context: str = "") -> Dict:
        """
        核心推理流程（重构版）

        Returns:
            {
                "answer": 最终答案,
                "reasoning_text": 完整推理链路文本,
                "graph": 推理图对象,
                "consistency_score": 一致性得分,
                "iterations": 迭代次数,
                "execution_plan": 执行计划,
                "sub_qa_chain": 子问题-答案链
            }
        """
        if self.model is None:
            graph = ReasoningGraph(question=question)
            return {
                "answer": "",
                "reasoning_text": "[Demo模式] 需要配置LLM模型",
                "graph": graph,
                "consistency_score": 0.0,
                "iterations": 0,
                "execution_plan": {"execution_order": [], "key_paths": [],
                                   "branch_points": [], "merge_points": []},
                "sub_qa_chain": [],
            }

        context_section = f"\nContext: {context[:1500]}" if context else ""

        # ── ① 分解问题为子问题 ──────────────────────────────────────────────
        sub_questions = self._decompose(question, context_section)

        # ── ② 构建推理依赖图 ───────────────────────────────────────────────
        graph = self._build_graph(question, sub_questions)

        # ── ③ 按拓扑顺序逐步回答子问题 ────────────────────────────────────
        execution_plan = self.path_planner.plan(graph)
        sub_answers = {}    # id → answer text
        sub_qa_chain = []   # 完整推理链

        for node_id in execution_plan["execution_order"]:
            node = graph.get_node(node_id)
            if node is None:
                continue
            from ..graph_representation.node import NodeType
            if node.node_type != NodeType.STEP:
                continue  # 只对 Step 节点（子问题）调用 LLM

            # 构建前驱答案上下文
            prev_answers_text = "" if self._no_context else \
                self._format_previous_answers(node_id, graph, sub_answers)

            sub_q_text = node.content
            sub_prompt = ANSWER_SUBQ_PROMPT.format(
                original_question=question,
                context_section=context_section,
                previous_answers=prev_answers_text,
                sub_question=sub_q_text,
            )

            raw_response = self.model.generate(sub_prompt, max_tokens=300, temperature=0.3)
            sub_ans = self._extract_sub_answer(raw_response)

            sub_answers[node_id] = sub_ans
            node.metadata["answer"] = sub_ans
            node.confidence = 0.8 if sub_ans else 0.2

            sub_qa_chain.append({
                "node_id": node_id,
                "sub_question": sub_q_text,
                "sub_answer": sub_ans,
                "raw_response": raw_response,
            })

        # ── ④ 汇总最终答案 ─────────────────────────────────────────────────
        reasoning_chain_text = self._format_reasoning_chain(sub_qa_chain)
        final_prompt = FINAL_ANSWER_PROMPT.format(
            original_question=question,
            reasoning_chain=reasoning_chain_text,
        )
        final_response = self.model.generate(final_prompt, max_tokens=200, temperature=0.1)
        answer = self._extract_final_answer(final_response)
        reasoning_text = reasoning_chain_text + "\n\n" + final_response

        # ── ⑤ 一致性校验 ──────────────────────────────────────────────────
        check_result = self.consistency_checker.check(graph, reasoning_text)
        score = check_result["consistency_score"]

        return {
            "answer": answer,
            "reasoning_text": reasoning_text,
            "graph": graph,
            "consistency_score": score,
            "iterations": 0,
            "execution_plan": execution_plan,
            "sub_qa_chain": sub_qa_chain,
        }

    # ─── 辅助方法 ─────────────────────────────────────────────────────────────

    def _decompose(self, question: str, context_section: str) -> List[Dict]:
        """将问题分解为有依赖的子问题列表"""
        import json, re
        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            context_section=context_section
        )
        response = self.model.generate(prompt, max_tokens=600, temperature=0.2)

        # 解析 JSON
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            sub_qs = data.get("sub_questions", [])
            if sub_qs:
                return sub_qs
        except Exception:
            pass

        # 降级：把整个问题作为唯一子问题
        return [{"id": 1, "question": question, "depends_on": [], "type": "inference"}]

    def _build_graph(self, question: str, sub_questions: List[Dict]) -> ReasoningGraph:
        """将子问题及其依赖关系构建为推理图"""
        graph = ReasoningGraph(question=question)

        # 事实节点：原始问题
        fact_id = graph.add_fact(
            content=f"Question: {question}",
            source="question"
        )

        # id → node_id 映射
        id_to_nodeid = {}

        # 为每个子问题创建 Step 节点
        for sq in sub_questions:
            node_id = graph.add_step(
                content=sq.get("question", ""),
                operation=sq.get("type", "inference")
            )
            id_to_nodeid[sq["id"]] = node_id

            # 无依赖的子问题 → 从原始事实推导
            if not sq.get("depends_on"):
                graph.add_derive(fact_id, node_id, desc="decompose")

        # 建立子问题之间的依赖边
        for sq in sub_questions:
            cur_nid = id_to_nodeid.get(sq["id"])
            for dep_id in sq.get("depends_on", []):
                dep_nid = id_to_nodeid.get(dep_id)
                if cur_nid and dep_nid:
                    graph.add_derive(dep_nid, cur_nid, desc="depends_on")

        # 最终结论节点
        conc_id = graph.add_conclusion(
            content=f"Final answer to: {question}",
        )
        # 最后一个子问题 → 结论
        if sub_questions:
            last_sq = sub_questions[-1]
            last_nid = id_to_nodeid.get(last_sq["id"])
            if last_nid:
                graph.add_derive(last_nid, conc_id, desc="conclude")

        return graph

    def _format_previous_answers(self, node_id: str, graph: ReasoningGraph,
                                   sub_answers: Dict[str, str]) -> str:
        """格式化前驱节点的已知答案，作为当前子问题的上下文"""
        preds = graph.get_predecessors(node_id)
        if not preds:
            return ""

        lines = ["Previously established:"]
        for pred_id in preds:
            pred_node = graph.get_node(pred_id)
            if pred_node and pred_id in sub_answers:
                lines.append(f"- Q: {pred_node.content}")
                lines.append(f"  A: {sub_answers[pred_id]}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_reasoning_chain(self, sub_qa_chain: List[Dict]) -> str:
        """将子问题-答案链格式化为推理文本"""
        if not sub_qa_chain:
            return ""
        lines = []
        for i, item in enumerate(sub_qa_chain, 1):
            lines.append(f"Step {i}: {item['sub_question']}")
            lines.append(f"Answer: {item['sub_answer']}")
        return "\n".join(lines)

    def _extract_sub_answer(self, text: str) -> str:
        """从子问题回答中提取简洁答案"""
        import re
        m = re.search(r'Sub-answer[：:]\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('*#.,').strip()
        # 回退：最后非空短句
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and 2 < len(line) < 200:
                return line
        return text.strip()[:100]

    def _extract_final_answer(self, text: str) -> str:
        """从最终回答中提取答案"""
        import re
        patterns = [
            r'Final Answer[：:]\s*(.+?)(?:\n|$)',
            r'The answer is[：:]\s*(.+?)(?:\n|$)',
            r'Therefore[,，]\s*(.+?)(?:\n|$)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                ans = m.group(1).strip().strip('*#.,。').strip()
                if ans and len(ans) < 200:
                    return ans
        # 回退
        lines = text.strip().split("\n")
        noise = [r'^如需', r'^如有', r'^需要', r'^\$\$', r'^---']
        for line in reversed(lines):
            line = line.strip().lstrip('#*|>-').strip()
            if not line or len(line) < 2 or len(line) > 150:
                continue
            if re.match(r'^[\d\s\.\:步骤]+$', line):
                continue
            if any(re.match(p, line) for p in noise):
                continue
            return line
        return ""
