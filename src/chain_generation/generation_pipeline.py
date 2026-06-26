"""
图约束推理生成流水线 —— GERS核心Pipeline（完整版）

完整推理流程：
1. 用 LLM 将问题分解为有依赖关系的子问题（构图）
2. 构建推理依赖图（DAG）
3. 拓扑排序 + GraphPromptBuilder 生成结构化路径描述
4. 按拓扑序逐步回答每个子问题（ConstrainedDecoder 增强约束）
5. 将所有子答案作为上下文，汇总得出最终答案
6. 一致性校验：ConsistencyChecker 评估推理链路完整性
7. 闭环修正：若 Score < 阈值，FeedbackLoop 触发回溯修正 → 再校验

图结构真正参与了推理过程，而非仅作为 Prompt 装饰文字。
"""

import logging
from typing import Dict, List, Optional
from ..graph_representation.reasoning_graph import ReasoningGraph
from .path_planner import PathPlanner
from .prompt_builder import GraphPromptBuilder
from .constrained_decoder import ConstrainedDecoder
from ..consistency_check.consistency_score import ConsistencyChecker
from ..consistency_check.feedback_loop import FeedbackLoop

logger = logging.getLogger(__name__)


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

{reasoning_path_hint}
{previous_answers}

Current sub-question (Step {step_index}): {sub_question}

Answer in 1-3 sentences. End with:
Sub-answer: <concise answer>"""


# ─── 最终汇总 Prompt ──────────────────────────────────────────────────────────

FINAL_ANSWER_PROMPT = """Based on the step-by-step reasoning below, give the final answer.

Original question: {original_question}

Step-by-step reasoning:
{reasoning_chain}

Now answer the original question directly and concisely.
Final Answer: <your answer>"""


# ─── 自适应复杂度判断 Prompt ──────────────────────────────────────────────────

COMPLEXITY_PROMPT = """Analyze the complexity of the following question.

Question: {question}
{context_section}

Classify this question:
- "simple": ONLY if it requires a single calculation or a single direct fact lookup with NO need to connect multiple pieces of information.
- "complex": If it requires connecting multiple facts, comparing entities, multi-hop reasoning, or any multi-step logical chain.

When in doubt, choose "complex". Most questions requiring reading context are complex.

Answer with ONLY one word: simple or complex."""


# ─── 简单问题直接回答 Prompt ──────────────────────────────────────────────────

SIMPLE_ANSWER_PROMPT = """Answer the following question step by step.

Question: {question}
{context_section}

Think step by step, then give the final answer.
Final Answer: <your answer>"""


class GraphGuidedGenerator:
    """
    图约束推理生成器 —— GERS系统核心（完整版）

    完整的图约束推理流程：
    ⓪ 自适应判断：判断问题复杂度，简单题直接CoT回答，复杂题走完整流程
    ① 分解：将问题分解为有依赖关系的子问题（LLM调用×1）
    ② 构图：依赖关系形成有向图（DAG）
    ③ 路径规划：拓扑排序 + GraphPromptBuilder 生成结构化路径描述
    ④ 按拓扑序逐步回答子问题（ConstrainedDecoder 增强约束）（LLM调用×N）
    ⑤ 汇总：将子答案链汇总为最终答案（LLM调用×1）
    ⑥ 一致性校验：ConsistencyChecker 评估推理链路完整性
    ⑦ 闭环修正：若 Score < 阈值且未禁用，FeedbackLoop 触发回溯修正 → 再校验
    """

    def __init__(self,
                 model=None,
                 constraint_mode: str = "soft",
                 max_iterations: int = 1,
                 consistency_threshold: float = 0.75,
                 enable_nli: bool = False,
                 adaptive: bool = True,
                 _no_context: bool = False,
                 _no_constraint: bool = False,
                 _no_feedback: bool = False):
        """
        Args:
            model: LLM 模型实例
            constraint_mode: 约束模式 (soft/hard/hybrid)
            max_iterations: 闭环修正最大迭代次数（含首次生成）
            consistency_threshold: 触发修正的一致性阈值（默认0.75）
            enable_nli: 是否启用 NLI 语义校验
            adaptive: 是否启用自适应分解策略（简单题跳过分解）
            _no_context: 消融-不传递前驱答案
            _no_constraint: 消融-不使用约束解码器
            _no_feedback: 消融-不使用闭环修正
        """
        self.model = model
        self.path_planner = PathPlanner()
        self.prompt_builder = GraphPromptBuilder(format_type="hybrid")
        self.constraint_decoder = ConstrainedDecoder(constraint_mode=constraint_mode)
        self.consistency_checker = ConsistencyChecker(
            enable_nli=enable_nli,
            nli_model=None,
            llm=model,
        )
        self.feedback_loop = FeedbackLoop(
            model=model,
            consistency_checker=self.consistency_checker,
            max_iterations=max(max_iterations - 1, 0),
            threshold=consistency_threshold,
        )
        self.max_iterations = max_iterations
        self.consistency_threshold = consistency_threshold
        self.adaptive = adaptive
        self._no_context = _no_context
        self._no_constraint = _no_constraint
        self._no_feedback = _no_feedback

    def reason(self, question: str, context: str = "") -> Dict:
        """
        核心推理流程（完整版）

        Returns:
            {
                "answer": 最终答案,
                "reasoning_text": 完整推理链路文本,
                "graph": 推理图对象,
                "consistency_score": 一致性得分,
                "consistency_detail": 校验详情,
                "iterations": 修正迭代次数（0=未触发修正）,
                "execution_plan": 执行计划,
                "sub_qa_chain": 子问题-答案链,
                "token_count": 总 Token 消耗,
            }
        """
        if self.model is None:
            graph = ReasoningGraph(question=question)
            return {
                "answer": "",
                "reasoning_text": "[Demo模式] 需要配置LLM模型",
                "graph": graph,
                "consistency_score": 0.0,
                "consistency_detail": None,
                "iterations": 0,
                "execution_plan": {"execution_order": [], "key_paths": [],
                                   "branch_points": [], "merge_points": []},
                "sub_qa_chain": [],
                "token_count": 0,
            }

        context_section = f"\nContext: {context[:1500]}" if context else ""
        total_tokens = 0

        # ── ⓪ 自适应复杂度判断 ─────────────────────────────────────────────
        if self.adaptive:
            is_simple = self._judge_complexity(question, context_section)
            if is_simple:
                logger.debug("GERS: ⓪ 判定为简单问题，直接CoT回答...")
                result = self._answer_simple(question, context_section)
                result["complexity"] = "simple"
                return result
            logger.debug("GERS: ⓪ 判定为复杂问题，走完整GERS流程...")

        # ── ① 分解问题为子问题 ──────────────────────────────────────────────
        logger.debug("GERS: ① 分解子问题...")
        sub_questions = self._decompose(question, context_section)
        logger.debug(f"GERS: 分解出 {len(sub_questions)} 个子问题")

        # ── ② 构建推理依赖图 ───────────────────────────────────────────────
        logger.debug("GERS: ② 构建推理DAG...")
        graph = self._build_graph(question, sub_questions)

        # ── ③ 路径规划 + GraphPromptBuilder 生成路径描述 ────────────────────
        logger.debug("GERS: ③ 拓扑排序 + 路径规划...")
        execution_plan = self.path_planner.plan(graph)

        # 用 GraphPromptBuilder 生成结构化路径提示（注入到子问题 Prompt 中）
        if not self._no_constraint:
            reasoning_path_hint = self.prompt_builder.build(
                question=question,
                graph=graph,
                execution_plan=execution_plan,
                context=context,
            )
            # 只取路径描述部分，避免 Prompt 过长
            path_hint_lines = reasoning_path_hint.split("\n")
            # 提取 [Reasoning Path] 部分
            in_path = False
            path_hint = []
            for line in path_hint_lines:
                if "[Reasoning Path]" in line:
                    in_path = True
                    continue
                if in_path and line.strip() and not line.startswith("Please"):
                    path_hint.append(line)
                elif in_path and line.startswith("Please"):
                    break
            reasoning_path_hint = "\n".join(path_hint[:15])  # 限制长度
        else:
            reasoning_path_hint = ""

        # ── ④ 按拓扑顺序逐步回答子问题 ────────────────────────────────────
        logger.debug("GERS: ④ 按拓扑序逐步回答子问题...")
        sub_answers = {}    # node_id → answer text
        sub_qa_chain = []   # 完整推理链
        step_index = 0

        for node_id in execution_plan["execution_order"]:
            node = graph.get_node(node_id)
            if node is None:
                continue
            from ..graph_representation.node import NodeType
            if node.node_type != NodeType.STEP:
                continue  # 只对 Step 节点（子问题）调用 LLM

            step_index += 1

            # 构建前驱答案上下文
            prev_answers_text = "" if self._no_context else \
                self._format_previous_answers(node_id, graph, sub_answers)

            sub_q_text = node.content
            base_prompt = ANSWER_SUBQ_PROMPT.format(
                original_question=question,
                context_section=context_section,
                reasoning_path_hint=f"[Reasoning Path]\n{reasoning_path_hint}" if reasoning_path_hint else "",
                previous_answers=prev_answers_text,
                sub_question=sub_q_text,
                step_index=step_index,
            )

            # ConstrainedDecoder 增强约束
            if not self._no_constraint:
                sub_prompt = self.constraint_decoder.apply_constraint(
                    graph, execution_plan, base_prompt
                )
            else:
                sub_prompt = base_prompt

            raw_response = self.model.generate(sub_prompt, max_tokens=300, temperature=0.3)
            sub_ans = self._extract_sub_answer(raw_response)

            # Token 计数
            total_tokens += self.model.count_tokens(raw_response) if hasattr(self.model, 'count_tokens') else 0

            sub_answers[node_id] = sub_ans
            node.metadata["answer"] = sub_ans
            node.confidence = 0.8 if sub_ans else 0.2

            sub_qa_chain.append({
                "node_id": node_id,
                "sub_question": sub_q_text,
                "sub_answer": sub_ans,
                "raw_response": raw_response,
            })

        # ── ⑤ 汇总最终答案 ─────────────────────────────────────────────────
        logger.debug("GERS: ⑤ 汇总最终答案...")
        reasoning_chain_text = self._format_reasoning_chain(sub_qa_chain)
        final_prompt = FINAL_ANSWER_PROMPT.format(
            original_question=question,
            reasoning_chain=reasoning_chain_text,
        )
        final_response = self.model.generate(final_prompt, max_tokens=200, temperature=0.1)
        answer = self._extract_final_answer(final_response)
        reasoning_text = reasoning_chain_text + "\n\n" + final_response
        total_tokens += self.model.count_tokens(final_response) if hasattr(self.model, 'count_tokens') else 0

        # ── ⑥ 一致性校验 ──────────────────────────────────────────────────
        logger.debug("GERS: ⑥ ConsistencyChecker 校验...")
        check_result = self.consistency_checker.check(graph, reasoning_text)
        score = check_result["consistency_score"]

        iterations = 0

        # ── ⑦ 闭环修正 ────────────────────────────────────────────────────
        if not self._no_feedback and score < self.consistency_threshold and self.max_iterations > 1:
            logger.debug(f"GERS: ⑦ 触发闭环修正 (score={score:.3f} < {self.consistency_threshold})...")
            refine_result = self.feedback_loop.refine(
                graph=graph,
                reasoning_text=reasoning_text,
                question=question,
                context=context,
            )
            reasoning_text = refine_result["reasoning_text"]
            answer = refine_result["answer"]
            score = refine_result["consistency_score"]
            iterations = refine_result["iterations"]
            total_tokens += self.model.count_tokens(reasoning_text) if hasattr(self.model, 'count_tokens') else 0
            logger.debug(f"GERS: 修正完成 (iterations={iterations}, score={score:.3f})")
        else:
            logger.debug(f"GERS: 无需修正 (score={score:.3f} >= {self.consistency_threshold} 或已禁用)")

        return {
            "answer": answer,
            "reasoning_text": reasoning_text,
            "graph": graph,
            "consistency_score": score,
            "consistency_detail": check_result,
            "iterations": iterations,
            "execution_plan": execution_plan,
            "sub_qa_chain": sub_qa_chain,
            "token_count": total_tokens,
        }

    # ─── 辅助方法 ─────────────────────────────────────────────────────────────

    def _judge_complexity(self, question: str, context_section: str) -> bool:
        """判断问题是否为简单问题（单步可解）"""
        prompt = COMPLEXITY_PROMPT.format(
            question=question,
            context_section=context_section
        )
        try:
            response = self.model.generate(prompt, max_tokens=20, temperature=0.0)
            label = response.strip().lower().strip("*#., \n")
            return "simple" in label
        except Exception:
            return False  # 出错时走完整流程

    def _answer_simple(self, question: str, context_section: str) -> Dict:
        """简单问题直接用CoT方式回答，跳过分解+构图+拓扑排序"""
        prompt = SIMPLE_ANSWER_PROMPT.format(
            question=question,
            context_section=context_section
        )
        reasoning_text = self.model.generate(prompt, max_tokens=500, temperature=0.3)
        answer = self._extract_simple_answer(reasoning_text)
        total_tokens = self.model.count_tokens(reasoning_text) if hasattr(self.model, 'count_tokens') else 0

        graph = ReasoningGraph(question=question)
        graph.add_fact(content=f"Question: {question}", source="question")
        conc_id = graph.add_conclusion(content=f"Final answer to: {question}")

        return {
            "answer": answer,
            "reasoning_text": reasoning_text,
            "graph": graph,
            "consistency_score": 1.0,
            "consistency_detail": None,
            "iterations": 0,
            "execution_plan": {"execution_order": [], "key_paths": [],
                               "branch_points": [], "merge_points": []},
            "sub_qa_chain": [],
            "token_count": total_tokens,
        }

    def _extract_simple_answer(self, text: str) -> str:
        """从CoT风格回答中提取答案（兼容数学答案和文本答案）"""
        import re
        # 1. 尝试 Final Answer 模式
        patterns = [
            r'Final Answer[：:]\s*(.+?)(?:\n|$)',
            r'The answer is[：:]\s*(.+?)(?:\n|$)',
            r'Therefore[,，]\s*(.+?)(?:\n|$)',
            r'答案是[：:]\s*(.+?)(?:\n|$)',
            r'####\s*(.+?)(?:\n|$)',  # GSM8K 标准格式
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                ans = m.group(1).strip().strip('*#.,。').strip()
                # 去掉 LaTeX 包裹 $$...$$
                ans = re.sub(r'\$+', '', ans).strip()
                if ans and len(ans) < 200:
                    return ans
        # 2. 尝试提取最后一个数字（适用于数学题）
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if numbers:
            return numbers[-1]
        # 3. 回退到 _extract_final_answer
        return self._extract_final_answer(text)

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
        noise = [r'^如需', r'^如有', r'^需要', r'^\$\$', r'^---',
                 r'^通过补充', r'^合理处理', r'^The question',
                 r'^Please', r'^Note:', r'^\*']
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
