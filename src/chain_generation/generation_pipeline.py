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

Now answer the ORIGINAL question directly. Output ONLY the concise answer.

CRITICAL — your answer must match the entity type the original question asks for:
- If the question asks "which film/book/song/show", answer with the film/book/song/show NAME.
- If the question asks "who/which person/director/author", answer with the person NAME.
- If the question asks "which place/city/country", answer with the place NAME.
- Do NOT output an intermediate entity computed in the sub-questions (e.g. a director, a date, a year) when the original question asks for a different entity type. Always return the entity the original question requests.
- For yes/no questions, output "yes" or "no".
- Otherwise output just the answer (a name, number, or short phrase).

Final Answer: """


# ─── 反向验证 Prompt（方向1：子答案双向交叉验证）──────────────────────────────

BACKWARD_VERIFY_PROMPT = """You are verifying one reasoning step in reverse.

Original question: {original_question}
{context_section}
The final answer to the original question is known to be: {final_answer}

Now, independently re-derive the answer to this specific sub-question, based ONLY on the context above. Do NOT simply copy the final answer — derive the sub-answer from the context. If the final answer is correct, your sub-answer should be consistent with it; if the final answer is wrong, your independent sub-answer may legitimately differ.

Sub-question (Step {step_index}): {sub_question}

Answer this sub-question concisely from the context. End with:
Sub-answer: <your answer>"""


# ─── 反向验证 Prompt（P2.4 控制：仅用上下文，不用最终答案）─────────────────────

BACKWARD_VERIFY_PROMPT_CONTEXT_ONLY = """You are independently verifying one reasoning step.

Original question: {original_question}
{context_section}

Independently answer this specific sub-question based ONLY on the context above. Do not assume any final answer; derive the sub-answer purely from the context. If the context does not provide the information, say so explicitly.

Sub-question (Step {step_index}): {sub_question}

Answer concisely from the context. End with:
Sub-answer: <your answer>"""


# ─── 证据约束反向验证 Prompt（默认关闭，供后续实验）──────────────────────────

BACKWARD_VERIFY_WITH_EVIDENCE_PROMPT = """You are verifying one reasoning step with evidence grounding.

Original question: {original_question}
{context_section}
The final answer to the original question is known to be: {final_answer}

Independently re-derive the answer to this specific sub-question using ONLY the context above. Do not copy the final answer. In addition to the answer, quote the shortest evidence span from the context that supports it. If the context does not support an answer, say so explicitly.

Sub-question (Step {step_index}): {sub_question}

End with exactly:
Sub-answer: <your answer>
Evidence: <short quote from context, or "not supported">"""


# ─── 验证驱动局部修复 Prompt（默认关闭，供后续实验）──────────────────────────

REPAIR_SUBQ_PROMPT = """You are repairing one unreliable reasoning step.

Original question: {original_question}
{context_section}

Previous reliable answers:
{previous_answers}

The original answer to this sub-question may be unreliable because forward and backward verification disagreed.

Sub-question (Step {step_index}): {sub_question}
Original forward answer: {forward_answer}
Backward verification answer: {backward_answer}

Re-answer this sub-question using ONLY the context and reliable previous answers. If the context does not support the answer, say so explicitly. End with:
Sub-answer: <repaired answer>"""


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
                 self_consistency_k: int = 0,
                 _no_context: bool = False,
                 _no_constraint: bool = False,
                 _no_feedback: bool = False,
                 dataset: str = None,
                 enable_backward_verify: bool = False,
                 enable_llm_match: bool = False,
                  cs_struct_weight: float = 0.3,
                  cs_crossval_weight: float = 0.7,
                  enable_confidence_weighting: bool = False,
                  confidence_threshold: float = 0.5,
                  uniform_crossval_weight: bool = False,
                  enable_soft_match: bool = False,
                  enable_evidence_grounding: bool = False,
                  enable_verification_repair: bool = False,
                  repair_threshold: float = 0.75,
                  backward_anchor_mode: str = "answer_context"):
        """
        Args:
            model: LLM 模型实例
            constraint_mode: 约束模式 (soft/hard/hybrid)
            max_iterations: 闭环修正最大迭代次数（含首次生成）
            consistency_threshold: 触发修正的一致性阈值（默认0.75）
            enable_nli: 是否启用 NLI 语义校验
            adaptive: 是否启用自适应分解策略（简单题跳过分解）
            self_consistency_k: 图级Self-Consistency采样数（0=关闭，K>0=生成K条DAG选优）
            _no_context: 消融-不传递前驱答案
            _no_constraint: 消融-不使用约束解码器（消融证明约束解码负贡献，默认True）
            _no_feedback: 消融-不使用闭环修正
            dataset: 数据集名称（gsm8k/hotpotqa/2wikimultihopqa），透传给答案提取
            enable_backward_verify: 是否启用子答案双向交叉验证（方向1创新点）。
                以最终答案+上下文反向重答子问题，对比正反向一致性，修复 CS 区分度。
            enable_llm_match: 一致性判断中，字符串/数值匹配失败时是否再用 LLM 语义判断（成本敏感默认关）
            cs_struct_weight: 新 CS 中结构分权重（默认0.3）
            cs_crossval_weight: 新 CS 中正反向一致性分权重（默认0.7）
            enable_soft_match: 是否启用软匹配分数（默认关闭，保持论文实验口径）。
                开启后，高 token-F1 的近似答案可获得部分分数，缓解字符串匹配过硬。
            enable_evidence_grounding: 是否要求反向验证同时给出上下文证据（默认关闭）。
                开启后，crossval 会乘以 evidence grounding 分数，避免“自洽但无依据”。
            enable_verification_repair: 是否启用验证驱动局部修复（默认关闭）。
            repair_threshold: 低于该 match_score 的子问题会触发局部重答及下游重汇总。
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
        self.self_consistency_k = self_consistency_k
        self._no_context = _no_context
        self._no_constraint = _no_constraint
        self._no_feedback = _no_feedback
        self.dataset = dataset
        # 方向1：子答案双向交叉验证
        self.enable_backward_verify = enable_backward_verify
        self.enable_llm_match = enable_llm_match
        self.cs_struct_weight = cs_struct_weight
        self.cs_crossval_weight = cs_crossval_weight
        # 方向2：子答案置信度加权汇总（低置信子答案标注 uncertain，降低错误传播）
        self.enable_confidence_weighting = enable_confidence_weighting
        self.confidence_threshold = confidence_threshold
        # P2.3 消融：crossval 权重均匀(uniform) vs 下游加权(downstream,默认)
        self.uniform_crossval_weight = uniform_crossval_weight
        self.enable_soft_match = enable_soft_match
        self.enable_evidence_grounding = enable_evidence_grounding
        self.enable_verification_repair = enable_verification_repair
        self.repair_threshold = repair_threshold
        # P2.4 控制：反向验证锚点 answer_context(默认,用A+context) vs context_only(仅context)
        self.backward_anchor_mode = backward_anchor_mode
        # Self-Consistency 时每条 DAG 的分解温度（None=用默认 0.2）。
        # 注意：generate() 用的是参数 temperature，不读 model.temperature，
        # 所以必须在这里显式传给 _decompose，否则 K 条 DAG 无温度多样性。
        self._decompose_temperature: Optional[float] = None

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

        # ── 图级 Self-Consistency：生成K条DAG，用Consistency Score选优 ────
        if self.self_consistency_k > 1:
            return self._reason_with_self_consistency(question, context, context_section)

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
            # 方向2：估计子答案置信度（轻量启发式，零额外LLM调用）
            if self.enable_confidence_weighting:
                conf = self._estimate_sub_answer_confidence(sub_ans, raw_response, context)
            else:
                conf = 0.8 if sub_ans else 0.2
            node.confidence = conf

            sub_qa_chain.append({
                "node_id": node_id,
                "sub_question": sub_q_text,
                "sub_answer": sub_ans,
                "raw_response": raw_response,
                "confidence": conf,
            })

        # ── ⑤ 汇总最终答案 ─────────────────────────────────────────────────
        logger.debug("GERS: ⑤ 汇总最终答案...")
        # 方向2：置信度加权汇总——低置信子答案在汇总时标注 uncertain，降低错误传播
        if self.enable_confidence_weighting:
            reasoning_chain_text = self._format_reasoning_chain_weighted(sub_qa_chain)
        else:
            reasoning_chain_text = self._format_reasoning_chain(sub_qa_chain)
        final_prompt = FINAL_ANSWER_PROMPT.format(
            original_question=question,
            reasoning_chain=reasoning_chain_text,
        )
        final_response = self.model.generate(final_prompt, max_tokens=200, temperature=0.1)
        answer = self._extract_final_answer(final_response, question=question)
        reasoning_text = reasoning_chain_text + "\n\n" + final_response
        total_tokens += self.model.count_tokens(final_response) if hasattr(self.model, 'count_tokens') else 0

        # ── ⑥ 一致性校验 ──────────────────────────────────────────────────
        logger.debug("GERS: ⑥ ConsistencyChecker 校验...")
        check_result = self.consistency_checker.check(graph, reasoning_text)
        struct_score = check_result["structural_score"]
        score = check_result["consistency_score"]

        # ── ⑥.5 子答案双向交叉验证（方向1：内容层 CS，修复区分度）────────
        if self.enable_backward_verify and sub_qa_chain:
            cv = self._backward_verify(question, context, answer, sub_qa_chain)
            crossval_score = cv["crossval_score"]
            # 新 CS = w_struct·结构分 + w_crossval·正反向一致性分
            score = self.cs_struct_weight * struct_score + self.cs_crossval_weight * crossval_score
            score = round(max(0.0, min(1.0, score)), 4)
            check_result["crossval_score"] = crossval_score
            check_result["crossval_detail"] = cv["verifications"]
            check_result["consistency_score"] = score
            check_result["struct_score"] = struct_score
            logger.debug(f"GERS: ⑥.5 反向验证 crossval={crossval_score:.3f}, 新CS={score:.3f}")

            if self.enable_verification_repair:
                repair = self._repair_inconsistent_steps(
                    question=question,
                    context=context,
                    graph=graph,
                    execution_plan=execution_plan,
                    sub_qa_chain=sub_qa_chain,
                    verification=cv,
                )
                if repair and repair.get("repaired"):
                    answer = repair["answer"]
                    reasoning_text = repair["reasoning_text"]
                    sub_qa_chain = repair["sub_qa_chain"]
                    total_tokens += repair.get("token_count", 0)
                    check_result = repair["consistency_detail"]
                    score = check_result["consistency_score"]
                    check_result["repair_detail"] = repair["repair_detail"]
                    logger.debug(
                        f"GERS: ⑥.6 局部修复完成 repaired={repair['repair_detail']['repaired_nodes']}, CS={score:.3f}"
                    )

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

    def _reason_with_self_consistency(self, question: str, context: str, context_section: str) -> Dict:
        """图级 Self-Consistency：生成K条推理DAG，用Consistency Score选最优答案

        对标 CoT-SC 的多数投票，但用图结构质量打分代替朴素投票：
        - 生成 K 条不同的推理 DAG（不同 temperature）
        - 对每条 DAG 计算 Consistency Score（连通性+覆盖度+子答案质量）
        - 选得分最高的那条 DAG 的答案作为最终答案
        """
        import random
        K = self.self_consistency_k
        logger.debug(f"GERS-SC: 生成 {K} 条推理DAG...")

        candidates = []
        original_decomp_temp = self._decompose_temperature

        for k in range(K):
            # 每条DAG使用不同分解温度增加多样性（真正传入 _decompose → generate）
            temp = 0.3 + k * 0.2  # 0.3, 0.5, 0.7, ...
            self._decompose_temperature = temp

            # 临时关闭self_consistency避免递归
            saved_k = self.self_consistency_k
            self.self_consistency_k = 0
            try:
                result = self._reason_single(question, context, context_section)
            finally:
                self.self_consistency_k = saved_k

            cs = result.get("consistency_score", 0)
            if isinstance(cs, dict):
                cs = cs.get("consistency_score", 0)

            candidates.append({
                "answer": result["answer"],
                "reasoning_text": result["reasoning_text"],
                "graph": result["graph"],
                "consistency_score": cs,
                "consistency_detail": result.get("consistency_detail"),
                "sub_qa_chain": result.get("sub_qa_chain", []),
                "token_count": result.get("token_count", 0),
                "temperature": temp,
            })
            logger.debug(f"  DAG {k+1}/{K}: CS={cs:.4f}, answer={result['answer'][:40]}")

        # 恢复原始分解温度
        self._decompose_temperature = original_decomp_temp

        # 选Consistency Score最高的候选
        best = max(candidates, key=lambda c: c["consistency_score"])

        # 如果最高分候选有相同答案的其他候选，增加置信度
        best_answer = best["answer"]
        same_answer_count = sum(1 for c in candidates if c["answer"] == best_answer)

        logger.debug(f"GERS-SC: 选最优DAG (CS={best['consistency_score']:.4f}, answer={best_answer[:40]})")
        logger.debug(f"  同答案候选数: {same_answer_count}/{K}")

        return {
            "answer": best["answer"],
            "reasoning_text": best["reasoning_text"],
            "graph": best["graph"],
            "consistency_score": best["consistency_score"],
            "consistency_detail": best["consistency_detail"],
            "iterations": 0,
            "execution_plan": best["graph"].get_execution_order() if hasattr(best["graph"], "get_execution_order") else [],
            "sub_qa_chain": best["sub_qa_chain"],
            "token_count": sum(c["token_count"] for c in candidates),
            "sc_candidates": K,
            "sc_same_answer_count": same_answer_count,
            "sc_all_scores": [c["consistency_score"] for c in candidates],
        }

    def _reason_single(self, question: str, context: str, context_section: str) -> Dict:
        """单次GERS推理流程（供Self-Consistency调用）"""
        # 保存原始self_consistency_k，临时设为0避免递归
        saved_k = self.self_consistency_k
        self.self_consistency_k = 0
        try:
            # 重新执行 reason 的主体逻辑（从分解开始）
            return self._reason_core(question, context, context_section)
        finally:
            self.self_consistency_k = saved_k

    def _reason_core(self, question: str, context: str, context_section: str) -> Dict:
        """GERS核心推理流程（分解→构图→执行→汇总→校验），不含自适应和Self-Consistency"""
        # 这个方法直接复用 reason() 中 ⓪ 之后的逻辑
        # 通过临时修改属性来跳过自适应和SC检查
        saved_adaptive = self.adaptive
        self.adaptive = False
        try:
            return self.reason(question, context)
        finally:
            self.adaptive = saved_adaptive

    def _judge_complexity(self, question: str, context_section: str) -> bool:
        """判断问题是否为简单问题（单步可解）

        保守策略：有上下文的问题默认为complex（需要阅读和连接信息）
        只有纯计算题或无上下文的直接事实查询才判定为simple
        """
        # 有上下文的问题几乎都需要多步推理，默认为complex
        if context_section and len(context_section.strip()) > 20:
            # 只有纯计算题（GSM8K风格）才判为simple
            ql = question.lower()
            is_arithmetic = any(k in ql for k in [
                "calculate", "how much", "how many", "sum", "total",
                "multiply", "divide", "subtract", "add", "average",
            ]) and not any(k in ql for k in ["who", "what", "which", "where", "when", "why"])
            if is_arithmetic:
                return True
            # 有上下文的非计算题 → complex
            return False

        # 无上下文：用 LLM 判断
        prompt = COMPLEXITY_PROMPT.format(
            question=question,
            context_section=context_section
        )
        try:
            response = self.model.generate(prompt, max_tokens=20, temperature=0.0)
            label = response.strip().lower().strip("*#., \n")
            # 保守：只有明确说simple才判为simple
            return label.startswith("simple")
        except Exception:
            return False  # 出错时走完整流程

    def _answer_simple(self, question: str, context_section: str) -> Dict:
        """简单问题直接用CoT方式回答，跳过分解+构图+拓扑排序"""
        prompt = SIMPLE_ANSWER_PROMPT.format(
            question=question,
            context_section=context_section
        )
        reasoning_text = self.model.generate(prompt, max_tokens=500, temperature=0.3)
        answer = self._extract_simple_answer(reasoning_text, question)
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

    def _extract_simple_answer(self, text: str, question: str = None) -> str:
        """从CoT风格回答中提取答案（使用统一答案提取工具）"""
        from ..utils.answer_extractor import extract_answer
        dataset = self.dataset
        if not dataset and question:
            ql = question.lower()
            if any(k in ql for k in ["calculate", "how many", "how much", "sum", "multiply", "divide"]):
                dataset = "gsm8k"
        return extract_answer(text, dataset=dataset, reference=None, question=question)

    def _decompose(self, question: str, context_section: str) -> List[Dict]:
        """将问题分解为有依赖的子问题列表"""
        import json, re
        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            context_section=context_section
        )
        # Self-Consistency 时由 _reason_with_self_consistency 注入不同温度，保证 K 条 DAG 多样
        decomp_temp = self._decompose_temperature if self._decompose_temperature is not None else 0.2
        response = self.model.generate(prompt, max_tokens=600, temperature=decomp_temp)

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

    def _format_reasoning_chain_weighted(self, sub_qa_chain: List[Dict]) -> str:
        """方向2：置信度加权汇总。
        低置信子答案标注 [LOW CONFIDENCE]，提示汇总时不要过度依赖该步，
        降低错误传播。"""
        if not sub_qa_chain:
            return ""
        lines = []
        low_conf_count = 0
        for i, item in enumerate(sub_qa_chain, 1):
            conf = item.get("confidence", 0.8)
            tag = " [LOW CONFIDENCE - may be unreliable, verify before relying on it]" \
                if conf < self.confidence_threshold else ""
            if conf < self.confidence_threshold:
                low_conf_count += 1
            lines.append(f"Step {i}: {item['sub_question']}")
            lines.append(f"Answer: {item['sub_answer']}{tag}")
        if low_conf_count > 0:
            lines.append(f"\n[Note: {low_conf_count} step(s) above have LOW confidence. "
                         f"Cross-check these before finalizing your answer; if a low-confidence "
                         f"step is a prerequisite for later steps, the later steps may also be unreliable.]")
        return "\n".join(lines)

    # ─── 方向2：子答案置信度估计 ────────────────────────────────────────────

    def _estimate_sub_answer_confidence(self, sub_answer: str, raw_response: str,
                                        context: str = "") -> float:
        """
        轻量启发式估计子答案置信度（零额外 LLM 调用）。

        考量维度：
        1. 答案非空且长度合理（2~80字符）→ 高
        2. 含不确定信号（uncertain/unknown/cannot/无法）→ 大幅降权
        3. 答案核心词出现在上下文中（实体落地）→ 加分
        4. 纯数字答案 → 中性偏高（算术题通常确定）

        Returns: [0, 1] 置信度
        """
        import re
        if not sub_answer or len(sub_answer.strip()) < 2:
            return 0.2
        ans = sub_answer.strip()
        score = 0.6  # 基线

        # 长度合理性
        if 2 <= len(ans) <= 80:
            score += 0.15
        elif len(ans) > 150:
            score -= 0.2  # 过长多为啰嗦/不确定

        # 不确定信号
        low_conf_kw = ["uncertain", "unknown", "cannot", "can't", "not sure",
                       "unclear", "无法", "不确定", "不清楚", "可能", "maybe",
                       "perhaps", "might be", "not specified", "not mentioned"]
        ans_lower = ans.lower()
        if any(k in ans_lower for k in low_conf_kw):
            score -= 0.35

        # 答案核心词在上下文中出现（实体落地，有据可查）
        if context:
            # 取答案前几个词作为核心实体
            core_words = re.findall(r'[A-Za-z]+', ans)
            if core_words:
                core = " ".join(core_words[:3])
                if core.lower() in context.lower():
                    score += 0.15

        # 纯数字（算术题，通常确定）
        if re.fullmatch(r'-?\d+\.?\d*', ans.replace(',', '')):
            score += 0.1

        return max(0.1, min(1.0, score))

    # ─── 方向1：子答案双向交叉验证 ────────────────────────────────────────────

    def _backward_verify(self, question: str, context: str,
                         final_answer: str, sub_qa_chain: list) -> Dict:
        """
        反向验证：以最终答案 A + 上下文 C 为锚，反向逐个重答子问题，
        对比正向子答案 a_i 与反向子答案 a'_i 的一致性，输出正反向一致性分。

        创新点：DAG 结构独有，线性 CoT 无法实现。让 CS 从"图是否合法"
        升级为"推理内容是否自洽"，修复 CS 区分度。

        Returns:
            {"crossval_score": float, "verifications": [...]}
        """
        if not sub_qa_chain or not final_answer:
            return {"crossval_score": 0.5, "verifications": []}

        context_section = f"\nContext: {context[:1500]}" if context else ""
        verifications = []
        weighted_match = 0.0
        total_weight = 0.0
        n = len(sub_qa_chain)

        for i, item in enumerate(sub_qa_chain):
            sub_q = item.get("sub_question", "")
            forward_ans = item.get("sub_answer", "")
            # P2.3 消融：下游加权(默认) vs 均匀权重
            if self.uniform_crossval_weight:
                weight = 1.0
            else:
                weight = 0.5 + 0.5 * (i + 1) / n

            # P2.4 控制：answer_context(默认,用A+context) vs context_only(仅context,不用A)
            if self.enable_evidence_grounding and self.backward_anchor_mode != "context_only":
                prompt = BACKWARD_VERIFY_WITH_EVIDENCE_PROMPT.format(
                    original_question=question[:400],
                    context_section=context_section,
                    final_answer=str(final_answer)[:200],
                    step_index=i + 1,
                    sub_question=sub_q[:300],
                )
            elif self.backward_anchor_mode == "context_only":
                prompt = BACKWARD_VERIFY_PROMPT_CONTEXT_ONLY.format(
                    original_question=question[:400],
                    context_section=context_section,
                    step_index=i + 1,
                    sub_question=sub_q[:300],
                )
            else:
                prompt = BACKWARD_VERIFY_PROMPT.format(
                    original_question=question[:400],
                    context_section=context_section,
                    final_answer=str(final_answer)[:200],
                    step_index=i + 1,
                    sub_question=sub_q[:300],
                )
            try:
                resp = self.model.generate(prompt, max_tokens=150, temperature=0.0)
                backward_ans = self._extract_sub_answer(resp)
                evidence = self._extract_evidence(resp) if self.enable_evidence_grounding else ""
            except Exception:
                backward_ans = ""
                evidence = ""

            match_score = self._answer_match_score(forward_ans, backward_ans) \
                if self.enable_soft_match else float(self._answers_match(forward_ans, backward_ans))
            grounding_score = self._evidence_grounding_score(evidence, context) \
                if self.enable_evidence_grounding else 1.0
            combined_score = match_score * grounding_score
            matched = match_score >= 1.0
            weighted_match += weight * combined_score
            total_weight += weight
            verifications.append({
                "node_id": item.get("node_id"),
                "sub_question": sub_q,
                "forward_answer": forward_ans,
                "backward_answer": backward_ans,
                "match": matched,
                "match_score": match_score,
                "evidence": evidence,
                "grounding_score": grounding_score,
                "combined_score": combined_score,
            })

        crossval_score = weighted_match / total_weight if total_weight > 0 else 0.5
        return {"crossval_score": round(crossval_score, 4), "verifications": verifications}

    def _repair_inconsistent_steps(self, question: str, context: str, graph: ReasoningGraph,
                                   execution_plan: Dict, sub_qa_chain: list,
                                   verification: Dict) -> Optional[Dict]:
        """验证驱动局部修复：重答低 crossval 节点及其下游节点，然后重新汇总。

        默认不启用；用于后续实验验证“CS 不仅可诊断，还可定位并修复错误传播”。
        """
        verifications = verification.get("verifications", []) if verification else []
        if not verifications or not sub_qa_chain:
            return None

        low_nodes = {
            v.get("node_id")
            for v in verifications
            if v.get("node_id") and v.get("combined_score", v.get("match_score", float(v.get("match", False)))) < self.repair_threshold
        }
        if not low_nodes:
            return None

        affected = set(low_nodes)
        stack = list(low_nodes)
        while stack:
            nid = stack.pop()
            for succ in graph.get_neighbors(nid):
                if succ not in affected:
                    affected.add(succ)
                    stack.append(succ)

        chain_by_node = {item.get("node_id"): dict(item) for item in sub_qa_chain}
        sub_answers = {
            item.get("node_id"): item.get("sub_answer", "")
            for item in sub_qa_chain
            if item.get("node_id")
        }
        verify_by_node = {v.get("node_id"): v for v in verifications if v.get("node_id")}
        context_section = f"\nContext: {context[:1500]}" if context else ""
        repaired_nodes = []
        added_tokens = 0

        for node_id in execution_plan.get("execution_order", []):
            if node_id not in affected or node_id not in chain_by_node:
                continue
            node = graph.get_node(node_id)
            if node is None:
                continue
            item = chain_by_node[node_id]
            v = verify_by_node.get(node_id, {})
            prev_answers = self._format_previous_answers(node_id, graph, sub_answers)
            prompt = REPAIR_SUBQ_PROMPT.format(
                original_question=question[:400],
                context_section=context_section,
                previous_answers=prev_answers or "(none)",
                step_index=len(repaired_nodes) + 1,
                sub_question=node.content[:300],
                forward_answer=str(item.get("sub_answer", ""))[:200],
                backward_answer=str(v.get("backward_answer", ""))[:200],
            )
            try:
                raw = self.model.generate(prompt, max_tokens=200, temperature=0.1)
                repaired_answer = self._extract_sub_answer(raw)
            except Exception:
                continue

            old_answer = item.get("sub_answer", "")
            item["sub_answer"] = repaired_answer
            item["raw_response"] = raw
            item["repaired"] = True
            item["old_sub_answer"] = old_answer
            item["repair_reason"] = "forward/backward mismatch"
            chain_by_node[node_id] = item
            sub_answers[node_id] = repaired_answer
            node.metadata["answer"] = repaired_answer
            repaired_nodes.append(node_id)
            if hasattr(self.model, "count_tokens"):
                added_tokens += self.model.count_tokens(raw)

        if not repaired_nodes:
            return None

        repaired_chain = [chain_by_node.get(item.get("node_id"), item) for item in sub_qa_chain]
        if self.enable_confidence_weighting:
            reasoning_chain_text = self._format_reasoning_chain_weighted(repaired_chain)
        else:
            reasoning_chain_text = self._format_reasoning_chain(repaired_chain)
        final_prompt = FINAL_ANSWER_PROMPT.format(
            original_question=question,
            reasoning_chain=reasoning_chain_text,
        )
        final_response = self.model.generate(final_prompt, max_tokens=200, temperature=0.1)
        answer = self._extract_final_answer(final_response, question=question)
        reasoning_text = reasoning_chain_text + "\n\n" + final_response
        if hasattr(self.model, "count_tokens"):
            added_tokens += self.model.count_tokens(final_response)

        check_result = self.consistency_checker.check(graph, reasoning_text)
        repair_cv = self._backward_verify(question, context, answer, repaired_chain) \
            if self.enable_backward_verify else None
        if repair_cv:
            struct_score = check_result["structural_score"]
            crossval_score = repair_cv["crossval_score"]
            score = self.cs_struct_weight * struct_score + self.cs_crossval_weight * crossval_score
            score = round(max(0.0, min(1.0, score)), 4)
            check_result["crossval_score"] = crossval_score
            check_result["crossval_detail"] = repair_cv["verifications"]
            check_result["consistency_score"] = score
            check_result["struct_score"] = struct_score

        return {
            "repaired": True,
            "answer": answer,
            "reasoning_text": reasoning_text,
            "sub_qa_chain": repaired_chain,
            "consistency_detail": check_result,
            "token_count": added_tokens,
            "repair_detail": {
                "trigger_nodes": sorted(low_nodes),
                "repaired_nodes": repaired_nodes,
                "repair_threshold": self.repair_threshold,
            },
        }

    def _answers_match(self, a: str, b: str) -> bool:
        """判断正反向子答案是否语义一致。
        分级策略：字符串归一化匹配 → 包含关系 → 数值相等 → (可选)LLM 语义判断。
        """
        if not a or not b:
            return False
        from ..utils.answer_normalizer import normalize_for_vote
        import re

        na = normalize_for_vote(a)
        nb = normalize_for_vote(b)
        if na and na == nb:
            return True
        # 包含关系（短答案是长答案的子串，或反之）
        if na and nb and (na in nb or nb in na):
            return True
        # yes/no 归一后一致
        if na in ("yes", "no") and na == nb:
            return True
        # 数值相等
        nums_a = re.findall(r'-?\d+\.?\d*', a.replace(',', ''))
        nums_b = re.findall(r'-?\d+\.?\d*', b.replace(',', ''))
        if nums_a and nums_b:
            try:
                if abs(float(nums_a[-1]) - float(nums_b[-1])) < 1e-6:
                    return True
            except ValueError:
                pass
        # 仍不匹配：可选 LLM 语义判断（成本敏感默认关）
        if self.enable_llm_match and self.model is not None:
            return self._llm_answers_match(a, b)
        return False

    def _extract_evidence(self, text: str) -> str:
        """从 evidence-grounded verification 输出中提取证据片段。"""
        import re
        m = re.search(r'Evidence[：:]\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('"').strip()
        return ""

    def _evidence_grounding_score(self, evidence: str, context: str) -> float:
        """轻量证据落地分数，避免自洽但上下文无依据的答案得高分。

        1.0: 证据片段直接出现在上下文，或核心词高度重叠
        0.5: 证据部分词落在上下文中
        0.0: 明确 not supported / unknown，或无证据
        """
        if not evidence or not context:
            return 0.0
        ev = evidence.strip().lower().strip('"')
        ctx = context.lower()
        unsupported = [
            "not supported", "not in context", "not provided", "not mentioned",
            "unknown", "cannot determine", "no evidence", "context does not",
        ]
        if any(k in ev for k in unsupported):
            return 0.0
        if len(ev) >= 8 and ev in ctx:
            return 1.0

        import re
        words = [w for w in re.findall(r"[a-z0-9]+", ev) if len(w) > 2]
        if not words:
            return 0.0
        hit = sum(1 for w in set(words) if w in ctx)
        ratio = hit / max(len(set(words)), 1)
        if ratio >= 0.7:
            return 1.0
        if ratio >= 0.4:
            return 0.5
        return 0.0

    def _answer_match_score(self, a: str, b: str) -> float:
        """软匹配分数，供后续实验开启；默认实验仍使用二值 `_answers_match`。

        Returns:
            1.0: 明确等价（复用二值匹配）
            0.5/0.75: token-F1 较高但未达到严格等价
            0.0: 不匹配
        """
        if self._answers_match(a, b):
            return 1.0
        if not a or not b:
            return 0.0

        import re
        from ..utils.answer_normalizer import normalize_for_vote

        na = normalize_for_vote(a)
        nb = normalize_for_vote(b)
        ta = re.findall(r"[a-z0-9]+", na.lower())
        tb = re.findall(r"[a-z0-9]+", nb.lower())
        if not ta or not tb:
            return 0.0
        common = len(set(ta) & set(tb))
        if common == 0:
            return 0.0
        precision = common / len(set(ta))
        recall = common / len(set(tb))
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 >= 0.85:
            return 0.75
        if f1 >= 0.65:
            return 0.5
        return 0.0

    def _llm_answers_match(self, a: str, b: str) -> bool:
        """用 LLM 判断两个答案是否语义一致（仅在 enable_llm_match=True 时调用）。"""
        prompt = (f"Determine if the two answers are semantically equivalent "
                  f"(same meaning, possibly different wording/format).\n"
                  f"Answer A: {a[:100]}\nAnswer B: {b[:100]}\n"
                  f"Reply with ONLY 'YES' or 'NO'.")
        try:
            resp = self.model.generate(prompt, max_tokens=5, temperature=0.0)
            return resp.strip().upper().startswith("YES")
        except Exception:
            return False

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

    def _extract_final_answer(self, text: str, question: str = None, reference: str = None) -> str:
        """从最终回答中提取答案（使用统一答案提取工具）"""
        from ..utils.answer_extractor import extract_answer
        dataset = self.dataset
        if not dataset and question:
            ql = question.lower()
            if any(k in ql for k in ["calculate", "how many", "sum", "multiply", "+", "-", "="]) and "what" not in ql[:10]:
                dataset = "gsm8k"
        return extract_answer(text, dataset=dataset, reference=reference, question=question)
