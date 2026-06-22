"""
图约束推理生成流水线 —— GERS核心Pipeline

完整推理流程：
1. 接收问题 → 构建推理图
2. 拓扑排序 → 路径规划
3. 图→Prompt转换 → 约束解码
4. 生成推理文本 → 一致性校验
5. 校验不通过 → 回溯修正（闭环）
"""

from typing import Dict, List, Optional, Tuple
from ..graph_representation.reasoning_graph import ReasoningGraph
from ..graph_representation.dynamic_builder import DynamicGraphBuilder
from ..graph_representation.extractor import EntityRelationExtractor
from .path_planner import PathPlanner
from .constrained_decoder import ConstrainedDecoder
from .prompt_builder import GraphPromptBuilder
from ..consistency_check.consistency_score import ConsistencyChecker
from ..consistency_check.feedback_loop import FeedbackLoop


class GraphGuidedGenerator:
    """
    图约束推理生成器 —— GERS系统核心
    
    实现"表征→生成→校验"闭环：
    
    输入问题 → 构建推理图 → 路径规划 → 约束生成 → 一致性校验 → (回溯修正) → 输出
    
    参数：
        model: LLM模型实例
        constraint_mode: 约束模式（soft/hard/hybrid）
        max_iterations: 最大校验-修正迭代次数
        consistency_threshold: 一致性校验阈值
    """

    def __init__(self,
                 model=None,
                 constraint_mode: str = "soft",
                 max_iterations: int = 3,
                 consistency_threshold: float = 0.7,
                 enable_nli: bool = False):
        self.model = model
        self.extractor = EntityRelationExtractor(model=model)
        self.graph_builder = DynamicGraphBuilder(extractor=self.extractor)
        self.path_planner = PathPlanner()
        self.decoder = ConstrainedDecoder(constraint_mode=constraint_mode)
        self.prompt_builder = GraphPromptBuilder()
        self.consistency_checker = ConsistencyChecker(
            enable_nli=enable_nli,
            nli_model=None  # NLI模型可选
        )
        self.feedback_loop = FeedbackLoop(
            model=model,
            consistency_checker=self.consistency_checker,
            max_iterations=max_iterations,
            threshold=consistency_threshold
        )
        self.max_iterations = max_iterations
        self.consistency_threshold = consistency_threshold

    def reason(self, question: str, context: str = "") -> Dict:
        """
        完整推理流程
        
        Returns:
            {
                "answer": 最终答案,
                "reasoning_text": 推理过程文本,
                "graph": 推理图对象,
                "consistency_score": 一致性得分,
                "iterations": 迭代修正次数,
                "execution_plan": 执行计划,
            }
        """
        # Step 1: 构建推理图
        graph = self.graph_builder.build(question, context)

        # Step 2: 路径规划
        execution_plan = self.path_planner.plan(graph)

        # Step 3: 构建约束Prompt
        base_prompt = self.prompt_builder.build(question, graph, execution_plan, context)
        constrained_prompt = self.decoder.apply_constraint(graph, execution_plan, base_prompt)

        # Step 4: 生成推理
        if self.model is None:
            reasoning_text = "[Demo模式] 需要配置LLM模型才能生成推理"
            answer = ""
        else:
            reasoning_text = self.model.generate(constrained_prompt)
            answer = self._extract_answer(reasoning_text)

        # Step 5: 一致性校验 + 闭环修正
        check_result = self.consistency_checker.check(graph, reasoning_text)
        score = check_result["consistency_score"]

        iteration_count = 0
        if score < self.consistency_threshold and self.model is not None:
            result = self.feedback_loop.refine(
                graph=graph,
                reasoning_text=reasoning_text,
                question=question,
                context=context
            )
            reasoning_text = result["reasoning_text"]
            answer = result["answer"]
            score = result["consistency_score"]
            iteration_count = result["iterations"]

        return {
            "answer": answer,
            "reasoning_text": reasoning_text,
            "graph": graph,
            "consistency_score": score,
            "iterations": iteration_count,
            "execution_plan": execution_plan,
        }

    def _extract_answer(self, reasoning_text: str) -> str:
        """从推理文本中提取最终答案

        优先找 'Final Answer: xxx'（全文扫描，不局限于最后几行）
        """
        import re

        # 全文扫描：找所有 Final Answer / 最终答案
        pattern = re.compile(
            r'(?:Final Answer|最终答案|The final answer is|Therefore[,，]\s*the answer is)'
            r'[：:\s]*(.+?)(?:\n|$)',
            re.IGNORECASE
        )
        matches = pattern.findall(reasoning_text)
        if matches:
            # 取最后一个匹配（最终结论）
            ans = matches[-1].strip().strip('*#.,。').strip()
            if ans and len(ans) < 200:
                return ans

        # 回退：其他答案关键词
        lines = reasoning_text.strip().split("\n")
        other_patterns = [
            re.compile(r'(?:The answer is|答案[是：:])[\s]*(.+)', re.IGNORECASE),
            re.compile(r'(?:结论[：:])[\s]*(.+)', re.IGNORECASE),
        ]
        for line in reversed(lines):
            for pat in other_patterns:
                m = pat.search(line)
                if m:
                    ans = m.group(1).strip().strip('*#.,。').strip()
                    if ans and len(ans) < 200:
                        return ans

        # 最终回退：取最后一个有意义的短句（排除闲聊）
        noise_patterns = [
            r'^如需', r'^如有', r'^需要', r'^欢迎', r'^可以.*扩展',
            r'^通过上述', r'^综上所述', r'^我们得到', r'^\$\$', r'^---'
        ]
        for line in reversed(lines):
            line = line.strip().lstrip('#*|>-').strip()
            if not line or len(line) < 2 or len(line) > 150:
                continue
            if re.match(r'^[\d\s\.\:步骤]+$', line):
                continue
            if re.match(r'^[|【】\[\]{}（）\(\)$$]+$', line):
                continue
            if any(re.match(p, line) for p in noise_patterns):
                continue
            return line

        return ""