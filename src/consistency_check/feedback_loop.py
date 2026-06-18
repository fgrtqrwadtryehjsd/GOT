"""
闭环修正机制 —— "生成→校验→回溯→修正"自动控制逻辑

核心思想：
当一致性校验得分低于阈值时，触发回溯修正：
1. 定位推理图中的问题节点/边
2. 重新规划受影响的推理路径
3. 重新生成受影响步骤的推理文本
4. 再次校验，循环直到得分达标或达到最大迭代次数
"""

from typing import Dict, Optional
from ..graph_representation.reasoning_graph import ReasoningGraph
from .consistency_score import ConsistencyChecker


FEEDBACK_PROMPT = """以下推理过程存在逻辑问题，请修正：

原始问题：{question}

原始推理过程：
{reasoning}

检测到的问题：
{issues}

请基于以下修正方向重新推理：
1. 补充断链处的缺失推理步骤
2. 解决循环论证，选择更合理的推理路径
3. 为证据不足的结论添加支撑
4. 消除逻辑矛盾

请给出修正后的完整推理过程："""


class FeedbackLoop:
    """
    生成-校验-修正闭环控制器
    
    工作流程：
    1. 校验推理结果 → 得到Consistency Score
    2. 如果 Score < threshold → 触发修正
    3. 定位问题 → 重新规划 → 重新生成 → 再次校验
    4. 循环直到 Score >= threshold 或达到最大迭代次数
    """

    def __init__(self,
                 model=None,
                 consistency_checker: Optional[ConsistencyChecker] = None,
                 max_iterations: int = 3,
                 threshold: float = 0.7):
        self.model = model
        self.consistency_checker = consistency_checker or ConsistencyChecker()
        self.max_iterations = max_iterations
        self.threshold = threshold

    def refine(self,
               graph: ReasoningGraph,
               reasoning_text: str,
               question: str,
               context: str = "") -> Dict:
        """
        执行闭环修正
        
        Returns:
            {
                "reasoning_text": 修正后的推理文本,
                "answer": 修正后的答案,
                "consistency_score": 最终一致性得分,
                "iterations": 实际迭代次数,
                "refinement_history": [每次迭代的记录],
            }
        """
        history = []
        current_text = reasoning_text
        current_score = 0.0

        for iteration in range(self.max_iterations):
            # 校验当前推理
            check_result = self.consistency_checker.check(graph, current_text)
            current_score = check_result["consistency_score"]
            issues = check_result["issues"]

            record = {
                "iteration": iteration + 1,
                "score": current_score,
                "issues": issues,
                "passed": check_result["passed"],
            }
            history.append(record)

            # 如果通过校验，停止迭代
            if current_score >= self.threshold:
                break

            # 如果没有模型，无法修正
            if self.model is None:
                break

            # 触发修正
            issues_text = "\n".join(f"- {issue}" for issue in issues)
            feedback_prompt = FEEDBACK_PROMPT.format(
                question=question,
                reasoning=current_text,
                issues=issues_text
            )

            new_text = self.model.generate(feedback_prompt)
            if new_text:
                current_text = new_text

            # 更新推理图（增量更新）
            from ..graph_representation.dynamic_builder import DynamicGraphBuilder
            from ..graph_representation.extractor import EntityRelationExtractor
            if self.model:
                extractor = EntityRelationExtractor(model=self.model)
                builder = DynamicGraphBuilder(extractor=extractor)
                graph = builder.update(graph, current_text)

        # 提取最终答案
        answer = self._extract_answer(current_text)

        return {
            "reasoning_text": current_text,
            "answer": answer,
            "consistency_score": current_score,
            "iterations": len(history),
            "refinement_history": history,
        }

    def _extract_answer(self, text: str) -> str:
        """提取最终答案（支持中英文、Markdown、数字回退）"""
        import re
        lines = text.strip().split("\n")
        keywords = [
            "答案是", "答案：", "最终答案", "Final Answer",
            "The answer is", "Therefore", "结论：", "结论:",
        ]
        for line in reversed(lines):
            line_s = line.strip().lstrip("#").strip()
            for kw in keywords:
                if kw in line_s:
                    after = line_s.split(kw, 1)[-1].strip(" :：*#")
                    if after:
                        num = re.search(r"-?\d[\d,\.]*", after)
                        return num.group().replace(",", "") if num else after
                    break
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            num = re.search(r"-?\d[\d,\.]*", line)
            if num:
                return num.group().replace(",", "")
        return lines[-1].strip() if lines else ""