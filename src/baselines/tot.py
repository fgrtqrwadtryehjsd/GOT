"""
Tree of Thoughts (ToT) 基线

基于树搜索的推理方法：
1. 生成多个候选思维步骤
2. 评估每个步骤的质量
3. 使用BFS搜索最优推理路径

修复：传入context上下文，使用英文prompt适配英文数据集
"""

import re
from typing import Dict, List


TOT_GENERATE_PROMPT = """You are solving a multi-step reasoning problem.

Question: {question}
{context_section}

Current reasoning state:
{current_state}

Generate {num_thoughts} distinct next reasoning steps. Each step should advance toward the answer.
Format each step on a new line:
1. [first reasoning step]
2. [second reasoning step]
3. [third reasoning step]"""

TOT_EVALUATE_PROMPT = """Evaluate the quality of each reasoning step for answering the question.

Question: {question}
{context_section}

Current reasoning path: {reasoning_path}

Candidate next steps:
{candidates}

Score each step 1-10 based on how much it contributes to answering the question.
Format: step_number:score (e.g., 1:8 2:5 3:9)"""

TOT_ANSWER_PROMPT = """Based on the reasoning below, provide a concise final answer.

Question: {question}
{context_section}

Reasoning:
{reasoning_path}

Final Answer: <your concise answer>"""


class TreeOfThoughts:
    """Tree of Thoughts基线"""

    def __init__(self, model=None, num_thoughts: int = 3, max_depth: int = 3,
                 search_strategy: str = "bfs", beam_width: int = 2):
        self.model = model
        self.num_thoughts = num_thoughts
        self.max_depth = max_depth
        self.search_strategy = search_strategy
        self.beam_width = beam_width

    def reason(self, question: str, context: str = "") -> Dict:
        if self.model is None:
            return {"answer": "", "reasoning_text": "[需要配置模型]", "method": "ToT"}

        if self.search_strategy == "bfs":
            return self._bfs_search(question, context)
        else:
            return self._dfs_search(question, context)

    def _bfs_search(self, question: str, context: str) -> Dict:
        """BFS搜索最优推理路径"""
        context_section = f"\nContext: {context[:1500]}" if context else ""
        current_states = [{"steps": [], "score": 0.0, "text": ""}]
        best_result = None

        for depth in range(self.max_depth):
            next_states = []

            for state in current_states:
                current_text = state["text"]
                gen_prompt = TOT_GENERATE_PROMPT.format(
                    question=question,
                    context_section=context_section,
                    current_state=current_text or "(starting point)",
                    num_thoughts=self.num_thoughts
                )
                candidates = self.model.generate(gen_prompt, max_tokens=300, temperature=0.7)

                eval_prompt = TOT_EVALUATE_PROMPT.format(
                    question=question,
                    context_section=context_section,
                    reasoning_path=current_text or "(starting point)",
                    candidates=candidates
                )
                eval_result = self.model.generate(eval_prompt, max_tokens=100, temperature=0.0)

                candidate_steps = self._parse_candidates(candidates)
                scores = self._parse_evaluations(eval_result)

                for i, (step_text, score) in enumerate(zip(candidate_steps, scores)):
                    new_state = {
                        "steps": state["steps"] + [step_text],
                        "score": state["score"] + score,
                        "text": (state["text"] + "\n" + step_text).strip(),
                    }
                    next_states.append(new_state)

            next_states.sort(key=lambda x: x["score"], reverse=True)
            current_states = next_states[:self.beam_width]

        if not current_states:
            return {"answer": "", "reasoning_text": "", "method": "ToT (bfs)"}

        best_result = current_states[0]
        reasoning_text = best_result["text"]

        # 用专门的答案提取prompt
        answer_prompt = TOT_ANSWER_PROMPT.format(
            question=question,
            context_section=context_section,
            reasoning_path=reasoning_text
        )
        answer_response = self.model.generate(answer_prompt, max_tokens=200, temperature=0.1)
        from ..utils.answer_extractor import extract_answer
        answer = extract_answer(answer_response, question=question)

        return {
            "answer": answer,
            "reasoning_text": reasoning_text + "\n\n" + answer_response,
            "search_strategy": self.search_strategy,
            "depth_explored": self.max_depth,
            "method": f"ToT ({self.search_strategy})",
        }

    def _dfs_search(self, question: str, context: str) -> Dict:
        """DFS搜索（简化版：贪心选择最高分路径）"""
        context_section = f"\nContext: {context[:1500]}" if context else ""
        current_text = ""
        for depth in range(self.max_depth):
            gen_prompt = TOT_GENERATE_PROMPT.format(
                question=question,
                context_section=context_section,
                current_state=current_text or "(starting point)",
                num_thoughts=self.num_thoughts
            )
            candidates = self.model.generate(gen_prompt, max_tokens=300, temperature=0.7)
            candidate_steps = self._parse_candidates(candidates)
            if candidate_steps:
                current_text = (current_text + "\n" + candidate_steps[0]).strip()

        answer_prompt = TOT_ANSWER_PROMPT.format(
            question=question,
            context_section=context_section,
            reasoning_path=current_text
        )
        answer_response = self.model.generate(answer_prompt, max_tokens=200, temperature=0.1)

        from ..utils.answer_extractor import extract_answer as _extract
        return {
            "answer": _extract(answer_response, question=question),
            "reasoning_text": current_text + "\n\n" + answer_response,
            "method": "ToT (dfs)",
        }

    def _parse_candidates(self, text: str) -> List[str]:
        """解析候选步骤"""
        steps = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 去掉编号前缀 (1. 2. 3. - * 等)
            content = re.sub(r'^[\d\.\-\*\)]+\s*', '', line)
            if content and len(content) > 5:
                steps.append(content)
        return steps[:self.num_thoughts]

    def _parse_evaluations(self, text: str) -> List[float]:
        """解析评估得分"""
        scores = []
        # 尝试匹配 "1:8" 或 "1: 8" 格式
        matches = re.findall(r'(\d+)\s*:\s*(\d+(?:\.\d+)?)', text)
        for _, score_str in matches:
            try:
                score = float(score_str)
                scores.append(min(10.0, max(0.0, score)))
            except ValueError:
                scores.append(5.0)
        # 补齐缺失分数
        while len(scores) < self.num_thoughts:
            scores.append(5.0)
        return scores[:self.num_thoughts]

    def _extract_answer(self, text: str) -> str:
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
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip().lstrip('#*|>-').strip()
            if line and 2 < len(line) < 150:
                return line
        return text.strip()[:100]
