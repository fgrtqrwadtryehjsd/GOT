"""
Tree of Thoughts (ToT) 基线

基于树搜索的推理方法：
1. 生成多个候选思维步骤
2. 评估每个步骤的质量
3. 使用BFS/DFS搜索最优推理路径
"""

from typing import Dict, List, Optional


TOT_GENERATE_PROMPT = """问题：{question}

当前推理状态：
{current_state}

请生成{num_thoughts}个不同的下一步推理方向（每个方向一步推理）："""


TOT_EVALUATE_PROMPT = """问题：{question}

当前推理路径：{reasoning_path}

候选下一步：
{candidates}

请评估每个候选步骤对解决问题的贡献，给出1-10的评分。
格式：步骤编号:评分
例如：1:8 2:5 3:9"""


class TreeOfThoughts:
    """Tree of Thoughts基线"""

    def __init__(self, model=None, num_thoughts: int = 3, max_depth: int = 5,
                 search_strategy: str = "bfs", beam_width: int = 2):
        self.model = model
        self.num_thoughts = num_thoughts
        self.max_depth = max_depth
        self.search_strategy = search_strategy
        self.beam_width = beam_width

    def reason(self, question: str, context: str = "") -> Dict:
        if self.model is None:
            return {"answer": "", "reasoning_text": "[需要配置模型]", "method": "ToT"}

        # BFS搜索
        if self.search_strategy == "bfs":
            return self._bfs_search(question, context)
        else:
            return self._dfs_search(question, context)

    def _bfs_search(self, question: str, context: str) -> Dict:
        """BFS搜索最优推理路径"""
        # 每层保留beam_width个最佳候选
        current_states = [{"steps": [], "score": 0.0, "text": ""}]
        best_result = None

        for depth in range(self.max_depth):
            next_states = []

            for state in current_states:
                # 生成候选步骤
                current_text = state["text"]
                gen_prompt = TOT_GENERATE_PROMPT.format(
                    question=question,
                    current_state=current_text or "（推理起点）",
                    num_thoughts=self.num_thoughts
                )
                candidates = self.model.generate(gen_prompt)

                # 评估候选
                eval_prompt = TOT_EVALUATE_PROMPT.format(
                    question=question,
                    reasoning_path=current_text or "（起点）",
                    candidates=candidates
                )
                eval_result = self.model.generate(eval_prompt)

                # 解析评分并选择最佳
                candidate_steps = self._parse_candidates(candidates)
                scores = self._parse_evaluations(eval_result)

                for i, (step_text, score) in enumerate(zip(candidate_steps, scores)):
                    new_state = {
                        "steps": state["steps"] + [step_text],
                        "score": state["score"] + score,
                        "text": (state["text"] + "\n" + step_text).strip(),
                    }
                    next_states.append(new_state)

            # 保留top-beam_width
            next_states.sort(key=lambda x: x["score"], reverse=True)
            current_states = next_states[:self.beam_width]

            # 检查是否有状态已得出答案
            for state in current_states:
                if "答案是" in state["text"] or "最终答案" in state["text"]:
                    best_result = state
                    break

            if best_result:
                break

        if best_result is None and current_states:
            best_result = current_states[0]

        answer = self._extract_answer(best_result["text"]) if best_result else ""
        reasoning_text = best_result["text"] if best_result else ""

        return {
            "answer": answer,
            "reasoning_text": reasoning_text,
            "search_strategy": self.search_strategy,
            "depth_explored": self.max_depth,
            "method": f"ToT ({self.search_strategy})",
        }

    def _dfs_search(self, question: str, context: str) -> Dict:
        """DFS搜索（简化版：贪心选择最高分路径）"""
        current_text = ""
        for depth in range(self.max_depth):
            gen_prompt = TOT_GENERATE_PROMPT.format(
                question=question,
                current_state=current_text or "（推理起点）",
                num_thoughts=self.num_thoughts
            )
            candidates = self.model.generate(gen_prompt)

            candidate_steps = self._parse_candidates(candidates)
            if candidate_steps:
                current_text = (current_text + "\n" + candidate_steps[0]).strip()

            if "答案是" in current_text or "最终答案" in current_text:
                break

        return {
            "answer": self._extract_answer(current_text),
            "reasoning_text": current_text,
            "method": "ToT (dfs)",
        }

    def _parse_candidates(self, text: str) -> List[str]:
        """解析候选步骤"""
        steps = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                # 去掉编号前缀
                content = line.lstrip("0123456789.-) *")
                if content:
                    steps.append(content)
        return steps[:self.num_thoughts]

    def _parse_evaluations(self, text: str) -> List[float]:
        """解析评估得分"""
        scores = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":")
                try:
                    score = float(parts[-1].strip().split()[0])
                    scores.append(min(10.0, max(0.0, score)))
                except (ValueError, IndexError):
                    scores.append(5.0)
        # 补齐缺失分数
        while len(scores) < self.num_thoughts:
            scores.append(5.0)
        return scores

    def _extract_answer(self, text: str) -> str:
        lines = text.strip().split("\n")
        keywords = [
            "答案是", "答案：", "答案:", "最终答案", "Final Answer",
            "The answer is", "Therefore,", "结论：",
        ]
        for line in reversed(lines):
            line_s = line.strip().lstrip("#*").strip()
            for kw in keywords:
                if kw in line_s:
                    after = line_s.split(kw, 1)[-1].strip(" :：*#\n")
                    after = after.split("\n")[0].split("（")[0].strip(" *")
                    if after and len(after) < 150:
                        return after
                    break
        for line in reversed(lines):
            line = line.strip().lstrip("#*").strip()
            if line and len(line) < 100:
                return line
        return lines[-1].strip() if lines else ""