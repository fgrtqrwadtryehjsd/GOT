"""
Standard Chain-of-Thought 基线

最简单的思维链推理：在Prompt中添加"Let's think step by step"
"""

from typing import Dict, Optional

COT_PROMPT = """Answer the following question step by step.

Question: {question}
{context_section}

Think step by step, then give the final answer.
Final Answer: """


class StandardCoT:
    """标准思维链推理基线"""

    def __init__(self, model=None):
        self.model = model

    def reason(self, question: str, context: str = "") -> Dict:
        context_section = f"\n参考信息：{context}" if context else ""
        prompt = COT_PROMPT.format(question=question, context_section=context_section)

        if self.model is None:
            return {"answer": "", "reasoning_text": "[需要配置模型]", "method": "Standard CoT"}

        reasoning_text = self.model.generate(prompt)
        from ..utils.answer_extractor import extract_answer
        answer = extract_answer(reasoning_text, question=question)

        return {
            "answer": answer,
            "reasoning_text": reasoning_text,
            "method": "Standard CoT",
        }

    def _extract_answer(self, text: str) -> str:
        import re
        lines = text.strip().split("\n")
        keywords = [
            "答案是", "答案：", "答案:", "最终答案", "Final Answer",
            "The answer is", "Therefore,", "结论：", "结论:",
            "**答案", "## 答案",
        ]
        for line in reversed(lines):
            line_s = line.strip().lstrip("#*").strip()
            for kw in keywords:
                if kw in line_s:
                    after = line_s.split(kw, 1)[-1].strip(" :：*#\n")
                    # 去掉末尾多余说明（如 \n\n解释...）
                    after = after.split("\n")[0].strip(" *")
                    if after and len(after) < 200:
                        return after
                    break
        # 末尾回退：取最后非空行（长度 < 100）
        for line in reversed(lines):
            line = line.strip().lstrip("#*").strip()
            if line and len(line) < 100:
                return line
        return lines[-1].strip() if lines else ""