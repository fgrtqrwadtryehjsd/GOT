"""
Standard Chain-of-Thought 基线

最简单的思维链推理：在Prompt中添加"Let's think step by step"
"""

from typing import Dict, Optional

COT_PROMPT = """请回答以下问题，要求逐步推理。

问题：{question}
{context_section}

请一步一步地思考并给出答案。"""


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
        answer = self._extract_answer(reasoning_text)

        return {
            "answer": answer,
            "reasoning_text": reasoning_text,
            "method": "Standard CoT",
        }

    def _extract_answer(self, text: str) -> str:
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if any(kw in line for kw in ["答案是", "答案：", "最终答案"]):
                return line
        return lines[-1].strip() if lines else ""