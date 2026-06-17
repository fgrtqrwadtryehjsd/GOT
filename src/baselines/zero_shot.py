"""Zero-Shot / Few-Shot基线"""
from typing import Dict

ZERO_SHOT_PROMPT = """请回答以下问题。

问题：{question}
{context_section}

答案："""

FEW_SHOT_PROMPT = """请参考以下示例回答问题。

示例1：
问题：{example_q1}
答案：{example_a1}

示例2：
问题：{example_q2}
答案：{example_a2}

问题：{question}
{context_section}

答案："""


class ZeroShot:
    """Zero-Shot / Few-Shot基线"""

    def __init__(self, model=None, mode: str = "zero_shot", examples: list = None):
        self.model = model
        self.mode = mode
        self.examples = examples or []

    def reason(self, question: str, context: str = "") -> Dict:
        context_section = f"\n参考信息：{context}" if context else ""

        if self.mode == "few_shot" and len(self.examples) >= 2:
            prompt = FEW_SHOT_PROMPT.format(
                example_q1=self.examples[0]["question"],
                example_a1=self.examples[0]["answer"],
                example_q2=self.examples[1]["question"],
                example_a2=self.examples[1]["answer"],
                question=question,
                context_section=context_section,
            )
        else:
            prompt = ZERO_SHOT_PROMPT.format(
                question=question,
                context_section=context_section,
            )

        if self.model is None:
            return {"answer": "", "reasoning_text": "[需要配置模型]", "method": self.mode}

        response = self.model.generate(prompt)
        return {
            "answer": response.strip(),
            "reasoning_text": response,
            "method": self.mode,
        }