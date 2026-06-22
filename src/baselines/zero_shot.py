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
        answer = self._extract_answer(response)
        return {
            "answer": answer,
            "reasoning_text": response,
            "method": self.mode,
        }

    def _extract_answer(self, text: str) -> str:
        """提取简洁答案，去掉解释性前缀和多余说明"""
        lines = text.strip().split("\n")
        keywords = [
            "答案：", "答案:", "答案是", "最终答案",
            "Final Answer:", "The answer is", "Therefore,",
            "**答案", "结论：",
        ]
        for line in reversed(lines):
            line_s = line.strip().lstrip("#*").strip()
            for kw in keywords:
                if kw in line_s:
                    after = line_s.split(kw, 1)[-1].strip(" :：*#\n")
                    after = after.split("\n")[0].strip(" *")
                    # 去掉解释性括号（如 "yes（是的）"）
                    after = after.split("（")[0].split("(")[0].strip(" *")
                    if after and len(after) < 150:
                        return after
                    break
        # 回退：取第一行简短内容
        for line in lines[:3]:
            line = line.strip().lstrip("#*").strip()
            if line and len(line) < 100:
                return line
        return lines[0].strip() if lines else ""