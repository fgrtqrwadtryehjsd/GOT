"""
Self-Consistency (CoT-SC) 基线

多次采样+投票机制：对同一问题进行N次CoT推理，取最频繁的答案
"""

from typing import Dict, List
from collections import Counter
from .standard_cot import StandardCoT


class CoTSC:
    """Self-Consistency基线"""

    def __init__(self, model=None, num_samples: int = 5, temperature: float = 0.7):
        self.model = model
        self.num_samples = num_samples
        self.temperature = temperature
        self.cot = StandardCoT(model=model)

    def reason(self, question: str, context: str = "") -> Dict:
        if self.model is None:
            return {"answer": "", "reasoning_text": "[需要配置模型]", "method": "CoT-SC"}

        # 多次采样
        answers = []
        all_reasoning = []
        for _ in range(self.num_samples):
            result = self.cot.reason(question, context)
            answers.append(result["answer"])
            all_reasoning.append(result["reasoning_text"])

        # 投票取众数
        counter = Counter(answers)
        most_common = counter.most_common(1)[0] if counter else ("", 0)
        answer, vote_count = most_common

        return {
            "answer": answer,
            "reasoning_text": all_reasoning[0] if all_reasoning else "",
            "all_answers": answers,
            "vote_distribution": dict(counter),
            "confidence": vote_count / self.num_samples,
            "method": f"CoT-SC (N={self.num_samples})",
        }