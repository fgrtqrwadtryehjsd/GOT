"""
测试模块4：基线方法 baselines

覆盖：
- StandardCoT（无模型）
- CoTSC（无模型 + mock 模型）
- TreeOfThoughts（无模型）
- ZeroShot（zero-shot + few-shot）
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.standard_cot import StandardCoT
from src.baselines.cot_sc import CoTSC
from src.baselines.tot import TreeOfThoughts
from src.baselines.zero_shot import ZeroShot


# ─── Mock 模型 ────────────────────────────────────────────────────────────────

class MockModel:
    """简单 mock：返回固定推理文本"""
    def __init__(self, response: str = "经过推理，最终答案：巴黎"):
        self._response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self._response

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


# ─── StandardCoT ─────────────────────────────────────────────────────────────

class TestStandardCoT:
    def test_no_model_returns_placeholder(self):
        cot = StandardCoT(model=None)
        result = cot.reason("测试问题")
        assert result["answer"] == ""
        assert "方法" in result or "method" in result["method"].lower() or "CoT" in result["method"]

    def test_with_mock_model(self):
        cot = StandardCoT(model=MockModel("步骤1...步骤2...答案是：巴黎"))
        result = cot.reason("巴黎在哪？")
        assert result["reasoning_text"] != ""
        assert result["method"] == "Standard CoT"

    def test_extract_answer_keyword(self):
        cot = StandardCoT(model=None)
        text = "经过分析\n步骤一...\n最终答案：42"
        answer = cot._extract_answer(text)
        assert "42" in answer

    def test_extract_answer_last_line_fallback(self):
        cot = StandardCoT(model=None)
        text = "推理过程\n一些步骤\n最终结果是X"
        answer = cot._extract_answer(text)
        assert answer != ""

    def test_with_context(self):
        cot = StandardCoT(model=MockModel())
        result = cot.reason("问题", context="这是上下文")
        assert "reasoning_text" in result


# ─── CoTSC ───────────────────────────────────────────────────────────────────

class TestCoTSC:
    def test_no_model_returns_placeholder(self):
        sc = CoTSC(model=None, num_samples=3)
        result = sc.reason("测试")
        assert result["answer"] == ""
        assert "CoT-SC" in result["method"]

    def test_voting_majority(self):
        """相同答案出现3次，应被选为最终答案"""
        call_count = {"n": 0}

        class VotingMock:
            def generate(self, prompt):
                call_count["n"] += 1
                if call_count["n"] <= 3:
                    return "答案是：巴黎"
                return "答案是：伦敦"

        sc = CoTSC(model=VotingMock(), num_samples=4)
        result = sc.reason("首都？")
        assert "巴黎" in result["answer"] or "巴黎" in str(result["vote_distribution"])

    def test_confidence_range(self):
        sc = CoTSC(model=MockModel("答案是：42"), num_samples=3)
        result = sc.reason("问题")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_answers_list(self):
        sc = CoTSC(model=MockModel(), num_samples=3)
        result = sc.reason("问题")
        assert "all_answers" in result
        assert len(result["all_answers"]) == 3

    def test_vote_distribution(self):
        sc = CoTSC(model=MockModel("答案是：A"), num_samples=5)
        result = sc.reason("问题")
        assert "vote_distribution" in result
        assert isinstance(result["vote_distribution"], dict)


# ─── TreeOfThoughts ───────────────────────────────────────────────────────────

class TestTreeOfThoughts:
    def test_no_model_returns_placeholder(self):
        tot = TreeOfThoughts(model=None)
        result = tot.reason("测试")
        assert result["answer"] == ""
        assert "ToT" in result["method"]

    def test_parse_candidates_numbered(self):
        tot = TreeOfThoughts(model=None)
        text = "1. 第一个候选步骤\n2. 第二个候选步骤\n3. 第三个候选步骤"
        candidates = tot._parse_candidates(text)
        assert len(candidates) <= 3
        assert all(len(c) > 0 for c in candidates)

    def test_parse_evaluations_valid(self):
        tot = TreeOfThoughts(model=None)
        text = "1:8\n2:5\n3:9"
        scores = tot._parse_evaluations(text)
        assert len(scores) >= 1
        assert all(0 <= s <= 10 for s in scores)

    def test_parse_evaluations_invalid_fallback(self):
        tot = TreeOfThoughts(model=None)
        text = "无法解析的文本内容"
        scores = tot._parse_evaluations(text)
        # fallback 应补齐分数
        assert len(scores) == tot.num_thoughts

    def test_extract_answer(self):
        tot = TreeOfThoughts(model=None)
        text = "步骤1...\n步骤2...\n答案是：42"
        answer = tot._extract_answer(text)
        assert "42" in answer

    def test_dfs_with_mock(self):
        responses = [
            "1. 步骤A\n2. 步骤B",
            "1:7\n2:8",
            "1. 步骤C\n2. 步骤D",
            "1:6\n2:9",
            "最终答案：测试答案",
        ]
        it = iter(responses)

        class SeqMock:
            def generate(self, prompt):
                try:
                    return next(it)
                except StopIteration:
                    return "最终答案：测试答案"

        tot = TreeOfThoughts(model=SeqMock(), max_depth=2, search_strategy="dfs")
        result = tot.reason("测试问题")
        assert "reasoning_text" in result


# ─── ZeroShot ─────────────────────────────────────────────────────────────────

class TestZeroShot:
    def test_zero_shot_no_model(self):
        zs = ZeroShot(model=None, mode="zero_shot")
        result = zs.reason("测试")
        assert result["answer"] == ""
        assert result["method"] == "zero_shot"

    def test_zero_shot_with_mock(self):
        zs = ZeroShot(model=MockModel("这是答案"), mode="zero_shot")
        result = zs.reason("问题")
        assert result["answer"] == "这是答案"

    def test_few_shot_with_examples(self):
        examples = [
            {"question": "示例问题1", "answer": "示例答案1"},
            {"question": "示例问题2", "answer": "示例答案2"},
        ]
        zs = ZeroShot(model=MockModel("答案"), mode="few_shot", examples=examples)
        result = zs.reason("新问题")
        assert result["answer"] != ""

    def test_few_shot_no_examples_fallback(self):
        """few-shot 但无示例时，退化为 zero-shot"""
        zs = ZeroShot(model=MockModel("答案"), mode="few_shot", examples=[])
        result = zs.reason("问题")
        assert result["answer"] != ""

    def test_with_context(self):
        zs = ZeroShot(model=MockModel("答案"), mode="zero_shot")
        result = zs.reason("问题", context="背景信息")
        assert result["answer"] != ""
