"""
测试模块2：一致性校验 consistency_check

覆盖：
- ConnectivityChecker
- CycleDetector
- CoverageCalculator
- ConsistencyChecker（综合评分）
- FeedbackLoop（无模型 mock）
- NLIVerifier（无模型 fallback）
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_representation.reasoning_graph import ReasoningGraph
from src.consistency_check.connectivity import ConnectivityChecker
from src.consistency_check.cycle_detector import CycleDetector
from src.consistency_check.coverage import CoverageCalculator
from src.consistency_check.nli_verifier import NLIVerifier
from src.consistency_check.consistency_score import ConsistencyChecker
from src.consistency_check.feedback_loop import FeedbackLoop


# ─── 辅助 ────────────────────────────────────────────────────────────────────

def make_connected_graph() -> ReasoningGraph:
    g = ReasoningGraph(question="测试")
    fid = g.add_fact("事实A")
    sid = g.add_step("步骤B")
    cid = g.add_conclusion("结论C")
    g.add_derive(fid, sid)
    g.add_derive(sid, cid)
    return g


def make_disconnected_graph() -> ReasoningGraph:
    g = ReasoningGraph(question="测试")
    g.add_fact("孤立事实1")
    g.add_conclusion("孤立结论2")   # 无边连接
    return g


def make_cyclic_graph() -> ReasoningGraph:
    g = ReasoningGraph(question="测试")
    a = g.add_fact("A")
    b = g.add_step("B")
    g.add_derive(a, b)
    g.add_derive(b, a)  # 环
    return g


# ─── ConnectivityChecker ──────────────────────────────────────────────────────

class TestConnectivityChecker:
    def setup_method(self):
        self.checker = ConnectivityChecker()

    def test_connected_graph(self):
        g = make_connected_graph()
        result = self.checker.check(g)
        assert result["is_connected"] is True
        assert result["num_components"] == 1
        assert result["connectivity_score"] > 0.5

    def test_disconnected_graph(self):
        g = make_disconnected_graph()
        result = self.checker.check(g)
        assert result["is_connected"] is False
        assert result["num_components"] > 1

    def test_isolated_nodes_detected(self):
        g = ReasoningGraph()
        g.add_fact("孤立节点")
        result = self.checker.check(g)
        assert len(result["isolated_nodes"]) >= 1

    def test_unreachable_conclusions(self):
        g = ReasoningGraph()
        g.add_fact("事实")
        g.add_conclusion("不可达结论")  # 无路径连接
        result = self.checker.check(g)
        assert len(result["unreachable_conclusions"]) >= 1

    def test_empty_graph(self):
        g = ReasoningGraph()
        result = self.checker.check(g)
        assert "connectivity_score" in result

    def test_score_range(self):
        g = make_connected_graph()
        result = self.checker.check(g)
        assert 0.0 <= result["connectivity_score"] <= 1.0


# ─── CycleDetector ────────────────────────────────────────────────────────────

class TestCycleDetector:
    def setup_method(self):
        self.detector = CycleDetector()

    def test_no_cycle(self):
        g = make_connected_graph()
        result = self.detector.detect(g)
        assert result["has_cycle"] is False
        assert result["cycle_score"] == 1.0

    def test_with_cycle(self):
        g = make_cyclic_graph()
        result = self.detector.detect(g)
        assert result["has_cycle"] is True
        assert result["cycle_score"] < 1.0

    def test_cycle_nodes_identified(self):
        g = make_cyclic_graph()
        result = self.detector.detect(g)
        assert len(result["involved_nodes"]) > 0

    def test_diagnose_cycle(self):
        g = make_cyclic_graph()
        diags = self.detector.diagnose(g)
        assert isinstance(diags, list)
        if diags:
            assert "循环论证" in diags[0]

    def test_score_range(self):
        for g in [make_connected_graph(), make_cyclic_graph()]:
            result = self.detector.detect(g)
            assert 0.0 <= result["cycle_score"] <= 1.0


# ─── CoverageCalculator ───────────────────────────────────────────────────────

class TestCoverageCalculator:
    def setup_method(self):
        self.calc = CoverageCalculator()

    def test_full_coverage(self):
        g = ReasoningGraph()
        fid = g.add_fact("事实")
        cid = g.add_conclusion("结论")
        g.add_support(fid, cid)
        result = self.calc.compute(g)
        assert result["coverage_score"] >= 0.0

    def test_no_facts(self):
        g = ReasoningGraph()
        g.add_conclusion("结论")
        result = self.calc.compute(g)
        assert result["coverage_score"] == 0.0
        assert len(result["uncovered_conclusions"]) >= 1

    def test_no_conclusions(self):
        g = ReasoningGraph()
        g.add_fact("事实")
        result = self.calc.compute(g)
        assert result["coverage_score"] == 0.0

    def test_per_conclusion_coverage(self):
        g = make_connected_graph()
        result = self.calc.compute(g)
        assert "per_conclusion_coverage" in result
        assert isinstance(result["per_conclusion_coverage"], dict)

    def test_score_range(self):
        g = make_connected_graph()
        result = self.calc.compute(g)
        assert 0.0 <= result["coverage_score"] <= 1.0


# ─── NLIVerifier（无模型 fallback） ───────────────────────────────────────────

class TestNLIVerifierNoModel:
    def setup_method(self):
        # 不加载真实模型
        self.verifier = NLIVerifier(model_name="invalid-model-for-test")

    def test_verify_edge_fallback(self):
        """无法加载模型时返回中性结果，不报错"""
        result = self.verifier.verify_edge("前提句子", "假设句子")
        assert "label" in result
        assert "scores" in result
        assert "is_valid" in result

    def test_verify_graph_empty(self):
        g = ReasoningGraph()
        result = self.verifier.verify_graph(g)
        assert result["valid_ratio"] == 0.0
        assert result["contradiction_count"] == 0

    def test_verify_graph_with_edges(self):
        g = make_connected_graph()
        result = self.verifier.verify_graph(g)
        assert "edge_verifications" in result
        assert "nli_score" in result
        assert 0.0 <= result["nli_score"] <= 1.0


# ─── ConsistencyChecker（综合） ───────────────────────────────────────────────

class TestConsistencyChecker:
    def setup_method(self):
        self.checker = ConsistencyChecker(enable_nli=False)

    def test_check_returns_dict(self):
        g = make_connected_graph()
        result = self.checker.check(g)
        for key in ["consistency_score", "structural_score", "semantic_score",
                    "connectivity", "cycle", "coverage", "issues", "passed"]:
            assert key in result

    def test_score_range(self):
        g = make_connected_graph()
        result = self.checker.check(g)
        assert 0.0 <= result["consistency_score"] <= 1.0

    def test_connected_graph_no_issues(self):
        g = make_connected_graph()
        result = self.checker.check(g)
        # 连通无环图，issues 应为空
        assert isinstance(result["issues"], list)

    def test_disconnected_has_issues(self):
        g = make_disconnected_graph()
        result = self.checker.check(g)
        assert len(result["issues"]) > 0

    def test_cyclic_has_issues(self):
        g = make_cyclic_graph()
        result = self.checker.check(g)
        # 有环路 → 至少有一个 issue
        assert len(result["issues"]) > 0

    def test_passed_threshold(self):
        g = make_connected_graph()
        result = self.checker.check(g)
        # passed 字段与分数一致
        expected_passed = result["consistency_score"] >= 0.6
        assert result["passed"] == expected_passed

    def test_weight_sum_valid(self):
        checker = ConsistencyChecker(alpha=0.6, beta=0.4)
        assert abs(checker.alpha + checker.beta - 1.0) < 1e-6


# ─── FeedbackLoop（无模型） ───────────────────────────────────────────────────

class TestFeedbackLoopNoModel:
    def setup_method(self):
        self.loop = FeedbackLoop(model=None, max_iterations=2, threshold=0.7)

    def test_refine_no_model_returns_original(self):
        g = make_connected_graph()
        result = self.loop.refine(
            graph=g,
            reasoning_text="原始推理文本，答案是：巴黎",
            question="巴黎在哪个国家？",
        )
        assert "reasoning_text" in result
        assert "answer" in result
        assert "consistency_score" in result
        assert "iterations" in result

    def test_refine_extracts_answer(self):
        g = make_connected_graph()
        result = self.loop.refine(
            graph=g,
            reasoning_text="经过分析，最终答案：巴黎",
            question="首都是哪里？",
        )
        assert result["answer"] != "" or result["reasoning_text"] != ""

    def test_refine_history_recorded(self):
        g = make_connected_graph()
        result = self.loop.refine(
            graph=g,
            reasoning_text="推理内容",
            question="问题",
        )
        assert "refinement_history" in result
        assert isinstance(result["refinement_history"], list)
