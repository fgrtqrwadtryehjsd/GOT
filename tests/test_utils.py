"""
测试模块5：工具类 utils

覆盖：
- Metrics（EM/F1/图覆盖率/综合计算）
- Config（默认值与字段）
- GraphVisualizer（导出DOT格式，不依赖显示器）
- Logger（日志器创建）
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import Metrics
from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.visualization import GraphVisualizer
from src.graph_representation.reasoning_graph import ReasoningGraph


# ─── Metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_exact_match_true(self):
        assert Metrics.exact_match("巴黎", "巴黎") == 1.0

    def test_exact_match_false(self):
        assert Metrics.exact_match("巴黎", "伦敦") == 0.0

    def test_exact_match_case_insensitive(self):
        assert Metrics.exact_match("Paris", "paris") == 1.0

    def test_exact_match_punctuation_ignored(self):
        assert Metrics.exact_match("巴黎。", "巴黎") == 1.0

    def test_f1_perfect(self):
        assert Metrics.f1_score("the cat sat", "the cat sat") == 1.0

    def test_f1_partial(self):
        score = Metrics.f1_score("the cat sat on mat", "the cat sat")
        assert 0.0 < score < 1.0

    def test_f1_no_overlap(self):
        assert Metrics.f1_score("apple orange", "cat dog") == 0.0

    def test_f1_empty_prediction(self):
        assert Metrics.f1_score("", "巴黎") == 0.0

    def test_f1_empty_reference(self):
        assert Metrics.f1_score("巴黎", "") == 0.0

    def test_graph_coverage_full(self):
        g = ReasoningGraph()
        f1 = g.add_fact("事实1")
        f2 = g.add_fact("事实2")
        cov = Metrics.graph_coverage(g, [f1, f2])
        assert cov == 1.0

    def test_graph_coverage_partial(self):
        g = ReasoningGraph()
        f1 = g.add_fact("事实1")
        f2 = g.add_fact("事实2")
        cov = Metrics.graph_coverage(g, [f1])
        assert cov == 0.5

    def test_graph_coverage_empty_facts(self):
        g = ReasoningGraph()
        assert Metrics.graph_coverage(g, []) == 0.0

    def test_consistency_score_extraction(self):
        check_result = {"consistency_score": 0.85, "other": "..."}
        assert Metrics.consistency_score(check_result) == 0.85

    def test_consistency_score_missing_key(self):
        assert Metrics.consistency_score({}) == 0.0

    def test_compute_all_basic(self):
        result = Metrics.compute_all(
            prediction="巴黎",
            reference="巴黎",
            token_count=10,
            latency=0.5,
        )
        assert result["em"] == 1.0
        assert result["f1"] == 1.0
        assert result["token_count"] == 10
        assert result["latency"] == 0.5

    def test_compute_all_with_graph(self):
        g = ReasoningGraph()
        f1 = g.add_fact("事实1")
        result = Metrics.compute_all(
            prediction="答案",
            reference="答案",
            graph=g,
            reasoning_path=[f1],
        )
        assert "graph_coverage" in result

    def test_normalize(self):
        # 英文：去标点、统一小写、合并空格
        assert Metrics._normalize("  Hello, World!  ") == "hello world"
        # 中文：去英文标点（句号是中文标点，re 的 \w 对中文有效）
        result = Metrics._normalize("巴  黎。")
        assert "巴" in result and "黎" in result


# ─── Config ───────────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_model_name(self):
        cfg = Config()
        assert cfg.model_name == "Qwen/Qwen3-8B"

    def test_default_constraint_mode(self):
        cfg = Config()
        assert cfg.constraint_mode == "soft"

    def test_default_max_iterations(self):
        cfg = Config()
        assert cfg.max_iterations == 3

    def test_default_threshold(self):
        cfg = Config()
        assert 0 < cfg.consistency_threshold <= 1

    def test_weight_sum(self):
        cfg = Config()
        assert abs(cfg.alpha + cfg.beta - 1.0) < 1e-6

    def test_structural_weight_sum(self):
        cfg = Config()
        total = cfg.w_connectivity + cfg.w_cycle + cfg.w_coverage
        assert abs(total - 1.0) < 1e-6

    def test_custom_config(self):
        cfg = Config(model_name="custom/model", num_samples=100)
        assert cfg.model_name == "custom/model"
        assert cfg.num_samples == 100


# ─── GraphVisualizer ─────────────────────────────────────────────────────────

class TestGraphVisualizer:
    def _make_graph(self):
        g = ReasoningGraph(question="可视化测试")
        fid = g.add_fact("事实节点")
        sid = g.add_step("推理步骤")
        cid = g.add_conclusion("最终结论")
        g.add_derive(fid, sid)
        g.add_derive(sid, cid)
        return g

    def test_export_dot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.dot")
            viz = GraphVisualizer(output_dir=tmpdir)
            g = self._make_graph()
            result = viz.export_dot(g, output_path=out_path)
            assert os.path.exists(out_path)
            content = open(out_path, encoding="utf-8").read()
            assert "digraph" in content
            assert "事实节点" in content or "fact" in content.lower()

    def test_export_dot_contains_edges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.dot")
            viz = GraphVisualizer(output_dir=tmpdir)
            g = self._make_graph()
            viz.export_dot(g, output_path=out_path)
            content = open(out_path, encoding="utf-8").read()
            assert "->" in content

    def test_export_dot_empty_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "empty.dot")
            viz = GraphVisualizer(output_dir=tmpdir)
            g = ReasoningGraph()
            viz.export_dot(g, output_path=out_path)
            assert os.path.exists(out_path)

    def test_plot_networkx_no_display(self):
        """headless 环境下测试不调用 plt.show()，保存到文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "graph.png")
            viz = GraphVisualizer(output_dir=tmpdir)
            g = self._make_graph()
            try:
                import matplotlib
                matplotlib.use("Agg")  # headless backend
                viz.plot_networkx(g, title="测试图", save_path=out_path)
                assert os.path.exists(out_path)
            except ImportError:
                pytest.skip("matplotlib 未安装")


# ─── Logger ───────────────────────────────────────────────────────────────────

class TestLogger:
    def test_get_logger_returns_logger(self):
        import logging
        logger = get_logger("test_gers")
        assert isinstance(logger, logging.Logger)

    def test_logger_no_duplicate_handlers(self):
        logger1 = get_logger("dedup_test")
        handler_count = len(logger1.handlers)
        logger2 = get_logger("dedup_test")
        assert len(logger2.handlers) == handler_count

    def test_logger_to_file(self):
        import logging
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            logger = get_logger("file_test_logger_unique", log_file=log_path)
            logger.info("测试日志信息")
            # 关闭文件 handler，释放文件锁（Windows 需要）
            for h in logger.handlers[:]:
                h.close()
                logger.removeHandler(h)
            assert os.path.exists(log_path)
            content = open(log_path, encoding="utf-8").read()
            assert "测试日志信息" in content

    def test_logger_custom_level(self):
        import logging
        logger = get_logger("level_test", level="WARNING")
        assert logger.level == logging.WARNING
