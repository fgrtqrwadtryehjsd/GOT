"""
测试模块1：推理状态图 graph_representation

覆盖：
- ReasoningGraph 节点/边增删查
- 拓扑排序
- 路径搜索
- 连通性检测
- 环路检测
- 最大流覆盖度
- DynamicGraphBuilder（无模型 mock）
- EntityRelationExtractor（无模型 mock）
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_representation.reasoning_graph import ReasoningGraph
from src.graph_representation.node import FactNode, StepNode, ConclusionNode, NodeType
from src.graph_representation.edge import DeriveEdge, SupportEdge, ConflictEdge, EdgeType
from src.graph_representation.extractor import EntityRelationExtractor
from src.graph_representation.dynamic_builder import DynamicGraphBuilder


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def make_simple_graph() -> ReasoningGraph:
    """构造一个简单三节点推理图用于复用：
       Fact(A) --derive--> Step(B) --derive--> Conclusion(C)
    """
    g = ReasoningGraph(question="测试问题")
    fid = g.add_fact("事实A：巴黎是法国首都")
    sid = g.add_step("过程B：法国首都在欧洲")
    cid = g.add_conclusion("结论C：巴黎在欧洲", answer="巴黎")
    g.add_derive(fid, sid, desc="地理推导")
    g.add_derive(sid, cid, desc="位置推导")
    return g


# ─── 节点与边基础测试 ──────────────────────────────────────────────────────────

class TestNodes:
    def test_fact_node_type(self):
        n = FactNode(content="事实内容")
        assert n.node_type == NodeType.FACT

    def test_step_node_type(self):
        n = StepNode(content="推理步骤")
        assert n.node_type == NodeType.STEP

    def test_conclusion_node_type(self):
        n = ConclusionNode(content="最终结论")
        assert n.node_type == NodeType.CONCLUSION

    def test_node_id_unique(self):
        n1 = FactNode(content="A")
        n2 = FactNode(content="B")
        assert n1.node_id != n2.node_id

    def test_node_repr(self):
        n = FactNode(content="测试节点内容")
        r = repr(n)
        assert "FACT" in r
        assert "测试节点内容" in r


class TestEdges:
    def test_derive_edge_type(self):
        e = DeriveEdge(src_id="a", dst_id="b")
        assert e.edge_type == EdgeType.DERIVE

    def test_support_edge_type(self):
        e = SupportEdge(src_id="a", dst_id="b")
        assert e.edge_type == EdgeType.SUPPORT

    def test_conflict_edge_type(self):
        e = ConflictEdge(src_id="a", dst_id="b")
        assert e.edge_type == EdgeType.CONFLICT

    def test_edge_id_unique(self):
        e1 = DeriveEdge(src_id="a", dst_id="b")
        e2 = DeriveEdge(src_id="a", dst_id="b")
        assert e1.edge_id != e2.edge_id


# ─── ReasoningGraph 基础测试 ───────────────────────────────────────────────────

class TestReasoningGraph:
    def test_add_fact(self):
        g = ReasoningGraph()
        nid = g.add_fact("事实内容")
        assert nid in g.nodes
        assert g.nodes[nid].node_type == NodeType.FACT

    def test_add_step(self):
        g = ReasoningGraph()
        nid = g.add_step("推理步骤")
        assert nid in g.nodes
        assert g.nodes[nid].node_type == NodeType.STEP

    def test_add_conclusion(self):
        g = ReasoningGraph()
        nid = g.add_conclusion("结论")
        assert nid in g.nodes
        assert g.nodes[nid].node_type == NodeType.CONCLUSION

    def test_add_derive_edge(self):
        g = ReasoningGraph()
        fid = g.add_fact("事实")
        sid = g.add_step("步骤")
        eid = g.add_derive(fid, sid)
        assert eid in g.edges
        assert g.edges[eid].edge_type == EdgeType.DERIVE

    def test_add_support_edge(self):
        g = ReasoningGraph()
        fid = g.add_fact("证据")
        cid = g.add_conclusion("结论")
        eid = g.add_support(fid, cid, strength=0.8)
        assert eid in g.edges
        assert g.edges[eid].edge_type == EdgeType.SUPPORT

    def test_add_conflict_edge(self):
        g = ReasoningGraph()
        n1 = g.add_fact("A是最大的")
        n2 = g.add_fact("B比A更大")
        eid = g.add_conflict(n1, n2, reason="矛盾")
        assert eid in g.edges
        assert g.edges[eid].edge_type == EdgeType.CONFLICT

    def test_get_node(self):
        g = ReasoningGraph()
        nid = g.add_fact("查询节点")
        node = g.get_node(nid)
        assert node is not None
        assert node.content == "查询节点"

    def test_get_nonexistent_node(self):
        g = ReasoningGraph()
        assert g.get_node("nonexistent") is None

    def test_get_neighbors(self):
        g = make_simple_graph()
        facts = g.get_fact_nodes()
        assert len(facts) > 0
        fid = facts[0].node_id
        neighbors = g.get_neighbors(fid)
        assert len(neighbors) > 0

    def test_get_predecessors(self):
        g = make_simple_graph()
        concs = g.get_conclusion_nodes()
        cid = concs[0].node_id
        preds = g.get_predecessors(cid)
        assert len(preds) > 0

    def test_node_type_filters(self):
        g = make_simple_graph()
        assert len(g.get_fact_nodes()) == 1
        assert len(g.get_step_nodes()) == 1
        assert len(g.get_conclusion_nodes()) == 1

    def test_edge_type_filters(self):
        g = make_simple_graph()
        assert len(g.get_derive_edges()) == 2
        assert len(g.get_support_edges()) == 0
        assert len(g.get_conflict_edges()) == 0

    def test_summary(self):
        g = make_simple_graph()
        s = g.summary()
        assert s["total_nodes"] == 3
        assert s["total_edges"] == 2
        assert s["fact_nodes"] == 1
        assert s["step_nodes"] == 1
        assert s["conclusion_nodes"] == 1

    def test_repr(self):
        g = make_simple_graph()
        r = repr(g)
        assert "ReasoningGraph" in r


# ─── 图论算法测试 ──────────────────────────────────────────────────────────────

class TestGraphAlgorithms:
    def test_topological_sort_order(self):
        """拓扑排序：事实节点应在结论节点之前"""
        g = make_simple_graph()
        order = g.topological_sort()
        assert len(order) == 3
        fact_ids = {n.node_id for n in g.get_fact_nodes()}
        conc_ids = {n.node_id for n in g.get_conclusion_nodes()}
        fact_pos = min(order.index(fid) for fid in fact_ids if fid in order)
        conc_pos = max(order.index(cid) for cid in conc_ids if cid in order)
        assert fact_pos < conc_pos

    def test_find_path_dfs(self):
        g = make_simple_graph()
        fid = g.get_fact_nodes()[0].node_id
        cid = g.get_conclusion_nodes()[0].node_id
        path = g.find_path(fid, cid, method="dfs")
        assert len(path) >= 2
        assert path[0] == fid
        assert path[-1] == cid

    def test_find_path_bfs(self):
        g = make_simple_graph()
        fid = g.get_fact_nodes()[0].node_id
        cid = g.get_conclusion_nodes()[0].node_id
        path = g.find_path(fid, cid, method="bfs")
        assert len(path) >= 2

    def test_find_path_shortest(self):
        g = make_simple_graph()
        fid = g.get_fact_nodes()[0].node_id
        cid = g.get_conclusion_nodes()[0].node_id
        path = g.find_path(fid, cid, method="shortest")
        assert len(path) >= 2

    def test_find_path_no_connection(self):
        """不相连节点之间找不到路径"""
        g = ReasoningGraph()
        n1 = g.add_fact("孤立事实1")
        n2 = g.add_fact("孤立事实2")
        path = g.find_path(n1, n2)
        assert path == []

    def test_find_all_paths(self):
        g = make_simple_graph()
        fid = g.get_fact_nodes()[0].node_id
        cid = g.get_conclusion_nodes()[0].node_id
        paths = g.find_all_paths(fid, cid)
        assert len(paths) >= 1

    def test_check_connectivity_single_component(self):
        g = make_simple_graph()
        components = g.check_connectivity()
        assert len(components) == 1

    def test_check_connectivity_disconnected(self):
        g = ReasoningGraph()
        g.add_fact("孤立节点A")
        g.add_fact("孤立节点B")
        # 两个孤立节点 → 两个连通分量
        components = g.check_connectivity()
        assert len(components) == 2

    def test_detect_cycles_no_cycle(self):
        g = make_simple_graph()
        cycles = g.detect_cycles()
        assert cycles == []

    def test_detect_cycles_with_cycle(self):
        """构造 A→B→C→A 的环路"""
        g = ReasoningGraph()
        a = g.add_fact("A")
        b = g.add_step("B")
        c = g.add_step("C")
        g.add_derive(a, b)
        g.add_derive(b, c)
        g.add_derive(c, a)
        cycles = g.detect_cycles()
        assert len(cycles) > 0

    def test_compute_coverage_full(self):
        """事实直接连到结论时覆盖度应为正值"""
        g = ReasoningGraph()
        fid = g.add_fact("事实")
        cid = g.add_conclusion("结论")
        g.add_support(fid, cid)
        coverage = g.compute_coverage([fid], [cid])
        assert coverage > 0.0

    def test_compute_coverage_empty(self):
        g = ReasoningGraph()
        assert g.compute_coverage([], []) == 0.0


# ─── DynamicGraphBuilder & Extractor 测试（无模型 mock） ─────────────────────

class TestExtractorNoModel:
    def test_default_extraction(self):
        extractor = EntityRelationExtractor(model=None)
        result = extractor.extract("任意问题")
        for key in ["facts", "steps", "conclusions", "relations"]:
            assert key in result
            assert isinstance(result[key], list)

    def test_parse_valid_json(self):
        extractor = EntityRelationExtractor(model=None)
        json_resp = '''```json
{
    "facts": [{"content": "事实1", "source": "question", "evidence": ""}],
    "steps": [{"content": "步骤1", "operation": "推导"}],
    "conclusions": [{"content": "结论1", "answer": "答案"}],
    "relations": [{"source": "事实1", "target": "步骤1", "type": "derive", "description": "推导"}]
}
```'''
        result = extractor._parse_response(json_resp)
        assert len(result["facts"]) == 1
        assert result["facts"][0]["content"] == "事实1"

    def test_parse_invalid_json_fallback(self):
        extractor = EntityRelationExtractor(model=None)
        result = extractor._parse_response("这不是JSON")
        assert result == extractor._default_extraction()


class TestDynamicBuilderNoModel:
    def test_build_empty_graph(self):
        """无模型时构建空图，节点为0"""
        extractor = EntityRelationExtractor(model=None)
        builder = DynamicGraphBuilder(extractor=extractor)
        g = builder.build("测试问题")
        assert isinstance(g, ReasoningGraph)
        assert g.question == "测试问题"

    def test_build_with_mock_extraction(self):
        """用 mock 抽取结果构建图"""
        class MockModel:
            def generate(self, prompt):
                return '''{"facts": [{"content": "mock事实", "source": "q"}],
                           "steps": [{"content": "mock步骤"}],
                           "conclusions": [{"content": "mock结论", "answer": "mock"}],
                           "relations": [{"source": "mock事实", "target": "mock步骤",
                                          "type": "derive", "description": ""}]}'''

        extractor = EntityRelationExtractor(model=MockModel())
        builder = DynamicGraphBuilder(extractor=extractor)
        g = builder.build("测试问题")
        assert len(g.nodes) >= 1

    def test_update_graph(self):
        extractor = EntityRelationExtractor(model=None)
        builder = DynamicGraphBuilder(extractor=extractor)
        g = builder.build("测试")
        initial_nodes = len(g.nodes)
        g2 = builder.update(g, "新信息补充")
        assert isinstance(g2, ReasoningGraph)
        # 节点数不应减少
        assert len(g2.nodes) >= initial_nodes
