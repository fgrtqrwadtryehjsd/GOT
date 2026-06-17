"""
证据覆盖度计算 —— 最大流方法

逻辑原理：
将事实节点视为"证据源"，结论节点视为"证据需求"，
通过最大流算法计算事实对结论的支撑覆盖程度。
覆盖度越高，说明推理链路越完整。

类比水网络：
- 事实节点 = 水源（产生证据流）
- 推导/支撑边 = 管道（传递证据流）
- 结论节点 = 需水点（需要证据支撑）
- 最大流 = 最大可传输的证据量
"""

from typing import Dict, List
from ..graph_representation.reasoning_graph import ReasoningGraph
from ..graph_representation.node import NodeType


class CoverageCalculator:
    """
    证据覆盖度计算器
    
    计算方法：
    1. 构建流网络（添加超级源和超级汇）
    2. 运行最大流算法
    3. 计算覆盖度 = 实际流量 / 理论最大流量
    """

    def compute(self, graph: ReasoningGraph) -> Dict:
        """
        计算证据覆盖度
        
        Returns:
            {
                "global_coverage": float,          # 全局覆盖度 [0, 1]
                "per_conclusion_coverage": {conc_id: float},  # 每个结论的覆盖度
                "uncovered_conclusions": [conc_ids],  # 未被覆盖的结论
                "coverage_score": float,            # 覆盖度得分
            }
        """
        fact_nodes = [n.node_id for n in graph.get_fact_nodes()]
        conclusion_nodes = [n.node_id for n in graph.get_conclusion_nodes()]

        if not fact_nodes or not conclusion_nodes:
            return {
                "global_coverage": 0.0,
                "per_conclusion_coverage": {},
                "uncovered_conclusions": conclusion_nodes,
                "coverage_score": 0.0,
            }

        # 全局覆盖度（最大流）
        global_coverage = graph.compute_coverage(fact_nodes, conclusion_nodes)

        # 每个结论的单独覆盖度
        per_conclusion = {}
        uncovered = []
        for conc_id in conclusion_nodes:
            single_coverage = graph.compute_coverage(fact_nodes, [conc_id])
            per_conclusion[conc_id] = single_coverage
            if single_coverage < 0.5:
                uncovered.append(conc_id)

        # 综合覆盖度得分
        if per_conclusion:
            avg_coverage = sum(per_conclusion.values()) / len(per_conclusion)
        else:
            avg_coverage = 0.0

        coverage_score = 0.6 * global_coverage + 0.4 * avg_coverage

        return {
            "global_coverage": global_coverage,
            "per_conclusion_coverage": per_conclusion,
            "uncovered_conclusions": uncovered,
            "coverage_score": max(0.0, min(1.0, coverage_score)),
        }