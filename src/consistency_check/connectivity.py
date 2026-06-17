"""
连通性检测 —— 识别推理链路中的断链（孤立节点）

逻辑原理：
如果推理图存在多个弱连通分量，说明推理链路中存在逻辑断层，
某些结论没有足够的事实支撑，或者某些事实没有被利用。
"""

from typing import Dict, List, Tuple
from ..graph_representation.reasoning_graph import ReasoningGraph
from ..graph_representation.node import NodeType


class ConnectivityChecker:
    """
    推理图连通性检测器
    
    检测内容：
    1. 弱连通分量数量（理想为1）
    2. 孤立节点（无入边也无出边的节点）
    3. 事实→结论的可达性
    """

    def check(self, graph: ReasoningGraph) -> Dict:
        """
        执行连通性检测
        
        Returns:
            {
                "num_components": int,           # 连通分量数
                "is_connected": bool,            # 是否完全连通
                "isolated_nodes": [node_ids],     # 孤立节点
                "unreachable_conclusions": [node_ids],  # 不可达的结论
                "connectivity_score": float,       # 连通性得分 [0, 1]
            }
        """
        components = graph.check_connectivity()
        num_components = len(components)

        # 识别孤立节点
        isolated_nodes = []
        for node_id, node in graph.nodes.items():
            if not graph.outgoing.get(node_id) and not graph.incoming.get(node_id):
                isolated_nodes.append(node_id)

        # 检查结论节点的可达性
        fact_ids = [n.node_id for n in graph.get_fact_nodes()]
        conclusion_ids = [n.node_id for n in graph.get_conclusion_nodes()]
        
        unreachable = []
        for conc_id in conclusion_ids:
            reachable = False
            for fact_id in fact_ids:
                path = graph.find_path(fact_id, conc_id)
                if path:
                    reachable = True
                    break
            if not reachable:
                unreachable.append(conc_id)

        # 计算连通性得分
        total_nodes = len(graph.nodes)
        if total_nodes == 0:
            connectivity_score = 0.0
        else:
            # 因子1：连通分量越少越好
            factor1 = 1.0 / max(num_components, 1)
            
            # 因子2：孤立节点越少越好
            factor2 = 1.0 - (len(isolated_nodes) / max(total_nodes, 1))
            
            # 因子3：结论可达率
            factor3 = 1.0 - (len(unreachable) / max(len(conclusion_ids), 1))
            
            connectivity_score = (0.3 * factor1 + 0.3 * factor2 + 0.4 * factor3)

        return {
            "num_components": num_components,
            "is_connected": num_components <= 1,
            "isolated_nodes": isolated_nodes,
            "unreachable_conclusions": unreachable,
            "connectivity_score": max(0.0, min(1.0, connectivity_score)),
        }