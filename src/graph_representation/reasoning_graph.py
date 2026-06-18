"""
推理状态图核心类 —— G=(V, E)

实现推理图的数据结构与图论算法：
- 动态节点/边的添加与更新
- 拓扑排序（路径规划）
- 连通分量检测（一致性校验）
- 环路检测（循环论证检测）
- 最大流计算（证据覆盖度）
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .node import NodeBase, FactNode, StepNode, ConclusionNode, NodeType
from .edge import EdgeBase, DeriveEdge, SupportEdge, ConflictEdge, EdgeType


class ReasoningGraph:
    """
    推理状态图 G = (V, E)
    
    V: 节点集合，包含事实节点、过程节点、目标节点
    E: 边集合，包含推导关系、支撑关系、互斥关系
    
    提供图论算法支撑：
    - 拓扑排序 → 路径规划
    - 连通分量 → 断链检测
    - 环路检测 → 循环论证检测
    - 最大流 → 证据覆盖度计算
    """

    def __init__(self, question: str = "", answer: str = ""):
        self.question = question
        self.answer = answer
        
        # 节点与边存储
        self.nodes: Dict[str, NodeBase] = {}
        self.edges: Dict[str, EdgeBase] = {}
        
        # 邻接表（前向 & 反向）
        self.outgoing: Dict[str, List[str]] = defaultdict(list)  # node_id -> [edge_ids]
        self.incoming: Dict[str, List[str]] = defaultdict(list)  # node_id -> [edge_ids]
        
        # NetworkX底层图（用于图论算法）
        self._nx_graph: nx.DiGraph = nx.DiGraph()

    def add_node(self, node: NodeBase) -> str:
        """添加节点到推理图"""
        self.nodes[node.node_id] = node
        self._nx_graph.add_node(node.node_id, 
                                content=node.content,
                                node_type=node.node_type.value,
                                confidence=node.confidence)
        return node.node_id

    def add_fact(self, content: str, source: str = None, evidence: str = None) -> str:
        """便捷方法：添加事实节点"""
        node = FactNode(content=content, source=source, evidence_text=evidence)
        return self.add_node(node)

    def add_step(self, content: str, operation: str = None) -> str:
        """便捷方法：添加过程节点"""
        node = StepNode(content=content, operation=operation)
        return self.add_node(node)

    def add_conclusion(self, content: str, answer: str = None) -> str:
        """便捷方法：添加目标节点"""
        node = ConclusionNode(content=content, answer=answer)
        return self.add_node(node)

    def add_edge(self, edge: EdgeBase) -> str:
        """添加边到推理图"""
        self.edges[edge.edge_id] = edge
        self.outgoing[edge.src_id].append(edge.edge_id)
        self.incoming[edge.dst_id].append(edge.edge_id)
        
        # 更新NetworkX图（仅非Conflict边参与拓扑排序）
        if edge.edge_type != EdgeType.CONFLICT:
            self._nx_graph.add_edge(edge.src_id, edge.dst_id,
                                    edge_type=edge.edge_type.value,
                                    weight=edge.weight,
                                    edge_id=edge.edge_id)
        else:
            # Conflict边用特殊标记
            self._nx_graph.add_edge(edge.src_id, edge.dst_id,
                                    edge_type=edge.edge_type.value,
                                    weight=-edge.weight,
                                    edge_id=edge.edge_id)
        return edge.edge_id

    def add_derive(self, src_id: str, dst_id: str, desc: str = None) -> str:
        """便捷方法：添加推导边"""
        edge = DeriveEdge(src_id=src_id, dst_id=dst_id, description=desc)
        return self.add_edge(edge)

    def add_support(self, src_id: str, dst_id: str, strength: float = 1.0) -> str:
        """便捷方法：添加支撑边"""
        edge = SupportEdge(src_id=src_id, dst_id=dst_id, support_strength=strength)
        return self.add_edge(edge)

    def add_conflict(self, src_id: str, dst_id: str, reason: str = None) -> str:
        """便捷方法：添加互斥边"""
        edge = ConflictEdge(src_id=src_id, dst_id=dst_id, conflict_reason=reason)
        return self.add_edge(edge)

    def get_node(self, node_id: str) -> Optional[NodeBase]:
        """获取节点"""
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[EdgeBase]:
        """获取边"""
        return self.edges.get(edge_id)

    def get_neighbors(self, node_id: str) -> List[str]:
        """获取节点的所有邻居节点ID"""
        neighbor_ids = []
        for edge_id in self.outgoing[node_id]:
            edge = self.edges[edge_id]
            neighbor_ids.append(edge.dst_id)
        return neighbor_ids

    def get_neighbor_content(self, node_id: str) -> str:
        """获取邻居节点的文本内容（用于 Prompt 构建）"""
        node = self.get_node(node_id)
        return node.content if node else node_id

    def get_predecessors(self, node_id: str) -> List[str]:
        """获取节点的所有前驱节点ID"""
        predecessor_ids = []
        for edge_id in self.incoming[node_id]:
            edge = self.edges[edge_id]
            predecessor_ids.append(edge.src_id)
        return predecessor_ids

    # ===================== 图论算法 =====================

    def topological_sort(self) -> List[str]:
        """
        拓扑排序 —— 获取推理执行顺序
        
        返回节点ID列表，按逻辑依赖关系排序：
        事实节点 → 过程节点 → 目标节点
        
        用于：路径规划模块（PathPlanner）
        """
        try:
            # 仅考虑非Conflict边构建拓扑排序
            derive_graph = nx.DiGraph()
            for node_id in self.nodes:
                derive_graph.add_node(node_id)
            for edge in self.edges.values():
                if edge.edge_type in (EdgeType.DERIVE, EdgeType.SUPPORT):
                    derive_graph.add_edge(edge.src_id, edge.dst_id)
            
            return list(nx.topological_sort(derive_graph))
        except nx.NetworkXUnfeasible:
            # 存在环路，返回部分排序
            return list(nx.lexicographical_topological_sort(
                derive_graph, key=lambda x: x))

    def find_path(self, src_id: str, dst_id: str, method: str = "dfs") -> List[str]:
        """
        路径搜索 —— 从src到dst的推理路径
        
        Args:
            method: "dfs" 深度优先 | "bfs" 广度优先 | "shortest" 最短路径
        
        Returns:
            路径上的节点ID列表
        """
        derive_graph = nx.DiGraph()
        for node_id in self.nodes:
            derive_graph.add_node(node_id)
        for edge in self.edges.values():
            if edge.edge_type in (EdgeType.DERIVE, EdgeType.SUPPORT):
                derive_graph.add_edge(edge.src_id, edge.dst_id)

        if method == "shortest":
            try:
                return nx.shortest_path(derive_graph, src_id, dst_id)
            except nx.NetworkXNoPath:
                return []
        elif method == "dfs":
            try:
                paths = list(nx.all_simple_paths(derive_graph, src_id, dst_id))
                return paths[0] if paths else []
            except nx.NetworkXNoPath:
                return []
        elif method == "bfs":
            try:
                return nx.shortest_path(derive_graph, src_id, dst_id)
            except nx.NetworkXNoPath:
                return []
        return []

    def find_all_paths(self, src_id: str, dst_id: str, cutoff: int = 10) -> List[List[str]]:
        """搜索所有可能的推理路径"""
        derive_graph = nx.DiGraph()
        for node_id in self.nodes:
            derive_graph.add_node(node_id)
        for edge in self.edges.values():
            if edge.edge_type in (EdgeType.DERIVE, EdgeType.SUPPORT):
                derive_graph.add_edge(edge.src_id, edge.dst_id)

        try:
            return list(nx.all_simple_paths(derive_graph, src_id, dst_id, cutoff=cutoff))
        except nx.NetworkXNoPath:
            return []

    def check_connectivity(self) -> Dict[str, List[List[str]]]:
        """
        连通分量检测 —— 识别推理链路中的断链
        
        返回各连通分量，如果存在多个弱连通分量，
        说明推理链路中存在逻辑断层（断链）
        
        用于：一致性校验模块
        """
        # 使用无向图检测弱连通分量
        undirected = self._nx_graph.to_undirected()
        components = list(nx.connected_components(undirected))
        
        result = {}
        for i, component in enumerate(components):
            result[f"component_{i}"] = sorted(list(component))
        
        return result

    def detect_cycles(self) -> List[List[str]]:
        """
        环路检测 —— 识别循环论证
        
        使用快慢指针（Floyd算法）检测有向图中的环路
        
        用于：一致性校验模块（循环论证检测）
        """
        try:
            cycles = list(nx.simple_cycles(self._nx_graph))
            return cycles
        except Exception:
            return []

    def compute_coverage(self, fact_nodes: List[str], conclusion_nodes: List[str]) -> float:
        """
        证据覆盖度计算 —— 最大流方法
        
        计算事实节点对结论节点的支撑覆盖程度
        
        用于：一致性校验模块
        """
        if not fact_nodes or not conclusion_nodes:
            return 0.0

        # 构建流网络：添加超级源和超级汇
        flow_graph = nx.DiGraph()
        
        super_source = "SUPER_SOURCE"
        super_sink = "SUPER_SINK"
        flow_graph.add_node(super_source)
        flow_graph.add_node(super_sink)

        # 超级源 → 事实节点（容量=1）
        for fact_id in fact_nodes:
            flow_graph.add_edge(super_source, fact_id, capacity=1.0)

        # 中间边（推导/支撑）
        for edge in self.edges.values():
            if edge.edge_type in (EdgeType.DERIVE, EdgeType.SUPPORT):
                if edge.src_id in self.nodes and edge.dst_id in self.nodes:
                    flow_graph.add_edge(edge.src_id, edge.dst_id, 
                                       capacity=edge.weight)

        # 结论节点 → 超级汇（容量=1）
        for conc_id in conclusion_nodes:
            flow_graph.add_edge(conc_id, super_sink, capacity=1.0)

        try:
            flow_value, flow_dict = nx.maximum_flow(flow_graph, super_source, super_sink)
            max_possible = min(len(fact_nodes), len(conclusion_nodes))
            return flow_value / max_possible if max_possible > 0 else 0.0
        except nx.NetworkXError:
            return 0.0

    def get_fact_nodes(self) -> List[NodeBase]:
        """获取所有事实节点"""
        return [n for n in self.nodes.values() if n.node_type == NodeType.FACT]

    def get_step_nodes(self) -> List[NodeBase]:
        """获取所有过程节点"""
        return [n for n in self.nodes.values() if n.node_type == NodeType.STEP]

    def get_conclusion_nodes(self) -> List[NodeBase]:
        """获取所有目标节点"""
        return [n for n in self.nodes.values() if n.node_type == NodeType.CONCLUSION]

    def get_derive_edges(self) -> List[EdgeBase]:
        """获取所有推导边"""
        return [e for e in self.edges.values() if e.edge_type == EdgeType.DERIVE]

    def get_support_edges(self) -> List[EdgeBase]:
        """获取所有支撑边"""
        return [e for e in self.edges.values() if e.edge_type == EdgeType.SUPPORT]

    def get_conflict_edges(self) -> List[EdgeBase]:
        """获取所有互斥边"""
        return [e for e in self.edges.values() if e.edge_type == EdgeType.CONFLICT]

    def summary(self) -> Dict:
        """推理图统计摘要"""
        return {
            "total_nodes": len(self.nodes),
            "fact_nodes": len(self.get_fact_nodes()),
            "step_nodes": len(self.get_step_nodes()),
            "conclusion_nodes": len(self.get_conclusion_nodes()),
            "total_edges": len(self.edges),
            "derive_edges": len(self.get_derive_edges()),
            "support_edges": len(self.get_support_edges()),
            "conflict_edges": len(self.get_conflict_edges()),
            "connected_components": len(self.check_connectivity()),
            "has_cycles": len(self.detect_cycles()) > 0,
        }

    def __repr__(self):
        s = self.summary()
        return (f"ReasoningGraph(nodes={s['total_nodes']}, "
                f"edges={s['total_edges']}, "
                f"components={s['connected_components']})")