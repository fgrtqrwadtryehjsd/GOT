"""
推理路径规划器 —— 基于拓扑排序的路径规划

核心思想：
1. 对推理图进行拓扑排序，得到逻辑执行顺序
2. 识别从事实节点到目标节点的关键路径
3. 生成"执行计划"—— 推理应该遵循的步骤序列
"""

from typing import Dict, List, Optional, Tuple
from ..graph_representation.reasoning_graph import ReasoningGraph
from ..graph_representation.node import NodeType


class PathPlanner:
    """
    基于图拓扑结构的推理路径规划器
    
    策略：
    1. 拓扑排序获取全局执行顺序
    2. DFS/BFS搜索从事实到结论的路径
    3. 多路径融合：识别汇聚点（多因一果）
    4. 生成结构化执行计划
    """

    def __init__(self, strategy: str = "topological_dfs"):
        """
        Args:
            strategy: 路径规划策略
                - "topological_dfs": 拓扑排序+DFS（默认）
                - "topological_bfs": 拓扑排序+BFS
                - "shortest_path": 最短路径优先
        """
        self.strategy = strategy

    def plan(self, graph: ReasoningGraph) -> Dict:
        """
        为推理图生成执行计划
        
        Returns:
            {
                "execution_order": [node_ids],     # 拓扑执行顺序
                "key_paths": [[node_ids], ...],     # 从事实到结论的关键路径
                "branch_points": [node_ids],        # 分支点（多出度节点）
                "merge_points": [node_ids],         # 汇聚点（多入度节点）
            }
        """
        # Step 1: 拓扑排序
        execution_order = graph.topological_sort()

        # Step 2: 搜索关键路径（事实→结论）
        fact_nodes = [n.node_id for n in graph.get_fact_nodes()]
        conclusion_nodes = [n.node_id for n in graph.get_conclusion_nodes()]

        key_paths = []
        for fact_id in fact_nodes:
            for conc_id in conclusion_nodes:
                paths = graph.find_all_paths(fact_id, conc_id, cutoff=10)
                key_paths.extend(paths)

        # 去重并按路径长度排序
        unique_paths = []
        seen = set()
        for path in key_paths:
            path_key = tuple(path)
            if path_key not in seen:
                seen.add(path_key)
                unique_paths.append(path)
        
        unique_paths.sort(key=len)

        # Step 3: 识别分支点和汇聚点
        branch_points = []
        merge_points = []
        for node_id in execution_order:
            out_degree = len(graph.outgoing.get(node_id, []))
            in_degree = len(graph.incoming.get(node_id, []))
            
            if out_degree > 1:
                branch_points.append(node_id)
            if in_degree > 1:
                merge_points.append(node_id)

        return {
            "execution_order": execution_order,
            "key_paths": unique_paths,
            "branch_points": branch_points,
            "merge_points": merge_points,
        }

    def get_optimal_path(self, graph: ReasoningGraph) -> List[str]:
        """
        获取最优推理路径（单条）
        
        策略：选择覆盖最多事实节点的最短路径
        """
        plan = self.plan(graph)
        
        if not plan["key_paths"]:
            # 退化为拓扑排序顺序
            return plan["execution_order"]
        
        # 选择最长路径（覆盖最多节点）
        best_path = max(plan["key_paths"], key=len)
        return best_path

    def format_execution_plan(self, graph: ReasoningGraph, plan: Dict) -> str:
        """
        将执行计划格式化为文本（用于Prompt构建）
        
        输出格式：
        步骤1: [事实] xxx
        步骤2: [过程] xxx → 推导自 步骤1
        步骤3: [目标] xxx ← 汇聚 步骤2
        """
        lines = []
        node_order = plan["execution_order"]
        
        for i, node_id in enumerate(node_order):
            node = graph.get_node(node_id)
            if node is None:
                continue
            
            type_label = {
                NodeType.FACT: "事实",
                NodeType.STEP: "过程",
                NodeType.CONCLUSION: "目标"
            }.get(node.node_type, "未知")
            
            # 查找前驱
            predecessors = graph.get_predecessors(node_id)
            pred_info = ""
            if predecessors:
                pred_indices = []
                for pred_id in predecessors:
                    if pred_id in node_order:
                        pred_indices.append(str(node_order.index(pred_id) + 1))
                if pred_indices:
                    action = "推导自" if node.node_type == NodeType.STEP else "支撑自"
                    pred_info = f" ← {action} 步骤{','.join(pred_indices)}"
            
            lines.append(f"步骤{i+1}: [{type_label}] {node.content}{pred_info}")
        
        return "\n".join(lines)