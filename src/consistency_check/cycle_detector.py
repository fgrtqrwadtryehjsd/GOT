"""
环路检测 —— 识别循环论证

逻辑原理：
如果推理图中存在有向环路，说明存在循环论证：
A→B→C→A，即A的结论依赖A本身，这是逻辑谬误。

实现：使用DFS+颜色标记算法（白-灰-黑标记法）
等价于NetworkX的simple_cycles，但提供更详细的诊断信息
"""

from typing import Dict, List, Tuple
from ..graph_representation.reasoning_graph import ReasoningGraph


class CycleDetector:
    """
    推理图环路检测器
    
    检测内容：
    1. 有向环路列表
    2. 涉及环路的节点
    3. 环路类型分类
    """

    def detect(self, graph: ReasoningGraph) -> Dict:
        """
        执行环路检测
        
        Returns:
            {
                "has_cycle": bool,
                "cycles": [[node_ids], ...],    # 环路列表
                "involved_nodes": set,           # 涉及环路的节点
                "cycle_score": float,            # 无环得分 [0, 1]
            }
        """
        cycles = graph.detect_cycles()
        has_cycle = len(cycles) > 0

        involved_nodes = set()
        for cycle in cycles:
            involved_nodes.update(cycle)

        # 计算无环得分
        total_nodes = len(graph.nodes)
        if total_nodes == 0:
            cycle_score = 1.0
        elif has_cycle:
            # 环路涉及的节点比例越低，得分越高
            involved_ratio = len(involved_nodes) / total_nodes
            cycle_score = 1.0 - involved_ratio
        else:
            cycle_score = 1.0

        return {
            "has_cycle": has_cycle,
            "cycles": cycles,
            "involved_nodes": list(involved_nodes),
            "cycle_score": max(0.0, min(1.0, cycle_score)),
        }

    def diagnose(self, graph: ReasoningGraph) -> List[str]:
        """
        对检测到的环路进行诊断，生成可读的解释
        
        Returns:
            每个环路的诊断信息列表
        """
        result = self.detect(graph)
        diagnoses = []

        for i, cycle in enumerate(result["cycles"]):
            nodes_text = []
            for node_id in cycle:
                node = graph.get_node(node_id)
                if node:
                    nodes_text.append(f"[{node.node_type.value}]{node.content}")
            
            chain = " → ".join(nodes_text)
            # 闭合环路
            if nodes_text:
                chain += f" → {nodes_text[0]}"
            
            diagnoses.append(
                f"循环论证 #{i+1}: {chain}\n"
                f"  该环路包含{len(cycle)}个节点，存在循环依赖。"
            )

        return diagnoses