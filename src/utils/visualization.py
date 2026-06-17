"""
推理图可视化工具

支持格式：
- NetworkX + Matplotlib（静态图）
- Graphviz（高质量输出）
- 交互式HTML（可选）
"""

from typing import Optional
from ..graph_representation.reasoning_graph import ReasoningGraph
from ..graph_representation.node import NodeType
from ..graph_representation.edge import EdgeType


# 节点颜色映射
NODE_COLORS = {
    NodeType.FACT: "#4CAF50",        # 绿色：事实节点
    NodeType.STEP: "#2196F3",        # 蓝色：过程节点
    NodeType.CONCLUSION: "#FF9800",  # 橙色：目标节点
}

# 边样式映射
EDGE_STYLES = {
    EdgeType.DERIVE: "solid",        # 实线：推导
    EdgeType.SUPPORT: "dashed",      # 虚线：支撑
    EdgeType.CONFLICT: "dotted",     # 点线：互斥
}

EDGE_COLORS = {
    EdgeType.DERIVE: "#333333",      # 黑色
    EdgeType.SUPPORT: "#666666",     # 灰色
    EdgeType.CONFLICT: "#F44336",    # 红色
}


class GraphVisualizer:
    """推理图可视化器"""

    def __init__(self, output_dir: str = "experiments/results/figures"):
        self.output_dir = output_dir

    def plot_networkx(self, graph: ReasoningGraph, 
                      title: str = "Reasoning Graph",
                      save_path: Optional[str] = None,
                      figsize: tuple = (14, 10)):
        """使用NetworkX + Matplotlib绘制推理图"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

        import networkx as nx

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # 仅绘制非Conflict边的子图
        G = nx.DiGraph()
        for node_id, node in graph.nodes.items():
            G.add_node(node_id, 
                       label=f"[{node.node_type.value[:3]}] {node.content[:30]}",
                       color=NODE_COLORS.get(node.node_type, "#999999"))

        for edge in graph.edges.values():
            G.add_edge(edge.src_id, edge.dst_id,
                       style=EDGE_STYLES.get(edge.edge_type, "solid"),
                       color=EDGE_COLORS.get(edge.edge_type, "#333333"),
                       label=edge.edge_type.value[:3])

        # 布局
        pos = nx.spring_layout(G, k=2, iterations=50)

        # 绘制节点
        node_colors = [NODE_COLORS.get(graph.nodes[n].node_type, "#999") for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                               node_size=1500, alpha=0.9, ax=ax)

        # 绘制标签
        labels = {n: graph.nodes[n].content[:25] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family="sans-serif", ax=ax)

        # 绘制边
        for edge in graph.edges.values():
            if edge.src_id in pos and edge.dst_id in pos:
                color = EDGE_COLORS.get(edge.edge_type, "#333")
                style = EDGE_STYLES.get(edge.edge_type, "solid")
                nx.draw_networkx_edges(G, pos, 
                                       edgelist=[(edge.src_id, edge.dst_id)],
                                       edge_color=color,
                                       style=style,
                                       arrows=True,
                                       ax=ax)

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=NODE_COLORS[NodeType.FACT], label="Fact Node"),
            Patch(facecolor=NODE_COLORS[NodeType.STEP], label="Step Node"),
            Patch(facecolor=NODE_COLORS[NodeType.CONCLUSION], label="Conclusion Node"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return fig

    def export_dot(self, graph: ReasoningGraph, output_path: str = "reasoning_graph.dot"):
        """导出为Graphviz DOT格式（高质量渲染）"""
        lines = ["digraph ReasoningGraph {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box, style=filled, fontname=\"Arial\"];")
        lines.append("  edge [fontname=\"Arial\"];")

        # 节点
        for node_id, node in graph.nodes.items():
            color = NODE_COLORS.get(node.node_type, "#FFFFFF")
            label = node.content.replace('"', '\\"').replace("\n", "\\n")
            lines.append(
                f'  "{node_id}" [label="{node.node_type.value[:3]}: {label[:40]}", '
                f'fillcolor="{color}", fontcolor="white"];'
            )

        # 边
        for edge in graph.edges.values():
            color = EDGE_COLORS.get(edge.edge_type, "#333333")
            style = EDGE_STYLES.get(edge.edge_type, "solid")
            label = edge.edge_type.value
            lines.append(
                f'  "{edge.src_id}" -> "{edge.dst_id}" '
                f'[label="{label}", color="{color}", style={style}];'
            )

        lines.append("}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path