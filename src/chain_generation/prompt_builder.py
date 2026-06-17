"""
图→Prompt转换器 —— 将推理图结构转化为LLM可理解的Prompt

核心思想：
将推理图的节点、边、执行计划等信息编码为结构化Prompt，
使LLM能够"看到"图结构，从而按照图约束进行推理
"""

from typing import Dict, List, Optional
from ..graph_representation.reasoning_graph import ReasoningGraph
from ..graph_representation.node import NodeType
from ..graph_representation.edge import EdgeType


class GraphPromptBuilder:
    """
    推理图 → Prompt 转换器
    
    支持三种编码格式：
    1. 自然语言描述（最通用）
    2. 结构化文本（JSON-like）
    3. Graph-Text混合（推荐）
    """

    def __init__(self, format_type: str = "hybrid"):
        self.format_type = format_type

    def build(self, 
              question: str,
              graph: ReasoningGraph,
              execution_plan: Dict,
              context: str = "") -> str:
        """
        构建图增强推理Prompt
        
        Args:
            question: 原始问题
            graph: 推理图
            execution_plan: 执行计划
            context: 补充上下文
        
        Returns:
            完整的推理Prompt
        """
        if self.format_type == "natural":
            return self._natural_format(question, graph, execution_plan, context)
        elif self.format_type == "structured":
            return self._structured_format(question, graph, execution_plan, context)
        else:
            return self._hybrid_format(question, graph, execution_plan, context)

    def _natural_format(self, question, graph, plan, context) -> str:
        """自然语言描述格式"""
        facts = [n.content for n in graph.get_fact_nodes()]
        steps = [n.content for n in graph.get_step_nodes()]
        conclusions = [n.content for n in graph.get_conclusion_nodes()]

        facts_text = "\n".join(f"- {f}" for f in facts)
        steps_text = "\n".join(f"- {s}" for s in steps)
        goals_text = "\n".join(f"- {g}" for g in conclusions)

        prompt = f"""请解答以下问题。

问题：{question}

{f'参考信息：{context}' if context else ''}

已知事实：
{facts_text}

需要执行的推理步骤：
{steps_text}

需要得出的结论：
{goals_text}

请按照逻辑顺序逐步推理，每步推理必须基于已知事实或前一步的结论。"""

        return prompt

    def _structured_format(self, question, graph, plan, context) -> str:
        """结构化文本格式"""
        # 构建邻接表描述
        adj_text = ""
        for node_id in plan["execution_order"]:
            node = graph.get_node(node_id)
            if node is None:
                continue
            neighbors = graph.get_neighbors(node_id)
            type_str = node.node_type.value
            neighbor_nodes = [graph.get_neighbor_content(nid) for nid in neighbors]
            adj_text += f"  [{type_str}] {node.content} → {', '.join(neighbor_nodes)}\n"

        prompt = f"""请基于以下推理图结构解答问题。

问题：{question}
{f'上下文：{context}' if context else ''}

推理图结构（节点 → 邻居）：
{adj_text}

请严格按照推理图的拓扑顺序进行推导。"""

        return prompt

    def _hybrid_format(self, question, graph, plan, context) -> str:
        """
        混合格式（推荐）—— 自然语言+结构化路径
        
        保留自然语言的可读性，同时嵌入结构化路径约束
        """
        from .path_planner import PathPlanner
        planner = PathPlanner()
        plan_text = planner.format_execution_plan(graph, plan)

        facts = [n.content for n in graph.get_fact_nodes()]
        facts_text = "\n".join(f"- {f}" for f in facts)

        prompt = f"""请解答以下问题。

问题：{question}
{f'参考信息：{context}' if context else ''}

【已知事实】
{facts_text}

【推理路径规划】
{plan_text}

请严格按照上述推理路径进行逐步推导。每个步骤必须：
1. 明确标注当前处于哪一步
2. 说明本步的推理依据（基于哪些事实或前驱步骤）
3. 给出本步的中间结论
4. 最终汇总所有步骤得出答案

开始推理："""

        return prompt