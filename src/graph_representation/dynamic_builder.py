"""
动态构图模块 —— 从自然语言问题构建推理图

流程：
1. 输入问题文本
2. 调用LLM抽取实体、关系与子任务
3. 将抽取结果映射为图的节点和边
4. 动态更新推理图（增量式添加）
"""

from typing import Dict, List, Optional, Tuple
from .reasoning_graph import ReasoningGraph
from .extractor import EntityRelationExtractor


class DynamicGraphBuilder:
    """
    动态推理图构建器
    
    核心能力：
    - 从自然语言问题中提取实体和关系
    - 将提取结果转换为推理图的节点和边
    - 支持图的增量式动态生长
    """

    def __init__(self, extractor: EntityRelationExtractor):
        self.extractor = extractor

    def build(self, question: str, context: str = "") -> ReasoningGraph:
        """
        完整构图流程
        
        Args:
            question: 待推理的问题
            context: 上下文信息（可选）
        
        Returns:
            构建好的推理图
        """
        graph = ReasoningGraph(question=question)

        # Step 1: 抽取实体和关系
        extraction = self.extractor.extract(question, context)

        # Step 2: 添加事实节点
        for fact in extraction.get("facts", []):
            node_id = graph.add_fact(
                content=fact["content"],
                source=fact.get("source", "question"),
                evidence=fact.get("evidence", "")
            )

        # Step 3: 添加子任务/过程节点
        for step in extraction.get("steps", []):
            node_id = graph.add_step(
                content=step["content"],
                operation=step.get("operation", "")
            )

        # Step 4: 添加目标节点
        for conclusion in extraction.get("conclusions", []):
            node_id = graph.add_conclusion(
                content=conclusion["content"],
                answer=conclusion.get("answer", "")
            )

        # Step 5: 添加边关系
        node_map = {n.content: n.node_id for n in graph.nodes.values()}
        
        for relation in extraction.get("relations", []):
            src_key = relation["source"]
            dst_key = relation["target"]
            rel_type = relation["type"]  # derive/support/conflict
            
            src_id = node_map.get(src_key)
            dst_id = node_map.get(dst_key)
            
            if src_id and dst_id:
                if rel_type == "derive":
                    graph.add_derive(src_id, dst_id, desc=relation.get("description"))
                elif rel_type == "support":
                    graph.add_support(src_id, dst_id, 
                                     strength=relation.get("strength", 1.0))
                elif rel_type == "conflict":
                    graph.add_conflict(src_id, dst_id, 
                                       reason=relation.get("reason"))

        return graph

    def update(self, graph: ReasoningGraph, new_info: str) -> ReasoningGraph:
        """
        增量式更新推理图
        
        当推理过程中获得新信息时，动态添加到图中
        （Dynamic Growth机制）
        """
        extraction = self.extractor.extract(new_info, graph.question)

        for fact in extraction.get("facts", []):
            # 检查是否已存在相同节点
            existing = [n for n in graph.nodes.values() 
                       if n.content == fact["content"]]
            if not existing:
                graph.add_fact(content=fact["content"])

        for step in extraction.get("steps", []):
            existing = [n for n in graph.nodes.values() 
                       if n.content == step["content"]]
            if not existing:
                graph.add_step(content=step["content"])

        # 重新建立边关系
        node_map = {n.content: n.node_id for n in graph.nodes.values()}
        for relation in extraction.get("relations", []):
            src_id = node_map.get(relation["source"])
            dst_id = node_map.get(relation["target"])
            if src_id and dst_id:
                rel_type = relation["type"]
                if rel_type == "derive":
                    graph.add_derive(src_id, dst_id)
                elif rel_type == "support":
                    graph.add_support(src_id, dst_id)
                elif rel_type == "conflict":
                    graph.add_conflict(src_id, dst_id)

        return graph