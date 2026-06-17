"""
推理状态图表示模块

核心数据结构：推理图 G=(V,E)
- 节点：FactNode（事实）、StepNode（过程）、ConclusionNode（目标）
- 边：DeriveEdge（推导）、SupportEdge（支撑）、ConflictEdge（互斥）
"""

from .reasoning_graph import ReasoningGraph
from .node import FactNode, StepNode, ConclusionNode, NodeBase
from .edge import DeriveEdge, SupportEdge, ConflictEdge, EdgeBase
from .dynamic_builder import DynamicGraphBuilder
from .extractor import EntityRelationExtractor