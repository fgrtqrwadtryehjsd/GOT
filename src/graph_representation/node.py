"""
推理状态图节点定义

三类节点：
- FactNode: 事实节点 —— 已知信息、证据、前提条件
- StepNode: 过程节点 —— 中间推理步骤、逻辑运算
- ConclusionNode: 目标节点 —— 最终结论、待证命题
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum


class NodeType(Enum):
    """节点类型枚举"""
    FACT = "fact"            # 事实节点
    STEP = "step"            # 过程节点
    CONCLUSION = "conclusion"  # 目标节点


@dataclass
class NodeBase:
    """推理图节点基类"""
    content: str                            # 节点文本内容
    node_type: NodeType                     # 节点类型
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0                  # 节点置信度 [0, 1]
    is_verified: bool = False               # 是否已被校验

    def __repr__(self):
        return f"{self.node_type.value.upper()}({self.node_id}): {self.content[:50]}"

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return isinstance(other, NodeBase) and self.node_id == other.node_id


@dataclass
class FactNode(NodeBase):
    """
    事实节点 —— 代表已知信息、证据或前提条件
    
    特征：
    - 来自问题文本或外部知识
    - 置信度通常较高
    - 是推理的起点
    """
    node_type: NodeType = field(default=NodeType.FACT, init=False)
    source: Optional[str] = None            # 事实来源（问题文本/知识库）
    evidence_text: Optional[str] = None      # 原始证据文本


@dataclass
class StepNode(NodeBase):
    """
    过程节点 —— 代表中间推理步骤
    
    特征：
    - 由前驱节点推导而来
    - 包含推理操作描述
    - 是推理链的中间环节
    """
    node_type: NodeType = field(default=NodeType.STEP, init=False)
    operation: Optional[str] = None          # 推理操作类型（比较/计算/检索等）
    depends_on: list = field(default_factory=list)  # 依赖的节点ID列表
    derivation: Optional[str] = None         # 推导过程描述


@dataclass
class ConclusionNode(NodeBase):
    """
    目标节点 —— 代表推理的最终结论
    
    特征：
    - 是推理图的终点
    - 需要被充分支撑
    - 对应问题的答案
    """
    node_type: NodeType = field(default=NodeType.CONCLUSION, init=False)
    answer: Optional[str] = None             # 最终答案
    support_evidence: list = field(default_factory=list)  # 支撑该结论的证据节点ID