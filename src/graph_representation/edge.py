"""
推理状态图边定义

三类边关系：
- DeriveEdge: 推导关系 —— A推导出B（因果/逻辑推导）
- SupportEdge: 支撑关系 —— A支撑B的成立（证据支撑）
- ConflictEdge: 互斥关系 —— A与B矛盾（逻辑冲突）
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class EdgeType(Enum):
    """边类型枚举"""
    DERIVE = "derive"       # 推导关系：A → B（因果推导）
    SUPPORT = "support"     # 支撑关系：A → B（证据支撑）
    CONFLICT = "conflict"   # 互斥关系：A ↔ B（逻辑矛盾）


@dataclass
class EdgeBase:
    """推理图边基类"""
    src_id: str                             # 源节点ID
    dst_id: str                             # 目标节点ID
    edge_type: EdgeType                     # 边类型
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    weight: float = 1.0                     # 边权重/强度
    description: Optional[str] = None       # 边关系描述
    entailment_score: Optional[float] = None  # NLI蕴含得分（校验后填充）

    def __repr__(self):
        return f"{self.edge_type.value.upper()}({self.src_id}→{self.dst_id})"

    def __hash__(self):
        return hash(self.edge_id)

    def __eq__(self, other):
        return isinstance(other, EdgeBase) and self.edge_id == other.edge_id


@dataclass
class DeriveEdge(EdgeBase):
    """
    推导关系边 —— 表示逻辑推导
    
    含义：src节点可以推导出dst节点
    例："巴黎是法国首都" →推导→ "法国的首都在巴黎"
    """
    edge_type: EdgeType = field(default=EdgeType.DERIVE, init=False)
    derivation_rule: Optional[str] = None   # 推导规则描述


@dataclass
class SupportEdge(EdgeBase):
    """
    支撑关系边 —— 表示证据支撑
    
    含义：src节点为dst节点提供证据支撑
    例："实验数据显示A增加" →支撑→ "A是增长趋势"
    """
    edge_type: EdgeType = field(default=EdgeType.SUPPORT, init=False)
    support_strength: float = 1.0           # 支撑强度


@dataclass
class ConflictEdge(EdgeBase):
    """
    互斥关系边 —— 表示逻辑矛盾
    
    含义：src节点与dst节点逻辑矛盾
    例："A是最大的" ↔矛盾↔ "B比A更大"
    """
    edge_type: EdgeType = field(default=EdgeType.CONFLICT, init=False)
    conflict_reason: Optional[str] = None   # 矛盾原因描述