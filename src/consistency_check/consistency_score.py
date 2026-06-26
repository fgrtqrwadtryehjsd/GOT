"""
Consistency Score —— 推理一致性综合评估

三层校验机制：
1. 结构层（图论算法）：连通性 + 环路检测 + 覆盖度
2. 语义层（NLI模型）：逻辑蕴含关系检测
3. 综合层：加权融合结构得分与语义得分

数学定义：
Consistency_Score = α · S_struct + β · S_semantic

其中：
S_struct = w1 · connectivity_score + w2 · cycle_score + w3 · coverage_score
S_semantic = NLI验证的有效边比例

默认权重：α=0.6, β=0.4（结构为主，语义为辅）
"""

from typing import Dict, Optional
from ..graph_representation.reasoning_graph import ReasoningGraph
from .connectivity import ConnectivityChecker
from .cycle_detector import CycleDetector
from .coverage import CoverageCalculator
from .nli_verifier import NLIVerifier


class ConsistencyChecker:
    """
    推理一致性综合校验器
    
    对推理图和生成结果进行三层校验，
    输出Consistency_Score量化评估。
    """

    def __init__(self,
                 enable_nli: bool = True,
                 nli_model: Optional[NLIVerifier] = None,
                 llm=None,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 w_connectivity: float = 0.35,
                 w_cycle: float = 0.30,
                 w_coverage: float = 0.35):
        """
        Args:
            enable_nli: 是否启用NLI语义校验
            nli_model: NLI检测器实例
            llm: LLM 模型实例（用于 LLM-based NLI，当 nli_model 为 None 时自动创建）
            alpha: 结构层权重
            beta: 语义层权重
            w_connectivity: 连通性在结构层中的权重
            w_cycle: 无环性在结构层中的权重
            w_coverage: 覆盖度在结构层中的权重
        """
        self.connectivity_checker = ConnectivityChecker()
        self.cycle_detector = CycleDetector()
        self.coverage_calculator = CoverageCalculator()
        
        # NLI 校验器：优先使用传入的，否则用 LLM 创建
        if nli_model is not None:
            self.nli_verifier = nli_model
        elif llm is not None:
            self.nli_verifier = NLIVerifier(llm=llm)
        else:
            self.nli_verifier = NLIVerifier()
        
        self.enable_nli = enable_nli
        self.llm = llm
        self.alpha = alpha
        self.beta = beta
        self.w_connectivity = w_connectivity
        self.w_cycle = w_cycle
        self.w_coverage = w_coverage

    def check(self, graph: ReasoningGraph, reasoning_text: str = "") -> Dict:
        """
        执行综合一致性校验

        Returns:
            {
                "consistency_score": float,        # 综合得分 [0, 1]
                "structural_score": float,          # 结构层得分
                "semantic_score": float,            # 语义层得分
                "connectivity": {...},              # 连通性详情
                "cycle": {...},                     # 环路检测详情
                "coverage": {...},                 # 覆盖度详情
                "nli": {...},                      # NLI详情（如启用）
                "issues": [str],                   # 发现的问题列表
                "passed": bool,                    # 是否通过校验
            }
        """
        # === 第一层：结构校验 ===
        connectivity = self.connectivity_checker.check(graph)
        cycle = self.cycle_detector.detect(graph)
        coverage = self.coverage_calculator.compute(graph)

        structural_score = (
            self.w_connectivity * connectivity["connectivity_score"] +
            self.w_cycle * cycle["cycle_score"] +
            self.w_coverage * coverage["coverage_score"]
        )

        # 推理链长度惩罚：子问题越多，结构得分越低（防止过分解）
        step_nodes = graph.get_step_nodes()
        num_steps = len(step_nodes)
        if num_steps > 3:
            length_penalty = max(0.7, 1.0 - (num_steps - 3) * 0.1)
            structural_score *= length_penalty

        # === 第二层：语义校验 ===
        semantic_score = 0.5  # 默认中性得分
        nli_result = None
        if self.enable_nli:
            nli_result = self.nli_verifier.verify_graph(graph)
            semantic_score = nli_result["nli_score"]
        else:
            # NLI关闭时，用子答案质量作为语义得分的替代
            if step_nodes:
                empty_count = 0
                low_conf_count = 0
                for node in step_nodes:
                    ans = node.metadata.get("answer", "")
                    if not ans or len(ans.strip()) < 2:
                        empty_count += 1
                    elif any(kw in ans.lower() for kw in ["i don't know", "unclear", "unknown", "无法确定", "不确定", "insufficient"]):
                        low_conf_count += 1
                total = len(step_nodes)
                semantic_score = 1.0 - (empty_count * 0.5 + low_conf_count * 0.3) / max(total, 1)
                semantic_score = max(0.1, min(1.0, semantic_score))

        # === 第三层：综合评估 ===
        consistency_score = self.alpha * structural_score + self.beta * semantic_score

        # 收集问题
        issues = []
        if not connectivity["is_connected"]:
            issues.append(
                f"推理图存在{connectivity['num_components']}个连通分量，"
                f"存在逻辑断链"
            )
        if connectivity["isolated_nodes"]:
            issues.append(
                f"存在{len(connectivity['isolated_nodes'])}个孤立节点"
            )
        if connectivity["unreachable_conclusions"]:
            issues.append(
                f"存在{len(connectivity['unreachable_conclusions'])}个不可达结论"
            )
        if cycle["has_cycle"]:
            issues.append(
                f"存在{len(cycle['cycles'])}个循环论证环路"
            )
        if coverage["uncovered_conclusions"]:
            issues.append(
                f"存在{len(coverage['uncovered_conclusions'])}个证据覆盖不足的结论"
            )
        if nli_result and nli_result["contradiction_count"] > 0:
            issues.append(
                f"NLI检测到{nli_result['contradiction_count']}条矛盾边"
            )

        return {
            "consistency_score": round(max(0.0, min(1.0, consistency_score)), 4),
            "structural_score": round(structural_score, 4),
            "semantic_score": round(semantic_score, 4),
            "connectivity": connectivity,
            "cycle": cycle,
            "coverage": coverage,
            "nli": nli_result,
            "issues": issues,
            "passed": consistency_score >= 0.6,
        }