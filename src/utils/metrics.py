"""评估指标工具"""
from typing import Dict, List, Optional
import re
import string


class Metrics:
    """推理评估指标集合"""

    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """精确匹配率"""
        pred = Metrics._normalize(prediction)
        ref = Metrics._normalize(reference)
        return 1.0 if pred == ref else 0.0

    @staticmethod
    def f1_score(prediction: str, reference: str) -> float:
        """F1分数（基于token匹配）"""
        pred_tokens = set(Metrics._normalize(prediction).split())
        ref_tokens = set(Metrics._normalize(reference).split())

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = pred_tokens & ref_tokens
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def graph_coverage(graph, reasoning_path: List[str]) -> float:
        """
        图覆盖率：推理路径覆盖关键事实节点的比例
        
        Coverage = |covered_facts| / |total_facts|
        """
        from ..graph_representation.node import NodeType
        fact_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.FACT]
        
        if not fact_nodes:
            return 0.0
        
        covered = sum(1 for n in fact_nodes if n.node_id in reasoning_path)
        return covered / len(fact_nodes)

    @staticmethod
    def consistency_score(check_result: Dict) -> float:
        """一致性得分（直接从校验结果提取）"""
        return check_result.get("consistency_score", 0.0)

    @staticmethod
    def _normalize(text: str) -> str:
        """文本标准化"""
        # 去除标点、多余空格、统一小写
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def compute_all(prediction: str, reference: str, 
                    graph=None, check_result: Dict = None,
                    reasoning_path: List[str] = None,
                    token_count: int = 0,
                    latency: float = 0.0) -> Dict:
        """计算所有指标"""
        result = {
            "em": Metrics.exact_match(prediction, reference),
            "f1": Metrics.f1_score(prediction, reference),
            "token_count": token_count,
            "latency": latency,
        }

        if graph and reasoning_path:
            result["graph_coverage"] = Metrics.graph_coverage(graph, reasoning_path)

        if check_result:
            result["consistency_score"] = Metrics.consistency_score(check_result)

        return result