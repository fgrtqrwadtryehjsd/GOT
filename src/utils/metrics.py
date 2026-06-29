"""评估指标工具"""
from typing import Dict, List, Optional
import re
import string


class Metrics:
    """推理评估指标集合"""

    @staticmethod
    def exact_match(prediction: str, reference: str, dataset: str = None) -> float:
        """精确匹配率

        Args:
            prediction: 预测答案
            reference: 参考答案
            dataset: 数据集名称（gsm8k 走数值比较，其余走 SQuAD 式归一化 EM）
        """
        # GSM8K：抽取数值做数值相等比较
        if dataset and dataset.lower() == "gsm8k":
            return Metrics._numeric_em(prediction, reference)

        # HotpotQA / CLUTRR：SQuAD 式归一化 EM（不再使用双向子串匹配）
        pred = Metrics._normalize(prediction)
        ref = Metrics._normalize(reference)
        if pred == ref:
            return 1.0
        return 0.0

    @staticmethod
    def _numeric_em(prediction: str, reference: str) -> float:
        """GSM8K 数值 EM：抽取最后一个数字做数值相等比较"""
        pred_num = Metrics._extract_number(prediction)
        ref_num = Metrics._extract_number(reference)
        if pred_num is None or ref_num is None:
            # 回退到归一化字符串比较
            return 1.0 if Metrics._normalize(prediction) == Metrics._normalize(reference) else 0.0
        return 1.0 if abs(pred_num - ref_num) < 1e-6 else 0.0

    @staticmethod
    def _extract_number(text: str):
        """从文本中抽取最后一个数字（支持负数和小数）"""
        if not text:
            return None
        numbers = re.findall(r'[-+]?\d*\.?\d+', text.replace(',', ''))
        if not numbers:
            return None
        try:
            num = numbers[-1]
            return float(num) if '.' in num else int(num)
        except (ValueError, IndexError):
            return None

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
                    latency: float = 0.0,
                    dataset: str = None) -> Dict:
        """计算所有指标"""
        result = {
            "em": Metrics.exact_match(prediction, reference, dataset=dataset),
            "f1": Metrics.f1_score(prediction, reference),
            "token_count": token_count,
            "latency": latency,
        }

        if graph and reasoning_path:
            result["graph_coverage"] = Metrics.graph_coverage(graph, reasoning_path)

        if check_result:
            result["consistency_score"] = Metrics.consistency_score(check_result)

        return result