"""
NLI语义蕴含检测 —— 验证推理链路中的逻辑蕴含关系

核心思想：
推理图中每条边(A→B)代表一个逻辑关系，
NLI模型可以验证A是否真的蕴含B：
- Entailment（蕴含）：A逻辑上能推出B ✓
- Neutral（中立）：A和B无直接逻辑关系 ⚠️
- Contradiction（矛盾）：A和B逻辑矛盾 ✗

NLI模型推荐：DeBERTa-v3-large-mnli
"""

from typing import Dict, List, Optional, Tuple


class NLIVerifier:
    """
    基于NLI的语义蕴含检测器
    
    对推理图中每条边进行语义校验，
    判断源节点和目标节点之间是否存在真实的逻辑蕴含关系。
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large", 
                 device: str = "cpu"):
        """
        Args:
            model_name: NLI模型名称（HuggingFace）
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """延迟加载NLI模型"""
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                ).to(self.device)
                self._model.eval()
            except Exception as e:
                print(f"NLI模型加载失败: {e}")
                self._model = None
                self._tokenizer = None

    def verify_edge(self, premise: str, hypothesis: str) -> Dict:
        """
        验证一条边的逻辑蕴含关系
        
        Args:
            premise: 前提（源节点内容）
            hypothesis: 假设（目标节点内容）
        
        Returns:
            {
                "label": "entailment/neutral/contradiction",
                "scores": {"entailment": float, "neutral": float, "contradiction": float},
                "is_valid": bool,  # 蕴含关系是否成立
            }
        """
        self._load_model()

        if self._model is None:
            # 模型不可用时返回中性结果
            return {
                "label": "neutral",
                "scores": {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33},
                "is_valid": False,
            }

        import torch
        with torch.no_grad():
            inputs = self._tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

        # 标签顺序：contradiction(0), neutral(1), entailment(2)
        scores = {
            "contradiction": probs[0],
            "neutral": probs[1],
            "entailment": probs[2],
        }

        label = max(scores, key=scores.get)
        is_valid = label == "entailment" and scores["entailment"] > 0.5

        return {
            "label": label,
            "scores": scores,
            "is_valid": is_valid,
        }

    def verify_graph(self, graph) -> Dict:
        """
        验证推理图中所有边的逻辑蕴含关系
        
        Returns:
            {
                "edge_verifications": [{edge_id, label, scores, is_valid}, ...],
                "valid_ratio": float,     # 有效边比例
                "contradiction_count": int,  # 矛盾边数量
                "nli_score": float,        # NLI语义得分
            }
        """
        verifications = []
        valid_count = 0
        contradiction_count = 0

        for edge in graph.edges.values():
            src_node = graph.get_node(edge.src_id)
            dst_node = graph.get_node(edge.dst_id)
            
            if src_node is None or dst_node is None:
                continue

            result = self.verify_edge(src_node.content, dst_node.content)
            result["edge_id"] = edge.edge_id
            result["src_content"] = src_node.content[:50]
            result["dst_content"] = dst_node.content[:50]
            verifications.append(result)

            if result["is_valid"]:
                valid_count += 1
            if result["label"] == "contradiction":
                contradiction_count += 1

        total = len(verifications)
        valid_ratio = valid_count / total if total > 0 else 0.0
        nli_score = valid_ratio - (contradiction_count / max(total, 1)) * 0.5

        return {
            "edge_verifications": verifications,
            "valid_ratio": valid_ratio,
            "contradiction_count": contradiction_count,
            "nli_score": max(0.0, min(1.0, nli_score)),
        }