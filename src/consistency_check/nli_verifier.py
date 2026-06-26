"""
NLI语义蕴含检测 —— 验证推理链路中的逻辑蕴含关系

核心思想：
推理图中每条边(A→B)代表一个逻辑关系，
NLI验证A是否真的蕴含B：
- Entailment（蕴含）：A逻辑上能推出B ✓
- Neutral（中立）：A和B无直接逻辑关系 ⚠️
- Contradiction（矛盾）：A和B逻辑矛盾 ✗

支持两种模式：
1. HuggingFace NLI 模型（DeBERTa-v3-large-mnli）—— 离线精确
2. LLM-based NLI —— 用 LLM 自身做蕴含判断，无需额外模型下载
"""

from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NLIVerifier:
    """
    基于NLI的语义蕴含检测器
    
    对推理图中每条边进行语义校验，
    判断源节点和目标节点之间是否存在真实的逻辑蕴含关系。
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large", 
                 device: str = "cpu",
                 llm=None):
        """
        Args:
            model_name: NLI模型名称（HuggingFace）
            device: 运行设备
            llm: LLM 模型实例（用于 LLM-based NLI fallback）
        """
        self.model_name = model_name
        self.device = device
        self.llm = llm
        self._model = None
        self._tokenizer = None
        self._load_attempted = False  # 避免重复加载

    def _load_model(self):
        """延迟加载NLI模型（只尝试一次）"""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            self._model.eval()
        except Exception as e:
            logger.debug(f"NLI模型加载失败，使用LLM fallback: {e}")
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
        # 优先使用 HuggingFace NLI 模型
        self._load_model()

        if self._model is not None:
            return self._verify_with_hf_model(premise, hypothesis)

        # Fallback: 使用 LLM-based NLI
        if self.llm is not None:
            return self._verify_with_llm(premise, hypothesis)

        # 无可用模型时返回中性结果
        return {
            "label": "neutral",
            "scores": {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33},
            "is_valid": False,
        }

    def _verify_with_hf_model(self, premise: str, hypothesis: str) -> Dict:
        """使用 HuggingFace NLI 模型验证"""
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

    def _verify_with_llm(self, premise: str, hypothesis: str) -> Dict:
        """
        使用 LLM 做蕴含判断（无需下载额外模型）
        
        通过 Prompt 让 LLM 判断 premise 是否蕴含 hypothesis
        """
        prompt = f"""Determine the logical relationship between the following two statements.

Statement A (Premise): {premise[:500]}
Statement B (Hypothesis): {hypothesis[:500]}

Does Statement A logically entail Statement B?
- "entailment": A logically implies B
- "neutral": A and B are not directly related
- "contradiction": A and B contradict each other

Answer with ONLY one word: entailment, neutral, or contradiction."""

        try:
            response = self.llm.generate(prompt, max_tokens=20, temperature=0.0)
            label = response.strip().lower().strip("*#., \n")
            
            # 提取关键词
            if "entail" in label:
                label = "entailment"
            elif "contradict" in label:
                label = "contradiction"
            else:
                label = "neutral"

            is_valid = label == "entailment"
            
            # 构造分数
            if label == "entailment":
                scores = {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05}
            elif label == "contradiction":
                scores = {"entailment": 0.05, "neutral": 0.15, "contradiction": 0.8}
            else:
                scores = {"entailment": 0.2, "neutral": 0.6, "contradiction": 0.2}

            return {
                "label": label,
                "scores": scores,
                "is_valid": is_valid,
            }
        except Exception:
            return {
                "label": "neutral",
                "scores": {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33},
                "is_valid": False,
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
