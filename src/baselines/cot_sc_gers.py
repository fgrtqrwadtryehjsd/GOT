"""
CoT-SC + GERS 加权重排 (CoT-SC with GERS Weighted Rerank)
==========================================================

动机：
上一版 CoT-SC+GERS 只在"票数并列"时才用 GERS 校验分重排，且 gap 小就退回多数票，
导致 GERS 分绝大多数情况不影响结果，与标准 CoT-SC 几乎无差。

本版升级为 **GERS 加权投票**（对应《GERS 改进方案（文献支撑版）》第五节思路）：
1. N 次 Standard CoT 采样 → 投票前先归一化（yes/no + 小写 + 去标点），修复投票碎片化
2. 对每个 unique 归一化答案取一条代表 reasoning，用轻量 GERS 校验打分
   （实体覆盖 / 推理连贯 / 答案落地，[0,1]）
3. 加权得分：combined_score(a) = count(a) + λ · gers_score(a)
   - count 主导（多数票信号），λ 控制 GERS 质量分能多大程度覆盖票数
   - λ=1.0（默认）：GERS 仅在票数并列时翻盘；λ 越大，高质量少数答案越可能胜出
4. 选 combined_score 最高的答案；同时记录多数票答案作对照，输出诊断字段

对标 CoT-SC：CoT-SC 是"答案多数投票"（每票等权），本方法是"图结构质量加权投票"
（结构/语义质量高的答案获得更高权重）。

成本：N 次 CoT 采样 + U 次轻量校验（U = unique 答案数 ≤ N）≈ 1.5x 单次 GERS，
远低于 GERS-SC（K 次完整图分解）。

消融：enable_gers_rerank=False → 退化为（归一化后的）标准 CoT-SC。
"""
from typing import Dict, List
from collections import Counter, defaultdict
from .standard_cot import StandardCoT
from ..utils.answer_normalizer import normalize_for_vote


# ─── 轻量 GERS 校验 Prompt ─────────────────────────────────────────────

GERS_LITE_SCORING_PROMPT = """You are evaluating whether a candidate answer is factually consistent with the given context and the question.

Question:
{question}

Context:
{context}

Candidate Reasoning:
{reasoning}

Candidate Answer: {answer}

Evaluate on 3 criteria (give a score 0.0-1.0 each, then overall score):

1. Entity Coverage (0.0-1.0): Does the answer's key entities appear in the context?
2. Reasoning Coherence (0.0-1.0): Is the reasoning text internally consistent (no logical jumps, no contradictions)?
3. Answer Grounding (0.0-1.0): Is the final answer directly supported by the context AND the reasoning (not a hallucination)?

Output ONLY a JSON object (no other text):
{{
  "entity_coverage": 0.0,
  "reasoning_coherence": 0.0,
  "answer_grounding": 0.0,
  "overall": 0.0,
  "reason": "one-sentence justification"
}}

The "overall" score = average of the three sub-scores. Be strict: a hallucinated answer that happens to match keywords should score < 0.5 on answer_grounding."""


class CoTSCWithGERS:
    """
    CoT-SC + GERS 加权重排

    流程：
    1. N 次 Standard CoT 采样
    2. 答案归一化 + Counter 投票
    3. 对每个 unique 答案做轻量 GERS 校验打分
    4. combined_score = count + λ·gers_score → 选最高
    5. 返回：(最终答案, 投票分布, GERS 分, 诊断字段, 原始 reasoning)
    """

    def __init__(self, model=None, num_samples: int = 3,
                 gers_lambda: float = 1.0,
                 enable_gers_rerank: bool = True,
                 dataset: str = None):
        """
        Args:
            model: LLM 模型实例
            num_samples: CoT 采样次数（默认 3，与标准 CoT-SC 一致）
            gers_lambda: GERS 质量分权重 λ。combined = count + λ·gers_score。
                         λ=1.0：GERS 仅在票数并列时翻盘；调大可让高质量少数答案胜出。
            enable_gers_rerank: 是否启用 GERS 重排（False 时退化为归一化后的标准 CoT-SC）
            dataset: 数据集名称（透传给 StandardCoT 的答案提取）
        """
        self.model = model
        self.num_samples = num_samples
        self.gers_lambda = gers_lambda
        self.enable_gers_rerank = enable_gers_rerank
        self.dataset = dataset
        self.cot = StandardCoT(model=model, dataset=dataset)

    def reason(self, question: str, context: str = "") -> Dict:
        if self.model is None:
            return {
                "answer": "",
                "reasoning_text": "[需要配置模型]",
                "method": f"CoT-SC+GERS (N={self.num_samples})",
            }

        # ── 1. N 次 CoT 采样 ─────────────────────────────────────────
        candidates: List[Dict] = []
        for i in range(self.num_samples):
            result = self.cot.reason(question, context)
            candidates.append({
                "answer": result.get("answer", ""),
                "reasoning_text": result.get("reasoning_text", ""),
                "sample_idx": i,
            })

        # ── 2. 归一化 + 投票 ─────────────────────────────────────────
        norm_answers = [normalize_for_vote(c["answer"]) for c in candidates]
        counter = Counter(norm_answers)
        vote_distribution = dict(counter)

        # 多数票答案（对照信号）
        majority_answer = counter.most_common(1)[0][0] if counter else ""
        majority_count = counter.get(majority_answer, 0)
        confidence = majority_count / self.num_samples

        # ── 3. 未启用重排 → 退化为归一化 CoT-SC ──────────────────────
        if not self.enable_gers_rerank:
            rep = next((c for c in candidates
                        if normalize_for_vote(c["answer"]) == majority_answer), candidates[0])
            return {
                "answer": majority_answer,
                "reasoning_text": rep["reasoning_text"],
                "all_answers": [c["answer"] for c in candidates],
                "vote_distribution": vote_distribution,
                "confidence": confidence,
                "gers_rerank_used": False,
                "method": f"CoT-SC+GERS (N={self.num_samples}, rerank=off)",
            }

        # ── 4. 对每个 unique 答案做轻量 GERS 校验打分 ─────────────────
        # 去重：每个 unique 归一化答案取第一条代表 reasoning，省 LLM 调用
        unique_answers = list(counter.keys())
        rep_reasoning: Dict[str, str] = {}
        for c in candidates:
            na = normalize_for_vote(c["answer"])
            if na not in rep_reasoning:
                rep_reasoning[na] = c["reasoning_text"]

        gers_scores: Dict[str, float] = {}
        for ua in unique_answers:
            gers_scores[ua] = self._score_by_gers_lite(
                question, context, ua, rep_reasoning.get(ua, "")
            )

        # ── 5. 加权得分：combined = count + λ·gers_score ─────────────
        combined: Dict[str, float] = {}
        for ua in unique_answers:
            combined[ua] = counter[ua] + self.gers_lambda * gers_scores[ua]

        chosen = max(combined, key=combined.get) if combined else majority_answer
        rerank_triggered = (chosen != majority_answer)

        rep = next((c for c in candidates
                    if normalize_for_vote(c["answer"]) == chosen), candidates[0])

        return {
            "answer": chosen,
            "reasoning_text": rep["reasoning_text"],
            "all_answers": [c["answer"] for c in candidates],
            "vote_distribution": vote_distribution,
            "confidence": confidence,
            "gers_rerank_used": True,
            "gers_rerank_triggered": rerank_triggered,
            "gers_lambda": self.gers_lambda,
            "gers_scores": gers_scores,
            "combined_scores": combined,
            "majority_answer": majority_answer,
            "chosen_answer": chosen,
            "method": (f"CoT-SC+GERS (N={self.num_samples}, λ={self.gers_lambda}, "
                       f"rerank={'on' if rerank_triggered else 'no-override'})"),
        }

    def _score_by_gers_lite(self, question: str, context: str,
                            answer: str, reasoning: str) -> float:
        """
        轻量 GERS 校验：对单个 (question, context, reasoning, answer) 打分

        用 1 次 LLM 调用，按 3 维度（实体覆盖 / 推理连贯 / 答案落地）评分
        解析失败时返回保守中性分 0.5
        """
        prompt = GERS_LITE_SCORING_PROMPT.format(
            question=question[:500],
            context=context[:1500] if context else "(no context provided)",
            reasoning=(reasoning or "")[:1500],
            answer=(answer or "")[:200],
        )

        try:
            response = self.model.generate(prompt, max_tokens=200, temperature=0.0)
            return self._parse_gers_score(response)
        except Exception:
            return 0.5  # 解析失败时给中性分

    @staticmethod
    def _parse_gers_score(response: str) -> float:
        """
        解析 LLM 返回的 JSON 评分

        预期格式：
        {
          "entity_coverage": 0.0,
          "reasoning_coherence": 0.0,
          "answer_grounding": 0.0,
          "overall": 0.0,
          "reason": "..."
        }
        """
        import re
        import json

        if not response:
            return 0.5

        # 尝试提取 JSON 块
        text = response.strip()

        # 去掉 markdown 包裹
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # 尝试直接 JSON 解析
        try:
            data = json.loads(text)
            overall = data.get("overall")
            if isinstance(overall, (int, float)) and 0.0 <= overall <= 1.0:
                return float(overall)
            # overall 缺失时用三维度均值
            sub = [data.get(k) for k in
                   ("entity_coverage", "reasoning_coherence", "answer_grounding")]
            sub = [v for v in sub if isinstance(v, (int, float))]
            if sub:
                return max(0.0, min(1.0, sum(sub) / len(sub)))
        except (json.JSONDecodeError, ValueError):
            pass

        # 回退：正则提取 overall
        m = re.search(r'"overall"\s*:\s*([\d.]+)', text)
        if m:
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass

        # 回退：取 "0.7" 之类数字
        nums = re.findall(r"0?\.\d+|[01]\.?\d*", text)
        for n in nums[:3]:
            try:
                val = float(n)
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue

        return 0.5
