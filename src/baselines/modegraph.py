"""
MoDeGraph 基线 —— 基于 Multi-Hop Dependency Graphs 的 prompt 方法

复现 MoDeGraph (CIKM 2025, Oruche et al.) 的核心思想：
1. 引导 LLM 从复杂问题中提取实体及其关系
2. 构建实体关系依赖图（entity relationships as multi-hop dependency graph）
3. 基于图引导 LLM 逐步推理回答

与 GERS 的关键区别：
- MoDeGraph 的图是「实体关系图」（从问题中提取实体间关系）
- GERS 的图是「子问题依赖图」（将问题分解为有依赖关系的子问题 DAG）
- MoDeGraph 是 prompt 方法（一次性构建图并送入 LLM）
- GERS 是完整 pipeline（分解→构图→拓扑排序执行→一致性校验→闭环修正）
"""

import re
import json
from typing import Dict, List

MODEGRAPH_EXTRACT_PROMPT = """You are an expert at analyzing complex multi-hop questions.

Question: {question}
{context_section}

Step 1: Extract all entities mentioned in the question and context.
Step 2: Identify the relationships between these entities (how they connect).
Step 3: Build a dependency graph showing the reasoning path from known entities to the answer.

Output the entity-relationship dependency graph as JSON:

```json
{{
  "entities": ["entity1", "entity2", ...],
  "relationships": [
    {{"from": "entity1", "to": "entity2", "relation": "relationship description"}},
    ...
  ],
  "reasoning_path": ["entity1 -> entity2 -> ... -> answer"]
}}
```

Output ONLY the JSON, no other text."""

MODEGRAPH_ANSWER_PROMPT = """Answer the question using the entity-relationship dependency graph below.

Question: {question}
{context_section}

Entity-Relationship Dependency Graph:
{dependency_graph}

Follow the reasoning path in the graph to derive the answer step by step.
Each step should use the relationship between connected entities.

Reasoning:
"""

MODEGRAPH_FINAL_PROMPT = """Based on the reasoning above, provide the final answer.

Question: {question}

Final Answer: <your concise answer>"""


class MoDeGraphBaseline:
    """
    MoDeGraph 基线方法

    流程：
    1. LLM 提取实体关系依赖图（prompt approach）
    2. 基于图引导 LLM 推理
    3. 提取最终答案

    注意：这是基于 MoDeGraph 公开摘要实现的 prompt 基线，
    用于与 GERS（子问题 DAG + 拓扑排序执行 + 一致性校验）对比。
    """

    def __init__(self, model=None):
        self.model = model

    def reason(self, question: str, context: str = "") -> Dict:
        if self.model is None:
            return {"answer": "", "reasoning_text": "[需要配置模型]", "method": "MoDeGraph"}

        context_section = f"\nContext: {context[:1500]}" if context else ""

        # Step 1: 提取实体关系依赖图
        extract_prompt = MODEGRAPH_EXTRACT_PROMPT.format(
            question=question,
            context_section=context_section
        )
        graph_response = self.model.generate(extract_prompt, max_tokens=500, temperature=0.2)
        dependency_graph = self._parse_graph(graph_response)

        # Step 2: 基于图引导推理
        graph_text = self._format_graph(dependency_graph)
        answer_prompt = MODEGRAPH_ANSWER_PROMPT.format(
            question=question,
            context_section=context_section,
            dependency_graph=graph_text
        )
        reasoning_text = self.model.generate(answer_prompt, max_tokens=500, temperature=0.3)

        # Step 3: 提取最终答案
        final_prompt = MODEGRAPH_FINAL_PROMPT.format(question=question)
        final_response = self.model.generate(final_prompt, max_tokens=200, temperature=0.1)
        from ..utils.answer_extractor import extract_answer
        answer = extract_answer(final_response, question=question)

        full_reasoning = f"[Dependency Graph]\n{graph_text}\n\n[Reasoning]\n{reasoning_text}\n\n{final_response}"

        return {
            "answer": answer,
            "reasoning_text": full_reasoning,
            "method": "MoDeGraph",
            "dependency_graph": dependency_graph,
        }

    def _parse_graph(self, response: str) -> Dict:
        """解析 LLM 输出的实体关系图 JSON"""
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            return data
        except Exception:
            return {
                "entities": [],
                "relationships": [],
                "reasoning_path": [],
                "raw": response[:200]
            }

    def _format_graph(self, graph: Dict) -> str:
        """格式化图为文本"""
        lines = []
        entities = graph.get("entities", [])
        if entities:
            lines.append(f"Entities: {', '.join(entities[:10])}")

        rels = graph.get("relationships", [])
        for r in rels[:8]:
            lines.append(f"  {r.get('from', '?')} --[{r.get('relation', '?')}]--> {r.get('to', '?')}")

        path = graph.get("reasoning_path", [])
        if path:
            lines.append(f"Reasoning path: {' -> '.join(path[:5])}")

        return "\n".join(lines) if lines else "No graph extracted"

    def _extract_answer(self, text: str) -> str:
        """从最终回答中提取答案"""
        patterns = [
            r'Final Answer[：:]\s*(.+?)(?:\n|$)',
            r'The answer is[：:]\s*(.+?)(?:\n|$)',
            r'Therefore[,，]\s*(.+?)(?:\n|$)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                ans = m.group(1).strip().strip('*#.,。').strip()
                if ans and len(ans) < 200:
                    return ans
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip().lstrip('#*|>-').strip()
            if line and 2 < len(line) < 150:
                return line
        return text.strip()[:100]
