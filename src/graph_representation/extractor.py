"""
实体关系抽取器 —— 使用LLM从自然语言中提取结构化信息

通过Prompt Engineering驱动LLM输出结构化的实体-关系三元组
"""

import json
from typing import Dict, List, Optional


# 图构建Prompt模板（英文，避免翻译导致答案不匹配）
GRAPH_EXTRACTION_PROMPT = """You are a reasoning analysis expert. Analyze the following question and extract structured reasoning information.

Question: {question}
{context_section}

Output ONLY valid JSON in this exact format:

```json
{{
    "facts": [
        {{"content": "fact description (keep original language)", "source": "question/context", "evidence": "original text"}}
    ],
    "steps": [
        {{"content": "reasoning step description", "operation": "compare/calculate/lookup/infer"}}
    ],
    "conclusions": [
        {{"content": "target conclusion", "answer": "expected answer if known"}}
    ],
    "relations": [
        {{"source": "node A content", "target": "node B content", "type": "derive/support/conflict", "description": "relation description"}}
    ]
}}
```

Requirements:
1. facts: extract known facts and evidence from question and context (preserve original wording)
2. steps: identify reasoning sub-tasks needed
3. conclusions: the final goal/answer target
4. relations: logical connections between nodes
   - derive: A logically leads to B
   - support: A supports/evidences B
   - conflict: A contradicts B

Output ONLY the JSON, no other text."""


class EntityRelationExtractor:
    """
    基于LLM的实体关系抽取器
    
    使用Prompt Engineering从自然语言中提取：
    - 事实节点（Facts）
    - 推理步骤（Steps）
    - 目标结论（Conclusions）
    - 逻辑关系（Derive/Support/Conflict）
    """

    def __init__(self, model=None, custom_prompt: str = None):
        """
        Args:
            model: LLM模型实例（需实现generate方法）
            custom_prompt: 自定义抽取Prompt
        """
        self.model = model
        self.prompt_template = custom_prompt or GRAPH_EXTRACTION_PROMPT

    def extract(self, question: str, context: str = "") -> Dict:
        """
        从自然语言中提取结构化信息
        
        Args:
            question: 待分析的问题
            context: 补充上下文
        
        Returns:
            包含facts/steps/conclusions/relations的字典
        """
        if self.model is None:
            # 无模型时返回空结构（用于测试）
            return self._default_extraction()

        context_section = f"\n上下文信息：{context}" if context else ""
        prompt = self.prompt_template.format(
            question=question,
            context_section=context_section
        )

        response = self.model.generate(prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict:
        """解析LLM返回的JSON"""
        try:
            # 尝试提取JSON块
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            result = json.loads(json_str.strip())
            
            # 验证必需字段
            for key in ["facts", "steps", "conclusions", "relations"]:
                if key not in result:
                    result[key] = []
            
            return result
        except (json.JSONDecodeError, IndexError):
            return self._default_extraction()

    def _default_extraction(self) -> Dict:
        """返回默认空结构"""
        return {
            "facts": [],
            "steps": [],
            "conclusions": [],
            "relations": []
        }