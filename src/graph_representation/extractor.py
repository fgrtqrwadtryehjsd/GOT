"""
实体关系抽取器 —— 使用LLM从自然语言中提取结构化信息

通过Prompt Engineering驱动LLM输出结构化的实体-关系三元组
"""

import json
from typing import Dict, List, Optional


# 图构建Prompt模板
GRAPH_EXTRACTION_PROMPT = """你是一个推理分析专家。请分析以下问题，提取出推理所需的结构化信息。

问题：{question}
{context_section}

请按以下JSON格式输出：

```json
{{
    "facts": [
        {{"content": "事实描述", "source": "question/context", "evidence": "原始文本"}}
    ],
    "steps": [
        {{"content": "推理步骤描述", "operation": "比较/计算/检索/推导"}}
    ],
    "conclusions": [
        {{"content": "目标结论描述", "answer": "预期答案（如有）"}}
    ],
    "relations": [
        {{"source": "节点A的内容", "target": "节点B的内容", "type": "derive/support/conflict", "description": "关系说明"}}
    ]
}}
```

要求：
1. facts：从问题和上下文中提取已知事实和证据
2. steps：识别需要执行的推理步骤（子任务分解）
3. conclusions：明确问题的最终目标
4. relations：标注节点之间的逻辑关系
   - derive：A推导出B（因果/逻辑推导）
   - support：A支撑B（证据支撑）
   - conflict：A与B矛盾（逻辑冲突）

请确保提取的信息完整且逻辑关系准确。只输出JSON，不要输出其他内容。"""


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