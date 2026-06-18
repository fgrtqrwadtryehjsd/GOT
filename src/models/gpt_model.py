"""
GPT模型接口 —— 通过OpenAI API调用GPT-4/GPT-3.5

安全说明：
- API Key 必须通过环境变量 OPENAI_API_KEY 传入，禁止硬编码
"""

import os
from typing import Optional
from .base_model import BaseModel


class GPTModel(BaseModel):
    """
    OpenAI GPT 模型接口

    使用方式：
    1. 设置环境变量 OPENAI_API_KEY
    2. 可选设置 OPENAI_API_BASE（代理地址）
    """

    def __init__(self,
                 model_name: str = "gpt-4o",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 device: str = "api"):
        super().__init__(model_name, device)
        # 安全：优先使用传入参数，其次从环境变量读取，禁止硬编码
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        if not self.api_key:
            raise ValueError(
                "OpenAI API Key 未设置。请通过环境变量 OPENAI_API_KEY 配置，"
                "例如：export OPENAI_API_KEY=sk-xxx"
            )

    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        """调用 OpenAI Chat Completion API"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库：pip install openai>=1.0.0")

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def count_tokens(self, text: str) -> int:
        """估算 token 数（tiktoken 可用时精确计算）"""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model_name)
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4
