"""
阿里云百炼 DashScope 模型接口

完全兼容 OpenAI 格式：
  base_url = https://dashscope.aliyuncs.com/compatible-mode/v1

支持模型（部分）：
  qwen-max / qwen-plus / qwen-turbo
  qwen-long / qwen3-235b-a22b / qwen3-32b / qwen3-8b 等

API Key 从环境变量 DASHSCOPE_API_KEY 读取，禁止硬编码。
"""

import os
from typing import Optional
from .base_model import BaseModel

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeModel(BaseModel):
    """
    阿里云百炼 DashScope 模型接口（OpenAI 兼容）

    使用方式：
        # 方式1：环境变量（推荐）
        export DASHSCOPE_API_KEY=sk-xxx
        model = DashScopeModel()

        # 方式2：.env 文件（需安装 python-dotenv）
        model = DashScopeModel()   # 自动加载 .env
    """

    def __init__(self,
                 model_name: str = "qwen-plus",
                 api_key: Optional[str] = None,
                 device: str = "api"):
        super().__init__(model_name, device)

        # 尝试从 .env 加载
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # 安全：优先显式传入，其次环境变量，禁止硬编码
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API Key 未设置。\n"
                "请通过环境变量设置：export DASHSCOPE_API_KEY=sk-xxx\n"
                "或在项目根目录创建 .env 文件：DASHSCOPE_API_KEY=sk-xxx"
            )

    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        """调用 DashScope OpenAI 兼容接口生成文本"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库：pip install openai>=1.0.0")

        client = OpenAI(
            api_key=self.api_key,
            base_url=DASHSCOPE_BASE_URL,
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def count_tokens(self, text: str) -> int:
        """粗略估算 token 数"""
        return len(text) // 4
