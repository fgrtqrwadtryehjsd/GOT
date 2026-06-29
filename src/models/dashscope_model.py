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
import time
import logging
from typing import Optional
from .base_model import BaseModel

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 重试的异常类型（延迟导入，避免 openai 未安装时 import 报错）
def _retryable_exceptions():
    try:
        from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
        return (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    except ImportError:
        return ()


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

    # qwen3 系列是思考模型，非流式调用需关闭 thinking
    THINKING_MODELS = {
        "qwen3-235b-a22b", "qwen3-32b", "qwen3-14b",
        "qwen3-8b", "qwen3-4b", "qwen3-1.7b", "qwen3-0.6b",
    }

    def __init__(self,
                 model_name: str = "qwen-plus",
                 api_key: Optional[str] = None,
                 device: str = "api"):
        super().__init__(model_name, device)
        # qwen3 思考模型非流式调用需禁用 thinking
        self._disable_thinking = model_name in self.THINKING_MODELS

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

        # 单例 OpenAI client：__init__ 在主线程一次性创建，并发场景下复用连接池，
        # 彻底避免多线程懒加载竞态。httpx client 线程安全，可被多线程共享。
        self._client = self._build_client()

    def _build_client(self):
        """创建 OpenAI 兼容 client（缓存复用）"""
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key,
            base_url=DASHSCOPE_BASE_URL,
            timeout=60.0,      # 单请求 60s 超时，避免并发下挂死
            max_retries=0,      # 关闭 SDK 内置重试，由下方 _call_with_retry 统一控制
        )

    def _get_client(self):
        """返回缓存的 client（兼容旧调用点）"""
        return self._client

    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        """调用 DashScope OpenAI 兼容接口生成文本（带重试）"""
        try:
            from openai import OpenAI  # noqa: F401  仅用于检测依赖
        except ImportError:
            raise ImportError("请安装 openai 库：pip install openai>=1.0.0")

        client = self._get_client()
        # qwen3 思考模型非流式调用必须设置 enable_thinking=False
        extra = {"extra_body": {"enable_thinking": False}} if self._disable_thinking else {}

        def _call():
            return client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )

        response = self._call_with_retry(_call)
        return response.choices[0].message.content.strip()

    @staticmethod
    def _call_with_retry(call_fn, max_retries: int = 3):
        """
        对可重试异常（限流/超时/连接/5xx）做指数退避重试。

        并发多线程场景下 DashScope 易触发 QPM 限流（429），
        靠重试 + 退避保证整体实验不被偶发限流打断。
        """
        retry_exc = _retryable_exceptions()
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return call_fn()
            except retry_exc as e:
                last_exc = e
                if attempt < max_retries:
                    wait = 2 ** attempt  # 1, 2, 4 秒
                    logger.warning(
                        f"DashScope 调用失败({type(e).__name__}),"
                        f"{wait}s 后重试 ({attempt+1}/{max_retries})"
                    )
                    time.sleep(wait)
                else:
                    raise
            except Exception:
                raise
        raise last_exc  # 不会执行到，保险

    def count_tokens(self, text: str) -> int:
        """粗略估算 token 数"""
        return len(text) // 4
