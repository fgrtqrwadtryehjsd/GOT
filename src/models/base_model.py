"""LLM模型基类"""


class BaseModel:
    """LLM模型基类，定义统一接口"""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device

    def generate(self, prompt: str, max_tokens: int = 2048, 
                 temperature: float = 0.7) -> str:
        """生成文本"""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """计算Token数"""
        return len(text) // 4  # 粗略估计