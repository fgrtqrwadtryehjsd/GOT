"""
Qwen模型接口 —— 支持本地Qwen模型推理

支持：
- Qwen-3-8B（本地推理）
- Qwen-3-70B-Int4（量化推理）
- 通过vLLM加速推理
"""

from typing import Optional
from .base_model import BaseModel


class QwenModel(BaseModel):
    """
    Qwen模型接口
    
    使用方式：
    1. 直接加载模型（transformers）
    2. 通过vLLM推理服务
    3. 通过API调用
    """

    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-8B",
                 device: str = "auto",
                 load_method: str = "transformers",
                 api_base: Optional[str] = None):
        super().__init__(model_name, device)
        self.load_method = load_method
        self.api_base = api_base
        self._model = None
        self._tokenizer = None

    def _load_transformers(self):
        """通过transformers加载本地模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def _load_vllm(self):
        """通过vLLM加载模型（更快的推理速度）"""
        from vllm import LLM, SamplingParams
        
        self._model = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_tokens: int = 2048, 
                 temperature: float = 0.7) -> str:
        """生成文本"""
        if self._model is None:
            if self.load_method == "transformers":
                self._load_transformers()
            elif self.load_method == "vllm":
                self._load_vllm()
            elif self.load_method == "api":
                return self._api_generate(prompt, max_tokens, temperature)

        if self.load_method == "vllm":
            return self._vllm_generate(prompt, max_tokens, temperature)
        else:
            return self._transformers_generate(prompt, max_tokens, temperature)

    def _transformers_generate(self, prompt: str, max_tokens: int, 
                                temperature: float) -> str:
        """transformers推理"""
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def _vllm_generate(self, prompt: str, max_tokens: int, 
                        temperature: float) -> str:
        """vLLM推理"""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        outputs = self._model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _api_generate(self, prompt: str, max_tokens: int, 
                       temperature: float) -> str:
        """通过API调用"""
        import requests

        url = f"{self.api_base}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(url, json=payload, timeout=120)
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()