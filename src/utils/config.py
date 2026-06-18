"""全局配置"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """GERS系统全局配置"""
    
    # 模型配置
    model_name: str = "qwen3-8b"
    model_load_method: str = "api"           # api = DashScope / transformers / vllm
    model_device: str = "auto"
    model_api_base: Optional[str] = None
    
    # 推理配置
    constraint_mode: str = "soft"  # soft / hard / hybrid
    max_iterations: int = 3        # 最大闭环修正次数
    consistency_threshold: float = 0.7  # 一致性校验阈值
    
    # NLI配置
    enable_nli: bool = False
    nli_model_name: str = "cross-encoder/nli-deberta-v3-large"
    
    # 实验配置
    dataset: str = "hotpotqa"
    num_samples: int = 500
    batch_size: int = 1
    
    # 输出配置
    output_dir: str = "experiments/results/"
    log_level: str = "INFO"
    
    # 权重配置
    alpha: float = 0.6   # 结构层权重
    beta: float = 0.4    # 语义层权重
    w_connectivity: float = 0.35
    w_cycle: float = 0.30
    w_coverage: float = 0.35