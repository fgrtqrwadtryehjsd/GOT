"""
推理一致性校验与闭环修正模块

核心功能：
- 图论结构校验（连通性、环路、覆盖度）
- NLI语义蕴含检测
- Consistency Score量化评估
- 生成-校验-修正闭环
"""

from .consistency_score import ConsistencyChecker
from .connectivity import ConnectivityChecker
from .cycle_detector import CycleDetector
from .coverage import CoverageCalculator
from .nli_verifier import NLIVerifier
from .feedback_loop import FeedbackLoop