"""日志工具"""

import logging
import sys
from pathlib import Path


def get_logger(name: str = "GERS", level: str = "INFO",
               log_file: str = None) -> logging.Logger:
    """
    获取统一格式的日志器

    Args:
        name: 日志器名称
        level: 日志级别 DEBUG/INFO/WARNING/ERROR
        log_file: 可选，日志文件路径（同时输出到文件和终端）
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重复添加 handler

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 终端 handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件 handler（可选）
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
