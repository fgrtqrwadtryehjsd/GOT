"""
统一答案后处理工具 —— 标准化各种格式的模型输出

支持清理：
- Markdown 加粗 **xxx**
- LaTeX: $xxx$, \\boxed{xxx}, \\$
- 中文前缀: 答案：xxx, 最终答案：xxx
- 编号前缀: 1. xxx, 步骤1: xxx
- 确认符号: ✅ xxx, > xxx
"""

import re
from typing import Optional


def clean_answer(raw: str) -> str:
    """
    将模型输出的各种格式规范化为干净的答案字符串

    规则（按优先级）：
    1. 提取关键词后的内容
    2. 清理 Markdown/LaTeX 格式
    3. 取第一行简短内容
    """
    if not raw:
        return ""

    text = raw.strip()

    # ─── 1. 关键词提取 ─────────────────────────────────────────────
    keywords = [
        r"(?:最终答案|Final Answer|The answer is|Therefore.*?answer is|答案是|答案：|答案:)[：:＊\s]*(.+)",
        r"(?:结论[：:]|Conclusion[：:])[：:＊\s]*(.+)",
    ]
    for kw_pat in keywords:
        m = re.search(kw_pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            text = m.group(1).strip()
            break

    # ─── 2. 格式清理 ────────────────────────────────────────────────
    # 去掉 LaTeX \boxed{xxx}
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
    # 去掉 $xxx$ (行内公式)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    # 去掉 \\$ → $
    text = re.sub(r'\\+\$', '$', text)
    # 去掉 Markdown 加粗/斜体 **xxx** / *xxx*
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # 去掉 >（引用块前缀）
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    # 去掉 ✅、✓、⚠️ 等前缀符号
    text = re.sub(r'^[✅✓⚠️⛔🔴🟢]+\s*', '', text)
    # 去掉数字编号前缀（如 "1. "）
    text = re.sub(r'^\d+[.、)]\s*', '', text)
    # 去掉"步骤X:"前缀
    text = re.sub(r'^步骤\d+[：:]\s*', '', text)
    # 去掉千位分隔符中的逗号（数字）
    # 保留：1,234,567 → 1234567
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    # 取第一行
    text = text.split('\n')[0].strip()
    # 去掉末尾句号/感叹号（中英文）
    text = text.rstrip('。！.!').strip()
    # 去掉中文括号说明 （...）
    text = re.sub(r'（[^）]{0,30}）$', '', text).strip()

    return text.strip()


def normalize_gsm8k_answer(raw: str) -> str:
    """GSM8K 专用：只提取数字答案"""
    cleaned = clean_answer(raw)
    # 优先取最后一个独立数字
    nums = re.findall(r'-?\d+(?:\.\d+)?', cleaned)
    if nums:
        return nums[-1]
    # 再从原文找
    nums = re.findall(r'-?\d+(?:\.\d+)?', raw)
    return nums[-1] if nums else cleaned


def normalize_hotpotqa_answer(raw: str) -> str:
    """HotpotQA 专用：提取简洁文本或 yes/no"""
    cleaned = clean_answer(raw)
    # yes/no 标准化
    lower = cleaned.lower().strip()
    if lower in ('yes', 'yes.', 'yes,', '是', '是的', '是。'):
        return 'yes'
    if lower in ('no', 'no.', 'no,', '否', '不是', '不', '没有'):
        return 'no'
    return cleaned


def normalize_clutrr_answer(raw: str) -> str:
    """CLUTRR 专用：提取亲属关系词"""
    cleaned = clean_answer(raw)
    return cleaned
