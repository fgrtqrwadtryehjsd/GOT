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
    lower = cleaned.lower().strip().rstrip('.,!?，。！？').strip()
    if lower in ('yes', '是', '是的', 'yes they are', 'yes they were',
                 'yes both', 'both are', 'both were'):
        return 'yes'
    if lower in ('no', '否', '不是', '不', '没有', 'no they are not',
                 'no they were not', 'neither', 'not both'):
        return 'no'
    # 去掉末尾的地名补充（如 ", New York City"）
    # 保留核心实体
    return cleaned


def normalize_clutrr_answer(raw: str) -> str:
    """CLUTRR 专用：提取亲属关系词，中文同义词标准化"""
    cleaned = clean_answer(raw)

    # 从完整句子中提取关系词（"X是Y的叔叔" → "叔叔"）
    m = re.search(r'是[^的]*的(.{2,6}?)(?:[。，,.]|$)', cleaned)
    if m:
        cleaned = m.group(1).strip()

    # 取第一个关系词（"叔叔和侄女" → "叔叔"）
    cleaned = re.split(r'[和与及,，、]', cleaned)[0].strip()

    # 同义词映射到标准词（与 data/processed/clutrr_test.json 的标签对齐）
    synonym_map = {
        "叔叔": "叔叔",
        "舅舅": "舅舅",
        "公公": "公公",
        "侄子": "侄子",
        "侄女": "侄女",
        "叔祖父": "叔祖父",
        "外祖父": "外祖父",
        "外公": "外祖父",
        "侄孙": "侄孙",
        "女儿": "女儿",
        "父亲": "父亲",
        "爸爸": "父亲",
        "母亲": "母亲",
        "妈妈": "母亲",
        "儿子": "儿子",
        "祖父": "祖父",
        "爷爷": "祖父",
        "祖母": "祖母",
        "奶奶": "祖母",
        "兄弟": "兄弟",
        "哥哥": "兄弟",
        "弟弟": "兄弟",
        "姐妹": "姐妹",
        "姐姐": "姐妹",
        "妹妹": "姐妹",
        "丈夫": "丈夫",
        "妻子": "妻子",
        "uncle": "叔叔",
        "aunt": "姑姑",
        "father": "父亲",
        "mother": "母亲",
        "son": "儿子",
        "daughter": "女儿",
        "grandfather": "祖父",
        "grandmother": "祖母",
        "nephew": "侄子",
        "niece": "侄女",
    }
    for k, v in synonym_map.items():
        if k == cleaned or k in cleaned.lower():
            return v
    return cleaned
