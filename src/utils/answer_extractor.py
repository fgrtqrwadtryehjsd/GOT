"""统一答案提取工具

解决各方法答案提取不一致、输出完整句子而非简短答案的问题。
针对 HotpotQA（yes/no + 短实体）、GSM8K（数值）、CLUTRR（关系词）分别优化。
"""
import re
from typing import Optional


def extract_answer(text: str, dataset: str = None, reference: str = None, question: str = None) -> str:
    """从 LLM 输出中提取简短答案

    Args:
        text: LLM 原始输出
        dataset: 数据集名称（gsm8k/hotpotqa/clutrr），用于选择提取策略
        reference: 参考答案（可选，用于 yes/no 检测）
        question: 原始问题（可选，用于 yes/no 问题检测）

    Returns:
        提取的简短答案
    """
    if not text:
        return ""

    text = text.strip()

    # 0. 提前检测是否为 yes/no 问题
    is_yesno = False
    if reference:
        ref_lower = reference.strip().lower()
        if ref_lower in ("yes", "no"):
            is_yesno = True
    if not is_yesno and question:
        ql = question.lower().strip()
        if ql.startswith(("are ", "is ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "have ", "has ", "will ")):
            is_yesno = True

    # 1. 先尝试 "Final Answer:" / "答案是" 等显式标记
    answer = _extract_by_marker(text)
    if answer:
        answer = _clean_answer(answer, dataset, reference)
        # yes/no 后检测：如果问题是 yes/no 类型，检查提取的答案
        if is_yesno:
            result = _detect_yesno(answer)
            if result in ("yes", "no"):
                return result
        return answer

    # 2. 取最后一行非空文本
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return _clean_answer(text, dataset, reference)

    last_line = lines[-1]

    # 3. 数据集特定提取
    if dataset and dataset.lower() == "gsm8k":
        num = _extract_last_number(text)
        if num is not None:
            return num

    # 4. yes/no 检测（基于 reference 或 question）
    if is_yesno:
        result = _detect_yesno(text)
        if result in ("yes", "no"):
            return result

    # 5. 尝试从最后一行提取
    answer = _clean_answer(last_line, dataset, reference)
    if answer and len(answer) <= 50:
        return answer

    # 6. 回退：截断到合理长度
    return _clean_answer(text, dataset, reference)


def _extract_by_marker(text: str) -> Optional[str]:
    """尝试从显式标记中提取答案"""
    patterns = [
        r'Final Answer[:\s]*(.+?)(?:\n|$)',
        r'The answer is[:\s]*(.+?)(?:\n|$)',
        r'Answer[:\s]*(.+?)(?:\n|$)',
        r'答案是[:\s]*(.+?)(?:\n|$)',
        r'答案[：:][\s]*(.+?)(?:\n|$)',
        r'最终答案[：:\s]*(.+?)(?:\n|$)',
        r'####\s*(.+?)(?:\n|$)',  # GSM8K 标准格式
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _detect_yesno(text: str) -> str:
    """检测 yes/no 答案"""
    text_lower = text.lower().strip()
    # 取前几个词判断
    first_words = text_lower[:50]
    if re.match(r'^(yes|yeah|correct|true|both|the same)', first_words):
        return "yes"
    if re.match(r'^(no|nope|false|not|different|neither)', first_words):
        return "no"
    # 整体判断
    if "yes" in text_lower and "no" not in text_lower:
        return "yes"
    if "no" in text_lower and "yes" not in text_lower:
        return "no"
    return text_lower[:3]


def _extract_last_number(text: str) -> Optional[str]:
    """提取最后一个数字"""
    text_clean = text.replace(',', '')
    numbers = re.findall(r'[-+]?\d*\.?\d+', text_clean)
    if numbers:
        return numbers[-1]
    return None


def _clean_answer(answer: str, dataset: str = None, reference: str = None) -> str:
    """清理答案：去标点包裹、去前缀、截断"""
    if not answer:
        return ""

    answer = answer.strip()

    # 去掉引号包裹
    answer = re.sub(r'^["""\']+|["""\']+$', '', answer).strip()

    # 去掉常见前缀
    prefixes = [
        r'^(?:The |the )?answer is[:\s]*',
        r'^(?:The |the )?final answer is[:\s]*',
        r'^final answer[:\s]*',
        r'^answer[:\s]*',
        r'^(?:based on|according to).{0,40}[,，:：]\s*',
        r'^therefore[,，:：]\s*',
        r'^so[,，:：]\s*',
        r'^thus[,，:：]\s*',
        r'^hence[,，:：]\s*',
    ]
    for p in prefixes:
        answer = re.sub(p, '', answer, flags=re.IGNORECASE).strip()

    # 去掉末尾句号
    answer = re.sub(r'[.。]+$', '', answer).strip()

    # 去掉 LaTeX 包裹
    answer = re.sub(r'\$+', '', answer).strip()

    # 如果仍然很长，取第一个逗号/分号前的部分
    if len(answer) > 50:
        for sep in [',', '，', ';', '；', ' - ', ' is ', ' was ', ' are ']:
            idx = answer.lower().find(sep)
            if 5 < idx < 50:
                answer = answer[:idx].strip()
                break

    # 最终截断
    if len(answer) > 80:
        answer = answer[:80].strip()

    return answer
