"""
数据集准备脚本 —— 下载并预处理实验数据集

支持数据集：
- HotpotQA（多跳问答）
- GSM8K（数学推理）
- 2WikiMultiHopQA（依赖结构密集多跳，含 comparison/bridge/inference/compositional 四类）

[2026-06-29 起移除] CLUTRR（程序化生成的自造数据，所有方法零区分度）
详见 docs/clutrr_changelog.md

使用方法：
    python data/prepare_data.py --dataset all --output_dir data/processed
    python data/prepare_data.py --dataset hotpotqa --num_samples 500
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── HotpotQA ────────────────────────────────────────────────────────────────

def prepare_hotpotqa(output_dir: Path, num_samples: int = 500, split: str = "validation"):
    """
    下载并预处理 HotpotQA 数据集

    支持两种方式：
    1. 自动从 HuggingFace 下载（需要网络访问 HF）
    2. 从本地 JSON 文件解析（手动下载后放到 data/hotpotqa/）
       - 下载地址：https://hotpotqa.github.io/
       - 文件名：hotpot_dev_distractor_v1.json
    """
    import json as _json

    # 优先尝试本地文件（两个可能的路径）
    local_candidates = [
        Path("data/hotpotqa/hotpot_dev_distractor_v1.json"),
        Path("data/hotpotqa_raw/raw/hotpot_dev_distractor_v1.json"),
        Path(__file__).parent / "hotpotqa" / "hotpot_dev_distractor_v1.json",
        Path(__file__).parent / "hotpotqa_raw" / "raw" / "hotpot_dev_distractor_v1.json",
    ]
    local_json = next((p for p in local_candidates if p.exists()), None)
    if local_json:
        print(f"[HotpotQA] 使用本地文件: {local_json}")
        with open(local_json, encoding="utf-8") as f:
            raw = _json.load(f)
        samples = []
        for item in raw:
            if num_samples and len(samples) >= num_samples:
                break
            context_parts = []
            for title, sentences in item.get("context", []):
                context_parts.append(f"[{title}] " + " ".join(sentences))
            context_full = " | ".join(context_parts)
            samples.append({
                "id": item["_id"],
                "question": item["question"],
                "context": context_full[:2000],
                "context_full": context_full,
                "answer": item["answer"],
                "type": item.get("type", ""),
                "supporting_facts": [f[0] for f in item.get("supporting_facts", [])],
            })
        out_path = output_dir / "hotpotqa_test.json"
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"[HotpotQA] 已保存 {len(samples)} 条样本 → {out_path}")
        return samples

    # 尝试 HuggingFace 下载
    print(f"[HotpotQA] 尝试从 HuggingFace 下载 {split} 集...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库：pip install datasets")

    try:
        ds = load_dataset("hotpot_qa", "distractor", split=split)
    except Exception:
        # 新版 datasets 包名
        ds = load_dataset("simonycl/hotpotqa-distractor", split=split)

    samples = []
    for i, item in enumerate(ds):
        if num_samples and len(samples) >= num_samples:
            break
        context_parts = []
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
        ):
            context_parts.append(f"[{title}] " + " ".join(sentences))
        context_full = " | ".join(context_parts)
        samples.append({
            "id": item.get("id", str(i)),
            "question": item["question"],
            "context": context_full[:2000],
            "context_full": context_full,
            "answer": item["answer"],
            "type": item.get("type", ""),
            "supporting_facts": item.get("supporting_facts", {}).get("title", []),
        })

    out_path = output_dir / "hotpotqa_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[HotpotQA] 已保存 {len(samples)} 条样本 → {out_path}")
    return samples


# ─── GSM8K ───────────────────────────────────────────────────────────────────

def prepare_gsm8k(output_dir: Path, num_samples: int = 500, split: str = "test"):
    """
    下载并预处理 GSM8K 数学推理数据集

    输出格式：
    {
        "id": str,
        "question": str,
        "context": "",
        "answer": str,       # 提取 #### 后的数字答案
        "full_solution": str # 完整解题过程
    }
    """
    print(f"[GSM8K] 正在下载 {split} 集...")
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    samples = []
    for i, item in enumerate(ds):
        if num_samples and len(samples) >= num_samples:
            break

        full_sol = item["answer"]
        # 提取 #### 后的最终数字答案
        if "####" in full_sol:
            answer = full_sol.split("####")[-1].strip()
        else:
            answer = full_sol.strip().split("\n")[-1].strip()

        samples.append({
            "id": str(i),
            "question": item["question"],
            "context": "",
            "answer": answer,
            "full_solution": full_sol,
        })

    out_path = output_dir / "gsm8k_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[GSM8K] 已保存 {len(samples)} 条样本 → {out_path}")
    return samples


# ─── CLUTRR ──────────────────────────────────────────────────────────────────
# 2026-06-29 起 CLUTRR 已被从主实验移除（见 docs/clutrr_changelog.md）
# 原因：所有方法 EM 在 0.13~0.19 区间贴地，零区分度，且为程序化生成的自造数据。
# 函数体保留以避免破坏历史脚本导入，但调用会立即报错。

def prepare_clutrr(output_dir: Path, num_samples: int = 500):
    """[已废弃] CLUTRR 已被从主实验移除。请改用 HotpotQA / GSM8K / 2WikiMultiHopQA。"""
    raise NotImplementedError(
        "CLUTRR 已被从主实验移除（2026-06-29）。详见 docs/clutrr_changelog.md。"
        "请改用 HotpotQA / GSM8K / 2WikiMultiHopQA。"
    )


def _generate_clutrr_samples(num_samples: int, start_id: int = 0):
    """[已废弃] CLUTRR 程序化生成器已停用，保留仅为 API 兼容。"""
    raise NotImplementedError("CLUTRR _generate_clutrr_samples 已停用")


# ─── 2WikiMultiHopQA ────────────────────────────────────────────────────────

def prepare_2wikimultihopqa(output_dir: Path, num_samples: int = 500,
                             split: str = "dev"):
    """
    下载并预处理 2WikiMultiHopQA 数据集（依赖结构密集多跳）

    2WikiMultiHopQA 是基于 Wikipedia 的多跳问答数据集，包含 4 类问题：
    - comparison：A 和 B 谁更...（对比型，GERS 杀手锏）
    - bridge：A 是 X，X 是 Y，那么 A 是 Y 的什么（桥接型）
    - inference：基于证据推断（GERS 也能受益）
    - compositional / bridge_comparison：组合推理（高阶 GERS）

    数据集来源（按优先级）：
    1. 本地 data/2wiki_raw/data/dev.json（Dropbox 官方 data.zip，2026-06-29 已下载）
    2. HuggingFace xanhho/2WikiMultihopQA（新版 datasets 已不支持 trust_remote_code）

    输出格式：
    {
        "id": str,
        "question": str,
        "context": str,       # 拼接的 Wikipedia 段落
        "answer": str,        # 短答案或 yes/no
        "type": str,          # comparison/bridge/inference/compositional/bridge_comparison
        "supporting_facts": list
    }
    """
    print(f"[2WikiMultiHopQA] 正在准备 {split} 集...")

    samples = []
    raw = None

    # 1) 本地 Dropbox 官方 data.zip
    local_candidates = [
        Path("data/2wiki_raw/data/dev.json"),
        Path("data/2wiki_raw/data/train.json"),
        Path("data/2wiki_raw/dev.json"),
    ]
    for p in local_candidates:
        if p.exists():
            print(f"[2WikiMultiHopQA] 使用本地文件: {p}")
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            break

    if raw is None:
        raise FileNotFoundError(
            "未找到 2WikiMultiHopQA 本地文件。请先下载官方 data.zip：\n"
            "https://www.dropbox.com/s/npidmtadreo6df2/data.zip\n"
            "解压后将 data/dev.json 放到 data/2wiki_raw/data/ 目录下。"
        )

    # 2WikiMultiHopQA 字段：_id, question, answer, supporting_facts, context, evidences, type, entity_ids
    for i, item in enumerate(raw):
        if num_samples and len(samples) >= num_samples:
            break

        # 解析 context：context 是 [[title, [sent1, sent2, ...]], ...]
        context_parts = []
        context_raw = item.get("context", [])
        if isinstance(context_raw, list):
            for ctx_item in context_raw:
                if isinstance(ctx_item, list) and len(ctx_item) >= 2:
                    title, sentences = ctx_item[0], ctx_item[1]
                    if isinstance(sentences, list):
                        context_parts.append(f"[{title}] " + " ".join(str(s) for s in sentences))
                    else:
                        context_parts.append(f"[{title}] {sentences}")

        # 解析 supporting_facts
        sf = item.get("supporting_facts", [])
        sf_titles = []
        if isinstance(sf, list):
            for s in sf:
                if isinstance(s, list) and s:
                    sf_titles.append(s[0])

        question = item.get("question", "")
        answer = item.get("answer", "")
        qtype = item.get("type", "")

        if not question or not answer:
            continue

        samples.append({
            "id": str(item.get("_id", i)),
            "question": question,
            "context": " | ".join(context_parts)[:2000] if context_parts else "",
            "answer": answer,
            "type": qtype,
            "supporting_facts": sf_titles,
        })

    out_path = output_dir / "2wikimultihopqa_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 统计分题型
    type_counts = {}
    for s in samples:
        t = s.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"[2WikiMultiHopQA] 已保存 {len(samples)} 条样本 → {out_path}")
    print(f"[2WikiMultiHopQA] 题型分布: {type_counts}")
    return samples


# ─── MuSiQue ─────────────────────────────────────────────────────────────────

def prepare_musique(output_dir: Path, num_samples: int = 500,
                    split: str = "dev"):
    """
    下载并预处理 MuSiQue 多跳问答数据集（H2 假设验证用）

    MuSiQue (Trivedi et al., TACL 2022) 是通过组合单跳问题构建的多跳问答数据集，
    包含 2-4 跳问题，每条样本带有完整的分解标注（question_decomposition），
    每跳子问题都有 gold answer——这对 SOP H2 假设的 Oracle 解剖实验至关重要：
    - Oracle 1（完美 DAG）：用 question_decomposition 替换 GERS 的分解
    - Oracle 3（完美中间答案）：用 question_decomposition[].answer 替换中间子答案

    数据集来源：本地 data/musique_raw/data/musique_ans_v1.0_dev.jsonl
    （Google Drive 官方 musique_v1.0.zip，gdown 下载）

    两个版本：
    - musique_ans（Answerable）：仅可回答的问题，用于主实验
    - musique_full（Full）：含不可回答问题，用于 unanswerability 检测（本实验不用）

    输出格式：
    {
        "id": str,                    # 含 hop 前缀，如 "2hop__460946_294723"
        "question": str,              # 完整多跳问题
        "context": str,               # 拼接段落（截断至 ~2000 字符）
        "context_full": str,          # 完整拼接段落（不截断）
        "answer": str,                # 最终答案
        "answer_aliases": list,       # 答案别名
        "hop_count": int,             # 跳数（2/3/4，从 id 前缀提取）
        "question_decomposition": [   # 分解标注（Oracle 实验关键）
            {"question": str, "answer": str, "paragraph_support_idx": int},
            ...
        ],
        "supporting_paragraphs": list, # is_supporting=True 的段落 idx
        "n_paragraphs": int           # 总段落数（含干扰段）
    }
    """
    print(f"[MuSiQue] 正在准备 {split} 集...")

    # 查找本地文件
    local_candidates = [
        Path("data/musique_raw/data/musique_ans_v1.0_dev.jsonl"),
        Path("data/musique_raw/data/musique_ans_v1.0_test.jsonl"),
        Path(__file__).parent / "musique_raw" / "data" / f"musique_ans_v1.0_{split}.jsonl",
    ]
    local_file = next((p for p in local_candidates if p.exists()), None)

    if local_file is None:
        raise FileNotFoundError(
            "未找到 MuSiQue 本地文件。请先下载官方数据：\n"
            "  pip install gdown\n"
            "  python -m gdown \"https://drive.google.com/uc?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h\" -O data/musique_v1.0.zip\n"
            "  # 解压到 data/musique_raw/"
        )

    print(f"[MuSiQue] 使用本地文件: {local_file}")

    # 先读取全部数据，再打乱（固定种子）后截取 num_samples，保证跳数多样性
    import random
    all_items = []
    with open(local_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_items.append(json.loads(line))

    random.seed(42)
    random.shuffle(all_items)

    samples = []
    for item in all_items:
        if num_samples and len(samples) >= num_samples:
            break

        # 跳过不可回答的问题（musique_ans 版本应全是 answerable=True）
        if not item.get("answerable", True):
            continue

        qid = item.get("id", str(len(samples)))
        # 从 id 前缀提取跳数（如 "2hop__460946_294723" → 2）
        hop_count = 2  # 默认
        if qid.startswith("2hop"):
            hop_count = 2
        elif qid.startswith("3hop"):
            hop_count = 3
        elif qid.startswith("4hop"):
            hop_count = 4

        # 构建上下文：拼接所有段落（含干扰段），标记 supporting
        paragraphs = item.get("paragraphs", [])
        context_parts = []
        supporting_indices = []
        for para in paragraphs:
            idx = para.get("idx", 0)
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            context_parts.append(f"[{title}] {text}")
            if para.get("is_supporting", False):
                supporting_indices.append(idx)

        context_full = " | ".join(context_parts)

        # 提取分解标注
        decomp = item.get("question_decomposition", [])
        decomposition = []
        for step in decomp:
            decomposition.append({
                "question": step.get("question", ""),
                "answer": step.get("answer", ""),
                "paragraph_support_idx": step.get("paragraph_support_idx", -1),
            })

        samples.append({
            "id": qid,
            "question": item.get("question", ""),
            "context": context_full[:2000],
            "context_full": context_full,
            "answer": item.get("answer", ""),
            "answer_aliases": item.get("answer_aliases", []),
            "hop_count": hop_count,
            "question_decomposition": decomposition,
            "supporting_paragraphs": supporting_indices,
            "n_paragraphs": len(paragraphs),
        })

    out_path = output_dir / "musique_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 统计跳数分布
    hop_counts = {}
    for s in samples:
        h = s.get("hop_count", 0)
        hop_counts[h] = hop_counts.get(h, 0) + 1

    print(f"[MuSiQue] 已保存 {len(samples)} 条样本 → {out_path}")
    print(f"[MuSiQue] 跳数分布: {hop_counts}")
    return samples


# ─── 验证数据集完整性 ─────────────────────────────────────────────────────────

def validate_dataset(file_path: Path) -> bool:
    """验证数据集 JSON 格式正确性"""
    if not file_path.exists():
        print(f"[验证] 文件不存在: {file_path}")
        return False

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print(f"[验证] 数据集为空或格式错误: {file_path}")
        return False

    required_keys = {"question", "answer"}
    first = data[0]
    missing = required_keys - set(first.keys())
    if missing:
        print(f"[验证] 缺少必需字段 {missing}: {file_path}")
        return False

    print(f"[验证] OK {file_path.name}: {len(data)} 条样本，字段正常")
    return True


# ─── 加载已处理数据集 ─────────────────────────────────────────────────────────

def load_processed_dataset(dataset_name: str,
                            data_dir: str = "data/processed",
                            num_samples: int = None):
    """
    加载已预处理的数据集（优先读取本地 processed，否则实时下载）

    Args:
        dataset_name: hotpotqa / gsm8k / 2wikimultihopqa
        data_dir: processed 数据目录
        num_samples: 最多加载条数

    Returns:
        list of dicts，每条含 question/context/answer 字段
    """
    data_dir = Path(data_dir)
    filename_map = {
        "hotpotqa": "hotpotqa_test.json",
        "gsm8k": "gsm8k_test.json",
        "2wikimultihopqa": "2wikimultihopqa_test.json",
        "musique": "musique_test.json",
    }

    fname = filename_map.get(dataset_name)
    if fname is None:
        raise ValueError(f"未知数据集: {dataset_name}，支持: {list(filename_map)}")

    local_path = data_dir / fname
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            samples = json.load(f)
        if num_samples:
            samples = samples[:num_samples]
        return samples

    # 本地不存在 → 实时下载
    print(f"[加载] 本地文件不存在，实时下载 {dataset_name}...")
    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset_name == "hotpotqa":
        return prepare_hotpotqa(data_dir, num_samples or 500)
    elif dataset_name == "gsm8k":
        return prepare_gsm8k(data_dir, num_samples or 500)
    elif dataset_name == "2wikimultihopqa":
        return prepare_2wikimultihopqa(data_dir, num_samples or 500)
    elif dataset_name == "musique":
        return prepare_musique(data_dir, num_samples or 500)


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="准备实验数据集")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "hotpotqa", "gsm8k", "2wikimultihopqa", "musique"],
        help="要准备的数据集名称，all=全部",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed",
        help="输出目录",
    )
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="每个数据集最多样本数",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="准备后验证数据集完整性",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_prepare = (
        ["hotpotqa", "gsm8k", "2wikimultihopqa", "musique"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for ds_name in datasets_to_prepare:
        print(f"\n{'='*50}")
        print(f"准备数据集: {ds_name}")
        print(f"{'='*50}")
        try:
            if ds_name == "hotpotqa":
                prepare_hotpotqa(output_dir, args.num_samples)
            elif ds_name == "gsm8k":
                prepare_gsm8k(output_dir, args.num_samples)
            elif ds_name == "2wikimultihopqa":
                prepare_2wikimultihopqa(output_dir, args.num_samples)
            elif ds_name == "musique":
                prepare_musique(output_dir, args.num_samples)
        except Exception as e:
            print(f"[错误] {ds_name} 准备失败: {e}")
            continue

        if args.validate:
            fname_map = {
                "hotpotqa": "hotpotqa_test.json",
                "gsm8k": "gsm8k_test.json",
                "2wikimultihopqa": "2wikimultihopqa_test.json",
                "musique": "musique_test.json",
            }
            validate_dataset(output_dir / fname_map[ds_name])

    print("\n数据准备完成！")


if __name__ == "__main__":
    main()
