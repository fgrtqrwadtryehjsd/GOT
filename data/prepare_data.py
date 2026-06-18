"""
数据集准备脚本 —— 下载并预处理实验数据集

支持数据集：
- HotpotQA（多跳问答）
- GSM8K（数学推理）
- CLUTRR（逻辑归纳推理）

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

    输出格式（每条样本）：
    {
        "id": str,
        "question": str,
        "context": str,      # 拼接的段落文本
        "answer": str,
        "type": str,         # bridge / comparison
        "supporting_facts": [[title, sent_idx], ...]
    }
    """
    print(f"[HotpotQA] 正在下载 {split} 集...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库：pip install datasets")

    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    samples = []
    for i, item in enumerate(ds):
        if num_samples and len(samples) >= num_samples:
            break

        # 拼接所有段落作为上下文
        context_parts = []
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
        ):
            context_parts.append(f"[{title}] " + " ".join(sentences))
        context = " | ".join(context_parts)

        samples.append({
            "id": item["id"],
            "question": item["question"],
            "context": context[:2000],   # 截断避免超长
            "answer": item["answer"],
            "type": item.get("type", ""),
            "supporting_facts": item.get("supporting_facts", {}).get("title", []),
        })

    # 保存
    out_path = output_dir / "hotpotqa_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

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

    ds = load_dataset("openai/gsm8k", "main", split=split, trust_remote_code=True)
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

def prepare_clutrr(output_dir: Path, num_samples: int = 500):
    """
    下载并预处理 CLUTRR 逻辑归纳推理数据集

    CLUTRR 测试模型通过家庭关系推理多跳亲属关系，例如：
    "A是B的父亲，B是C的母亲，那么A与C是什么关系？"

    数据来自 HuggingFace clutrr-fg 或使用内置样例。
    """
    print("[CLUTRR] 正在尝试下载...")
    try:
        from datasets import load_dataset
        ds = load_dataset("CLUTRR/v1", split="test", trust_remote_code=True)
        samples = []
        for i, item in enumerate(ds):
            if num_samples and len(samples) >= num_samples:
                break
            samples.append({
                "id": str(i),
                "question": item.get("query", item.get("question", "")),
                "context": item.get("story", item.get("context", "")),
                "answer": item.get("target", item.get("answer", "")),
            })
    except Exception as e:
        print(f"[CLUTRR] HuggingFace 下载失败（{e}），使用内置样例...")
        samples = _clutrr_builtin_samples(num_samples)

    out_path = output_dir / "clutrr_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[CLUTRR] 已保存 {len(samples)} 条样本 → {out_path}")
    return samples


def _clutrr_builtin_samples(num_samples: int):
    """CLUTRR 内置样例（用于离线测试）"""
    base = [
        {
            "id": "clutrr_001",
            "question": "基于以上信息，Alice与David是什么亲属关系？",
            "context": (
                "Bob是Alice的父亲。Carol是Bob的母亲。David是Carol的兄弟。"
            ),
            "answer": "叔祖父",
        },
        {
            "id": "clutrr_002",
            "question": "Eva与George是什么亲属关系？",
            "context": (
                "Frank是Eva的哥哥。Grace是Frank的女儿。George是Grace的儿子。"
            ),
            "answer": "侄孙",
        },
        {
            "id": "clutrr_003",
            "question": "Hannah与Jack是什么亲属关系？",
            "context": (
                "Ivan是Hannah的丈夫。Jack是Ivan的父亲。"
            ),
            "answer": "公公",
        },
        {
            "id": "clutrr_004",
            "question": "Karen与Mike是什么亲属关系？",
            "context": (
                "Linda是Karen的姐姐。Mike是Linda的儿子。"
            ),
            "answer": "外甥",
        },
        {
            "id": "clutrr_005",
            "question": "Nancy与Peter是什么亲属关系？",
            "context": (
                "Oscar是Nancy的父亲。Peter是Oscar的兄弟。"
            ),
            "answer": "叔叔/伯伯",
        },
    ]
    # 循环扩充到 num_samples
    samples = []
    for i in range(num_samples):
        s = dict(base[i % len(base)])
        s["id"] = f"clutrr_{i:04d}"
        samples.append(s)
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
        dataset_name: hotpotqa / gsm8k / clutrr
        data_dir: processed 数据目录
        num_samples: 最多加载条数

    Returns:
        list of dicts，每条含 question/context/answer 字段
    """
    data_dir = Path(data_dir)
    filename_map = {
        "hotpotqa": "hotpotqa_test.json",
        "gsm8k": "gsm8k_test.json",
        "clutrr": "clutrr_test.json",
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
    elif dataset_name == "clutrr":
        return prepare_clutrr(data_dir, num_samples or 500)


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="准备实验数据集")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "hotpotqa", "gsm8k", "clutrr"],
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
        ["hotpotqa", "gsm8k", "clutrr"]
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
            elif ds_name == "clutrr":
                prepare_clutrr(output_dir, args.num_samples)
        except Exception as e:
            print(f"[错误] {ds_name} 准备失败: {e}")
            continue

        if args.validate:
            fname_map = {
                "hotpotqa": "hotpotqa_test.json",
                "gsm8k": "gsm8k_test.json",
                "clutrr": "clutrr_test.json",
            }
            validate_dataset(output_dir / fname_map[ds_name])

    print("\n数据准备完成！")


if __name__ == "__main__":
    main()
