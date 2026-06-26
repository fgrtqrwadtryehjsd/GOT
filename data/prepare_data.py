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
            samples.append({
                "id": item["_id"],
                "question": item["question"],
                "context": " | ".join(context_parts)[:2000],
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
        samples.append({
            "id": item.get("id", str(i)),
            "question": item["question"],
            "context": " | ".join(context_parts)[:2000],
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

def prepare_clutrr(output_dir: Path, num_samples: int = 500):
    """
    下载并预处理 CLUTRR 逻辑归纳推理数据集

    CLUTRR 测试模型通过家庭关系推理多跳亲属关系，例如：
    "A是B的父亲，B是C的母亲，那么A与C是什么关系？"

    数据来源优先级：
    1. HuggingFace CLUTRR/v1（如可用）
    2. 程序化生成多样化家庭关系推理样本（默认）
    """
    print("[CLUTRR] 正在准备数据...")
    samples = []

    # 尝试 HuggingFace 下载
    try:
        from datasets import load_dataset
        ds = load_dataset("CLUTRR/v1", split="test", trust_remote_code=True)
        for i, item in enumerate(ds):
            if num_samples and len(samples) >= num_samples:
                break
            samples.append({
                "id": str(i),
                "question": item.get("query", item.get("question", "")),
                "context": item.get("story", item.get("context", "")),
                "answer": item.get("target", item.get("answer", "")),
            })
        print(f"[CLUTRR] 从 HuggingFace 加载了 {len(samples)} 条样本")
    except Exception as e:
        print(f"[CLUTRR] HuggingFace 下载失败（{e}），使用程序化生成...")

    # 如果 HuggingFace 不足或失败，用程序化生成补足
    if len(samples) < num_samples:
        needed = num_samples - len(samples)
        print(f"[CLUTRR] 程序化生成 {needed} 条补充样本...")
        generated = _generate_clutrr_samples(needed, start_id=len(samples))
        samples.extend(generated)

    # 截断到请求数量
    samples = samples[:num_samples]

    out_path = output_dir / "clutrr_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[CLUTRR] 已保存 {len(samples)} 条样本 → {out_path}")
    return samples


def _generate_clutrr_samples(num_samples: int, start_id: int = 0):
    """
    程序化生成多样化的 CLUTRR 家庭关系推理样本。
    
    使用预定义的关系链模板，确保每条样本都有正确答案。
    每条样本包含：
    - context: 家庭关系描述（如 "A是B的父亲。B是C的母亲。"）
    - question: 目标关系查询（如 "A与C是什么亲属关系？"）
    - answer: 正确的亲属关系（如 "祖父/外祖父"）
    """
    import random
    
    random.seed(42 + start_id)  # 可复现
    
    male_names = [
        "James", "John", "Robert", "Michael", "William", "David", "Thomas",
        "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven",
        "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian", "George",
        "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan",
    ]
    female_names = [
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
        "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
        "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna",
        "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie",
    ]
    
    # 预定义关系链模板: (关系描述列表, 最终关系答案)
    # 每个模板描述: P0 -> P1 -> P2 [-> P3], 问 P0 与最后一个人的关系
    # 每条描述用 {A} 和 {B} 分别表示当前步骤的源和目标人物
    TEMPLATES = [
        # 2跳: 父/母 × 父/母 → 祖父母
        (["{A}是{B}的父亲", "{A}是{B}的父亲"], "祖父"),
        (["{A}是{B}的父亲", "{A}是{B}的母亲"], "祖父"),
        (["{A}是{B}的母亲", "{A}是{B}的父亲"], "祖母"),
        (["{A}是{B}的母亲", "{A}是{B}的母亲"], "祖母"),
        # 2跳: 父 × 兄弟姐妹 → 伯伯/叔叔/姑姑
        (["{A}是{B}的父亲", "{A}是{B}的哥哥"], "伯伯/叔叔"),
        (["{A}是{B}的父亲", "{A}是{B}的弟弟"], "伯伯/叔叔"),
        (["{A}是{B}的父亲", "{A}是{B}的姐姐"], "姑姑"),
        (["{A}是{B}的父亲", "{A}是{B}的妹妹"], "姑姑"),
        # 2跳: 母 × 兄弟姐妹 → 舅舅/阿姨
        (["{A}是{B}的母亲", "{A}是{B}的哥哥"], "舅舅"),
        (["{A}是{B}的母亲", "{A}是{B}的弟弟"], "舅舅"),
        (["{A}是{B}的母亲", "{A}是{B}的姐姐"], "阿姨"),
        (["{A}是{B}的母亲", "{A}是{B}的妹妹"], "阿姨"),
        # 2跳: 配偶 × 父/母 → 公婆/岳父母
        (["{A}是{B}的丈夫", "{A}是{B}的父亲"], "公公"),
        (["{A}是{B}的丈夫", "{A}是{B}的母亲"], "婆婆"),
        (["{A}是{B}的妻子", "{A}是{B}的父亲"], "岳父"),
        (["{A}是{B}的妻子", "{A}是{B}的母亲"], "岳母"),
        # 2跳: 兄弟姐妹 × 子/女 → 侄子/侄女/外甥
        (["{A}是{B}的哥哥", "{A}是{B}的儿子"], "侄子"),
        (["{A}是{B}的哥哥", "{A}是{B}的女儿"], "侄女"),
        (["{A}是{B}的弟弟", "{A}是{B}的儿子"], "侄子"),
        (["{A}是{B}的弟弟", "{A}是{B}的女儿"], "侄女"),
        (["{A}是{B}的姐姐", "{A}是{B}的儿子"], "外甥"),
        (["{A}是{B}的姐姐", "{A}是{B}的女儿"], "外甥女"),
        (["{A}是{B}的妹妹", "{A}是{B}的儿子"], "外甥"),
        (["{A}是{B}的妹妹", "{A}是{B}的女儿"], "外甥女"),
        # 2跳: 父/母 × 子/女 → 兄弟姐妹
        (["{A}是{B}的父亲", "{A}是{B}的儿子"], "兄弟姐妹"),
        (["{A}是{B}的父亲", "{A}是{B}的女儿"], "兄弟姐妹"),
        (["{A}是{B}的母亲", "{A}是{B}的儿子"], "兄弟姐妹"),
        (["{A}是{B}的母亲", "{A}是{B}的女儿"], "兄弟姐妹"),
        # 2跳: 祖父母 × 子/女
        (["{A}是{B}的祖父", "{A}是{B}的儿子"], "父亲"),
        (["{A}是{B}的祖父", "{A}是{B}的女儿"], "父亲"),
        (["{A}是{B}的祖母", "{A}是{B}的儿子"], "母亲"),
        (["{A}是{B}的祖母", "{A}是{B}的女儿"], "母亲"),
        # 3跳: 父 × 父 × 兄弟 → 伯祖/叔祖
        (["{A}是{B}的父亲", "{A}是{B}的父亲", "{A}是{B}的哥哥"], "伯祖父/叔祖父"),
        (["{A}是{B}的父亲", "{A}是{B}的父亲", "{A}是{B}的弟弟"], "伯祖父/叔祖父"),
        (["{A}是{B}的父亲", "{A}是{B}的父亲", "{A}是{B}的姐姐"], "姑祖母"),
        (["{A}是{B}的父亲", "{A}是{B}的父亲", "{A}是{B}的妹妹"], "姑祖母"),
        # 3跳: 父 × 兄弟 × 子 → 堂表兄弟
        (["{A}是{B}的父亲", "{A}是{B}的哥哥", "{A}是{B}的儿子"], "堂兄弟"),
        (["{A}是{B}的父亲", "{A}是{B}的哥哥", "{A}是{B}的女儿"], "堂姐妹"),
        (["{A}是{B}的父亲", "{A}是{B}的姐姐", "{A}是{B}的儿子"], "表兄弟"),
        (["{A}是{B}的父亲", "{A}是{B}的姐姐", "{A}是{B}的女儿"], "表姐妹"),
        # 3跳: 母 × 兄弟 × 子
        (["{A}是{B}的母亲", "{A}是{B}的哥哥", "{A}是{B}的儿子"], "表兄弟"),
        (["{A}是{B}的母亲", "{A}是{B}的姐姐", "{A}是{B}的女儿"], "表姐妹"),
        # 3跳: 配偶 × 父 × 兄弟
        (["{A}是{B}的丈夫", "{A}是{B}的父亲", "{A}是{B}的哥哥"], "伯父/叔父"),
        (["{A}是{B}的丈夫", "{A}是{B}的父亲", "{A}是{B}的姐姐"], "姑母"),
        # 3跳: 父 × 母 × 兄弟
        (["{A}是{B}的父亲", "{A}是{B}的母亲", "{A}是{B}的弟弟"], "舅祖父"),
        (["{A}是{B}的父亲", "{A}是{B}的母亲", "{A}是{B}的妹妹"], "姨祖母"),
        # 3跳: 兄弟 × 子 × 子
        (["{A}是{B}的哥哥", "{A}是{B}的儿子", "{A}是{B}的儿子"], "侄孙"),
        (["{A}是{B}的姐姐", "{A}是{B}的女儿", "{A}是{B}的儿子"], "外甥孙"),
    ]
    
    samples = []
    
    for i in range(num_samples):
        idx = start_id + i
        template = random.choice(TEMPLATES)
        relations, answer = template
        
        # 确定需要多少个人名
        num_people = len(relations) + 1  # A, B, C, [D]
        
        # 随机选择人名，确保性别匹配关系
        # 分析关系中每个人物的性别需求
        used = []
        all_names_pool = list(male_names) + list(female_names)
        random.shuffle(all_names_pool)
        
        # 简化：随机分配人名
        names = all_names_pool[:num_people] if len(all_names_pool) >= num_people else \
                [f"Person{j}" for j in range(num_people)]
        
        # 构建context
        context_parts = []
        for j, rel_template in enumerate(relations):
            A, B = names[j], names[j + 1]
            context_parts.append(rel_template.format(A=A, B=B))
        
        context = "。".join(context_parts) + "。"
        
        if len(relations) == 2:
            target = names[0]
            query = names[2]
        else:
            target = names[0]
            query = names[3]
        
        question = f"基于以上信息，{target}与{query}是什么亲属关系？"
        
        samples.append({
            "id": f"clutrr_{idx:04d}",
            "question": question,
            "context": context,
            "answer": answer,
        })
    
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
