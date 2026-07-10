"""Download LongBench QA subsets from HuggingFace.

Downloads the official data.zip from HuggingFace `THUDM/LongBench` and extracts
the jsonl files into data/longbench_raw/data/. The 5 English generative-QA
subsets span the 4k/16k context-length range for SOP Stage-1 / H1 verification.

Subsets (in data/longbench_raw/data/):
  2wikimqa          n=200, 4-12k tok  (multi-hop, short)
  musique           n=200, 4-17k tok  (multi-hop, deep)
  multifieldqa_en   n=150, 0.5-10k tok (multi-domain)
  narrativeqa       n=200, 5-36k tok  (narrative, long)
  qasper            n=200, 1.5-15k tok (scientific QA)

Usage:
    python data/download_longbench.py
"""
import io
import zipfile
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent / "longbench_raw"
ZIP_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = OUT_DIR / "data"

    # Skip if already extracted
    target = data_dir / "2wikimqa.jsonl"
    if target.exists() and target.stat().st_size > 0:
        print(f"[LongBench] 数据已存在 ({data_dir}), 跳过下载")
        return

    print(f"[LongBench] 下载 {ZIP_URL} ...")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    print(f"      下载完成 ({len(raw) / 1e6:.1f} MB)")

    print(f"[LongBench] 解压到 {data_dir} ...")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(OUT_DIR)

    # Verify
    subsets = ["2wikimqa", "musique", "multifieldqa_en", "narrativeqa", "qasper"]
    for name in subsets:
        p = data_dir / f"{name}.jsonl"
        if p.exists():
            print(f"  ✓ {name}.jsonl ({p.stat().st_size / 1e3:.0f} KB)")
        else:
            print(f"  ✗ {name}.jsonl MISSING")

    print(f"\n[LongBench] 完成。数据在 {data_dir}/")
    print("  字段: input, context, answers, length, dataset, language, all_classes, _id")
    print("  下一步: 用 prepare_longbench() 转换为 GERS 统一格式")


if __name__ == "__main__":
    main()
