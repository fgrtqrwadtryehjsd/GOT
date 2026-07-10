"""Download LongBench QA subsets + probe format.

Run ONCE manually (bypasses the sandbox network restriction when invoked by the
user via the `!` prefix):
    !python data/download_longbench.py

Downloads the English generative-QA subsets that span the 4k/16k context-length
range (for SOP Stage-1 / H1 verification) from the LongBench GitHub repo into
data/longbench_raw/, then prints the schema so prepare_longbench() can be written
to match the real fields.

Subsets chosen:
  2wiki_multihop_qa  ~4-5k tok  (multi-hop, short)
  musique            ~4-11k tok (multi-hop)
  multifieldqa_en    ~5-12k tok (multi-domain)
  narrativeqa        ~15-20k tok (narrative, long)
  qasper             ~5-12k tok (scientific QA)

These cover 4k/8k/16k bins. 32k/64k are not well covered by generative-QA
subsets in LongBench (those lengths are summarization: gov_report/qmsum), so H1
is tested on the 4k-16k range where QA gold answers exist.
"""
import json
import urllib.request
from pathlib import Path

SUBSETS = [
    "2wiki_multihop_qa",
    "musique",
    "multifieldqa_en",
    "narrativeqa",
    "qasper",
]
BASE_URL = "https://raw.githubusercontent.com/THUDM/LongBench/main/data/{}.jsonl"
OUT_DIR = Path(__file__).parent / "longbench_raw"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in SUBSETS:
        url = BASE_URL.format(name)
        out = OUT_DIR / f"{name}.jsonl"
        if out.exists() and out.stat().st_size > 0:
            print(f"[{name}] 已存在 ({out.stat().st_size} bytes), 跳过下载")
        else:
            print(f"[{name}] 下载 {url} ...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                out.write_bytes(data)
                print(f"      → {out} ({len(data)} bytes)")
            except Exception as e:
                print(f"      下载失败: {e}")
                continue

    # ── 格式探测 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("格式探测（每子集第1条）")
    print("=" * 70)
    for name in SUBSETS:
        out = OUT_DIR / f"{name}.jsonl"
        if not out.exists():
            continue
        with open(out, encoding="utf-8") as f:
            first = json.loads(f.readline())
        print(f"\n--- {name} ---")
        print(f"  keys: {list(first.keys())}")
        inp = first.get("input", "")
        print(f"  input 字符数: {len(inp)}")
        print(f"  answers: {first.get('answers', '?')}")
        print(f"  length(token): {first.get('length', '?')}")
        print(f"  all_datasets: {first.get('all_datasets', '?')}")
        # 尝试拆出 question：LongBench 模板通常是 "...context...\n\nQuestion: Q\nAnswer:"
        for sep in ["\n\nQuestion: ", "\nQuestion: ", "\n\nQ: ", "\nQ: "]:
            if sep in inp:
                ctx, rest = inp.split(sep, 1)
                q = rest.rsplit("\nAnswer:", 1)[0] if "\nAnswer:" in rest else rest
                print(f"  [拆分 sep={sep!r}]")
                print(f"    context 字符数: {len(ctx)}")
                print(f"    question: {q[:120]!r}")
                break
        else:
            print(f"  [未找到已知 question 分隔符，input 尾部: {inp[-150:]!r}]")

    # ── length 分布（用于分桶）──────────────────────────────────
    print("\n" + "=" * 70)
    print("length(token) 分桶分布（决定 4k/8k/16k 怎么分）")
    print("=" * 70)
    import statistics
    for name in SUBSETS:
        out = OUT_DIR / f"{name}.jsonl"
        if not out.exists():
            continue
        lengths = []
        with open(out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lengths.append(json.loads(line).get("length", 0))
        if not lengths:
            continue
        buckets = {"<=4k": 0, "4-8k": 0, "8-16k": 0, "16-32k": 0, ">32k": 0}
        for L in lengths:
            if L <= 4000:
                buckets["<=4k"] += 1
            elif L <= 8000:
                buckets["4-8k"] += 1
            elif L <= 16000:
                buckets["8-16k"] += 1
            elif L <= 32000:
                buckets["16-32k"] += 1
            else:
                buckets[">32k"] += 1
        print(f"  {name:<22} n={len(lengths):>4}  median={statistics.median(lengths):.0f}  "
              f"min={min(lengths)} max={max(lengths)}  {buckets}")


if __name__ == "__main__":
    main()
