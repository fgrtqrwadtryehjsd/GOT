"""
统一 dump 所有实验结果文件的关键指标
"""
import json
import os
from pathlib import Path

RESULT_DIR = Path("d:/code/GOT/experiments/results")
ABLATION_DIR = RESULT_DIR / "ablation"

# ── 主实验 ──
print("=" * 80)
print("主实验结果（按数据集+方法）")
print("=" * 80)
print(f"{'file':<55} {'n':>5} {'EM':>7} {'F1':>7}")
print("-" * 80)

# GSM8K
gsm8k_methods = ["zero_shot", "standard_cot", "cot_sc", "cot_sc_gers", "tot", "gers", "gers_adaptive"]
print("[GSM8K]")
for m in gsm8k_methods:
    p = RESULT_DIR / f"gsm8k_{m}_results.json"
    if p.exists():
        try:
            data = json.load(open(p, encoding="utf-8"))
            if isinstance(data, dict) and "results" in data:
                n = len(data["results"])
                ems = [r.get("metrics", {}).get("em", 0) for r in data["results"] if r.get("error") is None]
                f1s = [r.get("metrics", {}).get("f1", 0) for r in data["results"] if r.get("error") is None]
                em = sum(ems) / max(len(ems), 1) if ems else 0
                f1 = sum(f1s) / max(len(f1s), 1) if f1s else 0
                print(f"  gsm8k_{m}_results.json{'':<30} {n:>5} {em:>7.3f} {f1:>7.3f}")
            else:
                print(f"  gsm8k_{m}_results.json: 格式未知")
        except Exception as e:
            print(f"  gsm8k_{m}_results.json: 读取失败 {e}")
    else:
        print(f"  gsm8k_{m}_results.json: 不存在")

# HotpotQA
hotpotqa_methods = ["zero_shot", "standard_cot", "cot_sc", "cot_sc_gers", "tot", "gers_adaptive", "gers_sc", "gers_nli", "gers_feedback"]
print("\n[HotpotQA]")
for m in hotpotqa_methods:
    p = RESULT_DIR / f"hotpotqa_{m}_results.json"
    if p.exists():
        try:
            data = json.load(open(p, encoding="utf-8"))
            if isinstance(data, dict) and "results" in data:
                n = len(data["results"])
                ems = [r.get("metrics", {}).get("em", 0) for r in data["results"] if r.get("error") is None]
                f1s = [r.get("metrics", {}).get("f1", 0) for r in data["results"] if r.get("error") is None]
                em = sum(ems) / max(len(ems), 1) if ems else 0
                f1 = sum(f1s) / max(len(f1s), 1) if f1s else 0
                print(f"  hotpotqa_{m}_results.json{'':<30} {n:>5} {em:>7.3f} {f1:>7.3f}")
            else:
                print(f"  hotpotqa_{m}_results.json: 格式未知")
        except Exception as e:
            print(f"  hotpotqa_{m}_results.json: 读取失败 {e}")
    else:
        print(f"  hotpotqa_{m}_results.json: 不存在")

# 2WikiMultiHopQA
print("\n[2WikiMultiHopQA]")
for m in ["zero_shot", "standard_cot", "cot_sc", "cot_sc_gers", "tot", "gers", "gers_adaptive"]:
    p = RESULT_DIR / f"2wikimultihopqa_{m}_results.json"
    print(f"  2wikimultihopqa_{m}_results.json: {'EXISTS' if p.exists() else 'MISSING'}")

# Qwen2.5
print("\n[Qwen2.5-7B]")
for m in ["zero_shot", "standard_cot", "cot_sc", "cot_sc_gers", "gers_adaptive"]:
    p = RESULT_DIR / f"qwen25_hotpotqa_{m}_results.json"
    if p.exists():
        print(f"  qwen25_hotpotqa_{m}_results.json: EXISTS")
    else:
        p2 = RESULT_DIR / f"hotpotqa_{m}_qwen25_results.json"
        if p2.exists():
            print(f"  hotpotqa_{m}_qwen25_results.json: EXISTS")
        else:
            print(f"  qwen25_hotpotqa_{m}_results.json: MISSING")
