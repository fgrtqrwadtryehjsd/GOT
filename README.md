# GERS: Graph-Enhanced Reasoning System

> 面向复杂推理任务的图结构思维增强大模型方法研究

## 项目概述

GERS（Graph-Enhanced Reasoning System）通过显式**推理状态图** G=(V,E) 对大模型推理过程进行建模、约束与校验，解决标准 CoT 方法在复杂推理任务中的三大问题：
- 非线性缺失（无法处理多分支/回溯）
- 错误累积（Error Cascading）
- 逻辑幻觉（Hallucination）

## 系统架构

```
输入问题
   ↓
模块1: 图结构驱动的推理状态表示
   LLM 抽取实体关系 → 动态构建推理图 G=(V,E)
   节点：Fact / Step / Conclusion
   边：  Derive / Support / Conflict
   ↓
模块2: 图约束思维链路生成
   拓扑排序路径规划 → 图→Prompt 转换 → 约束解码
   ↓
模块3: 图驱动一致性校验与闭环修正
   连通性 + 环路检测 + 最大流覆盖度 + NLI 语义蕴含
   Consistency_Score = α·S_struct + β·S_semantic
   若得分 < 阈值 → 回溯修正 → 再校验
   ↓
输出答案 + 推理图 + 一致性得分
```

## 目录结构

```
GOT/
├── src/
│   ├── graph_representation/   # 模块1：推理状态图表示
│   ├── chain_generation/       # 模块2：图约束链路生成
│   ├── consistency_check/      # 模块3：一致性校验
│   ├── baselines/              # 基线方法（CoT/CoT-SC/ToT/ZeroShot）
│   ├── models/                 # LLM接口（Qwen/GPT/LLaMA）
│   └── utils/                  # 工具（配置/指标/可视化/日志）
├── data/
│   ├── prepare_data.py         # 数据集下载与预处理脚本
│   └── processed/              # 预处理后数据（.json）
├── experiments/
│   ├── run_comparison.py       # 对比实验
│   ├── run_ablation.py         # 消融实验
│   ├── run_case_study.py       # 案例分析（可视化）
│   ├── configs/                # 实验配置文件
│   └── results/                # 实验结果
├── tests/                      # 单元测试
├── requirements.txt
└── setup.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 或使用 setup.py
pip install -e ".[dev]"
```

### 2. 准备数据集

```bash
# 下载全部数据集（各 500 条测试样本）
python data/prepare_data.py --dataset all --num_samples 500 --validate

# 只准备 GSM8K
python data/prepare_data.py --dataset gsm8k --num_samples 1000
```

### 3. 运行测试

```bash
# 全部测试
python -m pytest tests/ -v

# 单模块测试
python -m pytest tests/test_graph.py -v
python -m pytest tests/test_consistency.py -v
```

### 4. 运行实验（需配置 LLM 模型）

```bash
# 设置模型（API 方式示例）
export OPENAI_API_KEY=sk-xxx

# 对比实验
python experiments/run_comparison.py \
    --dataset hotpotqa \
    --methods gers,standard_cot,cot_sc,tot,zero_shot \
    --model qwen3-8b \
    --num_samples 500

# 消融实验
python experiments/run_ablation.py \
    --dataset hotpotqa \
    --model qwen3-8b \
    --num_samples 500

# 案例分析（含可视化）
python experiments/run_case_study.py \
    --dataset hotpotqa \
    --model qwen3-8b \
    --num_cases 10
```

## 模型配置

### 本地 Qwen（推荐）

```python
from src.models import QwenModel

model = QwenModel(
    model_name="Qwen/Qwen3-8B",
    load_method="transformers",  # 或 "vllm"
    device="auto",
)
```

### GPT-4（API）

```bash
export OPENAI_API_KEY=sk-xxx
```
```python
from src.models import GPTModel
model = GPTModel(model_name="gpt-4o")
```

### LLaMA 本地 API（Ollama）

```bash
export LLAMA_API_BASE=http://localhost:11434/v1
```
```python
from src.models import LlamaModel
model = LlamaModel(model_name="llama3", load_method="api")
```

## Demo（无模型）

```python
from src.chain_generation import GraphGuidedGenerator

gen = GraphGuidedGenerator(model=None)  # demo 模式
result = gen.reason("巴黎在哪个洲？")
print(result["graph"].summary())
print(result["consistency_score"])
```

## 评估指标

| 指标 | 说明 |
|------|------|
| EM | Exact Match 精确匹配率 |
| F1 | Token 级别 F1 分数 |
| Consistency Score | 图驱动一致性综合得分 |
| Graph Coverage | 推理路径覆盖关键事实比例 |
| Token Efficiency | Token 消耗量 |
| Latency | 推理耗时（秒） |

## 基线方法

| 方法 | 描述 |
|------|------|
| Zero-Shot | 直接调用 LLM 回答 |
| Standard CoT | "Let's think step by step" |
| CoT-SC | 多次采样 + 投票（N=5） |
| ToT | 思维树 BFS/DFS 搜索 |
| **GERS（本文）** | 图结构思维增强 + 一致性校验闭环 |

## 论文信息

- **题目**：面向复杂推理任务的图结构思维增强大模型方法研究
- **关键词**：大语言模型、复杂推理、图结构表征、思维链、一致性校验、过程监督
