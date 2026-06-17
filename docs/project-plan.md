# 毕业论文完成总体规划

## 论文题目
面向复杂推理任务的图结构思维增强大模型方法研究

## 当前状态（2026年6月）
处于第二阶段后期：核心算法设计与模块开发阶段

---

## 一、代码架构设计 —— GERS系统

### 1.1 项目结构
```
GOT/
├── src/
│   ├── graph_representation/      # 模块1：推理状态图表示
│   │   ├── __init__.py
│   │   ├── reasoning_graph.py     # 推理图类定义 G=(V,E)
│   │   ├── node.py               # 节点定义（Fact/Step/Conclusion）
│   │   ├── edge.py               # 边定义（Derive/Support/Conflict）
│   │   ├── dynamic_builder.py    # 动态构图机制
│   │   └── extractor.py          # 实体关系抽取（LLM驱动）
│   │
│   ├── chain_generation/          # 模块2：图约束链路生成
│   │   ├── __init__.py
│   │   ├── path_planner.py       # 拓扑排序路径规划
│   │   ├── constrained_decoder.py # 约束解码器
│   │   ├── prompt_builder.py     # 图→Prompt转换
│   │   └── generation_pipeline.py # 生成流水线
│   │
│   ├── consistency_check/         # 模块3：一致性校验
│   │   ├── __init__.py
│   │   ├── connectivity.py       # 连通性检测（断链）
│   │   ├── cycle_detector.py     # 环路检测（循环论证）
│   │   ├── coverage.py           # 最大流/证据覆盖度
│   │   ├── nli_verifier.py       # NLI语义蕴含检测
│   │   ├── consistency_score.py  # Consistency_Score计算
│   │   └── feedback_loop.py      # 生成-校验-修正闭环
│   │
│   ├── baselines/                 # 基线方法实现
│   │   ├── __init__.py
│   │   ├── standard_cot.py       # Standard CoT
│   │   ├── cot_sc.py             # CoT-SC (Self-Consistency)
│   │   ├── tot.py                # Tree of Thoughts
│   │   └── zero_shot.py          # Zero/Few-shot
│   │
│   ├── models/                    # LLM接口封装
│   │   ├── __init__.py
│   │   ├── base_model.py         # 模型基类
│   │   ├── qwen_model.py         # Qwen模型接口
│   │   ├── gpt_model.py          # GPT-4接口（可选）
│   │   └── llama_model.py        # LLaMA接口
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # 全局配置
│       ├── logger.py             # 日志工具
│       ├── metrics.py            # 评估指标（EM/F1/Consistency）
│       └── visualization.py      # 推理图可视化
│
├── data/
│   ├── hotpotqa/                  # HotpotQA数据集
│   ├── gsm8k/                    # GSM8K数据集
│   ├── clutrr/                   # CLUTRR数据集
│   └── processed/                # 预处理后数据
│
├── experiments/
│   ├── run_comparison.py          # 对比实验主脚本
│   ├── run_ablation.py            # 消融实验脚本
│   ├── run_case_study.py          # 案例分析脚本
│   ├── configs/                   # 实验配置文件
│   └── results/                   # 实验结果存储
│
├── tests/                         # 单元测试
├── docs/                          # 论文相关文档
├── requirements.txt
├── setup.py
└── README.md
```

### 1.2 核心模块代码设计要点

#### 模块1：推理状态图表示（reasoning_graph.py）

```python
class ReasoningGraph:
    """
    推理状态图 G=(V,E) 的核心数据结构
    
    节点类型：
    - FactNode: 事实节点（已知信息、证据）
    - StepNode: 过程节点（中间推理步骤）
    - ConclusionNode: 目标节点（最终结论）
    
    边类型：
    - DeriveEdge: 推导关系（A推导出B）
    - SupportEdge: 支撑关系（A支撑B的成立）
    - ConflictEdge: 互斥关系（A与B矛盾）
    """
    
    def __init__(self):
        self.nodes = {}          # node_id -> Node
        self.edges = {}          # edge_id -> Edge
        self.adjacency = {}      # node_id -> [edge_ids]
        self.reverse_adj = {}    # node_id -> [incoming_edge_ids]
    
    def add_node(self, node_type, content, metadata=None):
        """动态添加节点"""
        
    def add_edge(self, src_id, dst_id, edge_type, weight=1.0):
        """动态添加边关系"""
        
    def topological_sort(self):
        """拓扑排序，返回推理执行顺序"""
        
    def find_path(self, src_id, dst_id):
        """DFS/BFS搜索从src到dst的路径"""
        
    def connected_components(self):
        """连通分量检测（一致性校验用）"""
    
    def detect_cycles(self):
        """环路检测（一致性校验用）"""
    
    def max_flow_coverage(self, src_id, dst_id):
        """最大流计算证据覆盖度"""
    
    def to_visualization(self):
        """导出为可视化格式"""
```

#### 模块2：图约束链路生成（generation_pipeline.py）

```python
class GraphGuidedGenerator:
    """
    图约束思维链路生成流水线
    
    流程：
    1. 接收问题 → 构建推理图
    2. 拓扑排序 → 路径规划
    3. 图→Prompt转换 → 约束解码
    4. 生成推理文本 → 返回结果
    """
    
    def __init__(self, model, graph_builder, path_planner):
        self.model = model
        self.graph_builder = graph_builder
        self.path_planner = path_planner
    
    def reason(self, question):
        """完整推理流程"""
        # Step 1: 构建推理图
        graph = self.graph_builder.build(question)
        
        # Step 2: 路径规划
        planned_path = self.path_planner.plan(graph)
        
        # Step 3: 生成约束Prompt
        constrained_prompt = self.prompt_builder.build(
            question, graph, planned_path
        )
        
        # Step 4: 约束生成
        result = self.model.generate(constrained_prompt)
        
        # Step 5: 一致性校验（闭环）
        score = self.consistency_checker.check(graph, result)
        if score < threshold:
            # 回溯修正
            result = self.feedback_loop.refine(graph, result, score)
        
        return result, graph, score
```

#### 模块3：一致性校验（consistency_score.py）

```python
class ConsistencyChecker:
    """
    推理一致性校验与闭环修正
    
    三层校验机制：
    1. 结构层：连通性 + 环路 + 覆盖度（图论算法）
    2. 语义层：NLI蕴含关系检测（自然语言推理）
    3. 闭环层：校验→回溯→修正→再校验
    """
    
    def check(self, graph, reasoning_result):
        """综合一致性校验"""
        # 结构校验
        structural_score = self._structural_check(graph)
        
        # 语义校验
        semantic_score = self._semantic_check(graph, reasoning_result)
        
        # 综合得分
        consistency_score = α * structural_score + β * semantic_score
        
        return consistency_score
    
    def _structural_check(self, graph):
        """图论结构校验"""
        # 连通性：是否有断链（孤立节点）
        connectivity = self._check_connectivity(graph)
        
        # 环路：是否有循环论证
        has_cycle = self._detect_cycles(graph)
        
        # 覆盖度：证据对结论的支撑程度
        coverage = self._compute_coverage(graph)
        
        return weighted_score(connectivity, has_cycle, coverage)
    
    def _semantic_check(self, graph, result):
        """NLI语义蕴含校验"""
        # 对每条边，检查两端节点是否存在逻辑蕴含
        for edge in graph.edges:
            entailment = self.nli_model.predict(
                edge.src.content, edge.dst.content
            )
            edge.entailment_score = entailment
        
        return aggregate_scores(graph.edges)
```

---

## 二、实验方案设计

### 2.1 数据集准备

| 数据集 | 类型 | 样本量 | 下载方式 |
|--------|------|--------|----------|
| HotpotQA | 多跳问答 | ~90K训练/~7.4K测试 | HuggingFace datasets |
| GSM8K | 数学推理 | ~7.5K训练/~1K测试 | HuggingFace datasets |
| CLUTRR | 逻辑归纳 | ~10K | GitHub (CLUTRR repo) |

**注意**：论文实验通常只需要测试集（1K-7.4K样本），不需要训练集。

### 2.2 基线方法对比

| 方法 | 描述 | 实现方式 |
|------|------|----------|
| Zero-Shot | 直接调用LLM回答 | 简单Prompt |
| Few-Shot | 提供2-3个示例后回答 | 示例Prompt |
| Standard CoT | 线性思维链推理 | "Let's think step by step" |
| CoT-SC | 自洽性投票（多次采样取众数） | N次采样+投票 |
| ToT | 思维树搜索 | BFS/DFS + 评估 |

### 2.3 评估指标

**任务性能指标**：
- Exact Match (EM)：精确匹配率
- F1 Score：部分匹配率

**推理质量指标（本文创新）**：
- Logical Coherence：逻辑连贯性（GPT-4/人工评分）
- Graph Coverage：推理路径覆盖关键事实节点的比例
- Consistency Score：一致性校验模块输出的得分
- Token Efficiency：Token消耗量
- Latency：推理耗时

### 2.4 消融实验设计

| 配置 | 图表征 | 约束生成 | 一致性校验 | 说明 |
|------|--------|----------|------------|------|
| Full (GERS) | ✅ | ✅ | ✅ | 完整方法 |
| w/o Graph | ❌ | ✅ | ✅ | 移除图表征，退化为CoT+校验 |
| w/o Constraint | ✅ | ❌ | ✅ | 移除约束生成，自由生成+校验 |
| w/o Check | ✅ | ✅ | ❌ | 移除校验，图约束生成无闭环 |
| w/o Feedback | ✅ | ✅ | 仅打分 | 校验但不回溯修正 |

### 2.5 实验执行脚本设计

```bash
# 1. 对比实验
python experiments/run_comparison.py \
    --dataset hotpotqa \
    --methods gers,standard_cot,cot_sc,tot,zero_shot \
    --model qwen3-8b \
    --num_samples 500 \
    --output_dir experiments/results/

# 2. 消融实验
python experiments/run_ablation.py \
    --dataset hotpotqa \
    --model qwen3-8b \
    --num_samples 500

# 3. 案例分析（可视化）
python experiments/run_case_study.py \
    --dataset hotpotqa \
    --model qwen3-8b \
    --num_cases 10 \
    --visualize True
```

---

## 三、论文撰写路线图

### 3.1 推荐写作顺序

```
代码开发 → 实验运行 → 结果收集 → 论文撰写
```

**具体时间分配**：
| 周次 | 任务 | 产出 |
|------|------|------|
| W1-W2 | 搭建项目骨架 + 模块1代码 | reasoning_graph.py + extractor.py |
| W3-W4 | 模块2代码 + 模块3代码 | 生成流水线 + 校验机制 |
| W5 | 集成Pipeline + 跑初步Demo | GERS完整流程跑通 |
| W6-W7 | 对比实验 + 数据收集 | 实验结果表格 |
| W8 | 消融实验 + 案例分析 | 消融表格 + 可视化图 |
| W9-W10 | 撰写第3-5章（核心方法） | 方法章节初稿 |
| W11-W12 | 撰写第6章（实验）+ 第2章（相关工作） | 实验+相关工作章节 |
| W13 | 撰写第1章 + 第7章 | 绪论+总结 |
| W14 | 全文修改润色 | 最终稿 |

### 3.2 各章节字数参考

| 章节 | 页数 | 字数 |
|------|------|------|
| 第一章 绪论 | 15-20页 | ~8000字 |
| 第二章 相关工作 | 25-30页 | ~12000字 |
| 第三章 推理状态表示 | 30-40页 | ~15000字 |
| 第四章 链路生成算法 | 30-40页 | ~15000字 |
| 第五章 一致性校验 | 25-35页 | ~12000字 |
| 第六章 实验分析 | 35-45页 | ~16000字 |
| 第七章 总结展望 | 5-8页 | ~3000字 |
| **总计** | ~180-220页 | ~80000字 |

---

## 四、关键注意事项

1. **构图错误率问题**：LLM抽取实体关系可能有误差，需要设计容错机制（多轮校正）
2. **约束强度平衡**：过强约束导致僵化输出，过弱约束无法抑制幻觉——需要实验调参
3. **NLI模型选择**：建议使用DeBERTa-v3-large-mnli（轻量且效果好）
4. **Qwen vs GPT-4**：建议主实验用Qwen-3-8B（本地可控），补充实验用GPT-4 API
5. **推理图可视化**：使用NetworkX + matplotlib或graphviz，这是论文的视觉亮点
6. **实验样本量**：建议每个数据集取500-1000条测试，CoT-SC需要多次采样（计算开销大）
7. **Consistency Score公式**：需要给出明确的数学公式定义，不能只做定性描述