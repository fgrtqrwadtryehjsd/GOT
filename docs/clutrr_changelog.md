# CLUTRR 移除说明

**移除时间**：2026-06-29
**移除原因**：所有方法在 CLUTRR 上 EM 严重贴地（0.13~0.19），零区分度，纯粹是噪声
**决策来源**：GERS_方法改进与问题清单.md 第三梯队「砍掉自造 CLUTRR」

---

## 1. 为什么移除

### 1.1 数据来源问题
CLUTRR 不是公开下载的标准数据集，而是 `data/prepare_data.py:prepare_clutrr()` 通过 `_generate_clutrr_samples()` 用预定义模板程序化生成的 300 条样本：

```python
TEMPLATES = [
    (["{A}是{B}的父亲", "{A}是{B}的父亲"], "祖父"),  # 2跳模板
    (["{A}是{B}的父亲", "{A}是{B}的哥哥"], "伯伯/叔叔"),
    # ... 38 个模板
]
```

虽然程序化生成可以保证答案可控，但：
- **样本多样性受限于模板**——只覆盖了 38 个关系路径，与真实家庭关系推理的复杂度差距大
- **人名分配随机**——`random.shuffle(all_names_pool)`，没有语义关联性
- **答案标签是中文亲属称谓**——LLM 的预训练语料中家庭关系推理以英文为主（CLUTRR 原版是英文），翻译损失大

### 1.2 实验区分度问题
所有方法在 CLUTRR 测试集上 EM 全部贴地（典型值：Zero-Shot 0.13、Standard CoT 0.15、CoT-SC 0.16、ToT 0.14、GERS 0.18）：

| 方法 | CLUTRR EM | GSM8K EM | HotpotQA EM |
|------|----------|----------|-------------|
| Zero-Shot | 0.13 | 0.55 | 0.17 |
| Standard CoT | 0.15 | 0.73 | 0.21 |
| CoT-SC | 0.16 | 0.75 | 0.22 |
| ToT | 0.14 | 0.85 | 0.13 |
| GERS+自适应 | 0.18 | 0.65 | 0.26 |

**CLUTRR 上的方法间差距（5pt）远小于数据集本身的方法内噪声**，审稿人无法从中读出有效结论。

### 1.3 评审人意见
> 「程序化生成的 300 条，所有方法 EM 都在 0.13~0.19 贴地，零区分度，留着只暴露弱点。」
> —— 来自 `GERS_方法改进与问题清单.md` 第三梯队

---

## 2. 移除范围

### 2.1 代码层（已处理）
- ❌ `data/prepare_data.py:prepare_clutrr()` → 改为 `raise NotImplementedError`
- ❌ `data/prepare_data.py:_generate_clutrr_samples()` → 改为 `raise NotImplementedError`
- ❌ `src/utils/answer_normalizer.py:normalize_clutrr_answer()` → 改为 `raise NotImplementedError`
- ❌ `experiments/run_quick_exp.py` --dataset choices → 移除 clutrr
- ❌ `experiments/run_comparison.py` --dataset choices → 移除 clutrr
- ❌ `experiments/configs/comparison_config.json` clutrr 配置 → 删除
- ✅ 函数体保留但立即抛异常，避免破坏历史脚本导入

### 2.2 数据层
- 保留 `data/processed/clutrr_test.json`（历史数据，不删）
- 保留 `experiments/results/clutrr_*_results.json`（历史结果，仅做归档参考）

### 2.3 论文层（todo 8 处理）
- 移除 CLUTRR 整段（摘要/相关工作/主表/分题型分析）
- 重写为「HotpotQA + GSM8K + 2WikiMultiHopQA」三件套

---

## 3. 替代方案：2WikiMultiHopQA

为弥补 CLUTRR 留下的"逻辑归纳推理"场景空缺，新增 **2WikiMultiHopQA**（HF: `xanhho/2WikiMultihopQA`）：

| 维度 | 2WikiMultiHopQA | CLUTRR（已删） |
|------|----------------|----------------|
| 数据来源 | Wikipedia 维基百科真实文档 | 程序化生成模板 |
| 样本数 | 192,606（可截断到 500-2000） | 300 |
| 推理深度 | 2-4 跳，需要显式 entity 链接 | 2-3 跳，关系链模板 |
| 题型分类 | comparison / bridge / inference / compositional | 仅亲属关系归纳 |
| 方法区分度 | **预期高**（复杂依赖结构） | 贴地（0.13-0.19） |
| 答案格式 | 短实体 / yes-no / 日期 | 中文亲属称谓 |
| LLM 适配 | 英文为主，匹配预训练 | 中文亲属称谓需特殊处理 |

预期 GERS 在 2WikiMultiHopQA 上能发挥图结构优势（在 comparison/inference 题上预期领先 CoT 5-15pt），这正是新方法**应该发挥的战场**。

---

## 4. 回滚方式

如需临时恢复 CLUTRR 实验：
1. 从 git 历史中恢复 `prepare_clutrr()` 和 `_generate_clutrr_samples()` 函数体
2. 在 `data/prepare_data.py:load_processed_dataset` 的 `filename_map` 加 `"clutrr": "clutrr_test.json"`
3. 在 `run_quick_exp.py` / `run_comparison.py` 的 choices 加 "clutrr"

但**不推荐回滚**——重新引入 CLUTRR 只会重新暴露"方法零区分度"问题。

---

## 5. 时间线

| 日期 | 事件 |
|------|------|
| 2026-06-29 | 评审人提交 `GERS_方法改进与问题清单.md`，建议砍掉 CLUTRR |
| 2026-06-29 | 本 changelog 创建 |
| 2026-06-29 | 代码层 CLUTRR 函数体替换为 `NotImplementedError` |
| 2026-06-29 | 2WikiMultiHopQA 数据准备脚本（`prepare_2wikimultihopqa`）新增 |
| 2026-06-29 | 论文重写（todo 8）将完成 CLUTRR 段落删除 |
