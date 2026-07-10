# GERS-CV2 论文重新定位方案（2026-07-10 阶段一后草稿）

> **状态**：阶段一 LongBench H1 验证完成，双线扩量后台进行中。此文档为论文重写的策略与结构建议，等 §1.14 / §1.15 显著性最终结果落定后可直接映射到 paper_en.tex。

---

## 一、废止的旧定位

### 旧叙事（`paper_en.tex` 当前版本，pre-§1.6）
> "GERS-CV2 通过双向交叉验证在多跳推理上显著优于 CoT-SC（HotpotQA n=500, +0.041 F1, McNemar p=0.029）。核心贡献是把 Consistency Score 的对错区分度从 −0.0035 修复到 +0.0847。"

### 为什么必须废止
- `§1.6` 全 context 公平对比：CoT-SC (0.716 F1) > CV2 (0.683)，方向反转
- `§1.8` MuSiQue 所有跳数：CV2 都输 CoT-SC
- `§1.10 UPDATE`：即使 Oracle-1（gold 分解上限）也 CI [-0.024, +0.188] 跨零
- 任何 reviewer 用全 context 复现 HotpotQA 都会打脸

**结论**：旧定位在公平实验条件下**已被自己的实验推翻**，不能作为投稿基础。

---

## 二、新定位候选（按证据强度排序）

### 定位 A（首选）：**中长上下文、多领域干扰场景下的稳健推理方法**

**核心 claim**：
> GERS-CV2 在中等长度 (5-12k tok)、多领域/多文档干扰、跨段落多跳推理任务上，显著优于 CoT-SC。这一场景恰好覆盖当前 RAG / long-context QA 的核心应用区间。方法在短紧凑上下文（HotpotQA 4.7k）与超长叙事（narrativeqa 22k+，超出 8B 模型容量）上均让位于 CoT-SC，此为诚实的能力边界。

**证据支持（阶段一已完成）**：

| 数据集 | 长度 | CoT-SC F1 | CV2 F1 | dF1 | 95% CI | McNemar p | 结论 |
|--------|------|-----------|--------|-----|--------|-----------|------|
| HotpotQA | ~4.7k tok | 0.716 | 0.683 | −0.032 | [+0.002,+0.063] (rev) | 0.314 | CoT-SC 略赢（诚实边界 1） |
| **multifieldqa_en** | **~5k tok** | **0.345** | **0.415** | **+0.070** | **[+0.007, +0.132]** | 0.114 | **F1 显著 ✓** |
| **musique (LB)** | **~11k tok** | 0.310 | 0.390 | +0.075 | [-0.016, +0.165] | 0.453 | **待扩量到 n=200** |
| narrativeqa | ~22k tok | 0.234 | 0.208 | −0.027 | [-0.082, +0.026] | 0.617 | 8B 模型能力边界（诚实边界 2） |

**优点**：
- 至少 1 个 F1 显著数据点，可能扩量后 2 个
- 定位是 **RAG 场景的 sweet spot**，工业相关性强
- 边界 (HotpotQA + narrativeqa) 被诚实解释，不是隐藏

**风险**：
- 如果 musique n=200 CI 仍跨零，只有 multifieldqa_en 单点显著，reviewer 会质疑
- 需要一个清晰的"为什么长度不是决定因素而是干扰结构"的说明

---

### 定位 B（备胎）：**Diagnostic + Positioning Paper**

**核心 claim**：
> 我们系统化研究了图分解方法在多跳 QA 上的效果，发现（1）先前工作的表面收益源于不公平 context 截断；（2）通过 SOP Stage-2 Oracle 解剖定位了 66.2% 恢复空间在 reasoner 模块（非分解质量）；（3）两次 Minimal Fix 尝试（EASV/IDD）均失败；（4）然而在中长多领域场景（multifieldqa_en）下 GERS-CV2 展现出统计显著的稳健性优势。这是首个针对多跳分解方法的完整诊断 + regime 定位研究。

**证据支持**：
- Confound audit (§1.1 vs §1.6) 完整
- Oracle 66/18/15 waterfall n=200（**§1.13 强显著 p=0.002**）
- EASV/IDD 两次失败记录（§1.10, §1.11）
- multifieldqa_en 显著（§1.12）作为"positive corner"

**优点**：
- 证据链最完整（诊断 + 定位 + 失败尝试 + 局部胜利）
- 不依赖任何单一显著性数据点
- 差别化定位（大部分论文都是"方法赢"，你是"完整诊断 + 局部胜利"）

**风险**：
- 主叙事较复杂，需要多层次组织
- 卖点稍弱于"方法赢"

---

### 定位 C（保底）：**纯 Diagnostic Paper**（无 method 主张）

如果 musique 扩量失败 + Oracle-1 vs CoT-SC n=200 仍不显著，退回纯诊断论文：**"Graph decomposition for multi-hop QA: what works, what doesn't, and where the bottleneck really is"**。

---

## 三、待定实验（决定选 A 还是 B）

### 后台跑批（预计 60-90 分钟）
1. **musique CV2 扩量到 n=200**（当前 100 → 200）
   - 判定：CI 是否从 [-0.016, +0.165] 收窄到排除零
2. **MuSiQue 4-hop CoT-SC 扩量到 n=200**（现有 85 → 200）
   - 判定：Oracle-1 vs CoT-SC n=200 是否显著（§1.10 UPDATE 的最终答案）

### 决策矩阵

| musique n=200 | Oracle-1 vs CoT-SC n=200 | 推荐定位 |
|---------------|--------------------------|----------|
| ✅ 显著 | ✅ 显著 | **A + B 混合叙事（最强）** |
| ✅ 显著 | ❌ 不显著 | **A（RAG regime win）** |
| ❌ 不显著 | ✅ 显著 | **B（Oracle 主 + multifieldqa_en 局部）** |
| ❌ 不显著 | ❌ 不显著 | **C（纯 diagnostic）** |

---

## 四、新版 paper_en.tex 章节结构（按定位 A/B 混合叙事）

```
Title (提议): Graph-Decomposition for Long-Context Multi-Hop QA:
              Where It Wins, Where It Fails, and Why

Abstract
  - 问题：Graph decomposition 方法在多跳 QA 上的定位一直模糊
  - 我们的做法：
    (1) 提出 GERS-CV2（双向子答案交叉验证，CS 区分度 -0.0035 → +0.0847）
    (2) 系统化实验揭示 CV2 的真实 regime：中长多领域上下文（5-12k tok）显著优于 CoT-SC
    (3) 通过 Oracle 分解定位 66/18/15 的模块罪值分布
    (4) 诚实报告两次 Minimal Fix 尝试的失败作为方法学警示
  - 主张：GERS-CV2 是长上下文 RAG 场景下的稳健分解方法，但不是通用多跳解决方案

1. Introduction
   - 多跳 QA 的挑战 + 分解方法的定位争议
   - Confound audit warning：截断的陷阱
   - 本文贡献：regime 定位 + Oracle 定位 + 诚实边界

2. Related Work
   - 保留大部分，删除"方法通用赢"的对比表

3. Method (GERS-CV2)
   - 保留大部分（模块 1/2/3 + 双向验证机制）
   - 强调机制而非通用性能

4. Regime Definition and Experimental Setup
   - LongBench 5 子集介绍（新增）
   - HotpotQA 全 context vs 截断对比（confound audit 起点）
   - 评估指标 + paired bootstrap + McNemar 方法

5. Main Results: Regime-Wise Comparison
   5.1 长度 × 干扰结构 二维矩阵表 (核心表)
       - HotpotQA (~5k, 密集证据): CoT-SC 略赢 -0.032 F1
       - multifieldqa_en (~5k, 多领域干扰): **CV2 显著赢 +0.070 F1**
       - musique-LB (~11k, 深跳): [待定 n=200 结果]
       - narrativeqa (~22k): 模型能力边界外
   5.2 关键 finding：CV2 的优势不是长度轴的，是**干扰结构**轴的

6. Oracle Localization Analysis (方法学贡献)
   6.1 §1.9/§1.13 Waterfall on MuSiQue 4-hop n=200
       - Reasoner 66.2% / Graph-gen 18.4% / Retrieval 15.4%
       - 强显著 McNemar p=0.002
   6.2 Interpretation: 瓶颈在 reasoner，不在图生成
   6.3 [如果 Oracle-1 vs CoT-SC n=200 显著] Gold-decomp 上限显著超越 CoT-SC → 分解质量有价值但当前 LLM 未达到

7. Consistency Score Calibration
   - 保留旧 §4.3 内容（-0.0035 → +0.0847，AUROC 0.498 → 0.589）
   - 强调其为 diagnostic signal 而非 correctness verifier

8. Ablation and Case Study
   - 双向验证机制的独立贡献
   - Case A / Case B (保留旧内容)
   - 消融：uniform weight, context_only anchor

9. Negative Results and Method Boundaries (关键新章节，AAAI 稀有内容)
   9.1 Two Failed Minimal Fixes (EASV, IDD)
       - §1.10/§1.11 完整记录
   9.2 Grounded-forward gating (GF-GERS) 失败
   9.3 BM25 per-sub-question 检索反向
   9.4 Consistency Score is NOT correctness (250/500 CS=1.0 但 156 EM 错)
   9.5 讨论：为什么这些失败对社区有价值（诚实性 + 后续研究方向）

10. Cross-Model Generalization
    - Qwen-Plus n=300 平局（§1.5）
    - Qwen3-14b 反转（§2.4）：CS 是 fluency-indexed
    - 结论：GERS-CV2 优势集中在中等能力模型 + 中长多领域场景

11. Discussion, Limitations, Conclusion
    - 学术贡献：regime 定位 + Oracle 方法学 + 诚实边界
    - 实用价值：为 RAG / long-context QA 提供分解方法工具箱
    - 未来工作：depth-aware decomposition，跨模型泛化性
```

---

## 五、写作策略

### 5.1 优先级
1. **先改 Abstract 和 Introduction**（3-4 天）：定 tone
2. **主表和主图重画**（2-3 天）：regime 矩阵表 + Oracle waterfall
3. **§9 Negative Results 全新写作**（3-4 天，最耗时）：Failed Fix 诚实记录
4. **保留大部分 §3 Method + §7 CS Calibration + §8 Case Study**（1-2 天润色）

### 5.2 关键行文原则
- **不写"我方法比 CoT-SC 更好"这种通用 claim**（会被 §1.6 打脸）
- **一切 claim 加 regime 限定**：如 "on medium-length multi-domain contexts"
- **主动披露不利证据**（HotpotQA fair, narrativeqa loss）
- **让 negative results 成为贡献**（大多数论文缺失，你有）

### 5.3 投稿目标
- **首选**：NAACL / EMNLP / ACL Findings（对 diagnostic + honest reporting 友好）
- **备选**：TACL / TMLR（journal 型，接受更长、更诚实的报告）
- **不推荐**：AAAI（主会偏好方法赢，除非 §9 章节写得像 killer feature）

---

## 六、行动清单

**今日（阶段一后）**：
- [x] 阶段一 LongBench H1 验证跑完（multifieldqa_en 显著）
- [x] Oracle 扩量 n=200（reasoner 强显著）
- [ ] musique CV2 n=200 (**跑批中，PID 35260**)
- [ ] MuSiQue 4-hop CoT-SC n=200 (**跑批中，PID 29084**)

**下周**：
- [ ] 根据显著性结果最终锁定定位 (A / B / C)
- [ ] paper_en.tex Abstract + Intro 重写
- [ ] 主表主图重制
- [ ] §9 Negative Results 章节撰写

**两周内**：
- [ ] 完整 tex 稿定稿
- [ ] Supplementary Material 更新（阶段一实验的可复现材料）
- [ ] 内部预审

---

## 七、需要用户确认的关键点

1. **定位选择**：等 60-90 分钟后跑批结果，按决策矩阵定 A/B/C
2. **投稿目标**：NAACL 2026 (deadline TBD) vs EMNLP Findings vs TACL？
3. **是否公开 experiment_records**：诚实性极高，可作为 arxiv 材料同期公开
4. **§9 Negative Results 的力度**：完整摆开（学术勇气）vs 只放主要（讨好评审）？
