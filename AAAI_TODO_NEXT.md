# AAAI 2027 待办清单（2026-07-10 21:xx 起）

> **2026-07-13 更新：**本文档保留历史日程，但科学口径以 `docs/aaai_paper.tex` 和 `docs/experiment_records.md` 顶部的 submission-audit correction 为准。当前编译包是 `docs/overleaf_submission.zip`；旧 `66/18/15` Oracle waterfall 与“cross-check 导致端任务增益”的解释已废弃。

**关键时间线**：
- 📅 摘要 DDL：**2026-07-22 19:59 CST** — 剩 **12 天**
- 📅 全文 DDL：**2026-07-29 19:59 CST** — 剩 **19 天**
- 📅 补充材料 + 代码 DDL：**2026-08-01 19:59 CST** — 剩 **22 天**
- 🎯 会议：2027-02-16 加拿大蒙特利尔

---

## 🔴 明天早上第一优先（30 分钟内）

### 1. Overleaf 测编译 + 页数
- [ ] 上传到 Overleaf：`docs/aaai_paper.tex` + `docs/references.bib` + `docs/aaai2027.sty` + `docs/aaai2027.bst` + `docs/figures/` 整个目录
- [ ] 编译器选 pdfLaTeX，跑一次完整编译（先 pdfLaTeX → BibTeX → pdfLaTeX 两次）
- [ ] **记录真实正文页数**（含 refs 总页数）→ 回来告诉我
- [ ] 目标：正文 ≤ 7 页，总 ≤ 9 页

### 2. 检查图片视觉效果
- [ ] 打开 `docs/figures/image4.pdf`（LongBench 主图）→ 字号/颜色/布局是否顺眼
- [ ] 打开 `docs/figures/image6.pdf`（Oracle 瀑布图）
- [ ] 打开 `docs/figures/image7.pdf`（regime map 备用图）
- [ ] 如需调整：改 `docs/figures/generate_aaai_figures.py` 顶部的 `rcParams`，重跑

### 3. 处理 2wikimqa 后台跑批
- [ ] 检查 `experiments/results/longbench_2wikimqa_8b/` 是否已完成（PID 33132）
- [ ] 如完成，跑 `python experiments/_paired_stats_longbench.py` 看是否显著
- [ ] 显著 → 加入主表 Table 1（4 个 subsets）
- [ ] 不显著 → 加入 §4.10 additional diagnostic 里

---

## 🟡 Week 1 剩余（7/11–7/17）—— 论文冲刺

### 页数管理（依赖 Overleaf 结果）

**如果正文 ≤ 6 页**（有空间）：
- [ ] 加入 image7 (regime map) 到 §4.6 Regime Boundary Analysis
- [ ] Related Work 补充 1-2 段（RAG 相关工作、long-context reasoning）
- [ ] Introduction 加一段引出 RAG 场景

**如果正文 7 页刚好**：
- [ ] 微调格式，检查每张图有无跨栏问题
- [ ] Section 编号是否正确（`\setcounter{secnumdepth}{1}`）

**如果正文 > 7 页**（超页）：
- [ ] 压缩优先级顺序（我按需帮你做）：
  1. 砍 §4.10 Additional Diagnostic Experiments（转 supplementary）
  2. §4.6 Regime Boundary Analysis 合并 3 个 boundary 到 1 段
  3. §4.8 Case Study 精简（Case B 保留、Case A 一句话）
  4. §4.7 Cost 表转两栏 tabular
  5. 图 1/2/3/5 择一转 supplementary

### 内容优化

- [ ] **Related Work 补充引用**：
  - RAG 场景相关：Lewis et al. RAG 2020, Gao et al. RAG survey
  - Long-context LLM 相关：RULER benchmark, LongBench v2
  - DAG-based QA 相关：DecomP, Least-to-Most
- [ ] **Method 章节可能合并**：
  - `\subsection{Reasoning-State Graph}` (5 行) 和 `\subsection{Bidirectional Sub-Answer Cross-Checking}` 是否合并
- [ ] **Introduction 检查**：
  - 是否有太多"we propose"，AAAI reviewer 不喜欢重复
  - 4 个 Contributions 是否压到 3 个
- [ ] **Abstract 精炼**：
  - 目前 ~240 words，AAAI 建议 150-200，可能需压
- [ ] **双盲检查**：
  - 全文搜索 "zhouduomu"、"腾讯"、"清华"、"THU"、"BUAA" 等作者/机构线索
  - 致谢部分是否为空

---

## 🟢 Week 2（7/18–7/22）—— 摘要提交

- [ ] Day 1-2：找同学/导师内部审阅一遍
- [ ] Day 3-4：根据反馈修订
- [ ] Day 5（7/22）：**19:59 CST 前提交摘要**
  - 在 CMT/OpenReview 上注册账号
  - 填写：Title + Abstract + 作者信息占位（保匿名）+ Keywords + Category
  - **注意**：只提交 Title + Abstract，不提交 PDF

---

## 🟢 Week 3（7/23–7/29）—— 全文提交

- [ ] Day 1-6：最后打磨、按 Reviewer Bidding 微调关键词
- [ ] Day 7（7/29）：**19:59 CST 前提交全文 PDF**

---

## 🔵 Week 4（7/30–8/1）—— 补充材料 & 代码

### Reproducibility Checklist
- [ ] 参考 `docs/ReproducibilityChecklist.tex` 或 Author Kit 的模板
- [ ] 填写每一项（数据集、代码、超参、算力、seed、复现步骤）
- [ ] 单独上传（不计入正文页数）

### 匿名代码 Archive 打包

**新建 anonymous_code/ 目录，包含以下（去除作者信息、机构、API keys）**：

- [ ] `README.md`（匿名版）：项目介绍 + 数据来源引用 + 运行说明
- [ ] `LICENSE`（Apache 2.0 或 MIT）
- [ ] `requirements.txt`（同现有的）
- [ ] `src/`（整个目录）
- [ ] `data/prepare_data.py`
- [ ] `data/download_longbench.py`
- [ ] `experiments/`：
  - `run_parallel.py`
  - `run_quick_exp.py`
  - `run_oracle.py`
  - `run_comparison.py`
  - `_paired_stats_longbench.py`
  - `_extend_musique_cot_sc_hop4.py`
- [ ] `data/processed/`：processed JSON（去除 hf_cache）
- [ ] `experiments/results/`：核心结果 JSON（去除日志、smoke 结果、PID 文件）
- [ ] `tests/`（保留）
- [ ] `setup.py`

**必须删除**：
- [ ] `.env` / `.env.*`（DASHSCOPE_API_KEY 等）
- [ ] `docs/aaai_paper.tex`（论文另交，不放 code archive）
- [ ] `docs/experiment_records.md`
- [ ] `docs/paper_new_positioning.md`
- [ ] `docs/paper_draft.md`
- [ ] `docs/paper.docx`
- [ ] `docs/paper_cn.tex`
- [ ] `docs/AAAI_2027_TODO.md`（本文档 & 类似策略文档）
- [ ] `docs/GERS_*.md`, `GOT_improve.md`（内部改进笔记）
- [ ] `.git/`（git 历史含作者信息！）
- [ ] 任何提及具体作者名/机构的文件

### Technical Supplement PDF

- [ ] 补充材料 PDF（可选，reviewer 不强制看）：
  - 完整证明或推导（如果有数学部分）
  - 附加实验数据（如可扩展的 hop-curve n=500 完整表）
  - Prompt 模板（各 LLM prompt 具体文本）
  - Case Study 完整 traces（Case A/B 完整前向反向 sub-answers）
  - 更多 CS diagnostics 分桶（HotpotQA 500 samples 10-bucket 表）
- [ ] Day 22（8/1）：**19:59 CST 前提交补充材料 + 代码 archive**

---

## 📊 关键数字 sanity check（提交前必查）

以下数字必须与 `docs/experiment_records.md` 一致：

**Layer 1 LongBench**：
- multifieldqa_en n=100：CV2 F1 0.415 vs CoT-SC 0.345，dF1 **+0.070**，CI **[+0.007, +0.132]**
- musique n=193：CV2 F1 0.387 vs CoT-SC 0.325，dF1 **+0.063**，CI **[+0.001, +0.124]**，McNemar **p=0.049**
- narrativeqa n=71：dF1 −0.027，n.s.

**Layer 2 Oracle (MuSiQue 4-hop)**：
- Oracle-1 vs CoT-SC n=200：dF1 **+0.075**，CI **[+0.017, +0.134]**
- Oracle-1 vs Baseline n=199：dF1 +0.091，CI [+0.041, +0.142]，McNemar p=0.002
- Waterfall 分配：Reasoner **66.2%** / Graph-gen **18.4%** / Retrieval **15.4%**
- Model self-decomp vs Gold-decomp gap：~0.10 F1

**Layer 3 边界**：
- HotpotQA fair full-ctx n=500：CoT-SC 微赢 dF1 -0.032
- Qwen-Plus n=300：平局 dF1 -0.004
- 2Wiki bridge_comp n=100：CV2 0.524 vs CoT 0.905

**Mechanism (CS Calibration)**：
- Structural CS：discrimination **-0.0035**，AUROC **0.498**
- With cross-check：discrimination **+0.0847**，AUROC **0.589**
- CS=1.0 bucket：250/500 samples，其中 156 (62%) EM-wrong

---

## 🎯 决策点（明天你需要告诉我的）

1. **Overleaf 编译真实页数是多少？**
2. **2wikimqa 后台跑批结果如何？**（显著/不显著）
3. **图片视觉效果如何？**（image4/6/7 是否需要调整）
4. **是否有同学/导师帮忙审阅？**（Week 1 内部审阅关键）

---

## 📁 关键文件位置备忘

**投稿版**：
- `docs/aaai_paper.tex` — AAAI 主论文（就绪，等 Overleaf 编译）
- `docs/references.bib` — 12 条 bibitem
- `docs/figures/image4.pdf` — LongBench 主图（NEW）
- `docs/figures/image6.pdf` — Oracle 瀑布图（NEW）
- `docs/figures/image7.pdf` — Regime map（备用）
- `docs/figures/image1,2,3,5.pdf` — 已有的旧图（保留）
- `docs/aaai2027.sty` `docs/aaai2027.bst` — AAAI 官方样式（.gitignore 排除）

**内部参考**：
- `docs/experiment_records.md` — 完整实验记录，含未在论文中出现的 §1.7-1.11 等
- `docs/paper_new_positioning.md` — 论文重定位策略文档
- `docs/AAAI_2027_TODO.md` — 22 天日程表
- `AAAI_TODO_NEXT.md`（本文档） — 分优先级 TODO

**辅助工具**：
- `docs/figures/generate_aaai_figures.py` — AAAI 风格绘图脚本
- `experiments/_paired_stats_longbench.py` — 显著性检验脚本

---

## 🌙 今日成果最终盘点

**从"论文被自己实验推翻"到"AAAI 就绪版完稿" 一整天**：

| 阶段 | 状态 | 关键动作 |
|------|------|----------|
| Morning | 认识到旧 story 被自己 §1.6 推翻 | 决定 pivot |
| Afternoon | 跑 LongBench 3 subsets 验证 H1 | multifieldqa/musique 显著 |
| Evening | Oracle n=200 扩量 + CoT-SC 4-hop 扩量 | Oracle-1 vs CoT-SC 显著 (+0.075 F1) |
| Late | 论文重写为三层叙事 + AAAI 模板迁移 | aaai_paper.tex 就绪，3961 words |
| Final | AAAI 风格图生成 + git push | 全部就绪，7 个 commit push 到远端 |

**明天开工只有一件事：Overleaf 编译测页数。** 睡吧。
