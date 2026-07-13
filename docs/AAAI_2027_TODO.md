# AAAI 2027 投稿 TODO（今晚 2026-07-10 建立，DDL 2026-07-22 摘要）

> **2026-07-13 更新：**该文件是历史计划。当前科学口径以 `aaai_paper.tex` 和 `experiment_records.md` 顶部 correction 为准；当前 Overleaf 包为 `overleaf_submission.zip`。旧 additive Oracle waterfall 与 cross-check 因果性能解释不得用于投稿。

> **关键 DDL**：摘要 7/22（12 天后）、全文 7/29（19 天后）、补充材料 8/1（22 天后）

---

## 明天早上第一件事（必须先做）

### 1. 下载 AAAI 2027 Author Kit（获取 aaai2027.sty / aaai2027.bst）
访问 AAAI 2027 CFP 官网：https://aaai.org/aaai-conference/aaai-27/
- 找 "Author Kit" 或 "Style Files" 链接
- 下载 `aaai2027-latex-templates.zip`（大概叫这个名）
- 解压后拿到 `aaai2027.sty` 和 `aaai2027.bst`
- 复制到 `docs/` 目录（.gitignore 已排除，不会入库）

### 2. 迁移 paper_en.tex 内容到 aaai_paper.tex
- **策略**：不是替换 `aaai_paper.tex`，而是把 paper_en.tex 的**新叙事内容**逐节迁移到 aaai_paper.tex 骨架中
- 保留 aaai_paper.tex 的：`\documentclass`, `\usepackage{aaai2027}`, `\bibliographystyle{aaai2027}` 等 AAAI 官方样式设定
- 复制 paper_en.tex 的：Abstract、Introduction、Method、Experiments 4.1-4.10、Discussion、Conclusion
- 转换 `\begin{thebibliography}...\end{thebibliography}` → 使用 `\bibliography{references}` + BibTeX

### 3. 首次编译测页数
```bash
cd docs
tectonic aaai_paper.tex
# 或
pdflatex aaai_paper.tex && bibtex aaai_paper && pdflatex aaai_paper.tex && pdflatex aaai_paper.tex
```

**测量点**：
- 正文（不含 refs 和附录）多少页？必须 ≤ 7
- 总页数（含 refs）多少页？必须 ≤ 9

---

## 页数超出时的压缩优先级（根据编译结果动态调整）

**如果超 1-2 页**：
1. 压缩 §4.10（当前 ~600 words → ~300 words，或全部移 supplementary）
2. 合并 §4.5 Boundary Analysis 里的两个 boundary（HotpotQA + narrativeqa）
3. §4.7 Cost Comparison 表转两栏 tabular（省行）

**如果超 3+ 页**：
1. §4.10 完全移 supplementary
2. §4.6 Ablation 表压紧
3. §4.9 Cross-Model Generalization 转成一句话在 §5 Discussion 里
4. §4.8 Case Study 转到 supplementary（或删除 Case B 只留 A）

**绝不能砍**：
- Abstract, §1 Intro, §4.2 Main Results, §4.3 Oracle Analysis, §5 Discussion
- 三层显著性数字都必须保留

---

## 时间线（12 天计划，每天预算）

### Week 1 (7/11-7/17) — 论文冲刺

**Day 1 (7/11 周六)**：
- [ ] 下载 AAAI Author Kit + 迁移到 aaai_paper.tex
- [ ] 首次编译测页数
- [ ] 根据页数决定压缩策略
- [ ] 处理 2wikimqa n=200 结果（后台跑批完成后）

**Day 2 (7/12 周日)**：
- [ ] 压缩到 7 页正文（如需要）
- [ ] 制作 image4：LongBench 主表 F1 柱状图（vector, PDF）
- [ ] 更新 image1 (framework) 反映新叙事

**Day 3-4 (7/13-14 周一二)**：
- [ ] Related Work 补充：LongBench, RAG-QA, long-context LLM 相关引用
- [ ] Introduction 打磨（现在的 Contributions 4 条压到 3 条以省行数）
- [ ] Method 章节精简（3.4/3.5/3.6 是否合并）

**Day 5-6 (7/15-16 周三四)**：
- [ ] 内部审阅（找同学/导师看一遍）
- [ ] 修订
- [ ] Abstract 打磨到最精炼（150-200 words）

**Day 7 (7/17 周五)**：
- [ ] 冻结主要内容
- [ ] 检查所有数字与 experiment_records 一致
- [ ] 双盲审查（作者信息、致谢、self-citation）

### Week 2 (7/18-7/22) — 摘要提交

**Day 8-11 (7/18-21)**：
- [ ] 最后一轮打磨
- [ ] 补充 Reproducibility Checklist 草稿
- [ ] 准备 CMT/OpenReview 账号 & keywords

**Day 12 (7/22 周三)**：
- [ ] **19:59 前提交摘要** + 作者信息占位 + 分类

### Week 3 (7/23-7/29) — 全文提交

**Day 13-18**：
- [ ] 根据审稿人分配的领域微调 keywords
- [ ] 最后一轮 sanity check

**Day 19 (7/29 周三)**：
- [ ] **19:59 前提交全文**

### Week 4 (7/30-8/1) — 补充材料 & 代码

**Day 20-21**：
- [ ] Reproducibility Checklist 完整填写
- [ ] 匿名 code archive 打包：
  - 从当前 GOT repo 复制关键代码：`src/`, `data/prepare_data.py`, `experiments/run_parallel.py`, `experiments/_paired_stats_longbench.py`, `experiments/run_oracle.py`
  - **匿名化**：去除作者名、机构、`.env`、git 历史
  - README 匿名版（不提"清华/复旦/字节/腾讯"等真实机构线索）
  - 包含 `data/processed/` 里的 processed JSON（LongBench + MuSiQue 500 + 2417）
  - 包含 `experiments/results/` 里的 4 个 longbench dir + oracle_musique_8b
- [ ] Technical Supplement：Oracle 完整证明 / 更多分析 / 附录数据表
- [ ] Multimedia Archive：（可能不需要）
- [ ] **Day 22 (8/1) 19:59 前提交补充材料**

---

## 代码 archive 匿名化清单

**保留但脱敏**：
- `src/` 目录（所有 Python 代码）
- `data/prepare_data.py`
- `experiments/run_*.py`（主要跑批脚本）
- `experiments/_paired_stats_longbench.py`（复现主要统计结果）
- `data/processed/*.json`（LongBench + MuSiQue processed）
- `experiments/results/longbench_*_8b/*_results.json`
- `experiments/results/oracle_musique_8b/*_results.json`
- `experiments/results/musique_n500_8b/musique_{cot_sc,gers_cv2_fullctx}_results.json`

**必须删除**：
- `.env` / `.env.*`（DASHSCOPE_API_KEY）
- `docs/paper_en.tex`（正文另交，不放 code archive）
- `docs/experiment_records.md`（内部记录）
- `docs/paper_new_positioning.md`（策略文档）
- `docs/paper_draft.md`（旧稿）
- `.git/`（git 历史包含作者信息）
- 任何提及作者名、"zhouduomu"、真实机构的文件

**新建**：
- `README.md`（匿名版）：项目介绍 + 运行说明 + 数据来源引用
- `LICENSE`（选一个 open license, e.g. Apache 2.0）
- `requirements.txt`（已有）

---

## 未使用/存疑的实验（明天再决定归宿）

- **2wikimqa n=200 结果**（后台跑批中，明早查看）
  - 若 CV2 显著 → 主表加一行（Layer 1 三个显著）
  - 若不显著 → §4.10 additional diagnostic (like qasper)
- **qasper n=200 结果**（今晚跑完，dF1 = +0.018 n.s.）
  - 归宿：§4.10 已消化，作为"single-file scientific QA outside target regime"

---

## 关键数字备忘（论文里所有 SIG 结果的 canonical source）

**Layer 1**:
- LongBench multifieldqa_en (n=100): CV2 F1 0.415 vs CoT-SC 0.345, **dF1 +0.070, CI [+0.007, +0.132]**, McNemar p=0.114
- LongBench musique (n=193): CV2 F1 0.387 vs CoT-SC 0.325, **dF1 +0.063, CI [+0.001, +0.124]**, **McNemar p=0.049**
- LongBench narrativeqa (n=71): CV2 F1 0.208 vs CoT-SC 0.234, dF1 -0.027 (n.s., model-capacity ceiling)

**Layer 2**:
- Oracle-1 vs CoT-SC (MuSiQue 4-hop, n=200): F1 0.439 vs 0.363, **dF1 +0.075, CI [+0.017, +0.134]**, P(>0)=0.994
- Oracle-1 vs Baseline (n=199): F1 diff +0.091, CI [+0.041, +0.142], **McNemar p=0.002**
- Waterfall shares: Reasoner 66.2% / Graph-gen 18.4% / Retrieval 15.4%

**Layer 2 punch line (§4.3 Table 3)**:
- Model self-decomp (n=85): CV2 F1 0.348 vs CoT-SC 0.370, dF1 -0.022 (n.s.)
- Gold decomp (n=200): CV2 F1 0.439 vs CoT-SC 0.363, **dF1 +0.075 (SIG)**

**Layer 3 (Regime Boundaries)**:
- HotpotQA fair full-ctx (n=500): CoT-SC F1 0.716 vs CV2 0.683, dF1 -0.032 (F1 SIG in reverse)
- HotpotQA per-type: bridge 0.240 vs 0.218 (CV2 wins), comparison 0.562 vs 0.448 (CV2 wins)
- 2Wiki bridge_comp (n=100): CV2 0.524 vs CoT 0.905 (deep-bridge failure mode)
- Qwen-Plus (n=300): tied, dF1 -0.004

**Mechanism (§4.4 CS)**:
- Structural CS discrimination: -0.0035, AUROC 0.498
- With bidirectional cross-check: discrimination +0.0847, AUROC 0.589
- CS=1.0 bucket: 250/500 samples, but 156 (62%) still EM-wrong
