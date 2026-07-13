# GERS-CV2 论文待办交接文档

> **交接日期**：2026-07-13
> **项目目录**：`d:\code\GOT`
> **Git 仓库**：`https://github.com/fgrtqrwadtryehjsd/GOT.git`（最新 commit `de0e07a` 已 push）
> **投稿目标**：AAAI 2027，摘要 DDL 7/22，全文 DDL 7/29，补充材料 DDL 8/1

---

## 一、项目当前状态（已完成的部分）

### 论文叙事（三层结构，已锁定）
1. **Layer 1 — Method Win**：GERS-CV2 在 LongBench multifieldqa_en（F1 +0.070 SIG）和 musique（F1 +0.063, McNemar p=0.049 SIG）上显著优于 CoT-SC
2. **Layer 2 — Mechanism**：Oracle 分析（MuSiQue 4-hop n=200）证明 gold decomposition 显著超越 CoT-SC（F1 +0.075 SIG），66% 瓶颈在 reasoner
3. **Layer 3 — Regime Boundaries**：HotpotQA / narrativeqa / 2Wiki / qasper / 2wikimqa 五个边界数据点，诚实界定适用范围

### 已就绪的文件
| 文件 | 路径 | 状态 |
|------|------|------|
| 主论文 | `docs/aaai_paper.tex` | ✅ AAAI 2027 双栏格式，3961 words，7 页，已编译通过 |
| BibTeX | `docs/references.bib` | ✅ 12 条引用 |
| AAAI 样式 | `docs/aaai2027.sty` + `docs/aaai2027.bst` | ✅ 从 Author Kit 27 复制 |
| Framework 图 | `docs/figures/image1.pdf` | ✅ 旧版保留 |
| CS discrimination 图 | `docs/figures/image2.pdf` | ✅ 旧版保留 |
| Cross-checking 图 | `docs/figures/image3.pdf` | ✅ 旧版保留 |
| LongBench 主图 | `docs/figures/image4.pdf` | ✅ v2 已修复（去除 CI 重叠） |
| 2Wiki per-type 图 | `docs/figures/image5.pdf` | ✅ 旧版保留 |
| Oracle 瀑布图 | `docs/figures/image6.pdf` | ✅ v2 已修复（Y 轴留白+字号放大） |
| Regime map 图 | `docs/figures/image7.pdf` | ✅ v2 已修复（5 个数据点+字号放大） |
| 绘图脚本 | `docs/figures/generate_aaai_figures.py` | ✅ 可 re-run 生成 image4/6/7 |
| Overleaf zip | `docs/overleaf_aaai2027_gers_cv2.zip` | ✅ 385 KB，含 tex+bib+sty+7 图 |
| 实验记录 | `docs/experiment_records.md` | ✅ 完整（§1.1-§1.14 + §2.1-§2.4 + §4 定位） |
| 论文策略文档 | `docs/paper_new_positioning.md` | ✅ 决策矩阵+章节规划 |

### 尚未就绪的文件
| 文件 | 路径 | 状态 |
|------|------|------|
| 补充材料 | `docs/supplementary.tex` | ⚠️ 旧版 HotpotQA-centric，需更新到新叙事 |
| 可复现清单 | `docs/ReproducibilityChecklist.tex` | ⚠️ 模板存在，需按新数据填写 |
| 匿名代码 archive | 待打包 | ⚠️ 8/1 DDL 前必须完成 |

---

## 二、待办清单（按优先级排序）

### 🔴 P0 — 立刻做（影响 7/22 摘要提交）

#### 1. Overleaf 编译验证（30 分钟）
- [ ] 在 Overleaf 新建 project，上传 `docs/overleaf_aaai2027_gers_cv2.zip`
- [ ] 编译器选 pdfLaTeX
- [ ] 编译 `aaai_paper.tex`，确认：
  - 总页数 ≤ 7 页（正文）+ refs（AAAI 允许 7 正文 + 2 refs/appendix = 9 总）
  - Figure 7 (regime map) 出现在 §4.6
  - image4 柱子上无文字重叠
  - image6 顶部 "0.839" 有呼吸空间
  - 无编译 error（warning 可以接受）
- [ ] 如有编译 error，常见修复：
  - `aaai2027.sty not found` → 确认 sty 文件在根目录
  - `natbib` 报错 → 确认编译器是 pdfLaTeX（不是 XeLaTeX）
  - BibTeX 未生效 → 在 Overleaf 设置里开启 "Run BibTeX automatically"
  - `\citep` undefined → 确认 `\usepackage{natbib}` 在 preamble 里

#### 2. 摘要定稿（1 小时）
- [ ] 当前 Abstract ~240 words，AAAI 建议压到 200 words 以内
- [ ] 核心数字必须保留：
  - multifieldqa_en: F1 +0.070, CI [+0.007, +0.132]
  - musique: F1 +0.063, McNemar p=0.049
  - Oracle-1 vs CoT-SC: F1 +0.075, CI [+0.017, +0.134]
  - CS: -0.0035 → +0.0847
- [ ] 7/22 19:59 CST 前在 CMT/OpenReview 提交 Title + Abstract

### 🟡 P1 — 本周做（影响 7/29 全文提交）

#### 3. 补充材料更新 `supplementary.tex`（2 小时）
- [ ] 旧版是 HotpotQA-centric 叙事，需要更新到 LongBench + Oracle 三层叙事
- [ ] 应包含：
  - Qualitative positioning 表（CoT/CoT-SC/ToT/GoT/RwG/MoDeGraph/GERS 对比）
  - CS bucket diagnostics（HotpotQA n=500 完整分桶表）
  - HotpotQA per-type EM（bridge vs comparison）
  - Computational cost 表（HotpotQA n=500 参考成本）
  - 5-subset LongBench 完整 paired stats 表（含 qasper + 2wikimqa 的 n.s. 结果）
  - Oracle waterfall 完整数据（n=200 四个 config）
  - Rejected variants 详细数据（EASV, IDD, BM25 retrieval, GF-GERS）
  - Implementation details（模型 API、workers=4、seed=42、统计方法）
- [ ] 数据来源：`docs/experiment_records.md` §1.1-§1.14 + §2.1-§2.4
- [ ] 编译为独立 PDF（standalone `\documentclass[11pt]{article}`）

#### 4. 可复现清单填写 `ReproducibilityChecklist.tex`（1 小时）
- [ ] 从 AAAI Author Kit 的模板复制（`C:\Users\spiralzhou\Downloads\AuthorKit27.zip` 里有）
- [ ] 按新数据填写每一项：
  - 数据集：HotpotQA, LongBench (5 subsets), MuSiQue, 2WikiMultiHopQA, GSM8K
  - 代码：将在 8/1 前提供匿名 archive
  - 超参：Qwen3-8B, DashScope API, workers=4, N=3 (CoT-SC), K=3 (GERS-SC), B=10000 (bootstrap), seed=42
  - 算力：DashScope API（无本地 GPU）
  - 统计：paired bootstrap 95% CI, McNemar chi2-cc
- [ ] 编译为独立 PDF

#### 5. Related Work 补充引用（1 小时）
- [ ] 在 `references.bib` 添加：
  - RAG survey：Lewis et al. 2020 (RAG) 或 Gao et al. 2024 (RAG survey)
  - Long-context benchmark：RULER (Hsieh et al. 2024)
  - Decomposition QA：DecomP (Perez et al. 2020), Least-to-Most (Zhou et al. 2022)
- [ ] 在 `aaai_paper.tex` §2 Related Work 里加 1-2 段引用这些工作
- [ ] 重新编译确认页数不超

#### 6. 全文润色 + 内部审阅（2-3 天）
- [ ] 找同学/导师审阅 PDF（Overleaf 可直接分享 link 或下载 PDF）
- [ ] 审阅重点（可给同学的 checklist）：
  1. Abstract 是否清晰传达三层叙事？
  2. §4.2 主表数字是否准确？（与 experiment_records.md 对照）
  3. §4.3 Oracle 分析的逻辑链是否完整？（gold decomp → +0.075 SIG → model gap → cross-check 填补部分 gap）
  4. §4.6 Regime Boundary 是否诚实但不 self-sabotaging？
  5. §4.10 Additional Diagnostic 是否让 reviewer 觉得"thorough"而不是"cluttered"？
  6. 图片是否清晰可读？（特别关注 image4/6/7 在双栏 PDF 里的字号）
  7. 有无遗漏的作者/机构信息？（双盲检查）
- [ ] 根据反馈修订

#### 7. 双盲检查（30 分钟）
- [ ] 全文搜索以下关键词，确保无作者/机构泄漏：
  - "zhouduomu", "264154463", "qq.com"
  - "清华", "Tsinghua", "THU"
  - "腾讯", "Tencent"
  - 任何 GitHub/Gitee/个人主页 URL
- [ ] 确认 `\author{Anonymous Submission}` 且 `\affiliations{}` 为空
- [ ] 确认无 `\thanks` 致谢

### 🟢 P2 — 下周做（影响 8/1 补充材料 + 代码提交）

#### 8. 匿名代码 Archive 打包（3 小时）
- [ ] 新建目录 `anonymous_code/`
- [ ] 复制以下文件（去除作者信息）：
  ```
  anonymous_code/
  ├── README.md          ← 匿名版（不提作者名/机构）
  ├── LICENSE            ← Apache 2.0
  ├── requirements.txt   ← 从项目根目录复制
  ├── setup.py           ← 从项目根目录复制
  ├── src/               ← 整个目录
  ├── data/
  │   ├── prepare_data.py
  │   └── download_longbench.py
  ├── experiments/
  │   ├── run_parallel.py
  │   ├── run_quick_exp.py
  │   ├── run_oracle.py
  │   ├── run_comparison.py
  │   ├── _paired_stats_longbench.py
  │   └── _extend_musique_cot_sc_hop4.py
  ├── data/processed/    ← processed JSON（LongBench 5 + MuSiQue + HotpotQA + GSM8K + 2Wiki）
  └── experiments/results/
      ├── longbench_multifieldqa_en_8b/
      ├── longbench_musique_8b/
      ├── longbench_narrativeqa_8b/
      ├── longbench_qasper_8b/
      ├── longbench_2wikimqa_8b/
      ├── oracle_musique_8b/
      └── musique_n500_8b/
  ```
- [ ] **必须删除**：
  - `.env` / `.env.*`（含 DASHSCOPE_API_KEY）
  - `.git/`（git 历史含作者信息）
  - `docs/`（论文另交，不放 code archive）
  - `GOT_improve.md`, `GERS_*.md`（内部笔记）
  - `AAAI_TODO_NEXT.md`, `docs/AAAI_2027_TODO.md`（策略文档）
  - `docs/paper_new_positioning.md`（策略文档）
  - `experiments/results/` 里的 `_run_log*.txt`, `_err_log*.txt`, `_active_pids.json`（运行时日志）
  - `experiments/results/` 里的 smoke test 目录（`*_smoke*/`, `b_smoke*/`, `gf_gers_smoke*/` 等）
  - `data/hf_cache/`, `data/hotpotqa_raw/`, `data/2wiki_raw/`, `data/longbench_raw/`, `data/musique_raw/`（原始大数据，.gitignore 已排除）
- [ ] 打成 zip，8/1 19:59 CST 前上传

#### 9. Overleaf 提交 zip 更新（每次改完论文后做）
- [ ] 更新 `docs/overleaf_upload/` 里的文件
- [ ] 重新打包 `docs/overleaf_aaai2027_gers_cv2.zip`
- [ ] 在 Overleaf 上传覆盖（或新建 project）
- [ ] **注意**：如果加了 `supplementary.tex` 和 `ReproducibilityChecklist.tex`，需要在 Overleaf 里分别设置 "Main document" 编译三个 PDF

---

## 三、关键时间线

| 日期 | 事项 |
|------|------|
| **7/22 19:59 CST** | 摘要提交（Title + Abstract + 作者占位 + Keywords） |
| **7/29 19:59 CST** | 全文提交（aaai_paper.tex 编译的 PDF） |
| **8/1 19:59 CST** | 补充材料 + 代码 archive + Reproducibility Checklist |
| 10/19-10/26 | Rebuttal |
| 12/1 | 录用通知 |
| 2027/2/16-23 | 蒙特利尔会议 |

---

## 四、关键数字速查（论文里所有 SIG 结果，必须与 experiment_records.md 一致）

### Layer 1 — LongBench 主表（§4.2 Table 1）
| Subset | n | CoT-SC F1 | CV2 F1 | dF1 | F1 95% CI | McNemar p | Verdict |
|--------|---|-----------|--------|-----|-----------|-----------|---------|
| multifieldqa_en | 100 | 0.345 | 0.415 | +0.070 | [+0.007, +0.132] | 0.114 | F1 SIG |
| musique | 193 | 0.325 | 0.387 | +0.063 | [+0.001, +0.124] | 0.049 | EM+F1 SIG |
| narrativeqa | 71 | 0.234 | 0.208 | -0.027 | [-0.082, +0.026] | 0.617 | n.s. |

### Layer 1 扩展 — §4.10 Additional Diagnostic（5-subset 完整）
| Subset | n | dF1 | F1 95% CI | McNemar p | Verdict |
|--------|---|-----|-----------|-----------|---------|
| qasper | 200 | +0.018 | [-0.016, +0.054] | 1.000 | n.s. |
| 2wikimqa | 195 | -0.021 | [-0.076, +0.035] | 0.324 | n.s. |

### Layer 2 — Oracle（§4.3 Table 2 + Table 3）
| Comparison | n | F1 diff | F1 95% CI | McNemar p | Verdict |
|------------|---|---------|-----------|-----------|---------|
| Oracle-1 vs Baseline | 199 | +0.091 | [+0.041, +0.142] | 0.002 | F1+EM SIG |
| Oracle-1 vs CoT-SC (4-hop) | 200 | +0.075 | [+0.017, +0.134] | 0.188 | F1 SIG |
| Model self-decomp vs CoT-SC | 85 | -0.022 | [-0.115, +0.071] | 1.000 | n.s. |

Oracle waterfall shares: Reasoner 66.2% / Graph-gen 18.4% / Retrieval 15.4%

### Layer 3 — Regime Boundaries（§4.6）
- HotpotQA fair full-ctx n=500: CoT-SC F1 0.716 vs CV2 0.683, dF1 -0.032
- HotpotQA per-type: bridge 0.240 vs 0.218, comparison 0.562 vs 0.448
- 2Wiki bridge_comp n=100: CV2 0.524 vs CoT 0.905
- Qwen-Plus n=300: tied, dF1 -0.004

### Mechanism — CS Calibration（§4.4）
- Structural CS: discrimination -0.0035, AUROC 0.498
- With cross-check: discrimination +0.0847, AUROC 0.589
- CS=1.0 bucket: 250/500 samples, 156 (62%) EM-wrong

---

## 五、常用命令

### 重新生成图片
```bash
cd d:\code\GOT\docs\figures
python generate_aaai_figures.py
```

### 跑配对统计检验
```bash
cd d:\code\GOT
python experiments/_paired_stats_longbench.py
```

### 重新打包 Overleaf zip
```powershell
cd d:\code\GOT
Copy-Item docs\aaai_paper.tex docs\overleaf_upload\ -Force
Copy-Item docs\figures\image*.pdf docs\overleaf_upload\figures\ -Force
Compress-Archive -Path "docs\overleaf_upload\*" -DestinationPath docs\overleaf_aaai2027_gers_cv2.zip -CompressionLevel Optimal -Force
```

### Git 提交 + 推送
```bash
cd d:\code\GOT
git add <changed files>
git commit -m "描述"
git push origin main
```

---

## 六、注意事项

1. **AAAI 禁止的 LaTeX 包**：hyperref, multicol, geometry, setspace, float, balance, flushend, fullpage, wrapfig, tabu, titlesec, indentfirst, savetrees, CJK
2. **AAAI 禁止的命令**：`\vspace{-`, `\newpage`, `\clearpage`, `\pagebreak`, `\baselinestretch`, `\tiny`, `\addtolength`
3. **双盲**：不能出现作者名、机构、致谢、GitHub 链接
4. **页数**：正文 ≤ 7 页，总（含 refs）≤ 9 页，Reproducibility Checklist 不计页数
5. **补充材料**：reviewer 不强制看，核心论证必须在正文 7 页内
6. **代码**：8/1 前必须交，"录用后开源"承诺不再被接受为可复现性证据
7. **绘图风格**：serif 字体、中性色、无 top/right spine、单栏宽度 ~3.4 inches、PDF vector
