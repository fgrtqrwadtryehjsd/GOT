"""生成论文 Word 版（.docx）
基于 docs/paper_draft.md 内容，用 python-docx 生成完整排版的 Word 文档。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def set_cell_border(cell):
    """给单元格加边框"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        border = OxmlElement(f'w:{edge}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:color'), '000000')
        tcBorders.append(border)
    tcPr.append(tcBorders)


def set_font(run, name='宋体', size=10.5, bold=False, en_name='Times New Roman'):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = en_name
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), name)


def add_para(doc, text, size=10.5, bold=False, align=None, indent=True, space_after=6):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run(text)
    set_font(run, size=size, bold=bold)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    sizes = {1: 14, 2: 12, 3: 11}
    run = p.add_run(text)
    set_font(run, size=sizes.get(level, 11), bold=True)
    return p


def add_table(doc, headers, rows, caption=None):
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_before = Pt(8)
        cp.paragraph_format.space_after = Pt(4)
        run = cp.add_run(caption)
        set_font(run, size=10, bold=True)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 表头
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_font(run, size=9.5, bold=True)
        set_cell_border(cell)
    # 数据行
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ''
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            set_font(run, size=9.5)
            set_cell_border(cell)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def main():
    doc = Document()
    # 页边距
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # 标题
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(6)
    run = title.add_run('图约束推理链与子答案双向交叉验证：面向多跳推理的大模型方法')
    set_font(run, size=16, bold=True)

    # 作者
    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.paragraph_format.space_after = Pt(12)
    run = author.add_run('zhouduomu')
    set_font(run, size=11)

    # 摘要（英文，会议要求）
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run('Abstract: ')
    set_font(run, size=10.5, bold=True)
    run = p.add_run('Large language models (LLMs) suffer from error accumulation on multi-hop reasoning tasks. Existing graph-enhanced methods model reasoning as a DAG, but their consistency checks rely on pure graph-theoretic structural metrics that cannot reflect reasoning correctness, causing multi-path selection to degenerate into random sampling. We propose GERS, which models reasoning as a dependency DAG executed in topological order. The core contribution is sub-answer bidirectional cross-validation: using the final answer + context as an anchor, we re-derive each sub-question in reverse and compare forward/backward sub-answers, upgrading the Consistency Score from "is the graph legal" to "is the reasoning content self-consistent". On 500 HotpotQA samples, this raises the Consistency Score\'s correct/wrong discrimination from -0.0035 to +0.0847; the best configuration GERS-CV achieves EM=0.302, F1=0.413, outperforming CoT-SC by 4pt F1 (McNemar p=0.029). We honestly characterize the method\'s boundary: graph-level multi-path self-consistency brings no extra gain on medium-difficulty multi-hop, and error propagation limits GERS on deep bridging-comparison questions.')
    set_font(run, size=10)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run('Keywords: ')
    set_font(run, size=10.5, bold=True)
    run = p.add_run('Large Language Models; Multi-hop Reasoning; Chain-of-Thought; Graph Representation; Bidirectional Cross-Validation; Consistency Verification')
    set_font(run, size=10)

    # 1 引言
    add_heading(doc, '1 引言', 1)
    add_para(doc, '多跳推理要求模型跨多个证据片段组合推导出答案，是衡量 LLM 复杂推理能力的核心任务之一。Chain-of-Thought（CoT）提示通过引导模型输出中间推理步骤，显著提升了 LLM 的表现。然而，标准 CoT 以线性文本串接推理步骤，存在两类核心不足：')
    add_para(doc, '1. 非线性依赖缺失：CoT 以线性文本串接推理步骤，无法表达子问题间的分支、合流与并行依赖。当推理路径本应是 DAG 而非链时，线性化导致结构信息丢失。')
    add_para(doc, '2. 质量无差别投票：CoT-SC 通过多次采样取众数答案，但投票是"等权"的，无法利用推理路径本身的质量信息。')
    add_para(doc, '近期工作尝试用图结构增强推理：GoT 提出思维图但不保证执行顺序；RwG 将上下文结构化为实体关系图但未逐步执行；MoDeGraph 构建多跳依赖图但仍属 prompt 方法；GoV 将推理建模为 DAG 进行验证但侧重语义验证。这些方法普遍将图结构作为 Prompt 装饰或事后验证，而非真正参与推理执行与质量评估。')
    add_para(doc, '针对上述问题，本文提出 GERS，将推理显式建模为推理依赖图（DAG）并按拓扑序执行。我们发现，单纯用图论结构性质计算 Consistency Score 是无效的——LLM 生成的分解图几乎都是合法 DAG，导致所有样本得分趋同（聚集在 0.6~0.7），多路选优退化为随机抽取（500 条实测对错区分度仅 -0.0035）。为此，本文提出核心创新子答案双向交叉验证：以最终答案 + 上下文为锚，反向逐个重答每个子问题，对比正向与反向子答案的一致性，将 Consistency Score 从"图是否合法"升级为"推理内容是否自洽"。这一机制是 DAG 结构独有的能力——线性 CoT 没有可独立校验的子问题结构，无法实现双向校验。')
    add_para(doc, '本文的主要贡献：（1）子答案双向交叉验证：将 CS 对错区分度从 -0.0035 修复到 +0.0847，DAG 独有；（2）性能验证：GERS-CV 在 HotpotQA 500 条上 EM=0.302、F1=0.413，McNemar p=0.029 显著优于 CoT-SC；（3）诚实的适用边界：图级多路自一致性无额外增益，bridge_comparison 存在错误传播局限；（4）工程贡献：指标修复、答案类型回扣、提取公平性。')

    # 2 相关工作
    add_heading(doc, '2 相关工作', 1)
    add_para(doc, '思维链推理。Wei 等提出 CoT；Wang 等提出 Self-Consistency（CoT-SC），等权投票未利用推理路径质量；Yao 等提出 Tree of Thoughts（ToT），子问题间无显式依赖建模。这些方法均基于线性或树形结构。')
    add_para(doc, '图结构增强推理。GoT 将推理建模为图但侧重 Prompt 工程；RwG 将图作为一次性输入；MoDeGraph 本质仍是 prompt 方法；StepChain 子问题是线性序列而非 DAG。GERS 将推理依赖图与执行流程深度耦合，并用图结构质量作为多候选路径的选择信号。')
    add_para(doc, '推理验证与一致性校验。GoV 侧重语义验证而非图结构数学性质；Process Reward Models 依赖大量标注；DAG-Math 用节点置信度权重处理不确定步骤。GERS 用图论算法做结构化校验，并设计双向交叉验证使校验结果直接影响最终答案。')

    # 3 方法
    add_heading(doc, '3 方法', 1)
    add_heading(doc, '3.1 总体框架', 2)
    add_para(doc, 'GERS 推理流程（图1）：输入问题 Q + 上下文 C → ① 分解为子问题 {q_i} 及依赖 → ② 构建 DAG G=(V,E) → ③ 拓扑排序 τ → ④ 逐步执行生成子答案 a_i → ⑤ 汇总生成最终答案 A（答案类型回扣）→ ⑥ 一致性校验 S = 0.3·S_struct + 0.7·S_crossval →（GERS-SC：K 条 DAG 选 S 最高者）→ 输出 A + G + S。')

    add_heading(doc, '3.2 推理状态图表示', 2)
    add_para(doc, '节点类型：FactNode（事实/原始问题）、StepNode（子问题及答案）、ConclusionNode（最终结论）。边类型：DeriveEdge（推导）、SupportEdge（支撑）、ConflictEdge（互斥）。LLM 将问题分解为子问题列表并标注依赖，据此创建节点与边。')

    add_heading(doc, '3.3 图约束链路生成', 2)
    add_para(doc, '路径规划：拓扑排序得执行顺序 τ，确保每个子问题被回答时其前驱已获得答案。逐步执行与汇总：按拓扑序逐个回答子问题，依赖传递，最后汇总生成最终答案 A。答案类型回扣：约束最终答案匹配原问题要求的实体类型（如问 film 答电影名而非中间实体导演名）。')

    add_heading(doc, '3.4 一致性校验与子答案双向交叉验证', 2)
    add_para(doc, '结构层 Consistency Score（基线）：S_struct = 0.35·C_conn + 0.30·C_cycle + 0.35·C_cov（连通性、无环性、覆盖度），含推理链长度惩罚。')
    add_para(doc, '结构层的局限：LLM 生成的分解图几乎都是合法 DAG，导致 S_struct 高度聚集（500 条均值 0.662，stdev 0.098，区分度 -0.0035），无法区分推理好坏。')
    add_para(doc, '子答案双向交叉验证（核心创新）：以"最终答案 A + 上下文 C"为锚反向逐个重答子问题，得反向子答案 {a\'_i}，对比正反向一致性：S_crossval = Σ w_i·1[match(a_i, a\'_i)] / Σ w_i。反向 Prompt 要求"基于上下文独立重答，勿照抄最终答案"以缓解 confirmation bias。该机制为 DAG 独有。')
    add_para(doc, '新 Consistency Score：S = 0.3·S_struct + 0.7·S_crossval。结构层权重降至 0.3，内容层 0.7。')

    add_heading(doc, '3.5 子答案置信度加权汇总', 2)
    add_para(doc, '每个子答案置信度由轻量启发式估计（零额外 LLM 调用），综合考量答案非空、不含不确定信号、核心实体在上下文落地、是否纯数字。低于阈值 0.5 的子答案在汇总 Prompt 中标注 [LOW CONFIDENCE]，降低单步错误传播。')

    add_heading(doc, '3.6 图级自一致性（GERS-SC）', 2)
    add_para(doc, '生成 K 条不同推理 DAG（分解温度 T∈{0.3,0.5,0.7}），用 CS 选得分最高者：A* = argmax_k S(G_k)。其有效性严格依赖 CS 具备区分度——双向交叉验证正是使其生效的前提。')

    # 4 实验
    add_heading(doc, '4 实验', 1)
    add_heading(doc, '4.1 实验设置', 2)
    add_para(doc, '数据集：HotpotQA（500 条，bridge 404 + comparison 96）；2WikiMultiHopQA（100 条，四类题型）。基线：Zero-Shot、Standard CoT、CoT-SC(N=3)、CoT-SC+GERS 重排。本文方法：GERS+自适应、GERS-SC(K=3)、GERS-CV（+双向交叉验证）、GERS-CV2（+置信度加权，最优）。模型 Qwen3-8B，4 worker 并行，零失败。指标：EM、F1、CS。所有主结果报告 bootstrap 95% CI 与 McNemar 检验。')
    add_para(doc, '评估公平性保障：为避免评估链路缺陷扭曲对比，本文统一三项处理——(1) 修复 EM 双向子串匹配 bug（原使数值答案虚高）；(2) 所有方法采用统一简洁答案提取与归一化；(3) GERS 汇总强制答案类型回扣。这确保所有方法在同一公平口径下对比，GERS 增益来自方法本身而非评估偏差。')

    add_heading(doc, '4.2 主实验结果', 2)
    add_table(doc, ['方法', 'EM', 'F1', 'CS'], [
        ['Zero-Shot', '0.276', '0.389', '—'],
        ['Standard CoT', '0.264', '0.368', '—'],
        ['CoT-SC (N=3)', '0.262', '0.373', '—'],
        ['CoT-SC+GERS 重排', '0.264', '0.372', '—'],
        ['GERS+自适应', '0.284', '0.395', '0.996'],
        ['GERS-SC (K=3)', '0.282', '0.398', '0.662'],
        ['GERS-CV（+双向交叉验证）', '0.298', '0.409', '0.782'],
        ['GERS-CV2（+置信度加权）', '0.302', '0.413', '0.777'],
    ], caption='表 1  主实验结果（HotpotQA, n=500）')
    add_para(doc, 'GERS-CV2 取得最优 EM=0.302、F1=0.413，相比 CoT-SC EM +4pt、F1 +4pt。双向交叉验证（0.284→0.298）带来 +1.4pt EM 增益，是核心创新的直接贡献。HotpotQA 上 Zero-Shot（0.276）已接近 GERS，因上下文直接含答案证据压缩了结构化方法优势空间。')

    add_table(doc, ['对比', 'EM差值', '95% CI', 'McNemar p', '判定'], [
        ['GERS-CV2 vs CoT-SC', '+0.040', '[-0.014, +0.096]', '0.029', 'McNemar显著/CI微跨0'],
        ['GERS-CV2 vs StdCoT', '+0.038', '[-0.018, +0.094]', '0.040', 'McNemar显著/CI微跨0'],
        ['GERS-SC vs CoT-SC', '+0.020', '[-0.034, +0.076]', '0.275', '不显著'],
        ['CoT-SC vs StdCoT', '+0.030', '[-0.020, +0.080]', '0.453', '不显著'],
    ], caption='表 2  显著性检验（HotpotQA, n=500）')
    add_para(doc, 'GERS-CV2 McNemar 显著优于 CoT-SC（p=0.029），但 bootstrap 95% CI 微跨零。本文如实报告"配对显著、效应量临界"状态。未引入双向交叉验证的 GERS-SC（p=0.275）不显著，反向印证核心创新有效性。')

    add_heading(doc, '4.3 Consistency Score 区分度的修复（核心证据）', 2)
    add_table(doc, ['CS计算方式', '答对CS', '答错CS', '区分度', 'stdev'], [
        ['旧CS（纯结构, GERS-SC）', '0.6592', '0.6628', '-0.0035', '0.098'],
        ['新CS（+双向交叉验证, GERS-CV）', '0.7888', '0.7042', '+0.0847', '0.340'],
        ['对照：纯结构分 S_struct', '0.9967', '1.0000', '-0.0033', '—'],
    ], caption='表 3  CS 对错区分度（HotpotQA, n=500）')
    add_para(doc, '旧 CS 完全失效（区分度 -0.0035，反向噪声，分布聚集）。双向交叉验证成功修复（区分度 +0.0847，正向有效，分布拉开）。这是本文最硬的贡献证据——CS 从"无区分的图论指标"变为"有效的推理质量信号"，且为 DAG 独有。')

    add_heading(doc, '4.4 分题型分析与适用边界', 2)
    add_table(doc, ['方法', 'comparison', 'bridge_comp', 'compositional', 'inference'], [
        ['Standard CoT', '0.815', '0.905', '0.284', '0.361'],
        ['Zero-Shot', '0.800', '0.587', '0.366', '0.330'],
        ['CoT-SC', '0.720', '0.864', '0.268', '0.343'],
        ['GERS-CV2', '0.775', '0.524', '0.297', '0.288'],
    ], caption='表 4  2WikiMultiHopQA 分题型 F1（n=100）')
    add_para(doc, 'comparison（纯对比题）：GERS-CV2 F1=0.775 与 Standard CoT 接近，DAG 拓扑分解有效拆解对比双方。bridge_comparison（桥接对比题）：GERS 短板（0.524 vs 0.905），子问题错误传播导致。双向交叉验证将 bridge_comparison F1 从 0.476（无CV）提升至 0.524，部分缓解但未根除，剩余差距源于真实推理错误传播。')

    add_heading(doc, '4.5 计算成本分析', 2)
    add_table(doc, ['方法', '单题延迟', 'EM', 'F1'], [
        ['Zero-Shot', '0.7s', '0.276', '0.389'],
        ['Standard CoT', '3.0s', '0.264', '0.368'],
        ['CoT-SC (N=3)', '8.3s', '0.262', '0.373'],
        ['GERS+自适应', '4.8s', '0.284', '0.395'],
        ['GERS-SC (K=3)', '16.2s', '0.282', '0.398'],
        ['GERS-CV2（最优）', '6.6s', '0.302', '0.413'],
    ], caption='表 5  计算成本对比（HotpotQA, n=500）')
    add_para(doc, 'GERS-CV2 单路版以 6.6s 延迟取得最优 EM=0.302，成本约为 CoT-SC 的 80% 却领先 4pt F1。多路版 GERS-SC（16.2s）成本更高却无性能增益。')

    add_heading(doc, '4.6 消融实验', 2)
    add_table(doc, ['配置', 'EM', 'F1', '说明'], [
        ['GERS+自适应', '0.284', '0.395', '基线：DAG执行+自适应路由'],
        ['GERS-SC（+K=3）', '0.282', '0.398', '多路选优，旧CS无区分'],
        ['GERS-CV（+双向交叉验证）', '0.298', '0.409', '新CS有区分度'],
        ['GERS-CV2（+置信度加权）', '0.302', '0.413', '最优配置'],
        ['w/o 图执行（=CoT-SC+GERS重排）', '0.264', '0.372', '无DAG执行'],
    ], caption='表 6  消融实验（HotpotQA, n=500）')
    add_para(doc, '双向交叉验证是核心增益来源（+1.4pt EM）。图级自一致性无额外增益（p=1.000，诚实负面发现）。置信度加权贡献有限（+0.4pt 不显著）。图执行是基础（移除后退化为 0.264）。')

    add_heading(doc, '4.7 案例分析', 2)
    add_para(doc, '案例 1（交叉验证捕获推理不一致）：某多跳问题正向分解得最终答案后，反向验证以该答案+上下文反向重答子问题。正反向子答案一致时 crossval 升高、CS 升高，推理链被保留；当正向某子问题答错（如生卒年判断偏差），反向独立重答得到不同子答案，正反向不一致使 crossval 下降、CS 降低，错误推理链被识别为低质量。纯结构 CS 对两者都给 0.6 无法区分，而 crossval 能区分——这体现了双向交叉验证"内容自洽性"校验的价值。')
    add_para(doc, '案例 2（comparison 题图分解优于线性 CoT）：对比型问题"哪部电影更早上映，A 还是 B"天然对应"两条并行子链 + 汇合节点"的 DAG 结构。GERS 将其分解为"分别查 A/B 上映年份 → 比较年份"两路子问题，拓扑执行后正确汇合；而线性 CoT 易在汇合点混淆两部电影。这是 GERS 在 comparison 子集上优于 CoT-SC（+10.5pt，McNemar p=0.013）的结构性原因。')

    # 5 讨论
    add_heading(doc, '5 讨论与局限', 1)
    add_para(doc, '双向交叉验证的核心价值：将 CS 区分度从 -0.0035 修复到 +0.0847，使校验从"图是否合法"升级为"推理内容是否自洽"，DAG 独有。')
    add_para(doc, '性能的诚实定位：GERS-CV2 McNemar 显著（p=0.029）但 bootstrap CI 微跨零，"配对显著、效应量临界"。整体增益有限因 HotpotQA 上下文直接含答案，Zero-Shot 已接近 GERS。')
    add_para(doc, '图级自一致性的边界：即便修复 CS 区分度，K=3 多路选优仍无额外增益（p=1.000），单路 DAG 执行已足够。')
    add_para(doc, '深度复合桥接的局限：bridge_comparison 上 GERS 错误传播落后于线性 CoT，是图结构分解在深度多跳的固有局限。')
    add_para(doc, '与现有图方法对比：GERS 图结构既参与执行又参与内容校验，形成"构图-执行-校验"闭环。')

    # 6 结论
    add_heading(doc, '6 结论', 1)
    add_para(doc, '本文提出 GERS，核心创新是子答案双向交叉验证：以最终答案+上下文为锚反向重答子问题，对比正反向一致性，将 Consistency Score 从纯图论结构指标（区分度 -0.0035，等价随机）升级为有效的内容自洽信号（区分度 +0.0847）。该机制为 DAG 独有。最优配置 GERS-CV2 在 HotpotQA 500 条上取得 EM=0.302、F1=0.413，McNemar 显著优于 CoT-SC（p=0.029），F1 领先 4pt。本文诚实呈现方法适用边界，并通过三项工程改进量化了"表面失败"与"真实推理差距"的边界。')
    add_para(doc, '未来工作：(1) 针对深度复合桥接题设计局部重生成机制；(2) 在更大规模模型与更多数据集上验证双向交叉验证泛化性；(3) 探索反向验证对 confirmation bias 的进一步缓解策略。')

    # 参考文献
    add_heading(doc, '参考文献', 1)
    refs = [
        '[1] Wei, J., et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In NeurIPS, 2022.',
        '[2] Besta, M., et al. Graph of Thoughts: Solving Elaborate Problems with Large Language Models. In AAAI, 2024.',
        '[3] Han, H., et al. Reasoning with Graphs: Structuring Implicit Knowledge to Enhance LLMs Reasoning. In Findings of ACL, 2025.',
        '[4] Oruche, R., et al. Disentangling Complex Questions in LLMs via Multi-Hop Dependency Graphs. In CIKM, 2025.',
        '[5] Ni, T., et al. StepChain GraphRAG: Reasoning Over Knowledge Graphs for Multi-hop QA. arXiv:2510.02827, 2025.',
        '[6] Wang, X., et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In ICLR, 2023.',
        '[7] Yao, S., et al. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In NeurIPS, 2023.',
        '[8] Lightman, H., et al. Let\'s Verify Step by Step. In ICLR, 2024.',
        '[9] Madaan, A., et al. Self-Refine: Iterative Refinement with Self-Feedback. In NeurIPS, 2023.',
        '[10] DAG-Math: Modeling Chain-of-Thought as Directed Acyclic Graphs. arXiv:2510.19842, 2025.',
        '[11] Yang, Z., et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA. In EMNLP, 2018.',
        '[12] Ho, X., et al. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In COLING, 2020.',
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.3
        run = p.add_run(r)
        set_font(run, size=9.5)

    out = Path('docs/paper.docx')
    doc.save(out)
    print(f'Word 版已生成: {out} ({out.stat().st_size//1024} KB)')


if __name__ == '__main__':
    main()
