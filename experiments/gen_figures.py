"""生成论文数据图:图2(CS区分度对比) + 图5(2Wiki分题型F1)
用 matplotlib 生成,英文标注(避免中文字体问题)。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})

OUT = Path('docs/figures')

# ============ 图2: Consistency Score 区分度对比 ============
# 数据来自 4.3 节 (HotpotQA n=500)
fig, ax = plt.subplots(figsize=(7, 4.5))
methods = ['Old CS\n(pure structural)', 'New CS\n(+cross-validation)']
correct = [0.6592, 0.7888]
wrong = [0.6628, 0.7042]
x = np.arange(len(methods))
width = 0.32
b1 = ax.bar(x - width/2, correct, width, label='Correct answers', color='#4C72B0', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + width/2, wrong, width, label='Wrong answers', color='#DD8452', edgecolor='black', linewidth=0.5)

# 标注数值
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f'{bar.get_height():.4f}',
            ha='center', va='bottom', fontsize=10)
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f'{bar.get_height():.4f}',
            ha='center', va='bottom', fontsize=10)

# 标注区分度
ax.annotate('discrim.\n−0.0035\n(random)', xy=(0, 0.67), xytext=(0, 0.85),
            ha='center', fontsize=10, color='#C44E52', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#C44E52'))
ax.annotate('discrim.\n+0.0847\n(effective)', xy=(1, 0.75), xytext=(1, 0.90),
            ha='center', fontsize=10, color='#55A868', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#55A868'))

ax.set_ylabel('Consistency Score mean')
ax.set_title('Figure 2: Consistency Score Discrimination (HotpotQA, n=500)')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / 'image2.png', bbox_inches='tight', facecolor='white')
plt.close()
print('图2 已生成: docs/figures/image2.png')

# ============ 图5: 2WikiMultiHopQA 分题型 F1 ============
# 数据来自 4.4 节 (n=100)
fig, ax = plt.subplots(figsize=(8, 4.5))
types = ['comparison', 'bridge_comp', 'compositional', 'inference']
cot = [0.815, 0.905, 0.284, 0.361]
gers = [0.775, 0.524, 0.297, 0.288]
x = np.arange(len(types))
width = 0.36
b1 = ax.bar(x - width/2, cot, width, label='Standard CoT', color='#4C72B0', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + width/2, gers, width, label='GERS-CV2', color='#55A868', edgecolor='black', linewidth=0.5)

for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=9)
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=9)

ax.set_ylabel('F1')
ax.set_title('Figure 5: 2WikiMultiHopQA Per-Type F1 (n=100)')
ax.set_xticks(x)
ax.set_xticklabels(types)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.4)
# 标注 bridge_comparison 为 GERS 短板
ax.annotate('GERS weakness\n(error propagation)', xy=(1.18, 0.524), xytext=(1.6, 0.72),
            fontsize=9, color='#C44E52',
            arrowprops=dict(arrowstyle='->', color='#C44E52'))
plt.tight_layout()
plt.savefig(OUT / 'image5.png', bbox_inches='tight', facecolor='white')
plt.close()
print('图5 已生成: docs/figures/image5.png')
