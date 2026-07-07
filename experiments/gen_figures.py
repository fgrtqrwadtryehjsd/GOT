"""Generate publication-style figures for the GERS paper.

The script writes both PNG and PDF versions under docs/figures/:
image1: overall framework
image2: consistency-score discrimination
image3: bidirectional verification unit
image4: HotpotQA main F1 comparison
image5: 2Wiki per-type boundary analysis
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#4E79A7"
BLUE_DARK = "#1F4E79"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
RED = "#E15759"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#F5F6F8"
TEXT = "#222222"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"generated {OUT / (name + '.png')} and .pdf")


def rounded_box(ax, xy, w, h, text, fc="white", ec=GRAY, color=TEXT, lw=1.0, fontsize=9):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color)
    return box


def arrow(ax, start, end, color=GRAY, lw=1.0, style="-|>", ls="-"):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        linestyle=ls,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)
    return arr


def node(ax, xy, label, fc="#EEF4FB", ec=BLUE, color=TEXT, r=0.055, fontsize=9):
    c = Circle(xy, r, facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(c)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fontsize, color=color)
    return c


def panel_label(ax, x, y, text, color=TEXT):
    ax.text(x, y, text, ha="left", va="top", fontsize=8.5, fontweight="bold", color=color, linespacing=1.05)


def figure1_framework():
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    panels = [
        (0.02, 0.16, 0.30, 0.74, "1  Forward\nDAG execution", BLUE_DARK),
        (0.35, 0.16, 0.27, 0.74, "2  Answer\naggregation", BLUE_DARK),
        (0.65, 0.16, 0.33, 0.74, "3  Backward\nverification", ORANGE),
    ]
    for x, y, w, h, title, col in panels:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#D0D4DA", linewidth=0.9))
        panel_label(ax, x + 0.018, y + h - 0.04, title, col)

    # Panel 1: DAG
    rounded_box(ax, (0.08, 0.64), 0.18, 0.065, "$Q, C$", fc=LIGHT_GRAY, ec="#C9CDD3")
    node(ax, (0.17, 0.55), "$F$", r=0.040)
    node(ax, (0.17, 0.43), "$q_1, a_1$", r=0.050)
    node(ax, (0.10, 0.31), "$q_2, a_2$", r=0.050)
    node(ax, (0.24, 0.31), "$q_3, a_3$", r=0.050)
    node(ax, (0.17, 0.20), "$A$", fc="#F7FBFF", r=0.043)
    for s, e in [((0.17, 0.64), (0.17, 0.595)), ((0.17, 0.51), (0.17, 0.48)), ((0.145, 0.395), (0.115, 0.35)), ((0.195, 0.395), (0.225, 0.35)), ((0.115, 0.275), (0.145, 0.225)), ((0.225, 0.275), (0.195, 0.225))]:
        arrow(ax, s, e, color=BLUE_DARK, lw=1.0)

    # Panel 2: aggregation
    for i, y in enumerate([0.60, 0.47, 0.34], start=1):
        rounded_box(ax, (0.39, y - 0.03), 0.075, 0.06, f"$a_{i}$", fc="#EEF4FB", ec=BLUE)
        arrow(ax, (0.468, y), (0.545, 0.49), color=BLUE_DARK, lw=0.9)
    rounded_box(ax, (0.53, 0.435), 0.07, 0.11, "Final\n$A$", fc="#F7FBFF", ec=BLUE_DARK)
    rounded_box(ax, (0.48, 0.25), 0.105, 0.07, "answer-type\nalignment", fc="white", ec="#B9BDC4", fontsize=7)
    arrow(ax, (0.535, 0.32), (0.56, 0.43), color=GRAY, lw=0.9, ls="--")

    # Panel 3: backward verification
    rounded_box(ax, (0.71, 0.64), 0.11, 0.065, "Final $A$", fc=LIGHT_GRAY, ec="#C9CDD3")
    rounded_box(ax, (0.86, 0.64), 0.08, 0.065, "$C$", fc=LIGHT_GRAY, ec="#C9CDD3")
    ax.plot([0.765, 0.765, 0.90, 0.90], [0.64, 0.60, 0.60, 0.64], color=ORANGE, lw=1.0, ls="--")
    for i, y in enumerate([0.51, 0.39, 0.27], start=1):
        node(ax, (0.71, y), f"$q_{i}$", r=0.038, fc="#FFF7EF", ec=ORANGE)
        rounded_box(ax, (0.79, y - 0.025), 0.055, 0.05, f"$a'_{i}$", fc="white", ec="#C9CDD3")
        rounded_box(ax, (0.91, y - 0.025), 0.05, 0.05, f"$a_{i}$", fc="#EEF4FB", ec=BLUE)
        ax.text(0.875, y, r"$\approx$", ha="center", va="center", fontsize=12)
        arrow(ax, (0.79, y), (0.75, y), color=ORANGE, lw=0.9, ls="--")
    arrow(ax, (0.615, 0.52), (0.645, 0.52), color=GRAY, lw=1.0)

    rounded_box(
        ax,
        (0.23, 0.035),
        0.54,
        0.07,
        r"$S = 0.3\,S_{struct} + 0.7\,S_{crossval}$",
        fc="#F7FAF5",
        ec=GREEN,
        color="#1F6B2A",
        fontsize=10,
    )
    save(fig, "image1")


def figure2_cs_discrimination():
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    schemes = ["Structural CS", "Cross-validated CS"]
    correct = np.array([0.6592, 0.7888])
    wrong = np.array([0.6628, 0.7042])
    x = np.arange(2)

    ax.plot(x, correct, marker="o", color=BLUE, lw=1.8, label="Correct")
    ax.plot(x, wrong, marker="o", color=ORANGE, lw=1.8, label="Wrong")
    display_delta = [-0.0035, 0.0847]
    for xi, c, w, delta in zip(x, correct, wrong, display_delta):
        ax.vlines(xi, min(c, w), max(c, w), color="#555555", lw=1.0)
        ax.text(xi + 0.08, max(c, w) + 0.020, f"$\\Delta={delta:+.4f}$", fontsize=8, color=GREEN if delta > 0 else RED)
        ax.text(xi + 0.02, c - 0.032, f"{c:.4f}", fontsize=7, color=BLUE, ha="left")
        ax.text(xi + 0.12, w + 0.006, f"{w:.4f}", fontsize=7, color=ORANGE, ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.set_ylabel("Mean consistency score")
    ax.set_ylim(0.58, 0.86)
    ax.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.text(0.5, 0.585, "positive $\\Delta$ means correct answers score higher", ha="center", va="bottom", fontsize=7, color="#555555")
    save(fig, "image2")


def figure3_verification_unit():
    fig, ax = plt.subplots(figsize=(5.3, 2.7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rounded_box(ax, (0.06, 0.60), 0.18, 0.12, "Forward step\n$q_i \u2192 a_i$", fc="#EEF4FB", ec=BLUE)
    rounded_box(ax, (0.06, 0.30), 0.18, 0.12, "Final answer\n$A$", fc="#F7FBFF", ec=BLUE_DARK)
    rounded_box(ax, (0.06, 0.14), 0.18, 0.10, "Context $C$", fc=LIGHT_GRAY, ec="#C9CDD3")

    rounded_box(ax, (0.38, 0.43), 0.22, 0.16, "Backward query\n$(q_i, A, C)$", fc="#FFF7EF", ec=ORANGE)
    rounded_box(ax, (0.73, 0.56), 0.16, 0.10, "Forward\n$a_i$", fc="#EEF4FB", ec=BLUE)
    rounded_box(ax, (0.73, 0.34), 0.16, 0.10, "Backward\n$a'_i$", fc="#FFF7EF", ec=ORANGE)
    rounded_box(ax, (0.73, 0.12), 0.16, 0.10, "match?\n$0/1$", fc="white", ec="#B9BDC4")

    arrow(ax, (0.24, 0.66), (0.38, 0.52), BLUE_DARK)
    arrow(ax, (0.24, 0.36), (0.38, 0.50), BLUE_DARK)
    arrow(ax, (0.24, 0.19), (0.38, 0.48), BLUE_DARK)
    arrow(ax, (0.60, 0.51), (0.73, 0.39), ORANGE)
    arrow(ax, (0.81, 0.56), (0.81, 0.44), GRAY)
    arrow(ax, (0.81, 0.34), (0.81, 0.22), GRAY)
    ax.text(0.47, 0.77, "independent re-derivation", ha="center", fontsize=8, color=ORANGE)
    ax.text(0.50, 0.08, r"$S_{crossval}=\sum_i w_i\,match(a_i,a'_i) / \sum_i w_i$", ha="center", fontsize=10, color="#1F6B2A")
    save(fig, "image3")


def figure4_main_results():
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    methods = ["Zero-Shot", "Std CoT", "CoT-SC", "GERS-Adap.", "GERS-SC", "GERS-CV", "GERS-CV2"]
    f1 = np.array([0.389, 0.368, 0.373, 0.395, 0.398, 0.409, 0.413])
    colors = ["#B8B8B8", "#B8B8B8", "#8C8C8C", BLUE, BLUE, BLUE, ORANGE]
    x = np.arange(len(methods))
    ax.bar(x, f1, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0.373, color="#555555", lw=1.0, ls="--")
    ax.text(len(methods) - 0.1, 0.3745, "CoT-SC baseline", va="bottom", ha="right", fontsize=7, color="#555555")
    for xi, yi in zip(x, f1):
        ax.text(xi, yi + 0.003, f"{yi:.3f}", ha="center", va="bottom", fontsize=7)
    ax.annotate(
        "+0.041 F1 vs CoT-SC\n95% CI [-0.014, 0.095]",
        xy=(6, 0.413),
        xytext=(4.6, 0.427),
        ha="left",
        fontsize=8,
        color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9),
    )
    ax.set_ylabel("F1")
    ax.set_ylim(0.34, 0.44)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.8)
    save(fig, "image4")


def figure5_2wiki_boundary():
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    types = ["comparison\n(n=25)", "bridge-comp\n(n=21)", "compositional\n(n=39)", "inference\n(n=15)"]
    cot = np.array([0.815, 0.905, 0.284, 0.361])
    gers = np.array([0.775, 0.524, 0.297, 0.288])
    x = np.arange(len(types))
    width = 0.34
    ax.bar(x - width / 2, cot, width, label="Standard CoT", color="#A9A9A9", edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, gers, width, label="GERS-CV2", color=BLUE, edgecolor="white", linewidth=0.8)
    for xi, c, g in zip(x, cot, gers):
        ax.text(xi - width / 2, c + 0.015, f"{c:.3f}", ha="center", fontsize=7)
        ax.text(xi + width / 2, g + 0.015, f"{g:.3f}", ha="center", fontsize=7)
    ax.annotate(
        "largest gap\n(error propagation)",
        xy=(1 + width / 2, 0.524),
        xytext=(1.55, 0.72),
        fontsize=8,
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.9),
    )
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.8)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "image5")


def main():
    figure1_framework()
    figure2_cs_discrimination()
    figure3_verification_unit()
    figure4_main_results()
    figure5_2wiki_boundary()


if __name__ == "__main__":
    main()
