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

NAVY = "#1B4E7A"
BLUE = "#4B79A8"
BLUE_LIGHT = "#EAF1F8"
ORANGE = "#C95F1A"
ORANGE_LIGHT = "#FBF0E8"
GREEN = "#2D6B37"
RED = "#B33A3A"
GRAY = "#6F7378"
MID_GRAY = "#B8BDC4"
LIGHT_GRAY = "#F6F7F9"
GRID = "#E4E7EB"
TEXT = "#1F2328"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.015, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015, facecolor="white")
    fig.savefig(OUT / f"{name}.eps", bbox_inches="tight", pad_inches=0.015, facecolor="white")
    plt.close(fig)
    print(f"generated {OUT / (name + '.png')}, .pdf, and .eps")


def rounded_box(ax, xy, w, h, text, fc="white", ec=GRAY, color=TEXT, lw=0.8, fontsize=8):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.020",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color)
    return box


def arrow(ax, start, end, color=GRAY, lw=0.8, style="-|>", ls="-"):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8,
        linewidth=lw,
        color=color,
        linestyle=ls,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)
    return arr


def node(ax, xy, label, fc=BLUE_LIGHT, ec=BLUE, color=TEXT, r=0.055, fontsize=8):
    c = Circle(xy, r, facecolor=fc, edgecolor=ec, linewidth=0.8)
    ax.add_patch(c)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fontsize, color=color)
    return c


def panel_label(ax, x, y, text, color=TEXT):
    ax.text(x, y, text, ha="left", va="top", fontsize=8, fontweight="bold", color=color, linespacing=1.0)


def clean_axis(ax):
    ax.grid(axis="y", color=GRID, lw=0.45, alpha=1.0)
    ax.tick_params(width=0.7, length=3, color=TEXT)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)


def figure1_framework():
    fig, ax = plt.subplots(figsize=(7.05, 2.55))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    panels = [
        (0.02, 0.17, 0.30, 0.72, "1  Forward\nDAG execution", NAVY),
        (0.35, 0.17, 0.27, 0.72, "2  Answer\naggregation", NAVY),
        (0.65, 0.16, 0.33, 0.74, "3  Backward\nverification", ORANGE),
    ]
    for x, y, w, h, title, col in panels:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=MID_GRAY, linewidth=0.75))
        panel_label(ax, x + 0.018, y + h - 0.04, title, col)

    # Panel 1: DAG
    rounded_box(ax, (0.08, 0.64), 0.18, 0.065, "$Q, C$", fc=LIGHT_GRAY, ec=MID_GRAY)
    node(ax, (0.17, 0.55), "$F$", r=0.040)
    node(ax, (0.17, 0.43), "$q_1, a_1$", r=0.050)
    node(ax, (0.10, 0.31), "$q_2, a_2$", r=0.050)
    node(ax, (0.24, 0.31), "$q_3, a_3$", r=0.050)
    node(ax, (0.17, 0.20), "$A$", fc="#F8FBFF", r=0.043)
    for s, e in [((0.17, 0.64), (0.17, 0.595)), ((0.17, 0.51), (0.17, 0.48)), ((0.145, 0.395), (0.115, 0.35)), ((0.195, 0.395), (0.225, 0.35)), ((0.115, 0.275), (0.145, 0.225)), ((0.225, 0.275), (0.195, 0.225))]:
        arrow(ax, s, e, color=NAVY)

    # Panel 2: aggregation
    for i, y in enumerate([0.60, 0.47, 0.34], start=1):
        rounded_box(ax, (0.39, y - 0.03), 0.075, 0.06, f"$a_{i}$", fc=BLUE_LIGHT, ec=BLUE)
        arrow(ax, (0.468, y), (0.545, 0.49), color=NAVY, lw=0.75)
    rounded_box(ax, (0.53, 0.435), 0.07, 0.11, "Final\n$A$", fc="#F8FBFF", ec=NAVY)
    rounded_box(ax, (0.48, 0.25), 0.105, 0.07, "answer-type\nalignment", fc="white", ec=MID_GRAY, fontsize=6.7)
    arrow(ax, (0.535, 0.32), (0.56, 0.43), color=GRAY, lw=0.7, ls="--")

    # Panel 3: backward verification
    rounded_box(ax, (0.71, 0.64), 0.11, 0.065, "Final $A$", fc=LIGHT_GRAY, ec=MID_GRAY)
    rounded_box(ax, (0.86, 0.64), 0.08, 0.065, "$C$", fc=LIGHT_GRAY, ec=MID_GRAY)
    ax.plot([0.765, 0.765, 0.90, 0.90], [0.64, 0.60, 0.60, 0.64], color=ORANGE, lw=0.8, ls="--")
    for i, y in enumerate([0.51, 0.39, 0.27], start=1):
        node(ax, (0.71, y), f"$q_{i}$", r=0.038, fc=ORANGE_LIGHT, ec=ORANGE)
        rounded_box(ax, (0.79, y - 0.025), 0.055, 0.05, f"$a'_{i}$", fc="white", ec=MID_GRAY)
        rounded_box(ax, (0.91, y - 0.025), 0.05, 0.05, f"$a_{i}$", fc=BLUE_LIGHT, ec=BLUE)
        ax.text(0.875, y, r"$\approx$", ha="center", va="center", fontsize=9)
        arrow(ax, (0.79, y), (0.75, y), color=ORANGE, lw=0.75, ls="--")
    arrow(ax, (0.615, 0.52), (0.645, 0.52), color=GRAY)

    rounded_box(
        ax,
        (0.23, 0.035),
        0.54,
        0.07,
        r"$S = 0.3\,S_{struct} + 0.7\,S_{crossval}$",
        fc="#F7FAF5",
        ec=GREEN,
        color="#1F6B2A",
        fontsize=8.5,
    )
    save(fig, "image1")


def figure2_cs_discrimination():
    fig, ax = plt.subplots(figsize=(5.0, 2.35))
    schemes = ["Structural CS", "Cross-validated CS"]
    correct = np.array([0.6592, 0.7888])
    wrong = np.array([0.6628, 0.7042])
    x = np.arange(2)

    ax.plot(x, correct, marker="o", color=BLUE, lw=1.25, ms=4.2, label="Correct")
    ax.plot(x, wrong, marker="s", color=ORANGE, lw=1.25, ms=4.0, label="Wrong")
    display_delta = [-0.0035, 0.0847]
    for xi, c, w, delta in zip(x, correct, wrong, display_delta):
        ax.vlines(xi, min(c, w), max(c, w), color=GRAY, lw=0.7)
        ax.text(xi + 0.07, max(c, w) + 0.014, f"$\\Delta={delta:+.4f}$", fontsize=7, color=GREEN if delta > 0 else RED)
        ax.text(xi + 0.02, c - 0.026, f"{c:.4f}", fontsize=6.8, color=BLUE, ha="left")
        ax.text(xi + 0.12, w + 0.006, f"{w:.4f}", fontsize=6.8, color=ORANGE, ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.set_ylabel("Mean consistency score")
    ax.set_ylim(0.58, 0.86)
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.text(0.5, 0.585, "positive $\\Delta$: correct scores higher", ha="center", va="bottom", fontsize=6.6, color=GRAY)
    save(fig, "image2")


def figure3_verification_unit():
    fig, ax = plt.subplots(figsize=(5.3, 2.35))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rounded_box(ax, (0.06, 0.60), 0.18, 0.12, "Forward step\n$q_i \u2192 a_i$", fc=BLUE_LIGHT, ec=BLUE)
    rounded_box(ax, (0.06, 0.30), 0.18, 0.12, "Final answer\n$A$", fc="#F8FBFF", ec=NAVY)
    rounded_box(ax, (0.06, 0.14), 0.18, 0.10, "Context $C$", fc=LIGHT_GRAY, ec=MID_GRAY)

    rounded_box(ax, (0.38, 0.43), 0.22, 0.16, "Backward query\n$(q_i, A, C)$", fc=ORANGE_LIGHT, ec=ORANGE)
    rounded_box(ax, (0.73, 0.56), 0.16, 0.10, "Forward\n$a_i$", fc=BLUE_LIGHT, ec=BLUE)
    rounded_box(ax, (0.73, 0.34), 0.16, 0.10, "Backward\n$a'_i$", fc=ORANGE_LIGHT, ec=ORANGE)
    rounded_box(ax, (0.73, 0.12), 0.16, 0.10, "match?\n$0/1$", fc="white", ec=MID_GRAY)

    arrow(ax, (0.24, 0.66), (0.38, 0.52), NAVY)
    arrow(ax, (0.24, 0.36), (0.38, 0.50), NAVY)
    arrow(ax, (0.24, 0.19), (0.38, 0.48), NAVY)
    arrow(ax, (0.60, 0.51), (0.73, 0.39), ORANGE)
    arrow(ax, (0.81, 0.56), (0.81, 0.44), GRAY)
    arrow(ax, (0.81, 0.34), (0.81, 0.22), GRAY)
    ax.text(0.47, 0.76, "independent re-derivation", ha="center", fontsize=7.5, color=ORANGE)
    ax.text(0.50, 0.055, r"$S_{crossval}=\sum_i w_i\,match(a_i,a'_i) / \sum_i w_i$", ha="center", fontsize=8.5, color=GREEN)
    save(fig, "image3")


def figure4_main_results():
    fig, ax = plt.subplots(figsize=(5.8, 2.55))
    methods = ["Zero-Shot", "Std CoT", "CoT-SC", "MoDeGraph", "GERS-Adap.", "GERS-SC", "GERS-CV", "GERS-CV2"]
    f1 = np.array([0.389, 0.368, 0.373, 0.366, 0.395, 0.398, 0.409, 0.413])
    colors = ["#C8C8C8", "#C8C8C8", "#8E8E8E", "#B8A58A", BLUE, BLUE, BLUE, ORANGE]
    x = np.arange(len(methods))
    ax.bar(x, f1, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.373, color=GRAY, lw=0.75, ls="--")
    ax.text(3.0, 0.3745, "CoT-SC baseline", va="bottom", ha="left", fontsize=6.4, color=GRAY)
    for xi, yi in zip(x, f1):
        ax.text(xi, yi + 0.0024, f"{yi:.3f}", ha="center", va="bottom", fontsize=6.6)
    ax.annotate(
        "+0.041 F1 vs CoT-SC\n95% CI [-0.014, 0.095]",
        xy=(7, 0.413),
        xytext=(5.25, 0.427),
        ha="left",
        fontsize=6.8,
        color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.75),
    )
    ax.set_ylabel("F1")
    ax.set_ylim(0.34, 0.44)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=28, ha="right")
    clean_axis(ax)
    save(fig, "image4")


def figure5_2wiki_boundary():
    fig, ax = plt.subplots(figsize=(5.6, 2.65))
    types = ["comparison\n(n=25)", "bridge-comp\n(n=21)", "compositional\n(n=39)", "inference\n(n=15)"]
    cot = np.array([0.815, 0.905, 0.284, 0.361])
    gers = np.array([0.775, 0.524, 0.297, 0.288])
    x = np.arange(len(types))
    width = 0.34
    ax.bar(x - width / 2, cot, width, label="Standard CoT", color="#B9B9B9", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, gers, width, label="GERS-CV2", color=BLUE, edgecolor="white", linewidth=0.5)
    for xi, c, g in zip(x, cot, gers):
        ax.text(xi - width / 2, c + 0.014, f"{c:.3f}", ha="center", fontsize=6.5)
        ax.text(xi + width / 2, g + 0.014, f"{g:.3f}", ha="center", fontsize=6.5)
    ax.annotate(
        "largest gap\n(error propagation)",
        xy=(1 + width / 2, 0.524),
        xytext=(1.58, 0.72),
        fontsize=6.8,
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.75),
    )
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    clean_axis(ax)
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
