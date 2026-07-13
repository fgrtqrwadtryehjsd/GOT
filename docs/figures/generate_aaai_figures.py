"""Generate AAAI-2027 style figures for the main paper.

AAAI style conventions (learned from typical AAAI-quality figures):
- Serif fonts to match the paper body (Times/Nimbus Roman; matplotlib default cmr10 is fine)
- Neutral color palette (avoid rainbow / neon; use 2-4 colors, high contrast)
- Thick axes, tight layout, no top/right spines
- Font size 10-12pt for labels (small in double-column)
- Vector PDF output for LaTeX \\includegraphics
- Aspect ratio suited for one-column width (~3.3 inches wide)

Outputs (writes to docs/figures/):
  image4.pdf  - LongBench main results bar chart (Section 4.2)
  image6.pdf  - Oracle waterfall on MuSiQue 4-hop (Section 4.3)
  image7.pdf  - Regime map: dF1 vs context length x evidence structure (optional)

Run:
    cd docs/figures
    python generate_aaai_figures.py
"""
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------- AAAI style setup ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,  # TrueType, embed properly
    "ps.fonttype": 42,
})

# Neutral AAAI-friendly palette
COLOR_COT_SC = "#8ca0b3"       # cool gray-blue for baseline
COLOR_CV2 = "#c65b45"          # muted brick-red for our method (highlight)
COLOR_ORACLE_BAR = "#4c72b0"   # deep blue for oracle bars
COLOR_ANNOT = "#333333"

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Figure 4: LongBench main results ====================
def figure_longbench_main():
    """Bar chart: CoT-SC vs GERS-CV2 on 3 LongBench subsets + significance annotation."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    subsets = ["multifieldqa\\_en\n(n=100, $\\sim$5k tok)",
               "musique\n(n=193, $\\sim$11k tok)",
               "narrativeqa\n(n=71, $\\sim$22k tok)"]
    cot_sc = [0.345, 0.325, 0.234]
    cv2 = [0.415, 0.387, 0.208]

    x = np.arange(len(subsets))
    width = 0.36

    b1 = ax.bar(x - width / 2, cot_sc, width, label="CoT-SC (N=3)",
                color=COLOR_COT_SC, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, cv2, width, label="GERS-CV2 (ours)",
                color=COLOR_CV2, edgecolor="black", linewidth=0.5)

    # Numeric F1 above each bar (small, unobtrusive)
    for i, (c, v) in enumerate(zip(cot_sc, cv2)):
        ax.text(x[i] - width / 2, c + 0.008, f"{c:.3f}",
                ha="center", va="bottom", fontsize=7, color="#555555")
        ax.text(x[i] + width / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, color="#555555")

    # Significance markers ONLY (no CI text on bars — see caption/Table 1)
    # Positioned above the higher bar of each pair, well clear of bars
    sig_markers = [
        (0, r"$\ast$",          max(cot_sc[0], cv2[0])),   # F1 SIG
        (1, r"$\ast\!\ast$",    max(cot_sc[1], cv2[1])),   # EM+F1 SIG
        (2, "n.s.",             max(cot_sc[2], cv2[2])),   # model ceiling
    ]
    for idx, marker, top in sig_markers:
        ax.text(idx, top + 0.055, marker, ha="center", va="center",
                fontsize=13 if marker != "n.s." else 9,
                color=COLOR_ANNOT, fontweight="bold")

    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=8.5)
    ax.set_ylim(0, 0.58)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    # Legend at upper-LEFT to avoid CV2 bar overlap
    ax.legend(loc="upper left", frameon=False, ncol=1, fontsize=8.5)

    # Regime shading (light) + labels near top of plot
    ax.axvspan(-0.5, 1.5, alpha=0.06, color="green", zorder=0)
    ax.axvspan(1.5, 2.5, alpha=0.06, color="orange", zorder=0)
    ax.text(0.5, 0.555, "Target regime", ha="center", va="top",
            fontsize=8, color="#2a7a2a", style="italic")
    ax.text(2.0, 0.555, "Model ceiling", ha="center", va="top",
            fontsize=8, color="#a05a20", style="italic")

    plt.tight_layout()
    out = OUT_DIR / "image4.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"[figure] wrote {out}")


# ==================== Figure 6: Oracle waterfall ====================
def figure_oracle_waterfall():
    """Waterfall: MuSiQue 4-hop, n=200, F1 gains per module."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    labels = ["Baseline\n(model self)",
              "+Oracle-1\n(gold DAG)",
              "+Oracle-1+2\n(+gold retr.)",
              "+Oracle-1+3\n(+gold ans.)"]
    f1_vals = [0.348, 0.439, 0.514, 0.839]
    increments = [f1_vals[0]] + [f1_vals[i] - f1_vals[i - 1] for i in range(1, 4)]
    # Share labels: bumped from 7.5pt (too small) to 9pt bold; shortened for
    # narrow bars so text fits without overflowing.
    shares = [None, "18\\%\ngraph-gen", "15\\%\nretrieval", "66\\%\nreasoner"]

    x = np.arange(len(labels))
    bottoms = [0, f1_vals[0], f1_vals[1], f1_vals[2]]
    heights = [f1_vals[0]] + increments[1:]
    colors_bar = [COLOR_COT_SC, "#7fa5cc", "#5b8ac0", COLOR_ORACLE_BAR]

    for i in range(len(labels)):
        ax.bar(x[i], heights[i], bottom=bottoms[i], width=0.55,
               color=colors_bar[i], edgecolor="black", linewidth=0.5)
        # Cumulative F1 label on top (bumped from 8.5 to 9)
        ax.text(x[i], bottoms[i] + heights[i] + 0.018, f"{f1_vals[i]:.3f}",
                ha="center", va="bottom", fontsize=9, color=COLOR_ANNOT,
                fontweight="bold" if i == 3 else "normal")
        # In-bar share label (bumped from 7.5 to 9pt for legibility)
        if shares[i]:
            mid = bottoms[i] + heights[i] / 2
            ax.text(x[i], mid, shares[i], ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    # Connector lines between bar tops
    for i in range(len(labels) - 1):
        ax.plot([x[i] + 0.28, x[i + 1] - 0.28],
                [f1_vals[i], f1_vals[i]],
                "k--", linewidth=0.6, alpha=0.5)

    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    # Y-limit raised from 0.95 to 1.02 for annotation breathing room
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    # Highlight the reasoner dominance — moved to upper-left empty space
    # to avoid overlap with the "0.839" numeric label above the rightmost bar.
    ax.text(0.25, 0.98,
            "Reasoner is\nthe bottleneck\n(66\\% of\nrecoverable F1)",
            fontsize=8.5, color=COLOR_ANNOT, ha="left", va="top",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", linewidth=0.4, alpha=0.9))

    plt.tight_layout()
    out = OUT_DIR / "image6.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"[figure] wrote {out}")


# ==================== Figure 7 (optional): Regime map ====================
def figure_regime_map():
    """Scatter: dF1 (CV2 - CoT-SC) vs context length, coded by evidence structure."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    # (name, ctx_kilo_tokens, dF1, sig, structure, marker)
    # Updated 2026-07-13: 2wikimqa n=200 now measured (dF1 = -0.021 n.s.)
    points = [
        ("HotpotQA",       4.7, -0.032, True,  "single-file",  "o"),
        ("multifieldqa_en", 5.0, +0.070, True,  "multi-passage","s"),
        ("musique-LB",     11.4, +0.063, True,  "multi-passage","s"),
        ("qasper",          3.4, +0.018, False, "single-doc",   "^"),
        ("2wikimqa",        4.2, -0.021, False, "short-multi",  "v"),
        ("narrativeqa",    22.7, -0.027, False, "narrative",    "D"),
    ]

    # Plot each point
    for name, ctx, dF1, sig, struct, marker in points:
        if dF1 is None:
            continue
        color = {"single-file": "#a05a20",
                 "multi-passage": "#2a7a2a",
                 "single-doc": "#8888aa",
                 "short-multi": "#b07030",
                 "narrative": "#a03a3a"}[struct]
        edge = "black" if sig else "gray"
        lw = 1.2 if sig else 0.4
        ax.scatter(ctx, dF1, c=color, s=80, marker=marker,
                   edgecolor=edge, linewidth=lw, zorder=3)
        # Point label (bumped from 7.5 to 8.5 pt)
        offset_y = 0.010 if dF1 > 0 else -0.018
        offset_x = 0.6 if name != "narrativeqa" else -3.2
        ax.annotate(name, xy=(ctx, dF1),
                    xytext=(ctx + offset_x, dF1 + offset_y),
                    fontsize=8.5, color="#222222")

    # Zero line
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.text(25, 0.006, "CV2 wins above", fontsize=8, color="#2a7a2a",
            ha="right", style="italic")
    ax.text(25, -0.008, "CoT-SC wins below", fontsize=8, color="#a05a20",
            ha="right", style="italic")

    ax.set_xlabel("Median context length (k tokens)")
    ax.set_ylabel(r"$\Delta$F1 (CV2 $-$ CoT-SC)")
    ax.set_xlim(2, 26)
    ax.set_ylim(-0.06, 0.10)

    # Legend proxies for evidence structure
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", label="multi-passage (target)",
               markerfacecolor="#2a7a2a", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="single-file",
               markerfacecolor="#a05a20", markersize=6),
        Line2D([0], [0], marker="^", color="w", label="single-doc",
               markerfacecolor="#8888aa", markersize=6),
        Line2D([0], [0], marker="v", color="w", label="short-multi",
               markerfacecolor="#b07030", markersize=6),
        Line2D([0], [0], marker="D", color="w", label="narrative",
               markerfacecolor="#a03a3a", markersize=6),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=False,
              fontsize=7.5, ncol=2, columnspacing=0.6, handletextpad=0.2)

    plt.tight_layout()
    out = OUT_DIR / "image7.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"[figure] wrote {out}")


if __name__ == "__main__":
    figure_longbench_main()
    figure_oracle_waterfall()
    figure_regime_map()
    print("\nDone. Figures written to:", OUT_DIR)
