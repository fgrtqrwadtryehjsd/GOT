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
  image6.pdf  - Oracle interventions on MuSiQue 4-hop (Section 4.3)
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
COLOR_GERS = "#c65b45"         # muted brick-red for our method (highlight)
COLOR_ORACLE_BAR = "#4c72b0"   # deep blue for oracle bars
COLOR_ANNOT = "#333333"

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Figure 4: LongBench main results ====================
def figure_longbench_main():
    """Horizontal comparison that remains legible at one-column width."""
    fig, ax = plt.subplots(figsize=(3.4, 2.05))

    subsets = ["multifieldqa_en", "musique", "narrativeqa"]
    cot_sc = [0.345, 0.325, 0.234]
    cv2 = [0.415, 0.387, 0.208]

    y = np.arange(len(subsets))
    height = 0.32
    ax.axhspan(1.5, 2.5, color="#fbf0e8", zorder=0)
    ax.barh(y - height / 2, cot_sc, height, label="CoT-SC (N=3)",
            color=COLOR_COT_SC, edgecolor="black", linewidth=0.4)
    ax.barh(y + height / 2, cv2, height, label="GERS-DAG",
            color=COLOR_GERS, edgecolor="black", linewidth=0.4)

    markers = ["*", "*", "n.s."]
    for yi, (c, v, marker) in enumerate(zip(cot_sc, cv2, markers)):
        ax.text(c + 0.008, yi - height / 2, f"{c:.3f}", va="center", fontsize=7)
        ax.text(v + 0.008, yi + height / 2, f"{v:.3f}", va="center", fontsize=7)
        ax.text(0.485, yi, marker, va="center", ha="right", fontsize=8,
                color="#2a7a2a" if yi < 2 else "#a05a20")

    ax.set_xlabel("F1")
    ax.set_xlim(0, 0.51)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_yticks(y)
    ax.set_yticklabels(subsets, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e3e3e3", linewidth=0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, ncol=2, fontsize=7.5)

    plt.tight_layout()
    out = OUT_DIR / "image4.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"[figure] wrote {out}")


# ==================== Figure 6: Oracle interventions ====================
def figure_oracle_waterfall():
    """Independent Oracle interventions on MuSiQue 4-hop, n=200."""
    fig, ax = plt.subplots(figsize=(3.4, 2.25))

    labels = ["Model pipeline",
              "Gold decomposition",
              "Gold decomp. + retrieval",
              "Gold decomp. + sub-answers"]
    f1_vals = [0.348, 0.439, 0.514, 0.839]
    deltas = [None, 0.091, 0.166, 0.491]

    y = np.arange(len(labels))
    colors_bar = [COLOR_COT_SC, "#557da8", "#3f6f9f", "#244f87"]

    for i in range(len(labels)):
        ax.barh(y[i], f1_vals[i], height=0.56,
                color=colors_bar[i], edgecolor="black", linewidth=0.5)
        ax.text(f1_vals[i] + 0.015, y[i], f"{f1_vals[i]:.3f}",
                ha="left", va="center", fontsize=8, color=COLOR_ANNOT,
                fontweight="bold" if i == 3 else "normal")
        if deltas[i] is not None:
            ax.text(f1_vals[i] - 0.015, y[i], f"$\\Delta$ {deltas[i]:+.3f}",
                    ha="right", va="center", fontsize=7.2,
                    color="white", fontweight="bold")

    ax.set_xlabel("F1")
    ax.set_xlim(0, 0.92)
    ax.set_xticks(np.arange(0, 0.91, 0.2))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e3e3e3", linewidth=0.45)

    plt.tight_layout()
    out = OUT_DIR / "image6.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"[figure] wrote {out}")


# ==================== Figure 7 (optional): Regime map ====================
def figure_regime_map():
    """Scatter: dF1 (GERS-DAG - CoT-SC) vs context length and evidence structure."""
    fig, ax = plt.subplots(figsize=(3.4, 2.18))

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

    label_positions = {
        "HotpotQA": (5.1, -0.043),
        "multifieldqa_en": (5.6, 0.079),
        "musique-LB": (12.0, 0.057),
        "qasper": (2.8, 0.032),
        "2wikimqa": (5.2, -0.016),
        "narrativeqa": (19.0, -0.041),
    }

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
        label_x, label_y = label_positions[name]
        ax.annotate(name, xy=(ctx, dF1),
                    xytext=(label_x, label_y),
                    fontsize=7.2, color="#222222",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.45))

    # Zero line
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.text(25.5, 0.006, "GERS-DAG wins", fontsize=7, color="#2a7a2a",
            ha="right", style="italic")
    ax.text(25.5, -0.009, "CoT-SC wins", fontsize=7, color="#a05a20",
            ha="right", style="italic")

    ax.set_xlabel("Median context length (k tokens)")
    ax.set_ylabel(r"$\Delta$F1 (DAG $-$ CoT-SC)")
    ax.set_xlim(2, 26)
    ax.set_ylim(-0.06, 0.10)


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
