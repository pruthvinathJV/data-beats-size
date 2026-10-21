"""Render the README figures into docs/:

- results_{light,dark}.png  from results/eval_arms.csv. One accent for the judge-selected
  arm, gray for the rest; two panels instead of a dual axis (different scales).
- pipeline.png  the study design, drawn on a white background so it reads in any theme.

    python scripts/make_figure.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from datacurate.io import RESULTS, ROOT

ORDER = ["0", "A", "B", "C", "Dx", "E"]
NAMES = {"0": "No fine-tune", "A": "Random", "B": "Cleaned", "C": "CLIPScore top 25%",
         "Dx": "VLM-judge top 25%", "E": "Synthetic captions"}
HIGHLIGHT = "Dx"
THEMES = {
    "light": {"bg": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "gray": "#c3c2b7", "accent": "#2a78d6"},
    "dark": {"bg": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "gray": "#5d5c58", "accent": "#3987e5"},
}
PANELS = [("CIDEr", "CIDEr-D  (higher is better)", 1, "{:.2f}"),
          ("CHAIRs", "Captions with a hallucinated object  (lower is better)", 100, "{:.1f}%")]


# Pipeline diagram: one small box per step, straight elbow connectors, white background.
# Nodes: key -> (column, row, label lines; first line bold). Laid out in inches.
NODES = {
    "W": (0, 3, ["Web pool", "CC12M · 100K pairs"]),
    "F1": (1, 3, ["Filter", "junk images & captions"]),
    "F2": (2, 3, ["Deduplicate", "near-identical images"]),
    "F3": (3, 3, ["Decontaminate", "remove test images"]),
    "S": (4, 3, ["Choose 25K", "pairs per arm"]),
    "A": (5, 5, ["A · Random", "raw pool"]),
    "B": (5, 4, ["B · Random", "cleaned pool"]),
    "C": (5, 3, ["C · CLIPScore", "top 25%"]),
    "D": (5, 2, ["D* · VLM judge", "top 25%"]),
    "E": (5, 1, ["E · Synthetic", "captions rewritten by AI"]),
    "T": (6, 3, ["Fine-tune", "SmolVLM-256M · LoRA", "3 seeds per arm"]),
    "X": (7, 3, ["Exam", "COCO · 5,000 images"]),
    "H": (2, 0, ["Human labels", "200 blind ratings + Polaris"]),
    "J": (3, 0, ["VLM judge", "Qwen3-VL-8B, calibrated"]),
}
LINE_H, PAD_X, PAD_Y, GAP, ROW = 0.22, 0.14, 0.12, 0.42, 0.78  # inches


def _text_width_in(fig, line: str, bold: bool) -> float:
    tx = fig.text(0, 0, line, fontsize=10 if bold else 9, fontweight="bold" if bold else "normal")
    w = tx.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    tx.remove()
    return w


def pipeline():
    """Each step is its own box; connectors are straight horizontal/vertical segments."""
    t = THEMES["light"]
    probe = plt.figure()
    width = {k: max(_text_width_in(probe, ln, i == 0) for i, ln in enumerate(lines)) + 2 * PAD_X
             for k, (_, _, lines) in NODES.items()}
    plt.close(probe)
    ncol = 1 + max(v[0] for v in NODES.values())
    col_w = {c: max(width[k] for k, v in NODES.items() if v[0] == c) for c in range(ncol)}
    width = {k: col_w[v[0]] if k in "ABCDE" else width[k] for k, v in NODES.items()}  # arm boxes share one width
    col_x, x = {}, 0.1
    for c in range(ncol):
        col_x[c] = x + col_w[c] / 2
        x += col_w[c] + GAP
    fig_w, fig_h = x - GAP + 0.1, 5 * ROW + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#ffffff")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    box = {}
    for k, (c, row, lines) in NODES.items():
        cx, cy = col_x[c], 0.5 + row * ROW
        w, h = width[k], 2 * PAD_Y + LINE_H * len(lines)
        style = {"facecolor": "#ffffff", "edgecolor": t["gray"], "linewidth": 1.2, "linestyle": "-"}
        if k == "D":
            style.update(facecolor="#e8f1fc", edgecolor=t["accent"], linewidth=2)
        if k in ("H", "J"):
            style.update(facecolor="#f6f6f3", edgecolor=t["muted"], linestyle=(0, (4, 3)))
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    zorder=3, **style))
        for i, line in enumerate(lines):
            ax.text(cx, cy + (len(lines) - 1) * LINE_H / 2 - i * LINE_H, line, ha="center", va="center", zorder=4,
                    fontsize=10 if i == 0 else 9, color=t["ink"] if i == 0 else t["ink2"],
                    fontweight="bold" if i == 0 else "normal")
        box[k] = (cx, cy, w, h)

    def edge(k, side):
        cx, cy, w, h = box[k]
        return {"r": (cx + w / 2, cy), "l": (cx - w / 2, cy), "t": (cx, cy + h / 2), "b": (cx, cy - h / 2)}[side]

    def route(points, dashed=False, label=None):
        """Polyline of straight segments with an arrowhead on the last one."""
        ls = (0, (3, 3)) if dashed else "-"
        xs, ys = zip(*points)
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=t["muted"], linewidth=1.1, linestyle=ls, zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=10, color=t["muted"],
                                     linewidth=0, zorder=2, shrinkA=0, shrinkB=1))
        if label:
            (lx, ly), anchor = label
            ax.text(lx, ly, anchor, fontsize=8.5, color=t["ink2"], ha="left", va="center", zorder=4,
                    bbox={"facecolor": "#ffffff", "edgecolor": "none", "pad": 1})

    for a, b in [("W", "F1"), ("F1", "F2"), ("F2", "F3"), ("F3", "S"), ("T", "X"), ("H", "J")]:
        route([edge(a, "r"), edge(b, "l")], dashed=a == "H")
    sx = (edge("S", "r")[0] + min(edge(k, "l")[0] for k in "BCDE")) / 2
    for k in "BCDE":
        y = box[k][1]
        route([edge("S", "r"), (sx, box["S"][1]), (sx, y), edge(k, "l")])
    # A comes from the raw pool: up from the web pool, across above the cleaning steps.
    ya = box["A"][1]
    route([edge("W", "t"), (box["W"][0], ya), edge("A", "l")])
    tx = (max(edge(k, "r")[0] for k in "ABCDE") + edge("T", "l")[0]) / 2
    for k in "ABCDE":
        route([edge(k, "r"), (tx, box[k][1]), (tx, box["T"][1]), edge("T", "l")])
    # The judge scores the pool for D* and critiques the synthetic captions for E.
    jx, jy = edge("J", "r")
    route([(jx, jy), (box["S"][0], jy), edge("S", "b")], dashed=True,
          label=((box["S"][0] + 0.06, (jy + edge("S", "b")[1]) / 2), "grades → D*"))
    route([(jx, jy - 0.08), (box["E"][0], jy - 0.08), edge("E", "b")], dashed=True,
          label=((box["E"][0] + 0.06, (jy + edge("E", "b")[1]) / 2), "critic → E"))
    out = ROOT / "docs" / "pipeline.png"
    fig.savefig(out, dpi=200, facecolor="#ffffff")
    plt.close(fig)
    print(out)


def main():
    pipeline()
    arms = pd.read_csv(RESULTS / "eval_arms.csv", dtype={"arm": str}).set_index("arm")
    y = list(range(len(ORDER)))[::-1]
    for mode, t in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=t["bg"], gridspec_kw={"wspace": 0.08})
        for ax, (metric, title, scale, fmt) in zip(axes, PANELS):
            vals = [arms.at[k, f"{metric}_mean"] * scale for k in ORDER]
            errs = [0 if pd.isna(arms.at[k, f"{metric}_std"]) else arms.at[k, f"{metric}_std"] * scale for k in ORDER]
            ax.set_facecolor(t["bg"])
            ax.barh(y, vals, height=0.56, color=[t["accent"] if k == HIGHLIGHT else t["gray"] for k in ORDER],
                    edgecolor=t["bg"], linewidth=2, zorder=2)
            ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor=t["ink2"], elinewidth=1, capsize=0, zorder=3)
            xmax = max(v + e for v, e in zip(vals, errs))
            for yi, v, e, k in zip(y, vals, errs, ORDER):
                hi = k == HIGHLIGHT
                ax.text(v + e + xmax * 0.02, yi, fmt.format(v), va="center", fontsize=9.5,
                        color=t["ink"] if hi else t["ink2"], fontweight="bold" if hi else "normal")
            ax.set_xlim(0, xmax * 1.22)
            ax.set_title(title, loc="left", fontsize=10, color=t["ink2"], pad=8)
            ax.xaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
            ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(t["grid"])
            ax.set_yticks(y)
            ax.set_yticklabels([NAMES[k] for k in ORDER] if ax is axes[0] else [], fontsize=10, color=t["ink"])
        axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
        fig.suptitle("Same model, same 25K examples, only the data changes", x=0.125, ha="left",
                     fontsize=12.5, color=t["ink"], fontweight="bold", y=1.02)
        fig.text(0.125, -0.04, "SmolVLM-256M + LoRA, COCO Karpathy test (5,000 images). "
                 "Mean over 3 seeds; bars show ±1 sd.", fontsize=8.5, color=t["muted"])
        out = ROOT / "docs" / f"results_{mode}.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["bg"])
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
