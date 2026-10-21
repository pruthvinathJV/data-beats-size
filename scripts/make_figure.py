"""Render the README results chart (light and dark) from results/eval_arms.csv.

One accent for the judge-selected arm, gray for the rest; two panels instead of a
dual axis because CIDEr and hallucination rate are on different scales.

    python scripts/make_figure.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
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


def main():
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
