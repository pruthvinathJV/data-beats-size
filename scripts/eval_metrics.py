"""Stage 7b: CIDEr, CHAIR and caption length for every generated caption set,
with paired bootstrap CIs for the pre-registered contrasts.

    python scripts/eval_metrics.py
"""

import json

import numpy as np
import pandas as pd

from datacurate.captioning import Chair, cider
from datacurate.io import DATA, RESULTS

GEN = DATA / "eval" / "gen"
# Confirmatory (pre-registered): C-A (H3; D = C after the gate), E-A (H4), B-A. Exploratory (A2): Dx = D*.
CONTRASTS = [("C", "A"), ("E", "A"), ("B", "A"), ("Dx", "A"), ("Dx", "C"), ("E", "Dx"), ("E", "C")]
N_BOOT = 2000


def arm_of(run: str) -> str:
    return run.split("_")[0]


def main():
    test = pd.read_parquet(DATA / "coco" / "karpathy_test.parquet", columns=["cocoid", "refs"])
    refs = dict(zip(test.cocoid, test.refs.map(list)))
    chair = Chair(DATA / "coco" / "chair_synonyms.txt", DATA / "coco" / "instances_val2014.json")

    rows, per_image = [], {}
    for path in sorted(GEN.glob("*.parquet")):
        run = path.stem
        g = pd.read_parquet(path).set_index("cocoid").loc[test.cocoid]
        c, c_img = cider(refs, dict(zip(g.index, g.caption)))
        ch = chair.score(g.index, g.caption, test.refs)
        rows.append({"run": run, "arm": arm_of(run), "CIDEr": c, "CHAIRi": ch["CHAIRi"], "CHAIRs": ch["CHAIRs"],
                     "objects_per_caption": ch["objects_per_caption"],
                     "mean_words": float(g.caption.str.split().str.len().mean())})
        per_image[run] = {"cider": c_img, "chair": ch["per_image"].astype(float)}
        print(rows[-1], flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / "eval_runs.csv", index=False)
    arms = runs.groupby("arm").agg(["mean", "std"]).round(4)
    arms.columns = ["_".join(c) for c in arms.columns]
    arms.to_csv(RESULTS / "eval_arms.csv")

    # Paired bootstrap over test images on the seed-averaged per-image scores.
    rng = np.random.default_rng(0)
    n = len(test)
    boot = rng.integers(0, n, size=(N_BOOT, n))
    avg = {}
    for arm in runs.arm.unique():
        members = [r for r in per_image if arm_of(r) == arm]
        avg[arm] = {m: np.mean([per_image[r][m] for r in members], axis=0) for m in ("cider", "chair")}
    contrasts = []
    for a, b in CONTRASTS:
        if a not in avg or b not in avg:
            continue
        for metric in ("cider", "chair"):
            d = avg[a][metric] - avg[b][metric]
            draws = d[boot].mean(axis=1)
            contrasts.append({"contrast": f"{a} - {b}", "metric": "CIDEr" if metric == "cider" else "CHAIRs",
                              "diff": float(d.mean()), "lo": float(np.quantile(draws, 0.025)),
                              "hi": float(np.quantile(draws, 0.975))})
    pd.DataFrame(contrasts).to_csv(RESULTS / "eval_contrasts.csv", index=False)
    print(arms.to_string())
    print(json.dumps(contrasts, indent=1))


if __name__ == "__main__":
    main()
