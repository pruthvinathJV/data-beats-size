"""Stage 7c: independent evaluation judge (a third model family) on a fixed 2K
subset of the test images, for every generated caption set.

    python scripts/eval_judge.py
"""

import io

import numpy as np
import pandas as pd
from PIL import Image

from datacurate.io import DATA, RESULTS, load_config
from datacurate.judge import VLMJudge

EVAL_JUDGE = "microsoft/Phi-3.5-vision-instruct"
SUBSET = 2000


def main():
    cfg = load_config("configs/polaris_calibration.yaml")
    jc = cfg["judge"]
    test = pd.read_parquet(DATA / "coco" / "karpathy_test.parquet", columns=["cocoid", "img"])
    sub = test.sample(SUBSET, random_state=cfg["seed"]).reset_index(drop=True)
    images = [Image.open(io.BytesIO(b)) for b in sub.img]
    judge = VLMJudge(EVAL_JUDGE, jc["rubric"], max_image_side=jc["max_image_side"], max_model_len=jc["max_model_len"],
                     gpu_memory_utilization=jc["gpu_memory_utilization"], top_logprobs=jc["top_logprobs"],
                     seed=cfg["seed"], trust_remote_code=True)
    digits = judge.digit_token_ids()
    assert all(len(v) == 1 for v in digits.values()), f"eval judge digits are not single tokens: {digits}"

    rows = []
    for path in sorted((DATA / "eval" / "gen").glob("*.parquet")):
        g = pd.read_parquet(path).set_index("cocoid").loc[sub.cocoid]
        sc = judge.score(images, g.caption.tolist())
        vals = np.array([s.expected for s in sc])
        rows.append({"run": path.stem, "arm": path.stem.split("_")[0], "eval_judge": float(np.nanmean(vals)),
                     "parse_failure_rate": float(np.mean([not s.parse_ok for s in sc]))})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "eval_judge_runs.csv", index=False)
    print(df.groupby("arm").eval_judge.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
