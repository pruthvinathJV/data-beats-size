"""Stage 7c: independent evaluation judge (a model family no arm was selected or written with)
on a fixed 2K subset of the test images, for every generated caption set.

Runs in transformers, not vLLM: in vLLM 0.32 here only Qwen-family models loaded,
and non-Qwen models hung at engine start. One forward pass per caption; the score is
the expected value of the next-token distribution over "1".."5", as for the selection judge.

    python scripts/eval_judge.py            # all caption sets
    python scripts/eval_judge.py --sanity   # references vs shuffled captions on 64 images
"""

import argparse
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image

from datacurate.io import DATA, RESULTS, load_config
from datacurate.judge import DIGITS, load_rubric, summarize

EVAL_JUDGE = "HuggingFaceM4/Idefics3-8B-Llama3"
SUBSET = 2000
BATCH = 16


class HFJudge:
    def __init__(self, rubric_path):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.rubric, self.rubric_sha = load_rubric(rubric_path)
        self.proc = AutoProcessor.from_pretrained(EVAL_JUDGE)
        self.proc.image_processor.do_image_splitting = False
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(EVAL_JUDGE, torch_dtype=torch.bfloat16).cuda().eval()
        tok = self.proc.tokenizer
        # The answer may start with or without a leading space; both spellings count for a digit.
        self.digit_ids = []
        for d in DIGITS:
            encoded = [tok.encode(s, add_special_tokens=False) for s in (d, " " + d)]
            ids = sorted({e[0] for e in encoded if len(e) == 1})
            assert ids, f"digit {d} is not a single token"
            self.digit_ids.append(ids)

    def _prompt(self, caption: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "text", "text": self.rubric}, {"type": "image"},
                                             {"type": "text", "text": f"Caption: {caption}\nScore:"}]}]
        # Idefics3 answers " 4" as two tokens (space, digit); prefill the space so the next token is the digit.
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True) + " "

    @torch.inference_mode()
    def score(self, images, captions):
        out = []
        for s in range(0, len(images), BATCH):
            ims, caps = images[s : s + BATCH], captions[s : s + BATCH]
            enc = self.proc(text=[self._prompt(c) for c in caps], images=[[im.convert("RGB")] for im in ims],
                            return_tensors="pt", padding=True)
            enc = {k: v.cuda() for k, v in enc.items()}
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            probs = torch.softmax(self.model(**enc).logits[:, -1].float(), dim=-1).cpu().numpy()
            for p in probs:
                out.append(summarize(np.array([p[ids].sum() for ids in self.digit_ids])))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    cfg = load_config("configs/default.yaml")
    test = pd.read_parquet(DATA / "coco" / "karpathy_test.parquet", columns=["cocoid", "img", "refs"])
    judge = HFJudge(cfg["judge"]["rubric"])

    if args.sanity:
        t = test.head(64)
        imgs = [Image.open(io.BytesIO(b)) for b in t.img]
        good = judge.score(imgs, [r[0] for r in t.refs])
        bad = judge.score(imgs, [r[0] for r in t.refs.iloc[::-1]])  # captions of other images
        for name, sc in (("matching refs", good), ("mismatched refs", bad)):
            print(f"{name}: mean {np.mean([x.expected for x in sc]):.2f}, "
                  f"digit mass {np.mean([x.digit_mass for x in sc]):.3f}, parse ok {np.mean([x.parse_ok for x in sc]):.2f}")
        return

    sub = test.sample(SUBSET, random_state=cfg["seed"]).reset_index(drop=True)
    images = [Image.open(io.BytesIO(b)) for b in sub.img]
    rows = []
    for path in sorted((DATA / "eval" / "gen").glob("*.parquet")):
        g = pd.read_parquet(path).set_index("cocoid").loc[sub.cocoid]
        sc = judge.score(images, g.caption.tolist())
        rows.append({"run": path.stem, "arm": path.stem.split("_")[0],
                     "eval_judge": float(np.nanmean([s.expected for s in sc])),
                     "parse_failure_rate": float(np.mean([not s.parse_ok for s in sc]))})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "eval_judge_runs.csv", index=False)
    print(df.groupby("arm").eval_judge.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
