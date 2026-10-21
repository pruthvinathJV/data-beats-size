"""Stage 5: arm E, synthetic captions via a generator/critic loop.

Phases (each loads one model, so run them in order; outputs are resumable):
  generate   InternVL3-2B captions every arm-B image (greedy)
  critique   the selection judge scores every generated caption
  retry      regenerate captions below the critic threshold, or near-duplicates
             of an earlier accepted caption, once at temperature 0.7
  critique2  score the retries; keep the better of the two captions per image
  finalize   write data/arms/E.parquet and results/arm_e_stats.json

    python scripts/recaption.py --phase generate   # ... then critique, retry, critique2, finalize
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from datacurate.calibration import fit_isotonic
from datacurate.io import DATA, RESULTS, load_config
from datacurate.judge import resize_max_side

GENERATOR = "OpenGVLab/InternVL3-2B-hf"  # native transformers port; the remote-code variant hangs in vLLM 0.32
GEN_BATCH = 64
PROMPT = "Describe this image in one sentence."
THRESHOLD = 3.5  # calibrated critic score
JACCARD_DUP = 0.9
W = DATA / "arms" / "e_work"


_IMG: dict[str, bytes] = {}


def images_for(keys: pd.Series) -> list[Image.Image]:
    if not _IMG:  # read the pool's JPEG bytes once per process
        raw = pd.read_parquet(DATA / "pool" / "raw", columns=["key", "img"])
        _IMG.update(zip(raw.key, raw.img))
    return [Image.open(io.BytesIO(_IMG[k])).convert("RGB") for k in keys]


def generate(keys: pd.Series, temperature: float, seed: int, ckpt: Path) -> list[str]:
    """Caption with the generator via transformers (InternVL does not load in vLLM 0.32 here).

    Checkpoints every 20 batches to `ckpt` so a restart resumes instead of regenerating.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    done = pd.read_parquet(ckpt).caption.tolist() if ckpt.exists() else []
    torch.manual_seed(seed + len(done))
    proc = AutoProcessor.from_pretrained(GENERATOR)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(GENERATOR, torch_dtype=torch.bfloat16).cuda().eval()
    prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
                                      add_generation_prompt=True)
    sample = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 else {"do_sample": False}
    out = list(done)
    for b, s in enumerate(range(len(done), len(keys), GEN_BATCH)):
        imgs = [resize_max_side(im, 448) for im in images_for(keys.iloc[s : s + GEN_BATCH])]
        enc = proc(text=[prompt] * len(imgs), images=imgs, return_tensors="pt", padding=True, crop_to_patches=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            ids = model.generate(**enc, max_new_tokens=60, **sample)
        texts = proc.batch_decode(ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        out += [t.strip().split("\n")[0] for t in texts]
        if b % 20 == 0:
            pd.DataFrame({"caption": out}).to_parquet(ckpt)
            print(f"generated {len(out):,}/{len(keys):,}", flush=True)
    return out


def critique(keys: pd.Series, captions: list[str], cfg: dict) -> np.ndarray:
    from datacurate.judge import VLMJudge

    jc = cfg["judge"]
    judge = VLMJudge(jc["model"], jc["rubric"], max_image_side=jc["max_image_side"],
                     max_model_len=jc["max_model_len"], gpu_memory_utilization=jc["gpu_memory_utilization"],
                     top_logprobs=jc["top_logprobs"], seed=cfg["seed"])
    scores = []
    for s in range(0, len(keys), 2048):
        scores += [x.expected for x in judge.score(images_for(keys.iloc[s : s + 2048]), captions[s : s + 2048])]
    train = pd.read_parquet(DATA / "scores" / "polaris_train.parquet")
    iso = fit_isotonic(train.judge.fillna(train.judge.median()), train.human)
    raw = np.asarray(scores, dtype=float)
    return iso.predict(np.nan_to_num(raw, nan=float(np.nanmedian(raw))))


def jaccard_dups(captions: list[str]) -> np.ndarray:
    """True where a caption nearly repeats an earlier one (token Jaccard >= threshold)."""
    seen: dict[str, list[set]] = {}
    dup = np.zeros(len(captions), dtype=bool)
    for i, c in enumerate(captions):
        toks = set(c.lower().replace(".", "").split())
        bucket = seen.setdefault(" ".join(sorted(toks)[:3]), [])  # cheap blocking on 3 tokens
        dup[i] = any(len(toks & t) / max(len(toks | t), 1) >= JACCARD_DUP for t in bucket)
        bucket.append(toks)
    return dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["generate", "critique", "retry", "critique2", "finalize"])
    args = ap.parse_args()
    cfg = load_config("configs/default.yaml")
    W.mkdir(parents=True, exist_ok=True)
    keys = pd.read_parquet(DATA / "arms" / "B.parquet", columns=["key"]).key

    if args.phase == "generate":
        pd.DataFrame({"key": keys, "cap1": generate(keys, 0.0, cfg["seed"], W / "gen1.ckpt.parquet")}).to_parquet(W / "gen1.parquet")
    elif args.phase == "critique":
        g = pd.read_parquet(W / "gen1.parquet")
        g["score1"] = critique(g.key, g.cap1.tolist(), cfg)
        g["dup1"] = jaccard_dups(g.cap1.tolist())
        g.to_parquet(W / "crit1.parquet")
    elif args.phase == "retry":
        g = pd.read_parquet(W / "crit1.parquet")
        redo = g[(g.score1 < THRESHOLD) | g.dup1]
        caps = generate(redo.key.reset_index(drop=True), 0.7, cfg["seed"] + 1, W / "gen2.ckpt.parquet")
        pd.DataFrame({"key": redo.key.to_numpy(), "cap2": caps}).to_parquet(W / "gen2.parquet")
    elif args.phase == "critique2":
        r = pd.read_parquet(W / "gen2.parquet")
        r["score2"] = critique(r.key, r.cap2.tolist(), cfg)
        r.to_parquet(W / "crit2.parquet")
    else:
        g = pd.read_parquet(W / "crit1.parquet").merge(pd.read_parquet(W / "crit2.parquet"), on="key", how="left")
        use2 = g.cap2.notna() & ((g.score2 > g.score1) | (g.dup1 & (g.score2 >= THRESHOLD)))
        g["caption"] = np.where(use2, g.cap2, g.cap1)
        g["critic"] = np.where(use2, g.score2, g.score1)
        g[["key", "caption", "critic"]].to_parquet(DATA / "arms" / "E.parquet")
        words = g.caption.str.split()
        all_tokens = [t.lower() for ws in words for t in ws]
        bigrams = {tuple(ws[i : i + 2]) for ws in words for i in range(len(ws) - 1)}
        stats = {
            "n": len(g),
            "accepted_first_try": round(float(((g.score1 >= THRESHOLD) & ~g.dup1).mean()), 4),
            "retried": round(float(g.cap2.notna().mean()), 4),
            "retry_kept": round(float(use2.mean()), 4),
            "final_below_threshold": round(float((g.critic < THRESHOLD).mean()), 4),
            "near_duplicates_first_pass": int(g.dup1.sum()),
            "mean_words": round(float(words.str.len().mean()), 2),
            "distinct_1": round(len(set(all_tokens)) / max(len(all_tokens), 1), 4),
            "distinct_2": round(len(bigrams) / max(sum(max(len(ws) - 1, 0) for ws in words), 1), 4),
            "mean_critic": round(float(g.critic.mean()), 3),
        }
        (RESULTS / "arm_e_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
