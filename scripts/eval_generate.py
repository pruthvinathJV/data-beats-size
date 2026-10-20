"""Stage 7a: caption the 5K Karpathy test images with the base model and every trained adapter.

    python scripts/eval_generate.py            # all runs under runs/ plus the base model ("0")
"""

import argparse
import io

import pandas as pd
import torch
from PIL import Image
from train_lora import MODEL, RUNS, conversation, load_processor

from datacurate.io import DATA

OUT = DATA / "eval" / "gen"
BATCH = 64
MAX_NEW = 40


@torch.inference_mode()
def caption_all(model, proc, images: list[Image.Image]) -> list[str]:
    prompt = proc.apply_chat_template(conversation(None), add_generation_prompt=True)
    out = []
    for s in range(0, len(images), BATCH):
        chunk = images[s : s + BATCH]
        enc = proc(text=[prompt] * len(chunk), images=[[im] for im in chunk], return_tensors="pt", padding=True)
        enc = {k: v.cuda() for k, v in enc.items()}
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        ids = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        texts = proc.batch_decode(ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        out += [t.strip().split("\n")[0] for t in texts]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run names; default: base + every finished run")
    args = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    runs = args.runs or ["0"] + sorted(p.name for p in RUNS.iterdir() if (p / "done.json").exists())
    test = pd.read_parquet(DATA / "coco" / "karpathy_test.parquet", columns=["cocoid", "img"])
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in test.img]
    proc = load_processor()
    proc.tokenizer.padding_side = "left"
    OUT.mkdir(parents=True, exist_ok=True)
    for name in runs:
        path = OUT / f"{name}.parquet"
        if path.exists():
            continue
        model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
        if name != "0":
            model = PeftModel.from_pretrained(model, RUNS / name).merge_and_unload()
        model.eval()
        caps = caption_all(model, proc, images)
        pd.DataFrame({"cocoid": test.cocoid, "caption": caps}).to_parquet(path)
        print(f"{name}: {len(caps)} captions, e.g. {caps[0]!r}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
