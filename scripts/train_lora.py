"""Stage 6: LoRA fine-tune SmolVLM-256M-Instruct on one arm with one seed.

Identical hyperparameters for every arm; only the data and the seed differ.

    python scripts/train_lora.py --arm B --seed 0
"""

import argparse
import io
import json
import math
import time

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from datacurate.io import DATA, init_wandb, load_config

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
PROMPT = "Describe this image in one sentence."
HP = {"lr": 2e-4, "batch_size": 32, "epochs": 1, "warmup_steps": 50, "lora_r": 16, "lora_alpha": 32,
      "lora_dropout": 0.05, "weight_decay": 0.0, "max_caption_tokens": 64}
TARGETS = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
RUNS = DATA.parent / "runs"


def load_processor():
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    proc.image_processor.do_image_splitting = False  # one 512px tile -> fixed 64 image tokens
    return proc


def conversation(caption: str | None) -> list[dict]:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    if caption is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": caption}]})
    return msgs


class ArmData(Dataset):
    def __init__(self, arm: str):
        df = pd.read_parquet(DATA / "arms" / f"{arm}.parquet", columns=["key", "caption"])
        raw = pd.read_parquet(DATA / "pool" / "raw", columns=["key", "img"])
        img = dict(zip(raw.key, raw.img))
        self.items = [(img[k], c) for k, c in zip(df.key, df.caption)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        b, cap = self.items[i]
        return Image.open(io.BytesIO(b)).convert("RGB"), cap


class Collate:
    def __init__(self, proc, prompt_len: int):
        self.proc, self.prompt_len = proc, prompt_len

    def __call__(self, batch):
        images, caps = zip(*batch)
        tok = self.proc.tokenizer
        caps = [tok.convert_tokens_to_string(tok.tokenize(c)[: HP["max_caption_tokens"]]) for c in caps]
        texts = [self.proc.apply_chat_template(conversation(c)) for c in caps]
        enc = self.proc(text=texts, images=[[im] for im in images], return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[:, : self.prompt_len] = -100  # loss only on the caption (prompt + image tokens are a fixed length)
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test only; output goes to runs/_smoke")
    args = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup

    run_name = f"{args.arm}_s{args.seed}"
    out = RUNS / ("_smoke" if args.max_steps else run_name)
    if (out / "done.json").exists():
        print(f"{run_name} already trained")
        return
    torch.manual_seed(args.seed)
    proc = load_processor()
    proc.tokenizer.padding_side = "right"
    probe = proc(text=[proc.apply_chat_template(conversation(None), add_generation_prompt=True)],
                 images=[[Image.new("RGB", (64, 64))]], return_tensors="pt")
    prompt_len = probe["input_ids"].shape[1]

    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(r=HP["lora_r"], lora_alpha=HP["lora_alpha"],
                                             lora_dropout=HP["lora_dropout"], target_modules=TARGETS))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = ArmData(args.arm)
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(data, batch_size=HP["batch_size"], shuffle=True, generator=g, num_workers=4,
                        collate_fn=Collate(proc, prompt_len), drop_last=True, persistent_workers=True)
    steps = args.max_steps or HP["epochs"] * len(loader)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=HP["lr"],
                            weight_decay=HP["weight_decay"])
    sched = get_cosine_schedule_with_warmup(opt, HP["warmup_steps"], steps)
    cfg = load_config("configs/default.yaml") | {"run_name": run_name}
    run = init_wandb(cfg, "train", arm=args.arm, seed=args.seed, hp=HP, trainable_params=trainable)

    model.train()
    t0, step, losses = time.time(), 0, []
    for _ in range(HP["epochs"]):
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss.item())
            if step >= steps:
                break
            if step % 25 == 0:
                run.log({"loss": sum(losses[-25:]) / 25, "lr": sched.get_last_lr()[0]}, step=step)
                print(f"{run_name} step {step}/{steps} loss {sum(losses[-25:]) / 25:.4f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    summary = {"arm": args.arm, "seed": args.seed, "steps": step, "examples": step * HP["batch_size"],
               "final_loss": sum(losses[-50:]) / min(len(losses), 50), "minutes": round((time.time() - t0) / 60, 2),
               "trainable_params": trainable, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "hp": HP}
    assert math.isfinite(summary["final_loss"])
    (out / "done.json").write_text(json.dumps(summary, indent=2) + "\n")
    run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
