"""Stage 3: agent-proposed caption filter rules, human approval required.

The agent (an LLM served by vLLM) sees judge-vs-human disagreements and the
author's failure-mode tags, and can only act through tools:

  examples(tag, n)    hand-rated captions with a failure tag, with rating and note
  disagreements(n)    items where the calibrated judge and the author differ most
  test_rule(rule)     rule stats on the hand-rated set and its removal rate on the raw pool
  propose(rule, why)  record a rule that was tested first
  finish()

Proposals go to results/agent/proposals.json. Nothing is applied: the author
approves or rejects each one, and only approved rules are copied into
configs/filters.yaml (decisions in results/agent/decisions.json).

    python scripts/agent_rules.py
"""

import json
import re

import pandas as pd

from datacurate.filters import add_features, rule_mask
from datacurate.io import DATA, RESULTS, load_config

MAX_STEPS = 30
MAX_PROPOSALS = 6
MAX_POOL_REMOVAL = 0.15  # a rule removing more than this is too blunt to propose

SYSTEM = """You are a data-curation agent improving a filter for web image-caption pairs used to train an image captioning model.
A human rated 200 pairs from 1 (bad) to 5 (good) and tagged failure modes. Captions rated 1-2 are bad training data.
Your goal: propose up to {max_p} simple caption filter rules that remove bad captions while keeping good ones.

A rule is JSON: {{"name": "snake_case", "regex": "<case-insensitive Python regex on the caption>"}}
optionally with a numeric condition: {{"field": "n_words", "op": ">"|"<", "value": <number>}} (both must hold).

Tools (call exactly one per turn):
- examples: {{"tag": "<tag>", "n": 8}}  tags: spam, not_description, text_copy, unverifiable, hallucination, grammar, vague, missing_detail, wrong_subject
- disagreements: {{"n": 10}}
- test_rule: {{"rule": {{...}}}}  returns how many hand-rated captions it fires on, their mean rating, precision (share rated 1-2), and the share of the 100K raw pool it would remove
- propose: {{"rule": {{...}}, "why": "<one sentence>"}}  only for a rule you already tested; aim for precision >= 0.7 and pool removal <= {max_pool:.0%}
- finish: {{}}

Reply with ONLY a JSON object: {{"thought": "<short reasoning>", "tool": "<name>", "args": {{...}}}}"""


class Tools:
    def __init__(self, hand: pd.DataFrame, pool: pd.DataFrame):
        self.hand, self.pool = hand, pool
        self.tested: dict[str, dict] = {}
        self.proposals: list[dict] = []

    def examples(self, tag: str, n: int = 8):
        rows = self.hand[self.hand.tags.fillna("").str.contains(re.escape(tag))].head(int(n))
        return [{"caption": r.caption[:200], "rating": int(r.rating), "note": (r.note or "")[:150]}
                for r in rows.itertuples()]

    def disagreements(self, n: int = 10):
        d = self.hand.assign(gap=self.hand.judge_cal - self.hand.rating)
        d = d.reindex(d.gap.abs().sort_values(ascending=False).index).head(int(n))
        return [{"caption": r.caption[:200], "human": int(r.rating), "judge": round(r.judge_cal, 2),
                 "tags": r.tags or ""} for r in d.itertuples()]

    def test_rule(self, rule: dict):
        try:
            hit = rule_mask(self.hand, rule)
            pool_rate = float(rule_mask(self.pool, rule).mean())
        except (re.error, KeyError, TypeError) as e:
            return {"error": f"invalid rule: {e}"}
        fired = self.hand[hit]
        stats = {
            "fires_on_hand_rated": int(hit.sum()),
            "mean_rating_when_fired": round(float(fired.rating.mean()), 2) if len(fired) else None,
            "precision_bad": round(float((fired.rating <= 2).mean()), 2) if len(fired) else None,
            "recall_bad": round(float((fired.rating <= 2).sum() / max((self.hand.rating <= 2).sum(), 1)), 2),
            "pool_removal": round(pool_rate, 4),
        }
        self.tested[json.dumps(rule, sort_keys=True)] = stats
        return stats

    def propose(self, rule: dict, why: str):
        stats = self.tested.get(json.dumps(rule, sort_keys=True))
        if stats is None:
            return {"error": "test_rule this exact rule first"}
        if stats["pool_removal"] > MAX_POOL_REMOVAL:
            return {"error": f"removes {stats['pool_removal']:.1%} of the pool; too blunt"}
        self.proposals.append({"rule": rule, "why": why, "test": stats})
        return {"ok": True, "proposals": len(self.proposals)}


def main():
    cfg = load_config("configs/polaris_calibration.yaml")
    from vllm import LLM, SamplingParams

    from datacurate.calibration import fit_isotonic

    train = pd.read_parquet(DATA / "scores" / "polaris_train.parquet")
    iso = fit_isotonic(train.judge.fillna(train.judge.median()), train.human)
    hand = pd.read_parquet(DATA / "scores" / "handrate.parquet")
    hand["judge_cal"] = iso.predict(hand.judge.fillna(hand.judge.median()))
    labels = pd.read_csv(DATA / "handrate" / "ratings.csv", dtype={"key": str})[["key", "tags", "note"]]
    hand = hand.merge(labels, on="key")
    hand[["tags", "note"]] = hand[["tags", "note"]].astype(object).where(hand[["tags", "note"]].notna(), None)
    pool = add_features(pd.read_parquet(DATA / "pool" / "raw", columns=["caption", "n_words", "width", "height", "gray_std"]))
    tools = Tools(hand, pool)

    llm = LLM(model=cfg["judge"]["model"], max_model_len=16384, gpu_memory_utilization=0.85, seed=cfg["seed"])
    params = SamplingParams(temperature=0.2, max_tokens=600, seed=cfg["seed"])
    messages = [{"role": "system", "content": SYSTEM.format(max_p=MAX_PROPOSALS, max_pool=MAX_POOL_REMOVAL)},
                {"role": "user", "content": "Start by looking at the failure modes and disagreements."}]
    transcript = []
    for step in range(MAX_STEPS):
        reply = llm.chat(messages, params, use_tqdm=False)[0].outputs[0].text.strip()
        messages.append({"role": "assistant", "content": reply})
        try:
            call = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
            tool, args = call["tool"], call.get("args", {})
            if tool == "finish":
                transcript.append({"step": step, "call": call})
                break
            result = getattr(tools, tool)(**args) if tool in {"examples", "disagreements", "test_rule", "propose"} \
                else {"error": f"unknown tool {tool}"}
        except (ValueError, KeyError, TypeError) as e:
            call, result = {"raw": reply}, {"error": f"could not parse a tool call: {e}"}
        transcript.append({"step": step, "call": call, "result": result})
        print(f"[{step}] {call.get('tool', '?')}: {json.dumps(result)[:160]}")
        if len(tools.proposals) >= MAX_PROPOSALS:
            break
        messages.append({"role": "user", "content": json.dumps({"result": result})})

    out = RESULTS / "agent"
    out.mkdir(parents=True, exist_ok=True)
    (out / "proposals.json").write_text(json.dumps(tools.proposals, indent=2) + "\n")
    (out / "transcript.json").write_text(json.dumps(transcript, indent=2) + "\n")
    print(f"{len(tools.proposals)} proposals -> {out / 'proposals.json'}")


if __name__ == "__main__":
    main()
