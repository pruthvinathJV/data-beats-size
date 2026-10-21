"""Stage 3: agent-proposed caption filter rules, human approval required.

The agent (an LLM served by vLLM) sees judge-vs-human disagreements and the
author's failure-mode tags, and can only act through tools:

  examples(tag, n)    hand-rated captions with a failure tag, with rating and note
  disagreements(n)    items where the calibrated judge and the author differ most
  test_rule(rule)     rule stats on the hand-rated set and its removal rate on the raw pool
  pool_examples(rule) sample raw-pool captions the rule would remove (to spot false positives)
  propose(rule, why)  record a rule that was tested first, has enough evidence, and is not a near-duplicate
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

MAX_STEPS = 40
MAX_PROPOSALS = 6
MAX_POOL_REMOVAL = 0.15  # a rule removing more than this is too blunt to propose
MIN_SUPPORT = 4  # a rule must fire on at least this many hand-rated captions
MIN_PRECISION = 0.7  # share of its hand-rated hits that are rated 1-2 (added after round 2; see decisions.json)
MAX_OVERLAP = 0.8  # a proposal may share at most this much of its pool removals with an earlier one

SYSTEM = """You are a data-curation agent improving a filter for web image-caption pairs used to train an image captioning model.
A human rated 200 pairs from 1 (bad) to 5 (good) and tagged failure modes. Captions rated 1-2 are bad training data.
Your goal: propose up to {max_p} simple caption filter rules that remove bad captions while keeping good ones.

A rule is JSON: {{"name": "snake_case", "regex": "<case-insensitive Python regex on the caption>"}}
optionally with a numeric condition: {{"field": "n_words"|"n_commas", "op": ">"|"<", "value": <number>}} (both must hold).
The failure tags are the human's labels, NOT words that appear in captions. Write regexes for concrete caption text patterns,
e.g. stock-photo boilerplate ("stock photo", "royalty free", "vector illustration"), comma-separated SEO keyword lists
(many commas), product codes and sizes, prices, URLs, site names, listicle or how-to titles.
Each proposal should target a different failure pattern; near-duplicates of earlier proposals are rejected.

Tools (call exactly one per turn):
- examples: {{"tag": "<tag>", "n": 8}}  tags: spam, not_description, text_copy, unverifiable, hallucination, grammar, vague, missing_detail, wrong_subject
- disagreements: {{"n": 10}}
- test_rule: {{"rule": {{...}}}}  returns how many hand-rated captions it fires on, their mean rating, precision (share rated 1-2), and the share of the 100K raw pool it would remove
- pool_examples: {{"rule": {{...}}, "n": 8}}  random raw-pool captions the rule would remove; check they are really bad
- propose: {{"rule": {{...}}, "why": "<one sentence>"}}  only after test_rule on the identical rule; needs >= {min_support} hand-rated hits, precision >= 0.7, pool removal <= {max_pool:.0%}
- finish: {{}}

Reply with ONLY a JSON object: {{"thought": "<short reasoning>", "tool": "<name>", "args": {{...}}}}"""


class Tools:
    def __init__(self, hand: pd.DataFrame, pool: pd.DataFrame):
        self.hand = hand.assign(n_commas=hand.caption.str.count(","))
        self.pool = pool.assign(n_commas=pool.caption.str.count(","))
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

    def pool_examples(self, rule: dict, n: int = 8):
        try:
            hit = self.pool[rule_mask(self.pool, rule)]
        except (re.error, KeyError, TypeError) as e:
            return {"error": f"invalid rule: {e}"}
        return [c[:160] for c in hit.caption.sample(min(int(n), len(hit)), random_state=0)] if len(hit) else []

    def propose(self, rule: dict, why: str):
        stats = self.tested.get(json.dumps(rule, sort_keys=True))
        if stats is None:
            return {"error": "call test_rule with exactly this rule first, then propose it"}
        if stats["fires_on_hand_rated"] < MIN_SUPPORT:
            return {"error": f"fires on {stats['fires_on_hand_rated']} hand-rated captions; need >= {MIN_SUPPORT}"}
        if stats["pool_removal"] > MAX_POOL_REMOVAL:
            return {"error": f"removes {stats['pool_removal']:.1%} of the pool; too blunt"}
        if (stats["precision_bad"] or 0) < MIN_PRECISION:
            return {"error": f"precision {stats['precision_bad']} < {MIN_PRECISION}: it mostly removes captions rated 3+"}
        hit = rule_mask(self.pool, rule)
        for p in self.proposals:
            prev = rule_mask(self.pool, p["rule"])
            if (hit & prev).sum() / max(hit.sum(), 1) > MAX_OVERLAP:
                return {"error": f"near-duplicate of proposal '{p['rule'].get('name')}'; target a different pattern"}
        self.proposals.append({"rule": rule, "why": why, "test": stats})
        covered = sorted({p["rule"].get("name", "") for p in self.proposals})
        return {"ok": True, "proposals": len(self.proposals), "already_covered": covered}


def main():
    cfg = load_config("configs/default.yaml")
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
    messages = [{"role": "system", "content": SYSTEM.format(max_p=MAX_PROPOSALS, max_pool=MAX_POOL_REMOVAL, min_support=MIN_SUPPORT)},
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
            result = getattr(tools, tool)(**args) if tool in {"examples", "disagreements", "test_rule", "pool_examples", "propose"} \
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
