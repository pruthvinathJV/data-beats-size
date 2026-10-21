"""Stage 4b: build arms A-D and exploratory D* (file "Dx"), 25K each.
Arm E reuses arm B's images (scripts/recaption.py). See PREREGISTRATION.md, A2.

    python scripts/make_arms.py            # gate decision read from results/polaris_gate.json
"""

import json

import pandas as pd

from datacurate.io import DATA, RESULTS, load_config

N = 25_000


def main():
    cfg = load_config("configs/default.yaml")
    seed = cfg["seed"]
    pool = pd.read_parquet(DATA / "pool" / "pool.parquet", columns=["key", "caption", "removed_by", "clipscore"])
    b = pool[pool.removed_by == ""]
    gate = json.loads((RESULTS / "polaris_gate.json").read_text())

    arms = {
        "A": pool.sample(N, random_state=seed),
        "B": b.sample(N, random_state=seed),
        "C": b.nlargest(N, "clipscore"),
    }
    if gate["decision"] == "GO":
        j = b.merge(pd.read_parquet(DATA / "pool" / "judge_scores.parquet"), on="key")
        arms["D"] = j.nlargest(N, "judge")
        d_rule = "judge top-k"
    else:  # pre-registered fallback: CLIPScore top-k, judge as tiebreaker (or CLIPScore alone if unscored)
        path = DATA / "pool" / "judge_scores.parquet"
        j = b.merge(pd.read_parquet(path), on="key") if path.exists() else b.assign(judge=0.0)
        arms["D"] = j.sort_values(["clipscore", "judge"], ascending=False).head(N)
        d_rule = "CLIPScore top-k, judge tiebreak (gate failed)"

    judge_path = DATA / "pool" / "judge_scores.parquet"
    if judge_path.exists():  # A2: exploratory judge-selected arm, built whatever the gate said
        arms["Dx"] = b.merge(pd.read_parquet(judge_path), on="key").nlargest(N, "judge")

    out = DATA / "arms"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"pool": len(pool), "b_pool": len(b), "keep_rate_C_D": round(N / len(b), 3), "D_rule": d_rule}
    for name, df in arms.items():
        df[["key", "caption"]].to_parquet(out / f"{name}.parquet")
        summary[name] = {"n": len(df), "mean_words": round(float(df.caption.str.split().str.len().mean()), 2),
                         "mean_clipscore": round(float(df.clipscore.mean()), 4)}
    summary["C_D_overlap"] = round(len(set(arms["C"].key) & set(arms["D"].key)) / N, 3)
    if "Dx" in arms:
        summary["C_Dx_overlap"] = round(len(set(arms["C"].key) & set(arms["Dx"].key)) / N, 3)
    (RESULTS / "arms.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
