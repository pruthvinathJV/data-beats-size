"""Pool filtering: heuristic and agent-proposed caption rules, near-duplicate
removal and decontamination on CLIP image embeddings."""

import re

import numpy as np
import pandas as pd

OPS = {"<": np.less, ">": np.greater, "<=": np.less_equal, ">=": np.greater_equal}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        min_side=np.minimum(df.width, df.height),
        aspect=np.maximum(df.width, df.height) / np.maximum(np.minimum(df.width, df.height), 1),
    )


def rule_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    """True where the rule fires (the row should be removed).

    A rule is either a threshold {field, op, value} or a caption regex {regex}
    (case-insensitive), optionally both, in which case both must hold.
    """
    mask = pd.Series(True, index=df.index)
    if "regex" in rule:
        mask &= df.caption.str.contains(re.compile(rule["regex"], re.IGNORECASE), na=False)
    if "field" in rule:
        mask &= OPS[rule["op"]](df[rule["field"]].to_numpy(), rule["value"])
    return mask


def apply_rules(df: pd.DataFrame, rules: dict[str, dict]) -> pd.Series:
    """Name of the first rule that removes each row, or '' if kept."""
    reason = pd.Series("", index=df.index, dtype=object)
    for name, rule in rules.items():
        hit = rule_mask(df, rule) & (reason == "")
        reason[hit] = name
    return reason


def max_cosine(queries: np.ndarray, keys: np.ndarray, chunk: int = 8192, exclude_earlier: bool = False):
    """For each query, the max cosine to any key (embeddings are L2-normalized).

    With exclude_earlier (queries is keys), only keys with a smaller index count,
    so the first occurrence of a near-duplicate group is kept.
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k = torch.from_numpy(keys.astype(np.float32)).to(dev)
    out = np.empty(len(queries), dtype=np.float32)
    for s in range(0, len(queries), chunk):
        q = torch.from_numpy(queries[s : s + chunk].astype(np.float32)).to(dev)
        sim = q @ k.T
        if exclude_earlier:
            rows = torch.arange(s, s + len(q), device=dev)[:, None]
            sim = sim.masked_fill(torch.arange(len(k), device=dev)[None, :] >= rows, -1.0)
        out[s : s + len(q)] = sim.max(dim=1).values.cpu().numpy()
    return out
