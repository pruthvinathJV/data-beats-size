"""Rank correlations with image-clustered bootstrap CIs, and the pre-registered gate.

Captions of the same image are correlated, so we resample images, not rows.
Paired differences use the same resample for both scorers.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def corr(a, b, method: str = "spearman") -> float:
    if method == "spearman":
        return float(stats.spearmanr(a, b).statistic)
    if method == "kendall":
        return float(stats.kendalltau(a, b).statistic)
    raise ValueError(method)


def group_indices(groups: Sequence) -> list[np.ndarray]:
    codes, _ = pd.factorize(pd.Series(groups))
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes))[:-1])


def cluster_bootstrap(
    stat_fn: Callable[[np.ndarray], dict[str, float]],
    groups: Sequence,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> pd.DataFrame:
    by_group = group_indices(groups)
    rng = np.random.default_rng(seed)
    point = stat_fn(np.arange(len(groups)))
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(by_group), len(by_group))
        draws.append(stat_fn(np.concatenate([by_group[i] for i in pick])))
    draws = pd.DataFrame(draws)
    a = (1 - ci) / 2
    return pd.DataFrame({"point": pd.Series(point), "lo": draws.quantile(a), "hi": draws.quantile(1 - a)})


def correlation_table(
    df: pd.DataFrame,
    target: str,
    scorers: Sequence[str],
    groups: str,
    diffs: Sequence[tuple[str, str]] = (),
    method: str = "spearman",
    **boot,
) -> pd.DataFrame:
    """Correlation of each scorer with `target`, plus paired differences `a - b`."""
    y = df[target].to_numpy()
    xs = {s: df[s].to_numpy() for s in scorers}

    def stat(idx):
        vals = {s: corr(x[idx], y[idx], method) for s, x in xs.items()}
        vals.update({f"{a} - {b}": vals[a] - vals[b] for a, b in diffs})
        return vals

    return cluster_bootstrap(stat, df[groups].to_numpy(), **boot)


def go_no_go(
    polaris: pd.DataFrame,
    handrate: pd.DataFrame | None,
    judge: str,
    clip: str,
    min_gain: float,
) -> dict:
    """The gate from PREREGISTRATION.md for running the judge over the full pool.

    Polaris: judge Spearman beats CLIPScore by >= min_gain with non-overlapping CIs.
    Hand-rated CC12M: judge is at least tied, i.e. not significantly worse
    (upper bound of the paired CI on judge - clip is >= 0).
    """
    j, c, d = polaris.loc[judge], polaris.loc[clip], polaris.loc[f"{judge} - {clip}"]
    gain = float(j.point - c.point)
    polaris_pass = bool(gain >= min_gain and j.lo > c.hi)
    out = {
        "polaris_gain": gain,
        "polaris_ci_judge": [float(j.lo), float(j.hi)],
        "polaris_ci_clip": [float(c.lo), float(c.hi)],
        "polaris_paired_diff_ci": [float(d.lo), float(d.hi)],  # reported, not the gate
        "polaris_pass": polaris_pass,
    }
    if handrate is None:
        out.update(handrate_pass=None, decision="PENDING (hand-rated set not scored)")
        return out
    h = handrate.loc[f"{judge} - {clip}"]
    hand_pass = bool(h.hi >= 0)
    out.update(
        handrate_diff=float(h.point),
        handrate_diff_ci=[float(h.lo), float(h.hi)],
        handrate_pass=hand_pass,
        decision="GO" if polaris_pass and hand_pass else "NO-GO",
    )
    return out
