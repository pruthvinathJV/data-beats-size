import math

import numpy as np
import pandas as pd
import pytest

from datacurate.calibration import ece, fit_isotonic, mae
from datacurate.judge import digit_probs, summarize
from datacurate.polaris import collapse_ratings, rater_pairs
from datacurate.stats import cluster_bootstrap, corr, correlation_table, go_no_go, group_indices


def test_digit_probs_merges_whitespace_variants_and_ignores_other_tokens():
    top = [("4", math.log(0.5)), (" 4", math.log(0.1)), ("5", math.log(0.3)), ("The", math.log(0.1))]
    p = digit_probs(top)
    assert p == pytest.approx([0, 0, 0, 0.6, 0.3])


def test_summarize_expected_score_and_parse_flag():
    s = summarize(np.array([0, 0, 0, 0.6, 0.3]))
    assert s.expected == pytest.approx((4 * 0.6 + 5 * 0.3) / 0.9)
    assert s.argmax == 4 and s.parse_ok
    empty = summarize(np.zeros(5))
    assert math.isnan(empty.expected) and not empty.parse_ok


def test_isotonic_preserves_rank_order():
    # The reason calibration is reported as ECE/MAE, not Spearman.
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = np.clip(3 + x + rng.normal(scale=0.8, size=2000), 1, 5)
    iso = fit_isotonic(x, y)
    xs = np.sort(x)
    assert np.all(np.diff(iso.predict(xs)) >= 0)
    assert ece(iso.predict(x), y) < ece(x, y)
    assert mae(iso.predict(x), y) < mae(x, y)


def test_ece_zero_when_calibrated():
    y = np.linspace(1, 5, 100)
    assert ece(y, y) == pytest.approx(0)


def test_group_indices_cover_every_row_once():
    groups = ["a", "b", "a", "c", "b", "a"]
    idx = group_indices(groups)
    assert sorted(np.concatenate(idx).tolist()) == list(range(6))
    assert sorted(len(g) for g in idx) == [1, 2, 3]


def test_cluster_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    res = cluster_bootstrap(lambda i: {"mean": x[i].mean()}, np.arange(500) // 2, n_boot=300)
    r = res.loc["mean"]
    assert r.lo < r.point < r.hi


def test_correlation_table_paired_diff_and_gate():
    rng = np.random.default_rng(2)
    n = 3000
    human = rng.normal(size=n)
    df = pd.DataFrame({
        "human": human,
        "judge": human + rng.normal(scale=0.5, size=n),
        "clip": human + rng.normal(scale=1.5, size=n),
        "img": np.arange(n) // 3,
    })
    t = correlation_table(df, "human", ["judge", "clip"], "img", diffs=[("judge", "clip")], n_boot=200)
    assert t.loc["judge - clip", "point"] == pytest.approx(t.loc["judge", "point"] - t.loc["clip", "point"])
    gate = go_no_go(t, t, "judge", "clip", min_gain=0.05)
    assert gate["decision"] == "GO"
    assert go_no_go(t, None, "judge", "clip", 0.05)["decision"].startswith("PENDING")
    assert corr(df["judge"], df["human"]) > corr(df["clip"], df["human"])


def test_collapse_and_rater_pairs():
    df = pd.DataFrame({
        "img_hash": ["h1", "h1", "h1", "h2"],
        "cand": ["a dog", "a dog", "a cat", "a car"],
        "img_bytes": [b"x", b"x", b"x", b"y"],
        "refs": [["r"]] * 4,
        "human": [1.0, 3.0, 5.0, 2.0],
    })
    pairs = collapse_ratings(df)
    assert len(pairs) == 3
    dog = pairs[pairs.cand == "a dog"].iloc[0]
    assert dog.human == 2.0 and dog.n_ratings == 2 and dog.n_words == 2
    rp = rater_pairs(pairs)
    assert len(rp) == 1 and {rp.r1.iloc[0], rp.r2.iloc[0]} == {1.0, 3.0}


def test_rule_mask_regex_and_threshold():
    from datacurate.filters import apply_rules, rule_mask

    df = pd.DataFrame({"caption": ["stock photo of a cat", "a dog on grass", "Stock Illustration"],
                       "n_words": [4, 4, 2]})
    assert rule_mask(df, {"regex": r"stock (photo|illustration)"}).tolist() == [True, False, True]
    assert rule_mask(df, {"regex": "stock", "field": "n_words", "op": ">", "value": 3}).tolist() == [True, False, False]
    reasons = apply_rules(df, {"short": {"field": "n_words", "op": "<", "value": 3}, "stock": {"regex": "stock"}})
    assert reasons.tolist() == ["stock", "", "short"]  # first matching rule wins


def test_max_cosine_keeps_first_duplicate():
    pytest.importorskip("torch")
    from datacurate.filters import max_cosine

    e = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert max_cosine(e, e, exclude_earlier=True).tolist() == [-1.0, 1.0, 0.0]


def test_chair_counts_unsupported_objects(tmp_path):
    import json as _json

    from datacurate.captioning import Chair

    (tmp_path / "syn.txt").write_text("person, man, woman, people\ndog, puppy\nsurfboard, surf board\n")
    (tmp_path / "inst.json").write_text(_json.dumps({
        "categories": [{"id": 1, "name": "person"}, {"id": 18, "name": "dog"}],
        "annotations": [{"image_id": 7, "category_id": 1}]}))
    ch = Chair(tmp_path / "syn.txt", tmp_path / "inst.json")
    assert ch.objects("Two men carry a surf board") == {"person", "surfboard"}
    s = ch.score([7], ["A man walks two puppies"], [["a person walking"]])
    assert s["CHAIRi"] == 0.5 and s["CHAIRs"] == 1.0  # dog is not in the image
