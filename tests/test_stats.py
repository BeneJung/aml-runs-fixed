"""Tests for the statistical primitives used in merge_e2_stats."""

import pytest

from merge_e2_stats import holm_bonferroni, paired_t


def test_holm_preserves_smallest_ranks():
    """Smallest raw p gets the largest multiplier (m); largest gets m=1."""
    raw = {"a": 0.05, "b": 0.01, "c": 0.001}
    out = holm_bonferroni(raw, alpha=0.05)
    assert out["c"]["multiplier"] == 3
    assert out["b"]["multiplier"] == 2
    assert out["a"]["multiplier"] == 1


def test_holm_monotone():
    """Adjusted p must be monotone non-decreasing in raw-p order."""
    raw = {"a": 0.001, "b": 0.006, "c": 0.105}
    out = holm_bonferroni(raw, alpha=0.05)
    sorted_by_raw = sorted(raw.items(), key=lambda kv: kv[1])
    adj_seq = [out[label]["adjusted"] for label, _ in sorted_by_raw]
    assert all(adj_seq[i] <= adj_seq[i + 1] for i in range(len(adj_seq) - 1))


def test_holm_reproduces_reference():
    """The README's headline 1v4 / 2v4 / 1v2 family."""
    raw = {"2v4": 0.001, "1v4": 0.006, "1v2": 0.105}
    out = holm_bonferroni(raw, alpha=0.05)
    # 2v4: smallest, m=3 -> 0.003
    # 1v4: m=2 -> 0.012
    # 1v2: m=1 -> 0.105
    assert out["2v4"]["adjusted"] == pytest.approx(0.003, abs=1e-6)
    assert out["1v4"]["adjusted"] == pytest.approx(0.012, abs=1e-6)
    assert out["1v2"]["adjusted"] == pytest.approx(0.105, abs=1e-6)
    assert out["2v4"]["reject"] is True
    assert out["1v4"]["reject"] is True
    assert out["1v2"]["reject"] is False


def test_holm_clamps_at_one():
    """Adjusted p must never exceed 1."""
    raw = {"a": 0.6, "b": 0.7}
    out = holm_bonferroni(raw, alpha=0.05)
    for _, info in out.items():
        assert info["adjusted"] <= 1.0


def test_paired_t_recovers_known_value():
    """Hand-checked: diffs = [1, 2, 3]; mean=2, sd=1, n=3
    t = 2 / (1/sqrt(3)) = 3.4641"""
    res = paired_t([1.0, 2.0, 3.0])
    assert res["n"] == 3
    assert res["mean_diff"] == pytest.approx(2.0)
    assert res["sd_diff"] == pytest.approx(1.0)
    assert res["t"] == pytest.approx(2.0 * (3 ** 0.5), abs=1e-4)
