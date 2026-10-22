"""Mechanics of the model-ladder experiment (not its empirical claims).

The sweep is a measurement, not a monotonicity theorem: these tests pin the
protocol (paired seeds, fixed calibration, honest ranges, digests) so the
numbers it prints are comparable between runs.
"""
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))

import model_ladder as ML                                   # noqa: E402


def test_spearman_known_relations():
    assert ML._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ML._spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert ML._spearman([1, 1, 1], [1, 2, 3]) is None          # no variance
    assert ML._spearman([1, 2], [2, 1]) is None                # n < 3


def test_mean_ci_shrinks_with_more_seeds():
    m, ci = ML._mean_ci([0.9, 0.9, 0.9, 0.9])
    assert m == pytest.approx(0.9) and ci == pytest.approx(0.0)
    m, ci = ML._mean_ci([0.8, 1.0])
    assert m == pytest.approx(0.9) and ci > 0
    assert ML._mean_ci([None, None]) == (None, None)


def _rec(seed, cap, size, rec=0.2):
    return {"dataset": "d", "kind": "multiclass", "seed": seed,
            "capacity": cap, "size": size, "n_train": int(100 * size),
            "model_metric": 0.8, "prec": 0.9, "rec": rec,
            "claim_rate": rec, "haz_prec": 0.5, "haz_rec": 0.4,
            "haz_coverage": rec, "certified": 0.9, "n_test": 10}


def test_aggregate_groups_cells_and_reports_seed_count():
    recs = [_rec(0, "poor", 0.5), _rec(1, "poor", 0.5, rec=0.4),
            _rec(0, "rich", 1.0)]
    agg = ML.aggregate(recs)
    assert len(agg) == 2
    poor = next(r for r in agg if r["capacity"] == "poor")
    assert poor["n_seeds"] == 2
    assert poor["rec"] == pytest.approx(0.3)
    assert poor["rec_ci"] > 0
    rich = next(r for r in agg if r["capacity"] == "rich")
    assert rich["n_seeds"] == 1 and rich["rec_ci"] == 0.0


def test_paired_effect_requires_at_least_two_pairs():
    one_seed = [_rec(0, "poor", 1.0, rec=0.1), _rec(0, "rich", 1.0, rec=0.3)]
    eff = ML.paired_effect(one_seed)
    assert eff["rec"]["mean_diff"] == pytest.approx(0.2)
    assert eff["rec"]["excludes_zero"] is False        # cannot claim significance

    two = one_seed + [_rec(1, "poor", 1.0, rec=0.12),
                      _rec(1, "rich", 1.0, rec=0.32)]
    eff = ML.paired_effect(two)
    assert eff["rec"]["n_pairs"] == 2


def test_sweeps_are_protocol_consistent():
    sizes = (0.30, 1.0)
    caps = {"poor": ML.CAPACITIES["poor"], "rich": ML.CAPACITIES["rich"]}
    recs = (ML.sweep_multiclass(sizes, caps, (0,), log=lambda *_: None) +
            ML.sweep_regression(sizes, caps, (0,), log=lambda *_: None))
    assert len(recs) == 8

    for r in recs:
        assert 0.0 <= r["claim_rate"] <= 1.0
        assert r["n_test"] > 0
        for key in ("prec", "rec", "haz_prec", "haz_rec", "haz_coverage"):
            assert r[key] is None or 0.0 <= r[key] <= 1.0
        assert r["policy_digest"].startswith("sha256:")
        assert r["model_digest"].startswith("sha256:")
        assert r["calibration_digest"].startswith("sha256:")

    for kind in ("diabetes", "wine"):
        cells = [r for r in recs if r["dataset"] == kind]
        # the calibration ROWS are fixed across the sweep: the learner gets
        # the data, kappa does not (coverage stays comparable). kappa's own
        # digest moves because the model's scores change.
        assert len({r["cal_rows_digest"] for r in cells}) == 1
        assert len({r["n_cal"] for r in cells}) == 1
        assert len({r["n_test"] for r in cells}) == 1
        assert len({r["policy_digest"] for r in cells}) == 1
        # capacity actually changes the learner
        for size in sizes:
            at = [r for r in cells if r["size"] == size]
            assert len({r["model_digest"] for r in at}) == 2
            assert len({r["n_train"] for r in at}) == 1
