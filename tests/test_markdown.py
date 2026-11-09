import itertools

import pytest

from priceopt import baseline_cadences, optimize_markdown, simulate

PARAMS = dict(inventory=300, full_price=100.0, base_units=25, elasticity=-2.0, unit_cost=45.0,
              salvage_value=15.0, demand_decay=0.05)


def test_markdowns_only_deepen():
    plan = optimize_markdown(weeks=10, **PARAMS)
    depths = plan.schedule["markdown_pct"].tolist()
    assert depths == sorted(depths)


def test_beats_every_rule_of_thumb():
    plan = optimize_markdown(weeks=12, **PARAMS)
    for cadence in baseline_cadences(12).values():
        assert plan.total_value >= simulate(cadence, **PARAMS).total_value - 1e-6


def test_matches_brute_force_on_small_problem():
    ladder = (0.0, 0.2, 0.4)
    weeks = 5
    plan = optimize_markdown(weeks=weeks, ladder=ladder, **PARAMS)
    best = max(
        simulate(list(path), ladder=ladder, **PARAMS).total_value
        for path in itertools.product(ladder, repeat=weeks)
        if list(path) == sorted(path)
    )
    assert plan.total_value == pytest.approx(best)


def test_ample_demand_needs_no_markdown():
    plan = optimize_markdown(weeks=6, **{**PARAMS, "inventory": 50, "demand_decay": 0.0})
    assert plan.schedule["markdown_pct"].max() == 0
    assert plan.leftover_units == 0
