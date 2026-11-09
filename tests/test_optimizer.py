import numpy as np
import pytest

from priceopt.optimizer import candidate_prices, recommend_price, unconstrained_optimum


def test_closed_form_optimum():
    # e = -2, c = 50 -> P* = 50 * -2 / -1 = 100
    assert unconstrained_optimum(-2.0, 50.0) == pytest.approx(100.0)
    assert unconstrained_optimum(-0.8, 50.0) == np.inf


def test_lowers_price_when_above_optimum():
    rec = recommend_price(-2.0, 0.1, current_price=120.0, baseline_units=100, unit_cost=50.0, max_change=0.3)
    assert rec["action"] == "Lower"
    assert 99 <= rec["recommended_price"] <= 101   # nearest .99 price to P* = 100


def test_inelastic_demand_hits_guardrail():
    rec = recommend_price(-0.5, 0.1, current_price=100.0, baseline_units=100, unit_cost=40.0, max_change=0.10)
    assert rec["action"] == "Raise"
    assert rec["recommended_price"] <= 110.0
    assert "guardrail" in rec["rationale"]


@pytest.mark.parametrize("e, se", [(0.3, 0.1), (-2.0, 0.9), (np.nan, np.nan)])
def test_holds_when_estimate_unreliable(e, se):
    rec = recommend_price(e, se, current_price=100.0, baseline_units=100, unit_cost=40.0)
    assert rec["action"] == "Hold"
    assert rec["recommended_price"] == 100.0


def test_min_margin_constraint_respected():
    rec = recommend_price(-4.0, 0.1, current_price=100.0, baseline_units=100, unit_cost=60.0,
                          max_change=0.3, min_margin_pct=0.35)
    p = rec["recommended_price"]
    assert (p - 60.0) / p >= 0.35


def test_candidate_prices_use_charm_endings():
    grid = candidate_prices(42.5, 57.5)
    assert np.allclose(grid % 1, 0.99)
    assert grid.min() >= 42.5 and grid.max() <= 57.5
