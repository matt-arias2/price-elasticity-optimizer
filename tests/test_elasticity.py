import numpy as np
import pandas as pd

from priceopt import estimate_elasticities, generate_sales
from priceopt.elasticity import fit_poisson


def test_poisson_recovers_known_coefficient():
    rng = np.random.default_rng(0)
    n = 5000
    log_p = rng.normal(0, 0.2, n)
    X = np.column_stack([np.ones(n), log_p])
    y = rng.poisson(np.exp(3.0 - 2.0 * log_p))
    beta, cov, _, converged = fit_poisson(X, y.astype(float))
    assert converged
    assert abs(beta[1] - -2.0) < 3 * np.sqrt(cov[1, 1])


def test_estimates_track_ground_truth():
    sales, truth = generate_sales(seed=7)
    est = estimate_elasticities(sales).merge(truth, on="sku")
    assert (est["elasticity_shrunk"] - est["true_elasticity"]).abs().mean() < 0.3
    assert est["true_elasticity"].between(est["ci_low"], est["ci_high"]).mean() >= 0.8


def test_shrinkage_reduces_error_on_average():
    raw, shrunk = [], []
    for seed in range(5):
        sales, truth = generate_sales(seed=seed)
        est = estimate_elasticities(sales).merge(truth, on="sku")
        raw.append((est["elasticity"] - est["true_elasticity"]).abs().mean())
        shrunk.append((est["elasticity_shrunk"] - est["true_elasticity"]).abs().mean())
    assert np.mean(shrunk) < np.mean(raw)


def test_constant_price_is_flagged_not_estimated():
    weeks = pd.date_range("2025-01-06", periods=52, freq="W-MON")
    sales = pd.DataFrame({"week": weeks, "sku": "X", "category": "C", "price": 10.0,
                          "units": np.arange(52) % 7 + 1, "unit_cost": 5.0, "promo": 0})
    est = estimate_elasticities(sales)
    assert est.loc[0, "status"] == "price never changed"
    assert np.isnan(est.loc[0, "elasticity"])
