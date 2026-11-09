"""Own-price elasticity estimation.

Each SKU gets a Poisson pseudo-maximum-likelihood (PPML) regression:

    E[units] = exp(b0 + e*log(price) + x*log(competitor_price) + g*promo
                   + seasonality + trend)

The coefficient `e` on log(price) is the elasticity directly. PPML keeps
zero-sales weeks (a log-log OLS has to drop them, which biases slow movers),
and sandwich standard errors stay valid when counts are overdispersed.

Estimates are then shrunk toward their category mean (empirical Bayes), so a
noisy slow-mover borrows strength from its siblings instead of producing a
wild price recommendation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_WEEKS = 26
LOW_PRICE_VARIATION = 0.02   # std of log(price) below this = elasticity is barely identified
# Floor on between-SKU variance: SKUs in a category are assumed to differ by at
# least ~0.25 in elasticity. With only a handful of SKUs the moment estimate of
# tau² is often ~0, which over-shrinks and gives overconfident intervals.
MIN_TAU2 = 0.0625


def _design_matrix(g: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    t = (g["week"] - g["week"].min()).dt.days.to_numpy() / 7.0
    log_p = np.log(g["price"].to_numpy())
    cols = {
        "const": np.ones(len(g)),
        "log_price": log_p - log_p.mean(),
        "promo": g["promo"].fillna(0).to_numpy(dtype=float),
        "sin1": np.sin(2 * np.pi * t / 52),
        "cos1": np.cos(2 * np.pi * t / 52),
        "trend": t / 52,
    }
    if "competitor_price" in g.columns and g["competitor_price"].notna().any():
        comp = g["competitor_price"].ffill().bfill().to_numpy()
        cols["log_comp_price"] = np.log(comp) - np.log(comp).mean()
    # Drop controls that don't vary for this SKU (e.g. never promoted)
    names = [n for n, v in cols.items() if n == "const" or np.ptp(v) > 0]
    return np.column_stack([cols[n] for n in names]), names


def fit_poisson(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-8):
    """Poisson GLM via IRLS. Returns (beta, robust covariance, pseudo R², converged)."""
    n, k = X.shape
    beta = np.linalg.lstsq(X, np.log(y + 0.5), rcond=None)[0]
    converged = False
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XtW = X.T * mu
        new_beta = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(new_beta - beta)) < tol:
            beta, converged = new_beta, True
            break
        beta = new_beta

    mu = np.exp(np.clip(X @ beta, -30, 30))
    bread = np.linalg.pinv((X.T * mu) @ X)
    meat = (X.T * (y - mu) ** 2) @ X
    cov = bread @ meat @ bread * n / max(n - k, 1)

    def deviance(m):
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / m), 0.0)
        return 2 * np.sum(term - (y - m))

    null_dev = deviance(np.full_like(y, y.mean(), dtype=float))
    pseudo_r2 = 1 - deviance(mu) / null_dev if null_dev > 0 else np.nan
    return beta, cov, pseudo_r2, converged


def estimate_sku(g: pd.DataFrame) -> dict:
    g = g.sort_values("week")
    out = {
        "sku": g["sku"].iloc[0],
        "category": g["category"].iloc[0],
        "weeks": len(g),
        "avg_weekly_units": round(g["units"].mean(), 1),
        "price_variation": round(float(np.log(g["price"]).std()), 4),
    }
    if len(g) < MIN_WEEKS or g["units"].sum() == 0:
        return {**out, "elasticity": np.nan, "std_error": np.nan, "status": "insufficient data"}

    X, names = _design_matrix(g)
    if "log_price" not in names:
        return {**out, "elasticity": np.nan, "std_error": np.nan, "status": "price never changed"}

    beta, cov, r2, converged = fit_poisson(X, g["units"].to_numpy(dtype=float))
    i = names.index("log_price")
    out.update({
        "elasticity": beta[i],
        "std_error": float(np.sqrt(cov[i, i])),
        "pseudo_r2": round(r2, 3),
        "status": "ok" if converged else "did not converge",
    })
    if "promo" in names:
        out["promo_lift_pct"] = round((np.exp(beta[names.index("promo")]) - 1) * 100, 1)
    if "log_comp_price" in names:
        out["cross_elasticity"] = round(beta[names.index("log_comp_price")], 3)
    if out["price_variation"] < LOW_PRICE_VARIATION:
        out["status"] = "low price variation"
    return out


def shrink_to_category(est: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes partial pooling of SKU elasticities within each category.

    weight = tau² / (tau² + se²), where tau² is the between-SKU variance left
    after removing sampling noise. Precise SKUs keep their own estimate; noisy
    ones move toward the category mean.
    """
    est = est.copy()
    est["category_mean"] = np.nan
    est["category_mean_var"] = 0.0
    est["shrink_weight"] = 1.0
    ok = est["elasticity"].notna() & est["std_error"].notna()

    for cat, g in est[ok].groupby("category"):
        if len(g) < 3:
            continue
        se2 = g["std_error"] ** 2
        tau2 = max(g["elasticity"].var(ddof=1) - se2.mean(), MIN_TAU2)
        w = 1 / (tau2 + se2)
        mean = float((g["elasticity"] * w).sum() / w.sum())
        est.loc[g.index, "category_mean"] = mean
        est.loc[g.index, "category_mean_var"] = 1 / w.sum()
        est.loc[g.index, "shrink_weight"] = tau2 / (tau2 + se2)

    w = est["shrink_weight"]
    has_mean = est["category_mean"].notna()
    est["elasticity_shrunk"] = np.where(
        has_mean, est["category_mean"] + w * (est["elasticity"] - est["category_mean"]), est["elasticity"]
    )
    # Posterior variance, including uncertainty in the category mean itself
    post_var = w * est["std_error"] ** 2 + (1 - w) ** 2 * est["category_mean_var"]
    est["std_error_shrunk"] = np.where(has_mean, np.sqrt(post_var), est["std_error"])
    est = est.drop(columns="category_mean_var")
    est["ci_low"] = est["elasticity_shrunk"] - 1.96 * est["std_error_shrunk"]
    est["ci_high"] = est["elasticity_shrunk"] + 1.96 * est["std_error_shrunk"]
    return est


def estimate_elasticities(sales: pd.DataFrame, shrink: bool = True) -> pd.DataFrame:
    est = pd.DataFrame([estimate_sku(g) for _, g in sales.groupby("sku")])
    if shrink:
        est = shrink_to_category(est)
    else:
        est["elasticity_shrunk"] = est["elasticity"]
        est["std_error_shrunk"] = est["std_error"]
        est["ci_low"] = est["elasticity"] - 1.96 * est["std_error"]
        est["ci_high"] = est["elasticity"] + 1.96 * est["std_error"]
    for c in ["elasticity", "std_error", "elasticity_shrunk", "std_error_shrunk", "ci_low", "ci_high",
              "category_mean", "shrink_weight"]:
        if c in est.columns:
            est[c] = est[c].astype(float).round(3)
    return est
