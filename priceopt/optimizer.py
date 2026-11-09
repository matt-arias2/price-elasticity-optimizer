"""Price recommendations from estimated elasticities.

Demand is projected with the constant-elasticity curve fitted for each SKU:

    Q(P) = Q0 * (P / P0) ** e

With unit cost c, profit (P - c) * Q(P) peaks at P* = c * e / (1 + e) when
demand is elastic (e < -1). When demand is inelastic (e > -1) profit keeps
rising with price, so the guardrail is what stops it. Real pricing teams cap
moves for brand and competitive reasons, and the model is only trusted near
prices it has actually observed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def current_state(sales: pd.DataFrame, lookback_weeks: int = 13) -> pd.DataFrame:
    """Per-SKU current regular price, baseline weekly units, and unit cost."""
    rows = []
    for sku, g in sales.sort_values("week").groupby("sku"):
        recent = g.tail(lookback_weeks)
        regular = recent[recent["promo"] == 0] if (recent["promo"] == 0).any() else recent
        rows.append({
            "sku": sku,
            "current_price": round(float(regular.tail(4)["price"].median()), 2),
            "baseline_units": round(float(regular["units"].mean()), 2),
            "unit_cost": float(g["unit_cost"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def project(price, current_price, baseline_units, elasticity, unit_cost) -> dict:
    """Weekly units, revenue and gross margin at `price` (scalar or array)."""
    units = baseline_units * (np.asarray(price) / current_price) ** elasticity
    revenue = price * units
    return {"units": units, "revenue": revenue, "margin": revenue - unit_cost * units}


def unconstrained_optimum(elasticity: float, unit_cost: float) -> float:
    return unit_cost * elasticity / (1 + elasticity) if elasticity < -1 else np.inf


def candidate_prices(low: float, high: float, ending: float = 0.99) -> np.ndarray:
    """Charm-priced candidates (e.g. 42.99, 43.99, ...) between low and high."""
    grid = np.arange(np.floor(low), np.ceil(high) + 1) + ending
    grid = grid[(grid >= low) & (grid <= high)]
    return grid if grid.size else np.array([round(low, 2)])


def recommend_price(
    elasticity: float,
    std_error: float,
    current_price: float,
    baseline_units: float,
    unit_cost: float,
    max_change: float = 0.15,
    objective: str = "margin",
    min_margin_pct: float = 0.0,
    max_std_error: float = 0.5,
) -> dict:
    hold = {"recommended_price": current_price, "action": "Hold"}
    if not np.isfinite(elasticity):
        return {**hold, "rationale": "No usable elasticity estimate"}
    if elasticity >= 0:
        return {**hold, "rationale": f"Non-negative elasticity ({elasticity:.2f}); check data before acting"}
    if std_error > max_std_error:
        return {**hold, "rationale": f"Low confidence (SE {std_error:.2f}); collect more price variation first"}

    low, high = current_price * (1 - max_change), current_price * (1 + max_change)
    cands = candidate_prices(low, high)
    grid = np.append(cands, current_price)
    proj = project(grid, current_price, baseline_units, elasticity, unit_cost)
    score = proj["margin"] if objective == "margin" else proj["revenue"]
    margin_pct = np.where(grid > 0, (grid - unit_cost) / grid, -np.inf)
    score = np.where(margin_pct >= min_margin_pct, score, -np.inf)
    best = float(grid[int(np.argmax(score))])

    p_star = unconstrained_optimum(elasticity, unit_cost)
    capped = best >= cands.max() or best <= cands.min()
    if abs(best - current_price) < 0.005:
        action, why = "Hold", "Current price is already near the optimum within guardrails"
    elif elasticity > -1:
        action = "Raise"
        why = f"Inelastic (e={elasticity:.2f}): volume barely reacts, so a higher price grows revenue and margin"
    elif best > current_price:
        action = "Raise"
        why = f"Priced below the profit-maximizing level (P*=${p_star:,.2f})"
    else:
        action = "Lower"
        why = f"Elastic (e={elasticity:.2f}): priced above P*=${p_star:,.2f}, extra volume outweighs the cut"
    if capped and action != "Hold":
        why += f"; capped by the ±{max_change:.0%} guardrail"
    return {"recommended_price": round(best, 2), "action": action, "rationale": why}


def recommend_prices(
    estimates: pd.DataFrame,
    state: pd.DataFrame,
    max_change: float = 0.15,
    objective: str = "margin",
    min_margin_pct: float = 0.0,
    max_std_error: float = 0.5,
) -> pd.DataFrame:
    df = estimates[["sku", "category", "elasticity_shrunk", "std_error_shrunk"]].merge(state, on="sku")
    rows = []
    for r in df.itertuples(index=False):
        rec = recommend_price(
            r.elasticity_shrunk, r.std_error_shrunk, r.current_price, r.baseline_units, r.unit_cost,
            max_change, objective, min_margin_pct, max_std_error,
        )
        e = r.elasticity_shrunk if np.isfinite(r.elasticity_shrunk) else 0.0
        now = project(r.current_price, r.current_price, r.baseline_units, e, r.unit_cost)
        new = project(rec["recommended_price"], r.current_price, r.baseline_units, e, r.unit_cost)
        rows.append({
            "sku": r.sku,
            "category": r.category,
            "elasticity": r.elasticity_shrunk,
            "unit_cost": r.unit_cost,
            "current_price": r.current_price,
            "recommended_price": rec["recommended_price"],
            "price_change_pct": round((rec["recommended_price"] / r.current_price - 1) * 100, 1),
            "action": rec["action"],
            "current_units": round(float(now["units"]), 1),
            "projected_units": round(float(new["units"]), 1),
            "current_revenue": round(float(now["revenue"]), 2),
            "projected_revenue": round(float(new["revenue"]), 2),
            "current_margin": round(float(now["margin"]), 2),
            "projected_margin": round(float(new["margin"]), 2),
            "margin_lift": round(float(new["margin"] - now["margin"]), 2),
            "rationale": rec["rationale"],
        })
    return pd.DataFrame(rows)


def portfolio_summary(recs: pd.DataFrame, weeks_per_year: int = 52) -> dict:
    cur_m, new_m = recs["current_margin"].sum(), recs["projected_margin"].sum()
    cur_r, new_r = recs["current_revenue"].sum(), recs["projected_revenue"].sum()
    return {
        "skus": len(recs),
        "raise": int((recs["action"] == "Raise").sum()),
        "lower": int((recs["action"] == "Lower").sum()),
        "hold": int((recs["action"] == "Hold").sum()),
        "weekly_margin_current": cur_m,
        "weekly_margin_projected": new_m,
        "margin_lift_pct": (new_m / cur_m - 1) * 100 if cur_m else np.nan,
        "revenue_change_pct": (new_r / cur_r - 1) * 100 if cur_r else np.nan,
        "annualized_margin_lift": (new_m - cur_m) * weeks_per_year,
    }
