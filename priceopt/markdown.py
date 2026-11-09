"""Clearance markdown planning by dynamic programming.

Problem: N units must be sold over T weeks. Each week we pick a markdown depth
from a ladder (0%, 10%, 20%, ...). Markdowns can only deepen, matching how
retail clearance works, and whatever is left at the end goes for salvage.
Weekly demand follows the SKU's elasticity curve and can fade as the season ages.

State = (week, inventory left, current depth). Bellman recursion:

    V[t, d, inv] = max over d' >= d of
                   price(d') * sold + V[t+1, d', inv - sold],
                   sold = min(inv, demand[t, d'])

Every cadence is checked (not a heuristic), and it runs in milliseconds for
realistic sizes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_LADDER = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass
class MarkdownPlan:
    schedule: pd.DataFrame
    revenue: float
    salvage: float
    leftover_units: int
    sell_through_pct: float
    gross_margin: float

    @property
    def total_value(self) -> float:
        return self.revenue + self.salvage


def _demand_table(weeks, full_price, base_units, elasticity, ladder, demand_decay):
    prices = full_price * (1 - np.asarray(ladder))
    fade = (1 - demand_decay) ** np.arange(weeks)
    demand = np.rint(base_units * fade[:, None] * (prices / full_price)[None, :] ** elasticity)
    return prices, demand.astype(int)


def simulate(depth_by_week, inventory, full_price, base_units, elasticity, unit_cost,
             salvage_value=0.0, demand_decay=0.0, ladder=DEFAULT_LADDER) -> MarkdownPlan:
    """Play out a given markdown schedule (list of depths, one per week)."""
    ladder = list(ladder)
    prices, demand = _demand_table(len(depth_by_week), full_price, base_units, elasticity, ladder, demand_decay)
    inv, rows = inventory, []
    for t, depth in enumerate(depth_by_week):
        d = ladder.index(depth)
        sold = min(inv, demand[t, d])
        rows.append({
            "week": t + 1,
            "markdown_pct": round(depth * 100),
            "price": round(prices[d], 2),
            "demand": int(demand[t, d]),
            "units_sold": int(sold),
            "inventory_end": int(inv - sold),
            "revenue": round(sold * prices[d], 2),
        })
        inv -= sold
    schedule = pd.DataFrame(rows)
    revenue = float(schedule["revenue"].sum())
    salvage = inv * salvage_value
    return MarkdownPlan(
        schedule=schedule,
        revenue=revenue,
        salvage=salvage,
        leftover_units=int(inv),
        sell_through_pct=(1 - inv / inventory) * 100 if inventory else 0.0,
        gross_margin=revenue + salvage - inventory * unit_cost,
    )


def optimize_markdown(inventory, weeks, full_price, base_units, elasticity, unit_cost,
                      salvage_value=0.0, demand_decay=0.0, ladder=DEFAULT_LADDER) -> MarkdownPlan:
    """Revenue-maximizing markdown cadence (inventory cost is sunk)."""
    ladder = list(ladder)
    L = len(ladder)
    prices, demand = _demand_table(weeks, full_price, base_units, elasticity, ladder, demand_decay)
    inv = np.arange(inventory + 1)

    value = np.tile(inv * float(salvage_value), (L, 1))   # V at end of season, indexed [depth, inv]
    policy = np.zeros((weeks, L, inventory + 1), dtype=int)

    for t in range(weeks - 1, -1, -1):
        new_value = np.full((L, inventory + 1), -np.inf)
        for d in range(L):
            for d2 in range(d, L):                        # markdowns only deepen
                sold = np.minimum(inv, demand[t, d2])
                v = sold * prices[d2] + value[d2, inv - sold]
                better = v > new_value[d] + 1e-9          # ties go to the shallower depth
                new_value[d] = np.where(better, v, new_value[d])
                policy[t, d] = np.where(better, d2, policy[t, d])
        value = new_value

    d, left, path = 0, inventory, []
    for t in range(weeks):
        d = policy[t, d, left]
        path.append(ladder[d])
        left -= min(left, demand[t, d])

    return simulate(path, inventory, full_price, base_units, elasticity, unit_cost,
                    salvage_value, demand_decay, ladder)


def baseline_cadences(weeks: int) -> dict[str, list[float]]:
    """Common rules of thumb to benchmark the optimized plan against."""
    third = max(weeks // 3, 1)
    return {
        "Hold full price": [0.0] * weeks,
        "Standard cadence (0/30/50%)": [0.0] * third + [0.3] * third + [0.5] * (weeks - 2 * third),
        "Immediate 40% off": [0.4] * weeks,
    }
