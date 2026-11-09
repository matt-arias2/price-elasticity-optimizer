"""Sales data: synthetic generator (with known ground truth) and CSV loader."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["week", "sku", "category", "price", "units", "unit_cost"]
OPTIONAL_COLUMNS = ["promo", "competitor_price"]

# category: (SKU prefix, mean own-price elasticity, list price range, weekly base units range)
CATEGORY_PROFILES = {
    "Sofas": ("SOF", -1.6, (699, 1899), (15, 60)),
    "Dining Tables": ("DIN", -2.0, (399, 1299), (10, 45)),
    "Mattresses": ("MAT", -1.2, (499, 1999), (20, 80)),
    "Area Rugs": ("RUG", -2.6, (99, 499), (40, 160)),
    "Lighting": ("LGT", -1.9, (49, 299), (50, 200)),
}

CROSS_ELASTICITY = 0.5   # demand response to competitor price
PROMO_LIFT = 0.15        # log-lift from promo visibility, on top of the price cut


def generate_sales(
    n_skus_per_category: int = 6,
    n_weeks: int = 156,
    seed: int = 42,
    start: str = "2023-11-06",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate weekly SKU sales from a constant-elasticity demand model.

    Returns (sales, truth). `truth` holds each SKU's true elasticity so the
    estimator can be checked against it. The last SKU in each category is a
    slow mover, which is where category shrinkage earns its keep.
    """
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)
    sales, truth = [], []

    for category, (prefix, mean_e, (pmin, pmax), (umin, umax)) in CATEGORY_PROFILES.items():
        phase = rng.uniform(0, 2 * np.pi)
        season = 0.2 * np.sin(2 * np.pi * t / 52 + phase)

        for j in range(n_skus_per_category):
            sku = f"{prefix}-{j + 1:03d}"
            true_e = mean_e + rng.normal(0, 0.3)
            list_price = float(np.floor(rng.uniform(pmin, pmax))) + 0.99
            unit_cost = round(list_price * rng.uniform(0.42, 0.58), 2)
            slow_mover = j == n_skus_per_category - 1
            base_units = rng.uniform(umin, umax) * (0.12 if slow_mover else 1.0)

            # Regular price is piecewise constant with occasional permanent moves
            regular = np.empty(n_weeks)
            level = list_price
            for w in range(n_weeks):
                if w > 0 and rng.random() < 0.08:
                    level *= rng.uniform(0.88, 1.12)
                regular[w] = level

            promo = rng.random(n_weeks) < 0.15
            depth = rng.choice([0.10, 0.15, 0.20, 0.25, 0.30], n_weeks)
            price = np.round(np.where(promo, regular * (1 - depth), regular), 2)
            competitor = np.round(list_price * np.exp(np.cumsum(rng.normal(0, 0.02, n_weeks))), 2)

            mu = (
                base_units
                * (price / list_price) ** true_e
                * (competitor / list_price) ** CROSS_ELASTICITY
                * np.exp(season + PROMO_LIFT * promo + 0.002 * t)
            )
            # Gamma-Poisson noise: overdispersed counts, like real store data
            units = rng.poisson(mu * rng.gamma(50, 1 / 50, n_weeks))

            sales.append(pd.DataFrame({
                "week": weeks,
                "sku": sku,
                "category": category,
                "price": price,
                "units": units,
                "unit_cost": unit_cost,
                "promo": promo.astype(int),
                "competitor_price": competitor,
            }))
            truth.append({"sku": sku, "category": category, "true_elasticity": round(true_e, 3)})

    return pd.concat(sales, ignore_index=True), pd.DataFrame(truth)


def load_sales(path_or_buffer) -> pd.DataFrame:
    """Load and validate a sales CSV. See README for the expected schema."""
    df = pd.read_csv(path_or_buffer)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df["week"] = pd.to_datetime(df["week"])
    for col in ["price", "units", "unit_cost"] + [c for c in OPTIONAL_COLUMNS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "promo" not in df.columns:
        df["promo"] = 0

    bad = df["price"].le(0) | df["units"].lt(0) | df[["price", "units", "unit_cost"]].isna().any(axis=1)
    if bad.any():
        df = df[~bad]
    return df.sort_values(["sku", "week"]).reset_index(drop=True)
