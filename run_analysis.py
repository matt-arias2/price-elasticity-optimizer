"""Estimate elasticities and recommend prices from a sales CSV.

    python run_analysis.py                                  # synthetic demo data
    python run_analysis.py --input my_sales.csv             # your own data
    python run_analysis.py --max-change 0.10 --objective revenue --min-margin 0.35
"""
import argparse
from pathlib import Path

import pandas as pd

from priceopt import (current_state, estimate_elasticities, generate_sales, load_sales,
                      portfolio_summary, recommend_prices)

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="sales CSV (default: synthetic demo data)")
    ap.add_argument("--max-change", type=float, default=0.15, help="max price move per SKU, e.g. 0.15 = ±15%%")
    ap.add_argument("--objective", choices=["margin", "revenue"], default="margin")
    ap.add_argument("--min-margin", type=float, default=0.0, help="minimum gross margin %% as a fraction")
    ap.add_argument("--no-shrink", action="store_true", help="skip category shrinkage")
    ap.add_argument("--out", default=str(ROOT / "output"))
    args = ap.parse_args()

    if args.input:
        sales, truth = load_sales(args.input), None
    else:
        sales, truth = generate_sales()
    print(f"Loaded {len(sales):,} rows · {sales['sku'].nunique()} SKUs · "
          f"{sales['week'].min():%Y-%m-%d} to {sales['week'].max():%Y-%m-%d}")

    est = estimate_elasticities(sales, shrink=not args.no_shrink)
    recs = recommend_prices(est, current_state(sales), args.max_change, args.objective, args.min_margin)
    s = portfolio_summary(recs)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    est.to_csv(out / "elasticities.csv", index=False)
    recs.to_csv(out / "price_recommendations.csv", index=False)

    print(f"\nRecommendations: {s['raise']} raise · {s['lower']} lower · {s['hold']} hold")
    print(f"Weekly gross margin: ${s['weekly_margin_current']:,.0f} -> ${s['weekly_margin_projected']:,.0f} "
          f"({s['margin_lift_pct']:+.1f}%), revenue {s['revenue_change_pct']:+.1f}%")
    print(f"Annualized margin lift: ${s['annualized_margin_lift']:,.0f}")

    if truth is not None:
        cmp = est.merge(truth, on="sku")
        mae_raw = (cmp["elasticity"] - cmp["true_elasticity"]).abs().mean()
        mae_shrunk = (cmp["elasticity_shrunk"] - cmp["true_elasticity"]).abs().mean()
        coverage = ((cmp["ci_low"] <= cmp["true_elasticity"]) & (cmp["true_elasticity"] <= cmp["ci_high"])).mean()
        print(f"\nModel check vs ground truth: MAE raw {mae_raw:.3f} · shrunk {mae_shrunk:.3f} · "
              f"95% CI coverage {coverage:.0%}")

    top = recs.sort_values("margin_lift", ascending=False).head(10)
    with pd.option_context("display.width", 160, "display.max_colwidth", 70):
        print("\nTop 10 opportunities (weekly):")
        print(top[["sku", "elasticity", "current_price", "recommended_price", "action",
                   "margin_lift", "rationale"]].to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
