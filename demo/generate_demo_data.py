"""Write the synthetic demo dataset to sample_data/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from priceopt import generate_sales  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "sample_data"
    out.mkdir(exist_ok=True)
    sales, truth = generate_sales()
    sales.to_csv(out / "retail_sales_sample.csv", index=False)
    truth.to_csv(out / "true_elasticities.csv", index=False)
    print(f"Wrote {len(sales):,} rows ({sales['sku'].nunique()} SKUs) to {out}")
