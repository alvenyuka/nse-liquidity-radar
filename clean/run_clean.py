"""Stage 2: clean the raw price and sector tables and write parquet files plus the rule log."""
from pathlib import Path

from clean_prices import clean_prices
from clean_sectors import clean_sectors
from rule_log import clear_log, save_log

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "clean"


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    clear_log()

    print("prices")
    prices = clean_prices()

    print("sectors")
    sectors = clean_sectors()

    prices.to_parquet(CLEAN_DIR / "prices.parquet", index=False)
    sectors.to_parquet(CLEAN_DIR / "sectors.parquet", index=False)
    save_log(CLEAN_DIR / "rule_log.csv")

    print(f"\nclean rows: prices {len(prices)}, sectors {len(sectors)}")
    print(f"distinct stock codes in prices: {prices['stock_code'].nunique()}")
    print(f"date range: {prices['trade_date'].min().date()} to {prices['trade_date'].max().date()}")


if __name__ == "__main__":
    main()
