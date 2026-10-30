"""Stage 6: two baselines, scored on the test period only, before any model is fit.

persistence   flag = trailing_illiquid_5 (was this stock untraded on any of the last 5
              sessions?). The naive "today's regime is tomorrow's" guess -- and the one a
              model actually has to beat, because is_illiquid_next_5d and trailing_illiquid_5
              overlap heavily by construction (see docs/METHODOLOGY.md).
zeros_rule    flag = zeros_share_20d >= RULE_ZEROS_SHARE_THRESHOLD (untraded at least half
              of the last 20 sessions). Fixed in advance, not fit to the data. Named for the
              zeros_share_20d feature it thresholds, not to be confused with volume_zscore_60d.

Both are scored on two populations of the same test rows:
  full        every row with a valid forward label and complete features.
  transition  the harder, honest headline: only rows that are NOT already illiquid today
              (currently_liquid = 1). Persistence trivially scores well on "full" because
              most already-illiquid stocks stay illiquid; "transition" asks the real
              question -- which of today's liquid names turns illiquid next.

Accuracy is not reported: a rule that flags nothing can still score well on it.
"""
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "report"))

from rounding import round_numbers  # noqa: E402
from settings import RULE_ZEROS_SHARE_THRESHOLD, SPLIT_TEST_START_YEAR, EMBARGO_DAYS  # noqa: E402

DB_PATH = PROJECT_DIR / "data" / "nse_liquidity.duckdb"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "baseline_results.json"


def load_population():
    """Every (stock, session) with a complete feature set and a known forward label."""
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    population = connection.execute(
        """
        SELECT stock_code, session_no, trade_year, zeros_share_20d, volume_zscore_60d,
               avg_volume_20d, volatility_20d, trailing_illiquid_5, currently_liquid,
               is_illiquid_next_5d
        FROM mart_features_labels
        WHERE is_illiquid_next_5d IS NOT NULL
          AND avg_volume_20d IS NOT NULL
          AND volatility_20d IS NOT NULL
          AND zeros_share_20d IS NOT NULL
          AND volume_zscore_60d IS NOT NULL
          AND trailing_illiquid_5 IS NOT NULL
        ORDER BY stock_code, session_no
        """
    ).df()
    connection.close()
    return population


def test_cutoff_session(population):
    """The first session_no belonging to SPLIT_TEST_START_YEAR -- everything from here on
    is the test period; the embargo removes the EMBARGO_DAYS sessions just before it from
    training, so no training row's forward label reaches into the test period."""
    test_rows = population.loc[population["trade_year"] >= SPLIT_TEST_START_YEAR]
    return int(test_rows["session_no"].min())


def score_flags(actual, flagged):
    return {
        "flagged": int(flagged.sum()),
        "precision": float(precision_score(actual, flagged, zero_division=0)),
        "recall": float(recall_score(actual, flagged, zero_division=0)),
    }


def score_population(test_rows):
    actual = test_rows["is_illiquid_next_5d"]
    persistence_flags = test_rows["trailing_illiquid_5"] == 1
    rule_flags = test_rows["zeros_share_20d"] >= RULE_ZEROS_SHARE_THRESHOLD

    return {
        "rows_scored": int(len(test_rows)),
        "base_rate": float(actual.mean()),
        "persistence": {
            "pr_auc": float(average_precision_score(actual, test_rows["trailing_illiquid_5"])),
            **score_flags(actual, persistence_flags),
        },
        "zeros_rule": {
            "threshold": RULE_ZEROS_SHARE_THRESHOLD,
            "pr_auc": float(average_precision_score(actual, test_rows["zeros_share_20d"])),
            **score_flags(actual, rule_flags),
        },
    }


def main():
    population = load_population()
    cutoff_session = test_cutoff_session(population)
    embargo_end = cutoff_session - EMBARGO_DAYS - 1
    test_rows = population.loc[population["session_no"] >= cutoff_session]

    results = {
        "split_test_start_year": SPLIT_TEST_START_YEAR,
        "cutoff_session_no": cutoff_session,
        "embargo_days": EMBARGO_DAYS,
        "train_rows_available": int((population["session_no"] <= embargo_end).sum()),
        "full": score_population(test_rows),
        "transition": score_population(test_rows.loc[test_rows["currently_liquid"] == 1]),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = round_numbers(results)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
