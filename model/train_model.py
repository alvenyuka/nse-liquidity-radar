"""Stage 7: a logistic regression that flags stocks sliding into illiquidity, scored
against the two baselines in baseline/rule_of_thumb.py.

Same population and the same test period as the baseline (SPLIT_TEST_START_YEAR, with an
EMBARGO_DAYS gap between the end of training and the start of testing, because the forward
label looks 5 sessions ahead -- without the gap, the last few training rows' labels would
reach into days the test period is supposed to be judging).

Scored on both populations, same as the baseline:
  full        every row with a valid label.
  transition  only rows not already illiquid today (currently_liquid = 1) -- the honest
              headline, because persistence cannot even be defined usefully here.

Two ways to flag from the model's probability, since a fixed 0.5 cutoff is a poor fit for a
rare label: "same_flags_as_persistence" and "same_flags_as_rule" each take the model's
highest-probability rows, up to the exact count that baseline flagged on the same test rows
-- a like-for-like comparison, not a threshold picked after seeing the test outcomes.

A second run below walks forward one year at a time (WALKFORWARD_FIRST_TEST_YEAR to
WALKFORWARD_LAST_TEST_YEAR), refitting the model on an expanding window each time, scored
only on the transition population -- the single 2023-2024 split is one market regime; the
walk-forward table is the real evidence of whether the model holds up across years.
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "report"))

from rounding import round_numbers  # noqa: E402
from settings import (  # noqa: E402
    EMBARGO_DAYS, FEATURE_COLUMNS, RULE_ZEROS_SHARE_THRESHOLD, SPLIT_TEST_START_YEAR,
    WALKFORWARD_FIRST_TEST_YEAR, WALKFORWARD_LAST_TEST_YEAR,
)

DB_PATH = PROJECT_DIR / "data" / "nse_liquidity.duckdb"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "model_results.json"


def load_population():
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    population = connection.execute(
        """
        SELECT stock_code, session_no, trade_year, avg_volume_20d, volatility_20d,
               zeros_share_20d, volume_zscore_60d, trailing_illiquid_5, currently_liquid,
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


def fit_model(train_rows):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_rows[FEATURE_COLUMNS])
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, train_rows["is_illiquid_next_5d"])
    return scaler, model


def predict_probability(scaler, model, rows):
    x_scaled = scaler.transform(rows[FEATURE_COLUMNS])
    return model.predict_proba(x_scaled)[:, 1]


def flag_top_n(probability, n):
    """The n highest-probability rows flagged, the rest not -- a like-for-like comparison
    with a baseline that flagged exactly n rows on the same test set."""
    flags = np.zeros(len(probability), dtype=bool)
    if n > 0:
        top_indices = np.argsort(-probability)[:n]
        flags[top_indices] = True
    return flags


def score_flags(actual, flagged):
    true_positive = int((flagged & (actual == 1)).sum())
    predicted_positive = int(flagged.sum())
    actual_positive = int((actual == 1).sum())
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / actual_positive if actual_positive > 0 else 0.0
    return {"flagged": predicted_positive, "precision": precision, "recall": recall}


def score_population(test_rows, probability):
    actual = test_rows["is_illiquid_next_5d"]
    persistence_count = int((test_rows["trailing_illiquid_5"] == 1).sum())
    rule_count = int((test_rows["zeros_share_20d"] >= RULE_ZEROS_SHARE_THRESHOLD).sum())
    return {
        "rows_scored": int(len(test_rows)),
        "base_rate": float(actual.mean()),
        "pr_auc": float(average_precision_score(actual, probability)),
        "same_flags_as_persistence": score_flags(actual, flag_top_n(probability, persistence_count)),
        "same_flags_as_rule": score_flags(actual, flag_top_n(probability, rule_count)),
    }


def run_single_split(population):
    """The headline split: train on everything before the embargo, test on SPLIT_TEST_START_YEAR
    onward."""
    test_rows_all_years = population.loc[population["trade_year"] >= SPLIT_TEST_START_YEAR]
    cutoff_session = int(test_rows_all_years["session_no"].min())
    embargo_end = cutoff_session - EMBARGO_DAYS - 1

    train = population.loc[population["session_no"] <= embargo_end]
    test = population.loc[population["session_no"] >= cutoff_session]

    scaler, model = fit_model(train)
    test_probability = predict_probability(scaler, model, test)
    transition_test = test.loc[test["currently_liquid"] == 1]
    transition_probability = test_probability[test["currently_liquid"].values == 1]

    coefficients = {}
    for name, value in zip(FEATURE_COLUMNS, model.coef_[0]):
        coefficients[name] = float(value)

    return {
        "cutoff_session_no": cutoff_session,
        "embargo_days": EMBARGO_DAYS,
        "train_rows": int(len(train)),
        "train_base_rate": float(train["is_illiquid_next_5d"].mean()),
        "features": FEATURE_COLUMNS,
        "coefficients_standardised": coefficients,
        "full": score_population(test, test_probability),
        "transition": score_population(transition_test, transition_probability),
    }


def run_walkforward(population):
    """One row per test year: train on every session strictly before that year (minus the
    embargo), test on that year alone, scored on the transition population only."""
    rows = []
    for test_year in range(WALKFORWARD_FIRST_TEST_YEAR, WALKFORWARD_LAST_TEST_YEAR + 1):
        test_year_rows = population.loc[population["trade_year"] == test_year]
        if len(test_year_rows) == 0:
            continue
        cutoff_session = int(test_year_rows["session_no"].min())
        embargo_end = cutoff_session - EMBARGO_DAYS - 1
        train = population.loc[population["session_no"] <= embargo_end]
        if train["is_illiquid_next_5d"].nunique() < 2:
            continue  # not enough history yet to fit both classes

        transition_test = test_year_rows.loc[test_year_rows["currently_liquid"] == 1]
        if len(transition_test) == 0:
            continue

        scaler, model = fit_model(train)
        probability = predict_probability(scaler, model, transition_test)
        persistence_count = int((transition_test["trailing_illiquid_5"] == 1).sum())  # always 0
        rule_count = int((transition_test["zeros_share_20d"] >= RULE_ZEROS_SHARE_THRESHOLD).sum())
        actual = transition_test["is_illiquid_next_5d"]

        rows.append({
            "test_year": test_year,
            "rows_scored": int(len(transition_test)),
            "base_rate": float(actual.mean()),
            "model_pr_auc": float(average_precision_score(actual, probability)),
            "rule_pr_auc": float(average_precision_score(actual, transition_test["zeros_share_20d"])),
            "persistence_pr_auc": float(actual.mean()),  # persistence never flags here; its PR-AUC is the base rate
            "rule_flags": rule_count,
            "model_same_flags_as_rule": score_flags(actual, flag_top_n(probability, rule_count)),
        })
    return rows


def main():
    population = load_population()
    results = {
        "split_test_start_year": SPLIT_TEST_START_YEAR,
        "single_split": run_single_split(population),
        "walkforward_by_year_transition_only": run_walkforward(population),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = round_numbers(results)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
