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

The logistic regression's own coefficients are standardised but not comparable across
features: avg_volume_20d is heavy-tailed, zeros_share_20d is bounded [0, 1], so a standard
deviation means something different for each. permutation_importance_by_feature below
answers the same "which feature actually drives the separation" question in a way that is
scale-invariant -- it shuffles one raw feature at a time and reports how much PR-AUC drops,
which is comparable across features regardless of their original scale. run_single_split and
run_walkforward also take a build_model_fn, so model/train_challenger.py can reuse this same
split and scoring logic with a different model.
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
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


def build_model():
    """An unfitted scaler+logistic-regression pipeline. Bundled into one pipeline (rather
    than returned as a separate scaler and model) so permutation_importance can call it
    directly on the raw, unscaled feature columns -- it does its own scaling internally,
    exactly as it will on new data."""
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])


def predict_probability(model, rows):
    return model.predict_proba(rows[FEATURE_COLUMNS])[:, 1]


def logistic_regression_diagnostics(model):
    """Standardised coefficients from the fitted pipeline's logistic regression step -- kept
    for continuity with earlier runs. Not comparable across features (see module docstring);
    permutation_importance_by_feature below is scale-invariant, so it is comparable."""
    coefficients = {}
    for name, value in zip(FEATURE_COLUMNS, model.named_steps["model"].coef_[0]):
        coefficients[name] = float(value)
    return {"coefficients_standardised": coefficients}


def permutation_importance_by_feature(model, transition_test):
    """Mean drop in transition-set PR-AUC when one raw feature's values are shuffled, holding
    the rest fixed -- scale-invariant, so it can be compared across avg_volume_20d (heavy-
    tailed) and zeros_share_20d (bounded [0, 1]) directly."""
    outcome = permutation_importance(
        model, transition_test[FEATURE_COLUMNS], transition_test["is_illiquid_next_5d"],
        scoring="average_precision", n_repeats=20, random_state=42,
    )
    importances = {}
    for name, value in zip(FEATURE_COLUMNS, outcome.importances_mean):
        importances[name] = float(value)
    return importances


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


def run_single_split(population, build_model_fn=build_model, extra_diagnostics_fn=logistic_regression_diagnostics):
    """The headline split: train on everything before the embargo, test on SPLIT_TEST_START_YEAR
    onward. build_model_fn and extra_diagnostics_fn let model/train_challenger.py reuse this same
    split and scoring logic for a different model -- only the model-fitting step and its own
    diagnostics differ."""
    test_rows_all_years = population.loc[population["trade_year"] >= SPLIT_TEST_START_YEAR]
    cutoff_session = int(test_rows_all_years["session_no"].min())
    embargo_end = cutoff_session - EMBARGO_DAYS - 1

    train = population.loc[population["session_no"] <= embargo_end]
    test = population.loc[population["session_no"] >= cutoff_session]

    model = build_model_fn()
    model.fit(train[FEATURE_COLUMNS], train["is_illiquid_next_5d"])
    test_probability = predict_probability(model, test)
    transition_test = test.loc[test["currently_liquid"] == 1]
    transition_probability = test_probability[test["currently_liquid"].values == 1]

    result = {
        "cutoff_session_no": cutoff_session,
        "embargo_days": EMBARGO_DAYS,
        "train_rows": int(len(train)),
        "train_base_rate": float(train["is_illiquid_next_5d"].mean()),
        "features": FEATURE_COLUMNS,
    }
    if extra_diagnostics_fn is not None:
        result.update(extra_diagnostics_fn(model))
    result["permutation_importance_pr_auc_drop"] = permutation_importance_by_feature(model, transition_test)
    result["full"] = score_population(test, test_probability)
    result["transition"] = score_population(transition_test, transition_probability)
    return result


def run_walkforward(population, build_model_fn=build_model):
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

        model = build_model_fn()
        model.fit(train[FEATURE_COLUMNS], train["is_illiquid_next_5d"])
        probability = predict_probability(model, transition_test)
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


def train_years_feature_correlation(population):
    """Pearson correlation between avg_volume_20d and zeros_share_20d on train years only
    (before SPLIT_TEST_START_YEAR) -- backs the README/METHODOLOGY claim that the two
    features carrying most of the logistic regression's weight are only weakly correlated,
    so neither coefficient is just restating the other."""
    train_rows = population.loc[population["trade_year"] < SPLIT_TEST_START_YEAR]
    return float(train_rows["avg_volume_20d"].corr(train_rows["zeros_share_20d"]))


def main():
    population = load_population()
    results = {
        "split_test_start_year": SPLIT_TEST_START_YEAR,
        "single_split": run_single_split(population),
        "walkforward_by_year_transition_only": run_walkforward(population),
        "train_years_feature_correlation_avg_volume_zeros_share": train_years_feature_correlation(population),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = round_numbers(results)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
