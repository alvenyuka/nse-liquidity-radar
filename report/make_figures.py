"""Stage 9: charts from outputs/results.json, outputs/model_results.json and
outputs/challenger_results.json.

quintile_chart.png      Forward illiquidity rate by zeros_share_20d quintile, computed on
                         TRAIN years only (before SPLIT_TEST_START_YEAR) -- this chart, not
                         the PR-AUC number alone, is what shows whether the signal actually
                         separates risk, and computing it on train years keeps the test
                         period untouched by anything that shaped the chart.
walkforward_chart.png   Model vs. challenger vs. rule vs. persistence PR-AUC, one point per
                         test year, on the transition-only population -- the evidence that
                         the single 2023-2024 split is not one lucky year, and that neither
                         model consistently beats the other.
permutation_chart.png   Mean PR-AUC drop per feature when that feature alone is shuffled
                         (model/train_model.py's permutation_importance_by_feature), for both
                         models side by side -- scale-invariant, unlike the logistic
                         regression's own standardised coefficients, so it is the chart that
                         actually answers "which feature drives the separation".
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from settings import SPLIT_TEST_START_YEAR  # noqa: E402

DB_PATH = PROJECT_DIR / "data" / "nse_liquidity.duckdb"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = PROJECT_DIR / "figures"


def make_quintile_chart():
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    train_rows = connection.execute(
        f"""
        SELECT zeros_share_20d, is_illiquid_next_5d
        FROM mart_features_labels
        WHERE trade_year < {SPLIT_TEST_START_YEAR}
          AND zeros_share_20d IS NOT NULL
          AND is_illiquid_next_5d IS NOT NULL
        """
    ).df()
    connection.close()

    train_rows["quintile"] = pd_qcut_rank(train_rows["zeros_share_20d"])
    by_quintile = train_rows.groupby("quintile")["is_illiquid_next_5d"].mean()

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(by_quintile.index.astype(str), by_quintile.values, color="#3d5a00")
    axis.set_xlabel("zeros_share_20d quintile (1 = least untraded, 5 = most untraded)")
    axis.set_ylabel("share that became illiquid in the next 5 sessions")
    axis.set_title("Forward illiquidity rate by quintile (train years only, before 2023)")
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "quintile_chart.png", dpi=150)
    plt.close(figure)


def pd_qcut_rank(values):
    """1 to 5, by rank, not by pandas.qcut's bin edges -- a feature with many tied values
    at 0 (most rows have zeros_share_20d exactly 0) breaks qcut's usual bin-edge approach."""
    ranks = values.rank(method="first")
    raw_quintile = ranks * 5 / (len(values) + 1)
    quintile_numbers = []
    for rank_value in raw_quintile:
        quintile_number = int(rank_value) + 1
        if quintile_number > 5:
            quintile_number = 5
        quintile_numbers.append(quintile_number)
    return pd.Series(quintile_numbers, index=values.index)


def make_walkforward_chart():
    model_results = json.loads((OUTPUT_DIR / "model_results.json").read_text())
    challenger_results = json.loads((OUTPUT_DIR / "challenger_results.json").read_text())
    rows = model_results["walkforward_by_year_transition_only"]
    challenger_rows = challenger_results["walkforward_by_year_transition_only"]
    challenger_pr_auc_by_year = {}
    for row in challenger_rows:
        challenger_pr_auc_by_year[row["test_year"]] = row["model_pr_auc"]

    years = [row["test_year"] for row in rows]
    model_pr_auc = [row["model_pr_auc"] for row in rows]
    challenger_pr_auc = [challenger_pr_auc_by_year[year] for year in years]
    rule_pr_auc = [row["rule_pr_auc"] for row in rows]
    persistence_pr_auc = [row["persistence_pr_auc"] for row in rows]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(years, model_pr_auc, marker="o", label="logistic regression", color="#3d5a00")
    axis.plot(years, challenger_pr_auc, marker="s", label="challenger (gradient-boosted trees)", color="#1d3b6e")
    axis.plot(years, rule_pr_auc, marker="o", label="zeros_rule", color="#c2410c")
    axis.plot(years, persistence_pr_auc, marker="o", label="persistence (= base rate)", color="#9fbc00")
    axis.set_xlabel("test year")
    axis.set_ylabel("PR-AUC (transition population only)")
    axis.set_title("Walk-forward PR-AUC by year: both models vs. rule vs. persistence")
    axis.legend()
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "walkforward_chart.png", dpi=150)
    plt.close(figure)


def make_permutation_chart():
    model_results = json.loads((OUTPUT_DIR / "model_results.json").read_text())
    challenger_results = json.loads((OUTPUT_DIR / "challenger_results.json").read_text())
    model_importance = model_results["single_split"]["permutation_importance_pr_auc_drop"]
    challenger_importance = challenger_results["single_split"]["permutation_importance_pr_auc_drop"]

    feature_names = list(model_importance.keys())
    model_values = [model_importance[name] for name in feature_names]
    challenger_values = [challenger_importance[name] for name in feature_names]

    bar_positions = np.arange(len(feature_names))
    bar_width = 0.35

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(bar_positions - bar_width / 2, model_values, bar_width, label="logistic regression", color="#3d5a00")
    axis.bar(bar_positions + bar_width / 2, challenger_values, bar_width, label="challenger", color="#1d3b6e")
    axis.axhline(0, color="#12161f", linewidth=0.8)
    axis.set_xticks(bar_positions)
    axis.set_xticklabels(feature_names, rotation=20, ha="right")
    axis.set_ylabel("mean PR-AUC drop when shuffled (transition test set)")
    axis.set_title("Permutation importance: which feature actually drives the separation")
    axis.legend()
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "permutation_chart.png", dpi=150)
    plt.close(figure)


def main():
    make_quintile_chart()
    make_walkforward_chart()
    make_permutation_chart()
    print(f"wrote {(FIGURES_DIR / 'quintile_chart.png').relative_to(PROJECT_DIR)}, "
          f"{(FIGURES_DIR / 'walkforward_chart.png').relative_to(PROJECT_DIR)} "
          f"and {(FIGURES_DIR / 'permutation_chart.png').relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
