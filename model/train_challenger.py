"""Stage 7b: a challenger model, scored exactly the same way as model/train_model.py's
logistic regression, so the two are a fair comparison -- same population, same single split,
same walk-forward years, same flag-matching rules. The only thing that differs is the model
itself: HistGradientBoostingClassifier, a gradient-boosted tree ensemble already in
scikit-learn, picked over a tuned competition entry because this project's README is explicit
about carrying nothing extra to maintain.

Trees split on raw thresholds, so this model needs no StandardScaler (train_model.py's
logistic regression does, because its coefficients are compared on a common scale).

HistGradientBoostingClassifier has no feature_importances_, so permutation_importance (the
same one train_model.py now reports) stands in: how much transition-set PR-AUC drops when
one raw feature is shuffled.

random_state=42, otherwise scikit-learn's defaults -- this is a baseline challenger, not a
tuned competition entry, so no hyperparameter search.
"""
import json
import sys
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "report"))
sys.path.insert(0, str(PROJECT_DIR / "model"))

from rounding import round_numbers  # noqa: E402
from settings import SPLIT_TEST_START_YEAR  # noqa: E402
from train_model import load_population, run_single_split, run_walkforward  # noqa: E402

OUTPUT_PATH = PROJECT_DIR / "outputs" / "challenger_results.json"


def build_model():
    """HistGradientBoostingClassifier needs no scaler (see module docstring). early_stopping
    is turned off: its default internal validation split is a random 10% slice of these rows,
    which overlap heavily in time and by stock (rolling 20/60-day windows), not the
    time-respecting embargoed split used everywhere else in this project -- letting it decide
    when to stop would not be comparable to how the model is actually evaluated here."""
    return HistGradientBoostingClassifier(random_state=42, early_stopping=False)


def main():
    population = load_population()
    results = {
        "split_test_start_year": SPLIT_TEST_START_YEAR,
        "single_split": run_single_split(population, build_model_fn=build_model, extra_diagnostics_fn=None),
        "walkforward_by_year_transition_only": run_walkforward(population, build_model_fn=build_model),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = round_numbers(results)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
