"""Rounds the numbers in a results dictionary, so two runs write byte-identical files.

Copied from kenya-aid-flows/report/rounding.py (same house pattern)."""


def round_numbers(value):
    """Return a copy of value (dicts, lists, numbers) with every float rounded to six places."""
    if isinstance(value, dict):
        rounded = {}
        for key in value:
            rounded[key] = round_numbers(value[key])
        return rounded
    if isinstance(value, list):
        rounded_list = []
        for item in value:
            rounded_list.append(round_numbers(item))
        return rounded_list
    if isinstance(value, float):
        return round(value, 6)
    return value
