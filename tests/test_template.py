"""The SQL templating (sql/template.py) is what makes settings.py's window sizes actually
drive the SQL, instead of just describing hardcoded literals in a comment -- so it gets its
own direct test."""
from template import render, template_values


def test_render_substitutes_every_known_token():
    sql_text = "ROWS BETWEEN {{SHORT_WINDOW_MINUS_1}} PRECEDING AND CURRENT ROW"
    rendered = render(sql_text)
    assert rendered == f"ROWS BETWEEN {template_values()['SHORT_WINDOW_MINUS_1']} PRECEDING AND CURRENT ROW"
    assert "{{" not in rendered


def test_render_raises_on_an_unknown_token():
    try:
        render("SELECT {{NOT_A_REAL_TOKEN}}")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_template_values_match_settings():
    values = template_values()
    assert values["SHORT_WINDOW_MINUS_1"] == values["SHORT_WINDOW_DAYS"] - 1
    assert values["LONG_WINDOW_MINUS_1"] == values["LONG_WINDOW_DAYS"] - 1
    assert values["EARLIEST_VALID_DATE"] == "2013-01-01"
