"""Drive the pages headless, the way a team and the instructor would."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(tmp_path, monkeypatch, params):
    import streamlit as st

    monkeypatch.setenv("LAB_DATA_PATH", str(tmp_path / "state.json"))
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.secrets["instructor_password"] = "test-password"
    at.secrets["reactor"] = vars(params)
    return at


def open_page(at, name):
    at.switch_page(f"views/{name}.py")
    at.run()
    assert not at.exception, at.exception
    return at


def join(at, code="TEAM1"):
    at.run()
    assert not at.exception, at.exception
    at.text_input(key="team_code_input").set_value(code).run()
    assert not at.exception, at.exception
    return open_page(at, "experiment")


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception
    return at


def test_starts_on_introduction_without_a_team(app):
    app.run()
    assert not app.exception, app.exception
    assert app.title[0].value == "Introduction"
    assert len(app.latex) == 2
    ranges = app.table[0].value["Allowed range"].tolist()
    assert ranges == ["320 to 365 K", "35 to 110 min", "500 to 4000 mol/m³"]


def test_experiment_page_asks_for_team_code(app):
    app.run()
    open_page(app, "experiment")
    assert any("team code" in i.value for i in app.info)


def test_reserved_code_is_refused(app):
    join(app, "bo")
    assert any("reserved" in e.value for e in app.error)


def test_single_experiment_updates_budget(app):
    join(app)
    click(app, "Run experiment")
    assert app.metric[0].value == "14"
    assert len(app.dataframe) == 1


def test_allowed_ranges_are_shown_at_the_inputs(app):
    join(app)
    captions = [c.value for c in app.caption]
    assert "Allowed: 320 to 365 K" in captions
    assert "Allowed: 500 to 4000 mol/m³" in captions


def test_experiment_page_shows_charts_only_after_a_run(app):
    join(app)
    assert not app.get("plotly_chart")
    click(app, "Run experiment")
    assert len(app.get("plotly_chart")) == 4
    headings = [s.value for s in app.subheader]
    assert (
        headings.index("Run one experiment")
        < headings.index("Your search so far")
        < headings.index("Your experiments")
    )
    click(app, "Run experiment")
    next(s for s in app.selectbox if s.label == "Experiment").set_value(1).run()
    assert not app.exception, app.exception
    assert any("Measured margin" in m.value for m in app.markdown)


def test_strategy_rule_and_ofat_warning(app):
    join(app)
    assert any("any mix" in c.value for c in app.caption)
    next(s for s in app.selectbox if s.label == "Strategy").set_value("OFAT").run()
    assert any("one factor at a time" in c.value for c in app.caption)
    click(app, "Run experiment")
    app.number_input(key="experiment_wall_temperature").set_value(330.0)
    click(app, "Run experiment")
    assert not app.warning
    app.number_input(key="experiment_residence_time").set_value(90.0)
    app.number_input(key="experiment_feed_concentration").set_value(1000.0)
    click(app, "Run experiment")
    assert any("more than one factor" in w.value for w in app.warning)


def test_design_then_analyse_full_workflow(app):
    join(app)
    open_page(app, "design")
    click(app, "Run this design")
    assert app.metric[0].value == "0"
    open_page(app, "analyse")
    labels = [m.label for m in app.metric]
    assert "R²" in labels and "Adjusted R²" in labels
    assert not app.warning or all("coefficients" not in w.value for w in app.warning)


def test_too_large_design_cannot_run(app):
    join(app)
    click(app, "Run experiment")
    open_page(app, "design")
    assert any("you have 14 left" in w.value for w in app.warning)
    assert next(b for b in app.button if b.label == "Run this design").disabled


def test_analysis_refuses_model_with_too_few_runs(app):
    join(app)
    for _ in range(3):
        click(app, "Run experiment")
    open_page(app, "analyse")
    assert any("at least 11" in w.value for w in app.warning)


def test_recommendation_and_leaderboard(app):
    join(app)
    click(app, "Run experiment")
    click(app, "Save recommendation")
    assert any("saved" in s.value for s in app.success)
    open_page(app, "leaderboard")
    assert len(app.dataframe) == 1


def enter_password(app, password):
    next(t for t in app.text_input if t.label == "Password").set_value(password)
    click(app, "Log in")


def login(app):
    open_page(app, "instructor")
    enter_password(app, "test-password")


def test_instructor_needs_password(app):
    join(app)
    open_page(app, "instructor")
    enter_password(app, "wrong")
    assert any("Wrong password" in e.value for e in app.error)
    assert not any(b.label == "Reset the game" for b in app.button)


def test_live_bo_runs_to_the_end_of_the_budget(app):
    join(app)
    login(app)
    click(app, "Run 6 random experiments")
    for _ in range(9):
        click(app, "Show the next experiment")
        click(app, "Run it")
    assert any("recommendation" in s.value for s in app.success)


def test_reveal_shows_true_surface(app):
    join(app)
    click(app, "Run experiment")
    click(app, "Save recommendation")
    login(app)
    next(
        c for c in app.checkbox if c.label.startswith("Show the results")
    ).check().run()
    assert not app.exception, app.exception
    assert any("Best conditions" in m.value for m in app.markdown)


def test_reset_clears_the_game(app):
    join(app)
    click(app, "Run experiment")
    login(app)
    next(
        c for c in app.checkbox if c.label.startswith("I want to delete")
    ).check().run()
    click(app, "Reset the game")
    open_page(app, "experiment")
    assert app.metric[0].value == "15"


def test_four_factor_game_runs_from_design_to_reveal(app):
    join(app)
    open_page(app, "instructor")
    enter_password(app, "test-password")
    next(n for n in app.number_input if n.label == "Experiment budget").set_value(30)
    next(r for r in app.radio if r.label == "Number of factors").set_value(4)
    click(app, "Save settings")
    open_page(app, "design")
    assert any("Planned experiments: 27" in s.value for s in app.subheader)
    click(app, "Run this design")
    open_page(app, "analyse")
    assert "R²" in [m.label for m in app.metric]
    assert any(s.label.startswith("Hold") for s in app.slider)
    open_page(app, "introduction")
    assert "Catalyst loading" in app.table[0].value["Factor"].tolist()
    open_page(app, "experiment")
    click(app, "Save recommendation")
    open_page(app, "instructor")
    next(
        c for c in app.checkbox if c.label.startswith("Show the results")
    ).check().run()
    assert not app.exception, app.exception
    assert any("Best conditions" in m.value for m in app.markdown)
