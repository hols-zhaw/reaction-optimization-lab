import numpy as np
import pandas as pd
import pytest
from pyDOE3 import bbdesign, ff2n

from lab import bo
from lab.analysis import MODELS, code, fit_model, model_size, slice_best, slice_grid
from lab.designs import box_behnken, central_composite, full_factorial, to_conditions
from lab.reactor import (
    FACTORS,
    active_factors,
    simulate,
    temperature_profile,
    true_response,
)
from lab.store import BudgetError, GameStore, breaks_ofat
from lab.ui import surface_figure

F3 = active_factors(3)


def as_set(array):
    return {tuple(row) for row in np.round(array, 6) + 0.0}


@pytest.mark.parametrize("k", [3, 4])
def test_box_behnken_matches_reference_library(k):
    ours = box_behnken(k, center_points=0)
    assert as_set(ours) == as_set(bbdesign(k, center=1)) - {tuple([0.0] * k)}
    assert len(ours) == {3: 12, 4: 24}[k]


def test_standard_design_of_this_game_uses_exactly_the_budget():
    assert len(box_behnken(3, 3)) == 15


@pytest.mark.parametrize("k", [2, 3, 4])
def test_full_factorial_matches_reference_library(k):
    assert as_set(full_factorial(k)) == as_set(ff2n(k))


def test_central_composite_is_face_centred_and_inside_range():
    design = central_composite(3, 1)
    assert len(design) == 8 + 6 + 1
    assert np.abs(design).max() == 1.0


def test_to_conditions_uses_chosen_ranges():
    table = to_conditions(
        np.array([[-1.0, 0.0, 1.0]]), F3, {f.key: (f.low + 1, f.high - 1) for f in F3}
    )
    assert table.iloc[0].tolist() == [
        F3[0].low + 1,
        (F3[1].low + F3[1].high) / 2,
        F3[2].high - 1,
    ]


def test_coding_round_trip():
    for factor in FACTORS:
        assert factor.to_coded(factor.low) == -1 and factor.to_coded(factor.high) == 1
        assert factor.from_coded(factor.to_coded(factor.low + 1)) == pytest.approx(
            factor.low + 1
        )


def known_surface(coded):
    a, b, c = coded.T
    return 1.0 + 0.5 * a - 0.3 * b + 0.2 * c + 0.4 * a * b - 0.6 * a * a - 0.2 * c * c


def design_frame(coded, response):
    frame = pd.DataFrame({f.key: f.from_coded(coded[:, i]) for i, f in enumerate(F3)})
    frame["margin"] = response
    return frame


def test_fit_recovers_known_quadratic_and_optimum():
    design = box_behnken(3, 3)
    result = fit_model(
        design_frame(design, known_surface(design)), F3, "margin", MODELS[2]
    )
    table = result.table.set_index("Term")["Coefficient"]
    assert table["Intercept"] == pytest.approx(1.0)
    assert table["Tw × tau"] == pytest.approx(0.4)
    assert table["Tw²"] == pytest.approx(-0.6)
    assert table["c0"] == pytest.approx(0.2)
    assert table["tau²"] == pytest.approx(0.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0)
    grid = np.array(np.meshgrid(*[np.linspace(-1, 1, 41)] * 3)).reshape(3, -1).T
    assert result.optimum()["predicted"] == pytest.approx(known_surface(grid).max())


def test_fit_matches_plain_least_squares():
    rng = np.random.default_rng(1)
    design = box_behnken(3, 3)
    noisy = known_surface(design) + rng.normal(0, 0.05, len(design))
    result = fit_model(design_frame(design, noisy), F3, "margin", MODELS[2])
    a, b, c = design.T
    matrix = np.column_stack(
        [np.ones(len(a)), a, b, c, a * b, a * c, b * c, a * a, b * b, c * c]
    )
    assert result.fit.params == pytest.approx(
        np.linalg.lstsq(matrix, noisy, rcond=None)[0]
    )


def test_model_too_large_for_data_is_refused():
    design = full_factorial(3)
    assert model_size(3, MODELS[2]) == 10
    with pytest.raises(ValueError, match="at least 11"):
        fit_model(design_frame(design, design.sum(axis=1)), F3, "margin", MODELS[2])


def test_quadratic_on_two_level_design_is_refused_as_inseparable():
    design = full_factorial(3)
    doubled = np.vstack([design, design])
    with pytest.raises(ValueError, match="cannot separate"):
        fit_model(design_frame(doubled, doubled.sum(axis=1)), F3, "margin", MODELS[2])


def test_slice_grid_holds_other_factor_fixed():
    grid = slice_grid(F3, F3[0].key, F3[1].key, {F3[2].key: F3[2].low})
    assert grid["coded"].shape == (41 * 41, 3)
    assert np.all(grid["coded"][:, 2] == -1)


def test_simulation_balances_are_consistent(params):
    result = simulate(params, 330.0, 60.0, 1000.0)
    assert 0 <= result["yield"] <= result["conversion"] <= 1
    assert result["peak_temperature"] >= 330.0


def test_more_catalyst_speeds_up_conversion(params):
    low = simulate(params, 330.0, 30.0, 1000.0, catalyst_loading=0.5)["conversion"]
    high = simulate(params, 330.0, 30.0, 1000.0, catalyst_loading=2.0)["conversion"]
    assert high > low


def test_surface_margins_do_not_depend_on_the_labels():
    grid = np.linspace(0.0, 1.0, 3)
    short = surface_figure(grid, grid, np.eye(3), "x", "y", "Short")
    long = surface_figure(
        grid,
        grid,
        100 * np.eye(3),
        "A long x label",
        "A long y label",
        "A much longer value label",
    )
    assert short.layout.margin == long.layout.margin
    assert short.layout.margin.autoexpand is False
    assert (
        short.layout.yaxis.automargin is False
        and short.layout.xaxis.automargin is False
    )


def test_temperature_profile_matches_the_simulation(params):
    conditions = {
        "wall_temperature": 340.0,
        "residence_time": 40.0,
        "feed_concentration": 3000.0,
    }
    position, temperature = temperature_profile(params, conditions, points=2001)
    assert position[0] == 0.0 and position[-1] == 1.0
    assert temperature[0] == pytest.approx(340.0)
    assert temperature.max() == pytest.approx(
        true_response(params, conditions)["peak_temperature"], abs=0.1
    )


def test_ofat_check_compares_with_every_earlier_run():
    frame = pd.DataFrame(
        {
            "wall_temperature": [330.0, 340.0, 350.0],
            "residence_time": [60.0, 60.0, 60.0],
            "feed_concentration": [2000.0, 2000.0, 2000.0],
        }
    )
    assert not breaks_ofat(frame.iloc[:1], F3)
    assert not breaks_ofat(frame, F3)
    moved_to_best = pd.concat([frame, frame.iloc[[0]].assign(residence_time=90.0)])
    assert not breaks_ofat(moved_to_best, F3)
    two_changes = pd.concat(
        [frame, frame.iloc[[0]].assign(residence_time=90.0, feed_concentration=3000.0)]
    )
    assert breaks_ofat(two_changes, F3)


def test_real_constants_reproduce_the_feasibility_test(real_params, real_check):
    conditions = {f.key: real_check[f.key] for f in active_factors(3)}
    assert true_response(real_params, conditions)["margin"] == pytest.approx(
        real_check["margin"], abs=0.002
    )


def centre(factors):
    return {f.key: (f.low + f.high) / 2 for f in factors}


def test_budget_is_enforced_for_whole_batches(params, tmp_path):
    store = GameStore(tmp_path / "state.json", params)
    store.configure(budget=3)
    store.run("A", [centre(F3)] * 2, "manual")
    with pytest.raises(BudgetError):
        store.run("A", [centre(F3)] * 2, "manual")
    assert store.remaining("A") == 1


def test_noise_is_reproducible_and_differs_between_runs(params, tmp_path):
    one = GameStore(tmp_path / "a.json", params).run("A", [centre(F3)] * 2, "manual")
    two = GameStore(tmp_path / "b.json", params).run("A", [centre(F3)] * 2, "manual")
    assert one == two
    assert one[0]["margin"] != one[1]["margin"]


def test_conditions_are_clipped_to_reactor_limits(params, tmp_path):
    store = GameStore(tmp_path / "state.json", params)
    point = {**centre(F3), F3[0].key: 9999.0}
    assert store.run("A", [point], "manual")[0][F3[0].key] == F3[0].high


def test_state_survives_a_restart_of_the_store(params, tmp_path):
    path = tmp_path / "state.json"
    first = GameStore(path, params)
    first.run("A", [centre(F3)], "manual")
    first.recommend("A", centre(F3))
    second = GameStore(path, params)
    assert second.runs_frame("A").equals(first.runs_frame("A"))
    assert second.team("A").recommendation == centre(F3)


def test_csv_round_trip_restores_runs(params, tmp_path):
    store = GameStore(tmp_path / "state.json", params)
    store.run("A", [centre(F3)] * 2, "manual")
    frame = store.runs_frame("A")
    store.reset()
    assert store.restore("A", frame) == 2
    assert store.remaining("A") == store.budget - 2


def test_restore_rejects_other_files(params, tmp_path):
    store = GameStore(tmp_path / "state.json", params)
    with pytest.raises(ValueError, match="not a run table"):
        store.restore("A", pd.DataFrame({"x": [1]}))


def test_factor_count_cannot_change_during_a_game(params, tmp_path):
    store = GameStore(tmp_path / "state.json", params)
    store.team("A")
    store.configure(n_factors=4)  # no experiments yet, so this is allowed
    store.configure(n_factors=3)
    store.run("A", [centre(F3)], "manual")
    with pytest.raises(ValueError, match="Reset"):
        store.configure(n_factors=4)


def test_bo_prefers_the_promising_region():
    rng = np.random.default_rng(3)
    coded = rng.uniform(-1, 1, (12, 3))
    measured = -((coded - 0.5) ** 2).sum(axis=1)
    surrogate = bo.fit_surrogate(coded, measured)
    proposal = bo.propose(surrogate, 3, np.random.default_rng(4))
    assert np.abs(proposal["coded"]).max() <= 1
    assert (
        np.linalg.norm(bo.recommend(surrogate, 3, np.random.default_rng(5)) - 0.5) < 0.6
    )


def test_code_maps_limits_to_plus_minus_one():
    frame = pd.DataFrame({f.key: [f.low, f.high] for f in F3})
    assert code(frame, F3).tolist() == [[-1, -1, -1], [1, 1, 1]]


def test_slice_best_finds_the_highest_point_of_the_slice():
    x, y = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])
    z = np.array([[0.0, 1.0, 2.0], [3.0, 9.0, 4.0]])
    assert slice_best(x, y, z) == (2.0, 20.0, 9.0)


def test_snap_stays_on_allowed_values_and_inside_limits():
    factor = F3[2]
    assert factor.snap(1234.0) == 1250.0
    assert factor.snap(1e9) == factor.high
    assert factor.snap(-1e9) == factor.low
