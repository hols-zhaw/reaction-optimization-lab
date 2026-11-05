import hmac
from itertools import product

import numpy as np
import pandas as pd
import streamlit as st

from lab import bo
from lab.analysis import slice_best, slice_grid
from lab.reactor import ReactorParams, true_response
from lab.store import BO_TEAM
from lab.ui import CATEGORICAL, add_marker, add_points, get_store, slice_controls, surface_figure

store = get_store()
st.title("Instructor")

if not st.session_state.get("instructor"):
    with st.form("login"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            if hmac.compare_digest(password, str(st.secrets["instructor_password"])):
                st.session_state["instructor"] = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()

factors = store.factors
labels = {f.key: f"{f.label} ({f.unit})" for f in factors}


def from_coded(point) -> dict:
    return {f.key: float(f.from_coded(c)) for f, c in zip(factors, point)}


@st.cache_data(show_spinner="Computing the true surface...")
def true_slice(_params: ReactorParams, n_factors: int, x_key: str, y_key: str, fixed: tuple, levels: int = 25):
    grid = slice_grid(factors, x_key, y_key, dict(fixed), levels)
    values = [true_response(_params, from_coded(point))["margin"] for point in grid["coded"]]
    return grid["x"], grid["y"], np.array(values).reshape(grid["shape"])


@st.cache_data(show_spinner="Searching for the best conditions...")
def true_best(_params: ReactorParams, n_factors: int) -> dict:
    levels = 9 if n_factors <= 3 else 5
    best = (-np.inf, None)
    for point in product(np.linspace(-1, 1, levels), repeat=n_factors):
        conditions = from_coded(point)
        margin = true_response(_params, conditions)["margin"]
        if margin > best[0]:
            best = (margin, conditions)
    return {"margin": best[0], "conditions": best[1]}


bo_tab, reveal_tab, settings_tab = st.tabs(["Live BO", "Reveal", "Settings"])

with bo_tab:
    store.team(BO_TEAM, "Bayesian optimization")
    frame = store.runs_frame(BO_TEAM)
    n_factors = len(factors)
    start = bo.initial_runs(n_factors)
    st.markdown(f"Experiments used: **{len(frame)} of {store.budget}**")

    if len(frame) < start:
        st.markdown(f"BO needs some data first. It starts with {start} random experiments.")
        if st.button(f"Run {start} random experiments", type="primary"):
            rng = np.random.default_rng([store.seed, 0])
            points = [from_coded(p) for p in bo.random_start(n_factors, rng)]
            store.run(BO_TEAM, points, "BO: random start")
            st.rerun()
    else:
        coded = np.column_stack([f.to_coded(frame[f.key].to_numpy(dtype=float)) for f in factors])
        surrogate = bo.fit_surrogate(coded, frame["margin"].to_numpy(dtype=float))
        rng = np.random.default_rng([store.seed, len(frame)])
        proposal = bo.propose(surrogate, n_factors, rng)
        finished = len(frame) >= store.budget
        shown_key = f"bo_shown_{len(frame)}"
        shown = st.session_state.get(shown_key, False)

        x, y, fixed = slice_controls(factors, "bo")
        grid = slice_grid(factors, x, y, fixed)
        mean, sd = surrogate.predict(grid["coded"])
        ei = surrogate.expected_improvement(grid["coded"])
        tabs = st.tabs(["What BO believes", "How unsure it is", "Where it wants to look"])
        panels = ((mean, "Predicted margin", True), (sd, "Uncertainty", False), (ei, "Expected improvement", False))
        for panel, (values, title, diverging) in zip(tabs, panels):
            figure = surface_figure(grid["x"], grid["y"], values.reshape(grid["shape"]), labels[x], labels[y], title, diverging)
            add_points(figure, frame, x, y, "Experiments")
            if shown and not finished:
                point = from_coded(proposal["coded"])
                add_marker(figure, point[x], point[y], "Next experiment", CATEGORICAL[1])
            panel.plotly_chart(figure, width="stretch")

        if finished:
            best = from_coded(bo.recommend(surrogate, n_factors, np.random.default_rng([store.seed, 999])))
            store.recommend(BO_TEAM, best)
            st.success("Budget used. BO's recommendation: "
                       + ", ".join(f"{f.label} {best[f.key]:.4g} {f.unit}" for f in factors))
        elif not shown:
            st.info("Ask the class: where will BO sample next, and is that exploration or exploitation?")
            if st.button("Show the next experiment", type="primary"):
                st.session_state[shown_key] = True
                st.rerun()
        else:
            point = from_coded(proposal["coded"])
            st.markdown("**Next experiment:** " + ", ".join(f"{f.label} {point[f.key]:.4g} {f.unit}" for f in factors))
            st.markdown(f"Predicted margin: {proposal['predicted']:.3f}, higher than {proposal['prediction_rank']:.0%} "
                        f"of the region. Uncertainty: {proposal['uncertainty']:.3f}, higher than "
                        f"{proposal['uncertainty_rank']:.0%} of the region.")
            if st.button("Run it", type="primary"):
                store.run(BO_TEAM, [point], "BO: guided")
                st.rerun()

with reveal_tab:
    if not st.checkbox("Show the results on this screen"):
        st.info("This shows the true surface. Only tick it when you are ready for the reveal.")
    else:
        best = true_best(store.params, len(factors))
        st.markdown(f"**Best conditions found on a coarse grid:** margin {best['margin']:.3f} at "
                    + ", ".join(f"{f.label} {best['conditions'][f.key]:.4g} {f.unit}" for f in factors))
        rows = []
        for code, team in store.teams.items():
            value = true_response(store.params, team.recommendation)["margin"] if team.recommendation else None
            top_run = max(team.runs, key=lambda run: run["margin"], default=None)
            top_true = None
            if top_run is not None:
                top_true = true_response(store.params, {f.key: top_run[f.key] for f in factors})["margin"]
            rows.append({"Team": code, "Strategy": team.strategy, "Experiments used": len(team.runs),
                         "True margin of recommendation": value,
                         "Share of best": None if value is None else value / best["margin"],
                         "Best measured": None if top_run is None else top_run["margin"],
                         "True margin of best measured": top_true})
        table = pd.DataFrame(rows, columns=["Team", "Strategy", "Experiments used",
                                            "True margin of recommendation", "Share of best",
                                            "Best measured", "True margin of best measured"])
        st.dataframe(table.round(3), hide_index=True, width="stretch",
                     column_config={"Share of best": st.column_config.NumberColumn(format="percent")})

        x, y, fixed = slice_controls(factors, "reveal", best["conditions"])
        tx, ty, tz = true_slice(store.params, len(factors), x, y, tuple(sorted(fixed.items())))
        figure = surface_figure(tx, ty, tz, labels[x], labels[y], "True margin", height=560)
        for index, code in enumerate(store.teams):
            runs = store.runs_frame(code)
            if not runs.empty:
                add_points(figure, runs, x, y, code, CATEGORICAL[index % len(CATEGORICAL)])
        best_x, best_y, _ = slice_best(tx, ty, tz)
        add_marker(figure, best_x, best_y, "Best in this slice")
        st.plotly_chart(figure, width="stretch")
        st.caption("A slice through the true response, with all runs shown on the two axes, "
                   "whatever the value of the other factors.")

with settings_tab:
    budget = st.number_input("Experiment budget", 5, 60, store.budget, 1)
    n_factors = st.radio("Number of factors", [3, 4], index=[3, 4].index(store.n_factors), horizontal=True,
                         help="The number of factors can only be changed after a reset.")
    if st.button("Save settings"):
        try:
            store.configure(budget=budget, n_factors=n_factors)
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("Settings saved.")
            st.rerun()

    runs = [store.runs_frame(code).assign(team=code) for code in store.teams]
    if runs:
        st.download_button("Download all runs as CSV", pd.concat(runs).to_csv(index=False), "all_runs.csv", "text/csv")

    st.divider()
    confirm = st.checkbox("I want to delete all teams and runs")
    if st.button("Reset the game", disabled=not confirm):
        store.reset()
        st.success("The game was reset.")
        st.rerun()
