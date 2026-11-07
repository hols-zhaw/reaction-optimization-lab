import pandas as pd
import streamlit as st

from lab.reactor import temperature_profile
from lab.store import BudgetError, breaks_ofat
from lab.ui import (
    current_team,
    factor_inputs,
    get_store,
    outlet_figure,
    progress_figure,
    run_batch,
    settings_figure,
    show_runs,
    temperature_figure,
)

store = get_store()

st.title("Experiment")
st.markdown(
    "You operate a cooled tube reactor with a new catalyst. Its kinetics are unknown. "
    "**Goal:** find the conditions with the highest **margin** "
    f"within your budget of {store.budget} experiments. "
    "Margin is a measure of profit per reactor volume and hour. It falls when yield is low, "
    "and also when the reactor runs too hot. All measurements contain noise."
)

code = current_team()

with st.form("single_experiment"):
    st.subheader("Run one experiment")
    conditions = factor_inputs(store, "experiment")
    if st.form_submit_button("Run experiment", type="primary"):
        if run_batch(store, code, [conditions], "manual"):
            st.rerun()

frame = store.runs_frame(code)
if not frame.empty:
    if store.team(code).strategy == "OFAT" and breaks_ofat(frame, store.factors):
        st.warning(
            "Your strategy is OFAT, but your latest run changes more than one factor compared with every earlier run."
        )
    st.subheader("Your search so far")
    left, right = st.columns(2)
    left.plotly_chart(progress_figure(frame), width="stretch")
    right.plotly_chart(settings_figure(frame, store.factors), width="stretch")

st.subheader("Your experiments")
if frame.empty:
    st.caption("No experiments yet.")
else:
    show_runs(store, frame)
    st.download_button(
        "Download as CSV", frame.to_csv(index=False), f"runs_{code}.csv", "text/csv"
    )

    st.subheader("Look at one experiment")
    number = st.selectbox(
        "Experiment", frame["run"].tolist()[::-1], help="The latest one comes first."
    )
    run = frame[frame["run"] == number].iloc[0]
    selected = {f.key: float(run[f.key]) for f in store.factors}
    st.markdown(
        ", ".join(f"{f.label} {selected[f.key]:.4g} {f.unit}" for f in store.factors)
        + f". Measured margin: **{run['margin']:.3f}**."
    )
    left, right = st.columns(2)
    position, temperature = temperature_profile(store.params, selected)
    left.plotly_chart(
        temperature_figure(position, temperature, selected["wall_temperature"]),
        width="stretch",
    )
    right.plotly_chart(outlet_figure(run["conversion"], run["yield"]), width="stretch")
    st.caption(
        "The temperature curve leaves out measurement noise. The outlet composition is as measured."
    )

st.subheader("Your recommendation")
st.caption(
    "State your final recommended conditions. This uses no experiment, and you can change it until the reveal."
)
team = store.team(code)
with st.form("recommendation"):
    recommended = factor_inputs(store, "recommendation", team.recommendation)
    if st.form_submit_button("Save recommendation"):
        store.recommend(code, recommended)
        st.success("Recommendation saved.")

with st.expander("Restore runs from a CSV file"):
    st.caption(
        "If the app restarts, your runs may be lost. Upload your downloaded file to bring them back. "
        "This replaces the runs currently saved under your team code."
    )
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None and st.button("Restore"):
        try:
            count = store.restore(code, pd.read_csv(upload))
        except (ValueError, BudgetError) as error:
            st.error(str(error))
        else:
            st.success(f"Restored {count} runs.")
            st.rerun()
