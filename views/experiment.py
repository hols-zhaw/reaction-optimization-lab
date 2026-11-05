import pandas as pd
import streamlit as st

from lab.store import BudgetError
from lab.ui import current_team, factor_inputs, get_store, run_batch, show_runs

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

st.subheader("Your experiments")
frame = store.runs_frame(code)
if frame.empty:
    st.caption("No experiments yet.")
else:
    show_runs(store, frame)
    st.download_button("Download as CSV", frame.to_csv(index=False), f"runs_{code}.csv", "text/csv")

st.subheader("Your recommendation")
st.caption("State your final recommended conditions. This uses no experiment, and you can change it until the reveal.")
team = store.team(code)
with st.form("recommendation"):
    recommended = factor_inputs(store, "recommendation", team.recommendation)
    if st.form_submit_button("Save recommendation"):
        store.recommend(code, recommended)
        st.success("Recommendation saved.")

with st.expander("Restore runs from a CSV file"):
    st.caption("If the app restarts, your runs may be lost. Upload your downloaded file to bring them back. "
               "This replaces the runs currently saved under your team code.")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None and st.button("Restore"):
        try:
            count = store.restore(code, pd.read_csv(upload))
        except (ValueError, BudgetError) as error:
            st.error(str(error))
        else:
            st.success(f"Restored {count} runs.")
            st.rerun()
