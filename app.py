"""Reaction Optimization Lab: find good conditions for a black-box flow reactor."""

import streamlit as st

from lab.store import BO_TEAM, STRATEGIES, STRATEGY_RULES
from lab.ui import get_store

st.set_page_config(page_title="Reaction Optimization Lab", layout="wide")

if "reactor" not in st.secrets or "instructor_password" not in st.secrets:
    st.error("The reactor settings are missing. See README.md for how to set up the secrets.")
    st.stop()

store = get_store()

with st.sidebar:
    st.subheader("Your team")
    entered = st.text_input("Team code", key="team_code_input", max_chars=20,
                            help="Choose a short code, not your names. Your runs are saved under this code.")
    code = entered.strip().upper()
    if code == BO_TEAM:
        st.error("This code is reserved. Please choose another one.")
        code = ""
    st.session_state["team_code"] = code
    if code:
        team = store.team(code)
        choices = [s for s in STRATEGIES if s != "Bayesian optimization"]
        current = team.strategy if team.strategy in choices else choices[-1]
        strategy = st.selectbox("Strategy", choices, index=choices.index(current))
        st.caption(STRATEGY_RULES[strategy])
        store.team(code, strategy)
        used = len(store.team(code).runs)
        st.metric("Experiments left", store.budget - used, help=f"Budget: {store.budget} experiments")
        st.progress(min(used / store.budget, 1.0))

navigation = st.navigation([
    st.Page("views/introduction.py", title="Introduction", default=True),
    st.Page("views/experiment.py", title="Experiment"),
    st.Page("views/design.py", title="Design"),
    st.Page("views/analyse.py", title="Analyse"),
    st.Page("views/leaderboard.py", title="Leaderboard"),
    st.Page("views/instructor.py", title="Instructor"),
])
navigation.run()
