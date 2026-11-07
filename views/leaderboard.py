import streamlit as st

from lab.ui import get_store

store = get_store()
st.title("Leaderboard")
st.caption(
    "Best measured margin so far. Measurements contain noise, so the best measurement "
    "is not necessarily the best conditions."
)


@st.fragment(run_every="10s")
def board():
    frame = store.leaderboard()
    if frame.empty:
        st.info("No teams yet.")
        return
    ordered = frame.sort_values(
        "Best measured margin", ascending=False, na_position="last"
    )
    st.dataframe(
        ordered.round({"Best measured margin": 3}), hide_index=True, width="stretch"
    )


board()
