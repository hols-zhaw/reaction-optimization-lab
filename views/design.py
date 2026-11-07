import streamlit as st

from lab.designs import DESIGNS, to_conditions
from lab.ui import current_team, get_store, run_batch, run_labels

store = get_store()

st.title("Design")
st.markdown(
    "Plan a set of experiments that vary several factors at once, instead of one at a time. "
    "Choose a design and a range for each factor, then run all points together. "
    "You can use a narrower range for a second stage."
)

code = current_team()
n_factors = len(store.factors)
available = {
    name: spec
    for name, spec in DESIGNS.items()
    if name != "Box-Behnken" or n_factors >= 3
}

left, right = st.columns(2)
name = left.selectbox("Design", list(available))
builder, default_center = available[name]
center_points = right.number_input(
    "Centre points",
    0,
    10,
    default_center,
    1,
    key=f"center_{name}",
    help="Repeats in the middle of the range. They show the noise and any curvature.",
)

st.subheader("Range of each factor")
ranges = {}
for factor in store.factors:
    ranges[factor.key] = st.slider(
        f"{factor.label} ({factor.unit})",
        factor.low,
        factor.high,
        (factor.low, factor.high),
        factor.step,
        format=factor.number_format,
        key=f"range_{factor.key}",
    )

coded = builder(n_factors, center_points)
table = to_conditions(coded, store.factors, ranges)
remaining = store.remaining(code)

st.subheader(f"Planned experiments: {len(table)}")
st.dataframe(table.rename(columns=run_labels(store)), hide_index=True, width="stretch")

if len(table) > remaining:
    st.warning(
        f"This design needs {len(table)} experiments, but you have {remaining} left."
    )
if st.button("Run this design", type="primary", disabled=len(table) > remaining):
    if run_batch(store, code, table.to_dict("records"), name):
        st.toast(f"Ran {len(table)} experiments. Go to Analyse to look at the results.")
        st.rerun()
