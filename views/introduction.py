from pathlib import Path

import pandas as pd
import streamlit as st

from lab.ui import get_store

MEANINGS = {
    "wall_temperature": "Temperature of the cooled tube wall. The feed also enters at this temperature.",
    "residence_time": "Time the liquid spends in the tube. A shorter time means a higher feed rate.",
    "feed_concentration": "Concentration of A in the feed. More A can give more product, but also releases more heat.",
    "catalyst_loading": "Amount of catalyst, where 1 is the standard loading. It speeds up both reaction steps, and it costs money.",
}

store = get_store()

st.title("Introduction")
st.markdown(
    "Your team runs a cooled tube reactor with a new catalyst, and the numbers behind it are hidden. "
    f"**Your goal:** find the operating conditions with the highest margin, using at most {store.budget} experiments."
)

st.subheader("The reaction")
st.latex(r"\mathrm{A} \xrightarrow{\;k_1\;} \mathrm{P} \xrightarrow{\;k_2\;} \mathrm{S}")
st.markdown(
    "A is the feed, P is the product you want, and S is a by-product. P forms from A and then reacts further to S, "
    "so more reaction is not always better. Both steps release heat, and both run faster at higher temperature."
)

st.subheader("The reactor")
st.image(str(Path(__file__).resolve().parents[1] / "assets" / "reactor.svg"), width=880,
         alt="Cooled tube reactor: feed A enters on the left, A reacts to P and S along the tube, and a hot spot forms near the inlet.")
st.markdown(
    "This is a plug-flow tube: the liquid moves through like a series of small batches that do not mix with each other. "
    "The wall removes heat. Near the inlet, where the reaction is fastest, the temperature can rise above the wall "
    "temperature. This is the hot spot."
)

st.subheader("What you set")
factors = pd.DataFrame([
    {"Symbol": f.symbol, "Factor": f.label, "Allowed range": f"{f.low:g} to {f.high:g} {f.unit}", "Meaning": MEANINGS[f.key]}
    for f in store.factors
]).set_index("Symbol")
st.table(factors, alt="The factors you can set, with their allowed ranges and meaning")
st.caption("The Analyse page uses these symbols.")

st.subheader("What you measure")
results = pd.DataFrame([
    {"Result": "Margin", "Meaning": "The score you want to maximise: profit per reactor volume and hour, in relative units."},
    {"Result": "Yield of P", "Meaning": "Share of the feed that leaves as product P."},
    {"Result": "Conversion of A", "Meaning": "Share of the feed that has reacted, to either P or S."},
    {"Result": "Peak temperature", "Meaning": "Highest temperature along the tube, in K. This is the top of the hot spot."},
]).set_index("Result")
st.table(results, alt="The results of each experiment and their meaning")
st.latex(r"\text{margin} = \frac{c_0}{\tau}\,(v \cdot Y - 1) - C_\mathrm{cat} - C_\mathrm{hot}")
st.markdown(
    "Here, $c_0/\\tau$ is the feed rate in kmol/(m³ h), $Y$ is the yield, and $v$ is the value of P relative to the "
    "cost of A. $C_\\mathrm{cat}$ is the catalyst cost, and $C_\\mathrm{hot}$ is a penalty above a temperature limit."
)
st.markdown(
    "**Hidden from you:** the rate constants, activation energies, reaction heats, cooling performance, prices and "
    "the temperature limit. All measurements contain noise."
)

st.subheader("How to play")
st.markdown(
    "1. Enter a short team code in the sidebar. Do not use your names.\n"
    "2. Choose your strategy. The sidebar shows its rule.\n"
    "3. Run experiments on the Experiment page, or plan a set of them on the Design page. Each one uses up your budget.\n"
    "4. Look for patterns on the Analyse page.\n"
    "5. Save your recommendation on the Experiment page. At the reveal, it is judged by its true margin, without noise."
)
