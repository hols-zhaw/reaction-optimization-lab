import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lab.analysis import MODELS, fit_model, model_size, slice_best, slice_grid
from lab.ui import (CATEGORICAL, MEASURED_LABELS, MUTED, NEGATIVE, POSITIVE, add_marker, add_points, current_team,
                    get_store, plain_figure, run_batch, slice_controls, surface_figure)

store = get_store()
st.title("Analyse")
code = current_team()
frame = store.runs_frame(code)
if frame.empty:
    st.info("Run some experiments first.")
    st.stop()

factors = store.factors
labels = {f.key: f"{f.label} ({f.unit})" for f in factors}

left, right = st.columns([1, 2])
response = left.selectbox("Response", list(MEASURED_LABELS), format_func=MEASURED_LABELS.get)
selected = right.multiselect("Experiments to include", frame["run"].tolist(), default=frame["run"].tolist(),
                             help="Remove runs to fit a model to one stage or a smaller region only.")
data = frame[frame["run"].isin(selected)]

look, model_tab = st.tabs(["Look at the runs", "Fit a model"])

with look:
    x_key = st.selectbox("Plot against", list(labels), format_func=labels.get)
    fig = plain_figure()
    fig.add_trace(go.Scatter(x=data[x_key], y=data[response], mode="markers", marker=dict(size=10, color=CATEGORICAL[0]),
                             text=[f"Run {r}" for r in data["run"]],
                             hovertemplate="%{text}<br>%{x:.4g}, %{y:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title=labels[x_key], yaxis_title=MEASURED_LABELS[response])
    st.plotly_chart(fig, width="stretch")
    st.caption("The other factors differ between these runs, so a pattern here can be misleading.")

with model_tab:
    model = st.radio("Model", MODELS, index=2, horizontal=True,
                     captions=[f"{model_size(len(factors), m)} coefficients" for m in MODELS])
    try:
        result = fit_model(data, factors, response, model)
    except ValueError as error:
        st.warning(str(error))
        st.stop()

    a, b, c, d = st.columns(4)
    a.metric("R²", f"{result.r_squared:.3f}")
    b.metric("Adjusted R²", f"{result.adjusted_r_squared:.3f}")
    c.metric("Residual SD", f"{result.residual_sd:.3f}" if result.residual_df else "n/a")
    d.metric("Residual degrees of freedom", result.residual_df)
    if result.residual_df < 3:
        st.warning("Very few degrees of freedom are left. Standard errors and p-values are unreliable.")

    st.subheader("Effects")
    st.caption("Coefficients on coded factors: −1 is the lowest value in the game and +1 the highest. "
               "So their sizes are directly comparable.")
    table = result.table
    effects = table[table["Term"] != "Intercept"].iloc[::-1]
    bars = plain_figure(height=max(220, 34 * len(effects) + 70))
    bars.add_trace(go.Bar(x=effects["Coefficient"], y=effects["Term"], orientation="h",
                          marker_color=[POSITIVE if v >= 0 else NEGATIVE for v in effects["Coefficient"]],
                          hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    bars.update_layout(xaxis_title=f"Effect on {MEASURED_LABELS[response].lower()}")
    bars.update_yaxes(type="category")
    bars.update_xaxes(zerolinecolor=MUTED, zerolinewidth=1.5)
    st.plotly_chart(bars, width="stretch")
    st.dataframe(table.round(4), hide_index=True, width="stretch")

    st.subheader("Response surface")
    optimum = result.optimum()
    x, y, fixed = slice_controls(factors, "surface", optimum["conditions"])
    grid = slice_grid(factors, x, y, fixed)
    z = result.predict_coded(grid["coded"]).reshape(grid["shape"])
    surface = surface_figure(grid["x"], grid["y"], z, labels[x], labels[y], f"Predicted {MEASURED_LABELS[response].lower()}")
    add_points(surface, data, x, y, "Your runs", value_key=response)
    best_x, best_y, _ = slice_best(grid["x"], grid["y"], z)
    add_marker(surface, best_x, best_y, "Best in this slice")
    st.plotly_chart(surface, width="stretch")
    st.caption("This is a slice through the model, not a measurement. It opens at the predicted optimum. "
               "Runs are shown at their position on the two axes, whatever the value of the other factors.")

    st.subheader("Predicted optimum")
    summary = ", ".join(f"{f.label} {optimum['conditions'][f.key]:.4g} {f.unit}" for f in factors)
    st.markdown(f"{summary}. Predicted {MEASURED_LABELS[response].lower()}: **{optimum['predicted']:.3f}**.")
    if optimum["at_limit"]:
        st.warning(f"The optimum lies at the limit of the range for: {', '.join(optimum['at_limit'])}. "
                   "A polynomial is least reliable there.")
    if st.button("Test this point (uses one experiment)", disabled=store.remaining(code) < 1):
        if run_batch(store, code, [optimum["conditions"]], "predicted optimum"):
            st.rerun()

    with st.expander("Check the fit: measured against predicted"):
        predicted = result.fit.fittedvalues
        measured = data[response].to_numpy(dtype=float)
        limits = [float(np.min([predicted, measured])), float(np.max([predicted, measured]))]
        check = plain_figure()
        check.add_trace(go.Scatter(x=limits, y=limits, mode="lines", line=dict(color=MUTED, dash="dash", width=1),
                                   hoverinfo="skip"))
        check.add_trace(go.Scatter(x=predicted, y=measured, mode="markers", marker=dict(size=10, color=CATEGORICAL[0]),
                                   text=[f"Run {r}" for r in data["run"]],
                                   hovertemplate="%{text}<br>predicted %{x:.3f}<br>measured %{y:.3f}<extra></extra>"))
        check.update_layout(xaxis_title="Predicted", yaxis_title="Measured")
        st.plotly_chart(check, width="stretch")
