"""Streamlit helpers shared by the pages."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .reactor import ReactorParams
from .store import BudgetError, GameStore

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"
POSITIVE, NEGATIVE, MIDPOINT = "#2a78d6", "#e34948", "#f0efec"
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

MEASURED_LABELS = {
    "margin": "Margin",
    "yield": "Yield of P",
    "conversion": "Conversion of A",
    "peak_temperature": "Peak temperature (K)",
}


def reactor_params() -> ReactorParams:
    return ReactorParams(**{k: float(v) for k, v in st.secrets["reactor"].items()})


@st.cache_resource
def get_store() -> GameStore:
    path = Path(os.environ.get("LAB_DATA_PATH", Path(tempfile.gettempdir()) / "reaction_lab_state.json"))
    return GameStore(path, reactor_params())


def current_team() -> str:
    """The team code from the sidebar. Stops the page if there is none yet."""
    code = st.session_state.get("team_code", "")
    if not code:
        st.info("Enter your team code in the sidebar to start.")
        st.stop()
    return code


def run_labels(store: GameStore) -> dict:
    labels = {"run": "Run", "source": "Source"}
    labels.update({f.key: f"{f.label} ({f.unit})" for f in store.factors})
    labels.update(MEASURED_LABELS)
    return labels


def show_runs(store: GameStore, frame: pd.DataFrame) -> None:
    decimals = {"margin": 3, "yield": 3, "conversion": 3, "peak_temperature": 1}
    decimals.update({f.key: 2 if f.step < 1 else 1 for f in store.factors})
    st.dataframe(frame.round(decimals).rename(columns=run_labels(store)), hide_index=True, width="stretch")


def run_batch(store: GameStore, code: str, conditions: list[dict], source: str) -> bool:
    try:
        store.run(code, conditions, source)
    except BudgetError as error:
        st.error(str(error))
        return False
    return True


def factor_inputs(store: GameStore, prefix: str, defaults: dict | None = None) -> dict:
    """One number input per factor, within the limits of the reactor."""
    columns = st.columns(len(store.factors))
    values = {}
    for column, factor in zip(columns, store.factors):
        default = (defaults or {}).get(factor.key, (factor.low + factor.high) / 2)
        values[factor.key] = column.number_input(
            f"{factor.label} ({factor.unit})", min_value=factor.low, max_value=factor.high,
            value=factor.snap(default), step=factor.step, format=factor.number_format,
            key=f"{prefix}_{factor.key}")
        column.caption(f"Allowed: {factor.low:g} to {factor.high:g} {factor.unit}")
    return values


def slice_controls(factors, prefix: str, start: dict | None = None) -> tuple[str, str, dict]:
    """Choose two factors for the axes, and fix the others, by default at the given conditions."""
    labels = {f.key: f"{f.label} ({f.unit})" for f in factors}
    keys = list(labels)
    left, right = st.columns(2)
    x_key = left.selectbox("Horizontal axis", keys, index=0, format_func=labels.get, key=f"{prefix}_x")
    y_options = [k for k in keys if k != x_key]
    y_key = right.selectbox("Vertical axis", y_options, index=0, format_func=labels.get, key=f"{prefix}_y")
    fixed = {}
    for factor in factors:
        if factor.key not in (x_key, y_key):
            value = (start or {}).get(factor.key, (factor.low + factor.high) / 2)
            fixed[factor.key] = st.slider(
                f"Hold {labels[factor.key]} at", factor.low, factor.high, factor.snap(value),
                factor.step, format=factor.number_format, key=f"{prefix}_fix_{factor.key}")
    return x_key, y_key, fixed


def surface_figure(x, y, z, x_label: str, y_label: str, value_label: str,
                   diverging: bool = True, height: int = 460) -> go.Figure:
    z = np.asarray(z)
    top = float(np.nanmax(z))
    has_sign_change = diverging and float(np.nanmin(z)) < 0 < top
    scale = ([[0, NEGATIVE], [0.5, MIDPOINT], [1, POSITIVE]] if has_sign_change
             else [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)])
    contour = dict(x=x, y=y, z=z, colorscale=scale, line=dict(width=0.5, color=MUTED),
                   contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=11, color=INK)),
                   colorbar=dict(title=dict(text=value_label, side="right"), thickness=12),
                   hovertemplate=f"{x_label}: %{{x:.4g}}<br>{y_label}: %{{y:.4g}}<br>{value_label}: %{{z:.3f}}<extra></extra>")
    if has_sign_change:
        # Very negative regions would otherwise wash out the detail around the optimum.
        contour.update(zmid=0, zmin=max(float(np.nanmin(z)), -1.5 * top), zmax=top)
    fig = go.Figure(go.Contour(**contour))
    # Fixed margins, so that the plot area does not depend on the labels and surfaces line up across tabs.
    fig.update_layout(height=height, margin=dict(l=80, r=90, t=40, b=60, autoexpand=False), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, xaxis_title=x_label, yaxis_title=y_label,
                      font=dict(color=INK), legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(automargin=False)
    fig.update_yaxes(automargin=False)
    return fig


def add_points(fig: go.Figure, frame: pd.DataFrame, x_key: str, y_key: str, name: str,
               color: str = INK, value_key: str | None = "margin") -> None:
    text = None if value_key is None else [f"{name}, run {r}: {v:.3f}" for r, v in zip(frame["run"], frame[value_key])]
    fig.add_trace(go.Scatter(x=frame[x_key], y=frame[y_key], mode="markers", name=name, text=text,
                             hoverinfo="text" if text else "x+y+name",
                             marker=dict(size=9, color=color, line=dict(width=1.5, color="white"))))


def add_marker(fig: go.Figure, x: float, y: float, name: str, color: str = INK) -> None:
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=name, hoverinfo="x+y+name",
                             marker=dict(size=15, symbol="star", color=color, line=dict(width=1.5, color="white"))))


def plain_figure(height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(color=INK), showlegend=False)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED)
    return fig


def progress_figure(frame: pd.DataFrame) -> go.Figure:
    """Measured margin of each run, and the best value so far."""
    fig = plain_figure(height=340)
    fig.add_trace(go.Scatter(x=frame["run"], y=frame["margin"], mode="markers", name="Each experiment",
                             marker=dict(size=10, color=CATEGORICAL[0], line=dict(width=1.5, color="white")),
                             hovertemplate="Run %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=frame["run"], y=frame["margin"].cummax(), mode="lines", name="Best so far",
                             line=dict(width=2, color=CATEGORICAL[1], shape="hv"),
                             hovertemplate="Best after run %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0),
                      xaxis_title="Experiment", yaxis_title="Measured margin")
    fig.update_xaxes(dtick=1 if len(frame) <= 20 else 5)
    return fig


def settings_figure(frame: pd.DataFrame, factors) -> go.Figure:
    """The value of each factor over the runs, one panel per factor, each over its allowed range."""
    fig = make_subplots(rows=len(factors), cols=1, shared_xaxes=True, vertical_spacing=0.11)
    for row, factor in enumerate(factors, start=1):
        fig.add_trace(go.Scatter(x=frame["run"], y=frame[factor.key], mode="lines+markers",
                                 line=dict(width=2, color=CATEGORICAL[0]), marker=dict(size=8, color=CATEGORICAL[0]),
                                 hovertemplate=f"Run %{{x}}: %{{y:.4g}} {factor.unit}<extra></extra>"), row=row, col=1)
        margin = 0.06 * (factor.high - factor.low)
        fig.update_yaxes(title_text=factor.unit, range=[factor.low - margin, factor.high + margin],
                         gridcolor=GRID, zerolinecolor=MUTED, row=row, col=1)
        fig.add_annotation(text=factor.label, xref="x domain", yref="y domain", x=0, y=1.0, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=12, color=MUTED), row=row, col=1)
    fig.update_xaxes(gridcolor=GRID, dtick=1 if len(frame) <= 20 else 5)
    fig.update_xaxes(title_text="Experiment", row=len(factors), col=1)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(color=INK), showlegend=False)
    return fig


def temperature_figure(position: np.ndarray, temperature: np.ndarray, wall_temperature: float) -> go.Figure:
    fig = plain_figure(height=300)
    fig.add_trace(go.Scatter(x=100 * position, y=temperature, mode="lines", name="Reactor",
                             line=dict(width=2, color=CATEGORICAL[1]),
                             hovertemplate="%{x:.0f}% of the tube: %{y:.1f} K<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[wall_temperature, wall_temperature], mode="lines", name="Wall",
                             line=dict(width=2, color=MUTED, dash="dash"), hoverinfo="skip"))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.14, x=0),
                      xaxis_title="Position along the tube, from feed to outlet (%)", yaxis_title="Temperature (K)")
    return fig


def outlet_figure(conversion: float, yield_p: float) -> go.Figure:
    """Measured composition at the outlet, as shares of the feed."""
    parts = (("A, unreacted", 1 - conversion), ("P, product", yield_p), ("S, by-product", max(conversion - yield_p, 0.0)))
    fig = plain_figure(height=300)
    for index, (name, share) in enumerate(parts):
        fig.add_trace(go.Bar(x=[100 * share], y=["Outlet"], orientation="h", name=f"{name}: {share:.0%}",
                             marker=dict(color=CATEGORICAL[index], line=dict(width=2, color=SURFACE)),
                             hovertemplate=f"{name}: {share:.1%}<extra></extra>"))
    fig.update_layout(barmode="stack", showlegend=True, legend=dict(orientation="h", y=1.14, x=0, traceorder="normal"),
                      xaxis_title="Share of the feed (%)")
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(showticklabels=False)
    return fig
