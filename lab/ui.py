"""Streamlit helpers shared by the pages."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
    decimals.update({f.key: 2 if f.step < 1 else 0 for f in store.factors})
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
                   colorbar=dict(title=value_label, thickness=12),
                   hovertemplate=f"{x_label}: %{{x:.4g}}<br>{y_label}: %{{y:.4g}}<br>{value_label}: %{{z:.3f}}<extra></extra>")
    if has_sign_change:
        # Very negative regions would otherwise wash out the detail around the optimum.
        contour.update(zmid=0, zmin=max(float(np.nanmin(z)), -1.5 * top), zmax=top)
    fig = go.Figure(go.Contour(**contour))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, xaxis_title=x_label, yaxis_title=y_label,
                      font=dict(color=INK), legend=dict(orientation="h", y=1.08, x=0))
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
