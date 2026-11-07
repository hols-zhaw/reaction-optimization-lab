"""Experimental designs on coded levels (-1, 0, +1)."""

from itertools import combinations, product

import numpy as np

import pandas as pd

from .reactor import Factor


def full_factorial(n_factors: int, center_points: int = 0) -> np.ndarray:
    points = np.array(list(product([-1.0, 1.0], repeat=n_factors)))
    return _with_center(points, n_factors, center_points)


def box_behnken(n_factors: int, center_points: int = 3) -> np.ndarray:
    if n_factors < 3:
        raise ValueError("A Box-Behnken design needs at least 3 factors.")
    rows = []
    for i, j in combinations(range(n_factors), 2):
        for a, b in product([-1.0, 1.0], repeat=2):
            row = [0.0] * n_factors
            row[i], row[j] = a, b
            rows.append(row)
    return _with_center(np.array(rows), n_factors, center_points)


def central_composite(n_factors: int, center_points: int = 1) -> np.ndarray:
    """Face-centred, so that every point stays inside the chosen range."""
    corners = np.array(list(product([-1.0, 1.0], repeat=n_factors)))
    axial = np.vstack([sign * np.eye(n_factors) for sign in (-1.0, 1.0)])
    return _with_center(np.vstack([corners, axial]), n_factors, center_points)


def _with_center(points: np.ndarray, n_factors: int, center_points: int) -> np.ndarray:
    return np.vstack([points, np.zeros((center_points, n_factors))])


DESIGNS = {
    "Box-Behnken": (box_behnken, 3),
    "Central composite (face-centred)": (central_composite, 1),
    "Full factorial, 2 levels": (full_factorial, 3),
}


def to_conditions(
    coded: np.ndarray, factors: tuple[Factor, ...], ranges: dict
) -> pd.DataFrame:
    """Translate coded levels into real conditions for the chosen low and high values."""
    table = {}
    for i, factor in enumerate(factors):
        low, high = ranges[factor.key]
        table[factor.key] = low + (coded[:, i] + 1) / 2 * (high - low)
    return pd.DataFrame(table)
