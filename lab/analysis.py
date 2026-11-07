"""Response surface models, fitted on factors coded to -1...+1 over the game's full range."""

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .reactor import Factor

MODELS = ("Main effects", "Main effects + interactions", "Full quadratic")


def model_size(n_factors: int, model: str) -> int:
    size = 1 + n_factors
    if model != MODELS[0]:
        size += n_factors * (n_factors - 1) // 2
    if model == MODELS[2]:
        size += n_factors
    return size


def _terms(
    factors: tuple[Factor, ...], model: str
) -> list[tuple[str, tuple[int, ...]]]:
    terms = [("Intercept", ())] + [(f.symbol, (i,)) for i, f in enumerate(factors)]
    if model != MODELS[0]:
        terms += [
            (f"{factors[i].symbol} × {factors[j].symbol}", (i, j))
            for i, j in combinations(range(len(factors)), 2)
        ]
    if model == MODELS[2]:
        terms += [(f"{f.symbol}²", (i, i)) for i, f in enumerate(factors)]
    return terms


def _matrix(coded: np.ndarray, terms: list) -> np.ndarray:
    return np.column_stack(
        [
            np.prod(coded[:, list(idx)], axis=1) if idx else np.ones(len(coded))
            for _, idx in terms
        ]
    )


def code(frame: pd.DataFrame, factors: tuple[Factor, ...]) -> np.ndarray:
    return np.column_stack(
        [f.to_coded(frame[f.key].to_numpy(dtype=float)) for f in factors]
    )


@dataclass
class FitResult:
    factors: tuple[Factor, ...]
    terms: list
    fit: object

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Term": [name for name, _ in self.terms],
                "Coefficient": self.fit.params,
                "Std. error": self.fit.bse,
                "p-value": self.fit.pvalues,
            }
        )

    @property
    def r_squared(self) -> float:
        return float(self.fit.rsquared)

    @property
    def adjusted_r_squared(self) -> float:
        return float(self.fit.rsquared_adj)

    @property
    def residual_sd(self) -> float:
        return float(np.sqrt(self.fit.mse_resid))

    @property
    def residual_df(self) -> int:
        return int(self.fit.df_resid)

    def predict_coded(self, coded: np.ndarray) -> np.ndarray:
        return _matrix(np.atleast_2d(coded), self.terms) @ self.fit.params

    def optimum(self, levels: int = 41) -> dict:
        """Best predicted conditions on a grid over the full range."""
        n = len(self.factors)
        levels = levels if n <= 3 else 21
        grid = np.linspace(-1, 1, levels)
        coded = np.array(list(product(grid, repeat=n)))
        predicted = self.predict_coded(coded)
        best = coded[int(np.argmax(predicted))]
        return {
            "conditions": {
                f.key: float(f.from_coded(c)) for f, c in zip(self.factors, best)
            },
            "predicted": float(predicted.max()),
            "at_limit": [f.label for f, c in zip(self.factors, best) if abs(c) == 1.0],
        }


def fit_model(
    frame: pd.DataFrame, factors: tuple[Factor, ...], response: str, model: str
) -> FitResult:
    needed = model_size(len(factors), model)
    if len(frame) <= needed:
        raise ValueError(
            f"This model has {needed} coefficients, so it needs at least "
            f"{needed + 1} experiments. You selected {len(frame)}."
        )
    terms = _terms(factors, model)
    matrix = _matrix(code(frame, factors), terms)
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError(
            "These experiments cannot separate all terms of this model. "
            "Vary the factors more, or choose a simpler model."
        )
    return FitResult(
        factors, terms, sm.OLS(frame[response].to_numpy(dtype=float), matrix).fit()
    )


def slice_best(x, y, z) -> tuple[float, float, float]:
    """Location and value of the highest point in a two-factor slice."""
    row, column = np.unravel_index(int(np.argmax(z)), np.shape(z))
    return float(x[column]), float(y[row]), float(np.asarray(z)[row, column])


def slice_grid(
    factors: tuple[Factor, ...], x_key: str, y_key: str, fixed: dict, levels: int = 41
) -> dict:
    """Coded grid for a two-factor slice, with all other factors held at fixed real values."""
    by_key = {f.key: i for i, f in enumerate(factors)}
    x_factor, y_factor = factors[by_key[x_key]], factors[by_key[y_key]]
    x = np.linspace(x_factor.low, x_factor.high, levels)
    y = np.linspace(y_factor.low, y_factor.high, levels)
    xx, yy = np.meshgrid(x, y)
    coded = np.zeros((xx.size, len(factors)))
    for f in factors:
        coded[:, by_key[f.key]] = f.to_coded(fixed.get(f.key, 0.0))
    coded[:, by_key[x_key]] = x_factor.to_coded(xx.ravel())
    coded[:, by_key[y_key]] = y_factor.to_coded(yy.ravel())
    return {"x": x, "y": y, "shape": xx.shape, "coded": coded}
