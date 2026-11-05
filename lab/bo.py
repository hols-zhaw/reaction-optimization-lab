"""Bayesian optimization: a Gaussian process surrogate with expected improvement."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def initial_runs(n_factors: int) -> int:
    return n_factors + 3


def random_start(n_factors: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1, 1, (initial_runs(n_factors), n_factors))


@dataclass
class Surrogate:
    gp: GaussianProcessRegressor
    mean: float
    scale: float
    best: float

    def predict(self, coded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = self.gp.predict(np.atleast_2d(coded), return_std=True)
        return mu * self.scale + self.mean, sd * self.scale

    def expected_improvement(self, coded: np.ndarray) -> np.ndarray:
        mu, sd = self.predict(coded)
        sd = np.maximum(sd, 1e-9)
        z = (mu - self.best) / sd
        return (mu - self.best) * norm.cdf(z) + sd * norm.pdf(z)


def fit_surrogate(coded: np.ndarray, measured: np.ndarray) -> Surrogate:
    n_factors = coded.shape[1]
    mean, scale = float(measured.mean()), float(measured.std()) + 1e-9
    kernel = (ConstantKernel(1.0)
              * Matern(length_scale=[0.6] * n_factors, length_scale_bounds=(0.15, 5), nu=2.5)
              + WhiteKernel(1e-2, (1e-5, 1)))
    gp = GaussianProcessRegressor(kernel=kernel, random_state=1)
    gp.fit(coded, (measured - mean) / scale)
    return Surrogate(gp, mean, scale, float(measured.max()))


def propose(surrogate: Surrogate, n_factors: int, rng: np.random.Generator,
            candidates: int = 2000) -> dict:
    """The next experiment, with the numbers needed to discuss why it was chosen."""
    points = rng.uniform(-1, 1, (candidates, n_factors))
    improvement = surrogate.expected_improvement(points)
    index = int(np.argmax(improvement))
    mu, sd = surrogate.predict(points)
    return {
        "coded": points[index],
        "predicted": float(mu[index]),
        "uncertainty": float(sd[index]),
        "uncertainty_rank": float((sd < sd[index]).mean()),
        "prediction_rank": float((mu < mu[index]).mean()),
    }


def recommend(surrogate: Surrogate, n_factors: int, rng: np.random.Generator,
              candidates: int = 4000) -> np.ndarray:
    points = rng.uniform(-1, 1, (candidates, n_factors))
    return points[int(np.argmax(surrogate.predict(points)[0]))]
