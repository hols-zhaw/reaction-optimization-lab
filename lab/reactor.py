"""Black-box flow reactor: a series reaction A -> P -> S in a cooled plug-flow tube.

The model structure is public. The numerical constants are not part of this
repository: they come from Streamlit secrets, so that the game stays a black box.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

GAS_CONSTANT = 8.314  # J/(mol K)


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    symbol: str
    unit: str
    low: float
    high: float
    step: float

    def to_coded(self, value: float) -> float:
        return 2 * (value - self.low) / (self.high - self.low) - 1

    def from_coded(self, coded: float) -> float:
        return self.low + (coded + 1) / 2 * (self.high - self.low)

    @property
    def number_format(self) -> str:
        return "%.0f" if self.step >= 1 else "%.2f"

    def snap(self, value: float) -> float:
        """Nearest allowed input value within the limits."""
        snapped = self.low + round((value - self.low) / self.step) * self.step
        return float(min(max(snapped, self.low), self.high))


FACTORS = (
    Factor("wall_temperature", "Wall temperature", "Tw", "K", 320.0, 365.0, 1.0),
    Factor("residence_time", "Residence time", "tau", "min", 35.0, 110.0, 1.0),
    Factor("feed_concentration", "Feed concentration", "c0", "mol/m³", 500.0, 4000.0, 50.0),
    Factor("catalyst_loading", "Catalyst loading", "L", "relative", 0.5, 2.0, 0.05),
)
BASELINE_LOADING = 1.0


def active_factors(n_factors: int) -> tuple[Factor, ...]:
    return FACTORS[:n_factors]


@dataclass(frozen=True)
class ReactorParams:
    k10: float  # 1/s
    k20: float  # 1/s
    E1: float  # J/mol
    E2: float  # J/mol
    H1: float  # J/mol
    H2: float  # J/mol
    cp: float  # J/(kg K)
    rho: float  # kg/m3
    U: float  # W/(m2 K)
    tube_diameter: float  # m
    value_ratio: float  # product value relative to feed cost
    catalyst_cost: float  # margin units per unit of loading
    temperature_limit: float  # K
    temperature_penalty: float  # margin units per K above the limit
    noise_sd_margin: float  # margin units
    noise_sd_fraction: float  # yield and conversion
    noise_sd_temperature: float  # K


def _integrate(params: ReactorParams, wall_temperature, tau, feed_concentration, catalyst_loading, t_eval=None):
    """Solve the balances along the tube. Tau in seconds."""
    area_per_volume = 4.0 / params.tube_diameter

    def rhs(_, y):
        c_a, c_p, temp = y
        k1 = catalyst_loading * params.k10 * np.exp(-params.E1 / (GAS_CONSTANT * temp))
        k2 = catalyst_loading * params.k20 * np.exp(-params.E2 / (GAS_CONSTANT * temp))
        heat = -params.H1 * k1 * c_a - params.H2 * k2 * c_p
        exchange = params.U * (wall_temperature - temp) * area_per_volume
        return [-k1 * c_a, k1 * c_a - k2 * c_p, (heat + exchange) / (params.rho * params.cp)]

    return solve_ivp(rhs, [0.0, tau], [feed_concentration, 0.0, wall_temperature],
                     method="LSODA", rtol=1e-5, atol=1e-3, t_eval=t_eval)


def simulate(params: ReactorParams, wall_temperature, residence_time, feed_concentration,
             catalyst_loading=BASELINE_LOADING) -> dict:
    """Noise-free steady state for one set of conditions. Residence time in minutes."""
    tau = residence_time * 60.0
    sol = _integrate(params, wall_temperature, tau, feed_concentration, catalyst_loading)
    c_a, c_p = sol.y[0, -1], sol.y[1, -1]
    yield_p = c_p / feed_concentration
    conversion = 1.0 - c_a / feed_concentration
    peak_temperature = float(sol.y[2].max())
    feed_rate = (feed_concentration / 1000.0) / (tau / 3600.0)  # kmol/(m3 h)
    margin = (feed_rate * (params.value_ratio * yield_p - 1.0)
              - params.catalyst_cost * catalyst_loading
              - params.temperature_penalty * max(0.0, peak_temperature - params.temperature_limit))
    return {
        "margin": float(margin),
        "yield": float(yield_p),
        "conversion": float(conversion),
        "peak_temperature": peak_temperature,
    }


def conditions_to_args(conditions: dict) -> dict:
    """Fill in the baseline for any factor that is not part of the current game."""
    return {f.key: conditions.get(f.key, BASELINE_LOADING) for f in FACTORS}


def true_response(params: ReactorParams, conditions: dict) -> dict:
    return simulate(params, **conditions_to_args(conditions))


def temperature_profile(params: ReactorParams, conditions: dict, points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free temperature along the tube, against the fraction of its length.

    The app shows only temperature along the tube. Concentrations along the tube would
    give away the yield at every shorter residence time in a single experiment.
    """
    args = conditions_to_args(conditions)
    tau = args["residence_time"] * 60.0
    times = np.linspace(0.0, tau, points)
    sol = _integrate(params, args["wall_temperature"], tau, args["feed_concentration"],
                     args["catalyst_loading"], t_eval=times)
    return times / tau, sol.y[2]


def measure(params: ReactorParams, conditions: dict, rng: np.random.Generator) -> dict:
    """One experiment: the true result plus measurement noise."""
    true = true_response(params, conditions)
    return {
        "margin": true["margin"] + rng.normal(0, params.noise_sd_margin),
        "yield": float(np.clip(true["yield"] + rng.normal(0, params.noise_sd_fraction), 0, 1)),
        "conversion": float(np.clip(true["conversion"] + rng.normal(0, params.noise_sd_fraction), 0, 1)),
        "peak_temperature": true["peak_temperature"] + rng.normal(0, params.noise_sd_temperature),
    }
