"""Shared game state across all browser sessions, saved to a local file after every change.

On Streamlit Community Cloud this file does not survive a restart, so teams can
also download their runs as CSV and upload them again.
"""

import json
import threading
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .reactor import FACTORS, ReactorParams, active_factors, measure

STRATEGIES = ("OFAT", "Designed experiment", "Free choice", "Bayesian optimization")
BO_TEAM = "BO"
MEASURED = ("margin", "yield", "conversion", "peak_temperature")


class BudgetError(ValueError):
    pass


@dataclass
class Team:
    code: str
    strategy: str = STRATEGIES[2]
    runs: list = field(default_factory=list)
    recommendation: dict | None = None


class GameStore:
    def __init__(self, path: Path, params: ReactorParams):
        self.path = Path(path)
        self.params = params
        self.lock = threading.RLock()
        self.budget = 15
        self.n_factors = 3
        self.seed = 20261130
        self.teams: dict[str, Team] = {}
        self._load()

    # Settings and persistence

    @property
    def factors(self):
        return active_factors(self.n_factors)

    def configure(self, budget: int | None = None, n_factors: int | None = None) -> None:
        with self.lock:
            if budget is not None:
                self.budget = int(budget)
            if n_factors is not None:
                if any(team.runs for team in self.teams.values()) and int(n_factors) != self.n_factors:
                    raise ValueError("Reset the game before changing the number of factors.")
                self.n_factors = int(n_factors)
            self._save()

    def reset(self) -> None:
        with self.lock:
            self.teams = {}
            self._save()

    def _save(self) -> None:
        state = {"budget": self.budget, "n_factors": self.n_factors, "seed": self.seed,
                 "teams": {code: asdict(team) for code, team in self.teams.items()}}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        temporary.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        state = json.loads(self.path.read_text())
        self.budget = state["budget"]
        self.n_factors = state["n_factors"]
        self.seed = state["seed"]
        self.teams = {code: Team(**team) for code, team in state["teams"].items()}

    # Teams

    def team(self, code: str, strategy: str | None = None) -> Team:
        with self.lock:
            team = self.teams.get(code)
            if team is None:
                team = Team(code=code)
                self.teams[code] = team
            if strategy is not None and strategy != team.strategy:
                team.strategy = strategy
            self._save()
            return team

    def remaining(self, code: str) -> int:
        with self.lock:
            return self.budget - len(self.team(code).runs)

    # Experiments

    def _rng(self, code: str, run_number: int) -> np.random.Generator:
        """Noise is reproducible for a given team and run number, so a game can be replayed."""
        return np.random.default_rng([self.seed, zlib.crc32(code.encode()), run_number])

    def run(self, code: str, conditions: list[dict], source: str) -> list[dict]:
        """Run a batch of experiments. All of them fit into the budget, or none is run."""
        with self.lock:
            team = self.team(code)
            if len(team.runs) + len(conditions) > self.budget:
                raise BudgetError(
                    f"{len(conditions)} experiments do not fit into your remaining budget "
                    f"of {self.budget - len(team.runs)}.")
            new_runs = []
            for point in conditions:
                number = len(team.runs) + 1
                clean = {f.key: float(np.clip(point[f.key], f.low, f.high)) for f in self.factors}
                result = measure(self.params, clean, self._rng(code, number))
                row = {"run": number, "source": source, **clean, **result}
                team.runs.append(row)
                new_runs.append(row)
            self._save()
            return new_runs

    def recommend(self, code: str, conditions: dict) -> None:
        with self.lock:
            self.team(code).recommendation = {f.key: float(conditions[f.key]) for f in self.factors}
            self._save()

    def restore(self, code: str, csv_frame: pd.DataFrame) -> int:
        """Bring back runs from a downloaded CSV, for example after an app restart."""
        needed = ["run", "source", *[f.key for f in self.factors], *MEASURED]
        missing = [column for column in needed if column not in csv_frame.columns]
        if missing:
            raise ValueError(f"This file is not a run table. Missing: {', '.join(missing)}.")
        if len(csv_frame) > self.budget:
            raise BudgetError(f"This file has more runs than the budget of {self.budget}.")
        with self.lock:
            self.team(code).runs = csv_frame[needed].to_dict("records")
            self._save()
        return len(csv_frame)

    # Views

    def runs_frame(self, code: str) -> pd.DataFrame:
        with self.lock:
            runs = list(self.team(code).runs)
        columns = ["run", "source", *[f.key for f in self.factors], *MEASURED]
        return pd.DataFrame(runs, columns=columns)

    def leaderboard(self) -> pd.DataFrame:
        with self.lock:
            rows = []
            for team in self.teams.values():
                best = max((run["margin"] for run in team.runs), default=None)
                rows.append({"Team": team.code, "Strategy": team.strategy,
                             "Experiments used": len(team.runs),
                             "Best measured margin": best,
                             "Recommendation made": team.recommendation is not None})
        return pd.DataFrame(rows, columns=["Team", "Strategy", "Experiments used",
                                           "Best measured margin", "Recommendation made"])


def factor_labels() -> dict:
    return {f.key: f"{f.label} ({f.unit})" for f in FACTORS}
