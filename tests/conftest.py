import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.reactor import ReactorParams  # noqa: E402

SECRETS = ROOT / ".streamlit" / "secrets.toml"


@pytest.fixture
def params():
    """Made-up constants, so that the public tests do not reveal the real ones."""
    return ReactorParams(
        k10=2.0,
        k20=1e9,
        E1=25e3,
        E2=80e3,
        H1=-200e3,
        H2=-150e3,
        cp=3500.0,
        rho=900.0,
        U=400.0,
        tube_diameter=0.03,
        value_ratio=2.0,
        catalyst_cost=0.3,
        temperature_limit=400.0,
        temperature_penalty=0.2,
        noise_sd_margin=0.03,
        noise_sd_fraction=0.005,
        noise_sd_temperature=0.5,
    )


@pytest.fixture
def real_params():
    if not SECRETS.exists():
        pytest.skip("No local secrets file.")
    return ReactorParams(**tomllib.loads(SECRETS.read_text())["reactor"])


@pytest.fixture
def real_check():
    """Expected result for the real constants. It stays in the secrets file, as it gives away the optimum."""
    if not SECRETS.exists():
        pytest.skip("No local secrets file.")
    return tomllib.loads(SECRETS.read_text())["check"]
