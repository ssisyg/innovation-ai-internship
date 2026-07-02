"""
Shared pytest fixtures.

Two important pieces of setup happen here, both required because
backend/agent.py loads its models/data using paths relative to the
current working directory (e.g. "data/models/best_model.pkl"),
and backend/main.py uses an absolute import ("from backend.agent import ..."):

1. We chdir to the project root so relative data paths resolve correctly
   no matter where `pytest` is invoked from.
2. We make sure the project root is on sys.path so `backend` is importable
   as a package.

We also set placeholder env vars *before* backend.agent is imported
anywhere, since that import constructs a ChatOpenAI client at module
load time (no network call happens until .invoke() is used, but the
client constructor may still expect these vars to be present).
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
os.environ.setdefault("API_KEY", "test-api-key")


@pytest.fixture(scope="session")
def known_ticker() -> str:
    """A ticker guaranteed to exist in data/raw, sentiment data, and test_meta.csv."""
    return "AAPL"


@pytest.fixture(scope="session")
def unknown_ticker() -> str:
    """A ticker that exists nowhere in the local data."""
    return "ZZZZ_NOPE"
