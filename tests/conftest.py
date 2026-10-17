"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable so ``from modules import ...`` resolves when
# pytest is invoked from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.pipeline_harness import _get_model, run_demo_pipeline  # noqa: E402

SCENARIOS = ["Good Borrower", "Risky Borrower", "Borderline / Manual Review"]


@pytest.fixture(scope="session")
def model_bundle():
    """The trained RandomForest, shared so it is fitted once for the whole run."""
    return _get_model()


@pytest.fixture(scope="session")
def pipeline_results() -> dict:
    """Run all three demo scenarios once and share the results across tests.

    Session-scoped because training the RandomForest costs ~1s and the pipeline
    is fully deterministic (seeded at 42), so there is nothing to isolate.
    """
    return {name: run_demo_pipeline(name) for name in SCENARIOS}


@pytest.fixture(scope="session")
def good(pipeline_results) -> dict:
    return pipeline_results["Good Borrower"]


@pytest.fixture(scope="session")
def risky(pipeline_results) -> dict:
    return pipeline_results["Risky Borrower"]


@pytest.fixture(scope="session")
def borderline(pipeline_results) -> dict:
    return pipeline_results["Borderline / Manual Review"]
