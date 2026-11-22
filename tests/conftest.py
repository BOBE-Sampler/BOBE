"""Shared pytest configuration and custom CLI options for the BOBE test suite."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow end-to-end BO and determinism tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    markexpr = (config.getoption("-m") or "").strip()
    if "slow" in markexpr and "not slow" not in markexpr:
        return
    skip_slow = pytest.mark.skip(reason="slow test (pass --runslow or -m slow to run)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
