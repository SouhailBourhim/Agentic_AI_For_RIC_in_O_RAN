"""Smoke test for checkpoint M00.

Why it exists: pytest exits with code 5 ("no tests collected") when there are no tests,
which scripts treat as a failure. This test also proves the flat layout works: every
package in the repository layout can be imported from the tests.
"""

import importlib

import pytest

PACKAGES = [
    "sim",
    "rl",
    "verifier",
    "agent",
    "gateway",
    "console",
    "backends",
    "xapps",
    "eval",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    """Each top-level package of the layout must be importable."""
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} should have a docstring saying what it is for"
