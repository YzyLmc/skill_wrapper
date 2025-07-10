"""Test various utility functions from the refactored codebase."""

import pytest

from skillwrapper.refactored.utils import is_camelcase, is_snakecase


@pytest.fixture
def snakecase_strings() -> list[str]:
    """Create a list of snake_case strings."""
    return ["str", "snake_case_strings", "numbers1_2_3", "apple1"]


@pytest.fixture
def camelcase_strings() -> list[str]:
    """Create a list of CamelCase strings."""
    return ["Hello", "ChatGPT", "PredicateInstance", "Apple1"]


def test_is_camelcase(snakecase_strings: list[str], camelcase_strings: list[str]) -> None:
    """Verify that is_camelcase correctly classifies CamelCase and snake_case strings."""
    snakecase_results = [is_camelcase(s) for s in snakecase_strings]
    camelcase_results = [is_camelcase(s) for s in camelcase_strings]

    for was_camelcase, snakecase_str in zip(snakecase_results, snakecase_strings, strict=True):
        assert not was_camelcase, f"String '{snakecase_str}' is not CamelCase."

    for was_camelcase, camelcase_str in zip(camelcase_results, camelcase_strings, strict=True):
        assert was_camelcase, f"String '{camelcase_str}' is CamelCase."


def test_is_snakecase(snakecase_strings: list[str], camelcase_strings: list[str]) -> None:
    """Verify that is_camelcase correctly classifies snake_case and CamelCase strings."""
    snakecase_results = [is_snakecase(s) for s in snakecase_strings]
    camelcase_results = [is_snakecase(s) for s in camelcase_strings]

    for was_snakecase, snakecase_str in zip(snakecase_results, snakecase_strings, strict=True):
        assert was_snakecase, f"String '{snakecase_str}' is snake_case."

    for was_snakecase, camelcase_str in zip(camelcase_results, camelcase_strings, strict=True):
        assert not was_snakecase, f"String '{camelcase_str}' is not snake_case."
