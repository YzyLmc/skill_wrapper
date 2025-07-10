"""Test various utility functions from the refactored codebase."""

import pytest

from skillwrapper.refactored.utils import is_camel_case, is_snake_case


@pytest.fixture
def snake_case_strings() -> list[str]:
    """Create a list of snake_case strings."""
    return ["str", "snake_case_strings", "numbers1_2_3", "apple1"]


@pytest.fixture
def camel_case_strings() -> list[str]:
    """Create a list of CamelCase strings."""
    return ["CamelCaseStrings", "ChatGPT", "Apple1", "PredicateInstance3A"]


@pytest.fixture
def neither_strings() -> list[str]:
    """Create a list of strings that are neither snake_case nor CamelCase."""
    return ["Hello World", "lower case", "aBCD"]


def test_is_camel_case(snake_case_strings: list[str], camel_case_strings: list[str]) -> None:
    """Verify that is_camel_case(str) correctly classifies CamelCase and snake_case strings."""
    snake_results = [is_camel_case(s) for s in snake_case_strings]
    camel_results = [is_camel_case(s) for s in camel_case_strings]

    for was_camel_case, snake_str in zip(snake_results, snake_case_strings, strict=True):
        assert not was_camel_case, f"String '{snake_str}' is not CamelCase."

    for was_camel_case, camel_str in zip(camel_results, camel_case_strings, strict=True):
        assert was_camel_case, f"String '{camel_str}' is CamelCase."


def test_is_snake_case(snake_case_strings: list[str], camel_case_strings: list[str]) -> None:
    """Verify that is_snake_case(str) correctly classifies snake_case and CamelCase strings."""
    snake_results = [is_snake_case(s) for s in snake_case_strings]
    camel_results = [is_snake_case(s) for s in camel_case_strings]

    for was_snake_case, snake_str in zip(snake_results, snake_case_strings, strict=True):
        assert was_snake_case, f"String '{snake_str}' is snake_case."

    for was_snake_case, camel_str in zip(camel_results, camel_case_strings, strict=True):
        assert not was_snake_case, f"String '{camel_str}' is not snake_case."


def test_neither_snake_nor_camel_case(neither_strings: list[str]) -> None:
    """Verify that strings that are neither snake_case nor CamelCase are classified as such."""
    were_snake_case = [is_snake_case(s) for s in neither_strings]
    were_camel_case = [is_camel_case(s) for s in neither_strings]

    for was_snake_case, string in zip(were_snake_case, neither_strings, strict=True):
        assert not was_snake_case, f"String '{string}' is not snake_case."

    for was_camel_case, string in zip(were_camel_case, neither_strings, strict=True):
        assert not was_camel_case, f"String '{string}' is not CamelCase."
