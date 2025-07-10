"""Define various utility functions and type aliases for the refactored codebase."""

from typing import Any


def is_camelcase(string: str) -> bool:
    """Check whether the given string is CamelCase."""
    return string.lower() != string and ("_" not in string)


def is_snakecase(string: str) -> bool:
    """Check whether the given string is snake_case."""
    return string.lower() == string


YAMLData = dict[str, Any]
"""A map from YAML keys to YAML data."""
