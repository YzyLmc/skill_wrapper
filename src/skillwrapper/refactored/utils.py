"""Define various utility functions and type aliases for the refactored codebase."""

from typing import Any


def is_camel_case(string: str) -> bool:
    """Check whether the given string is CamelCase."""
    return string.lower() != string and ("_" not in string) and (" " not in string)


def is_snake_case(string: str) -> bool:
    """Check whether the given string is snake_case."""
    return string.lower() == string and (" " not in string)


YAMLData = dict[str, Any]
"""A map from YAML keys to YAML data."""
