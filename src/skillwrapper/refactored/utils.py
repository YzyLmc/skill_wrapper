"""Define various utility functions and type aliases for the refactored codebase."""

from typing import Any


def is_camel_case(string: str) -> bool:
    """Check whether the given string is CamelCase."""
    if not string:
        return False

    return string[0].isupper() and all(c not in string for c in [" ", "_", "-"])


def is_snake_case(string: str) -> bool:
    """Check whether the given string is snake_case."""
    if not string:
        return False

    return string.lower() == string and all(c not in string for c in [" ", "-"])


YAMLData = dict[str, Any]
"""A map from YAML keys to YAML data."""
