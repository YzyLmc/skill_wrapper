"""Define various utility functions and type aliases for the refactored codebase."""

from typing import Any


def is_camel_case(string: str) -> bool:
    """Check whether the given string is CamelCase."""
    if not string.isalnum() or not string[0].isupper():
        return False

    for i, char in enumerate(string):
        if not char.isalpha() and i + 1 < len(string) and string[i + 1].islower():
            return False

    return True


def is_snake_case(string: str) -> bool:
    """Check whether the given string is snake_case."""
    if not string.replace("_", "").isalnum():
        return False

    return string.lower() == string


YAMLData = dict[str, Any]
"""A map from YAML keys to YAML data."""
