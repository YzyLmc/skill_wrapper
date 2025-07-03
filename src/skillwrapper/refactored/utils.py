"""Define various utility functions for the refactored codebase."""

from __future__ import annotations

import platform
import re
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch
import yaml
from typing_extensions import Self


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    # Insert underscore before uppercase letters that follow lowercase letters
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return s1.lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase."""
    chunks = name.split("_")
    return "".join(word.capitalize() for word in chunks)


def import_yaml_into_dict(yaml_path: Path, required_keys: set[str]) -> dict[str, Any]:
    """Import data from a YAML file into a Python dictionary.

    :param yaml_path: Filepath to a YAML file containing data to be imported
    :param required_keys: Keys verified to exist in the imported dictionary
    :return: Dictionary mapping YAML keys to corresponding imported data
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Cannot import data from nonexistent YAML file: {yaml_path}")

    try:
        with yaml_path.open() as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
    except yaml.YAMLError as err:
        raise RuntimeError(f"Could not load data from YAML file: {yaml_file}") from err

    for key in required_keys:
        if key not in yaml_data:
            raise KeyError(f"Required key '{key}' is missing from the YAML file: {yaml_file}")

    return yaml_data


def parse_docstring_params(docstring: str) -> dict[str, str]:
    """Extract parameter semantics from a docstring.

    :param docstring: String containing the docstring of a skill function
    :return: Map from parameter names to their semantic descriptions
    """
    param_docs = {}
    param_pattern = r":param\s+(\w+):\s*([^\n]+)"

    for match in re.finditer(param_pattern, docstring):
        param_name = match.group(1)
        description = match.group(2)
        param_docs[param_name] = description

    return param_docs


@runtime_checkable
class FromYAMLProtocol(Protocol):
    """A protocol for classes supporting construction from a dictionary of YAML data."""

    @classmethod
    def from_yaml(cls, yaml_data: dict[str, Any]) -> Self:
        """Import a FromYAMLProtocol instance from data imported from YAML."""
        ...


StateT = TypeVar("StateT", bound=FromYAMLProtocol)  # Support any YAML-importable state type


def load_class_from_module(class_name: str, module_name: str) -> type:
    """Dynamically load a class from the specified module.

    :param class_name: Name of a class to load from a module (e.g., "MyClass")
    :param module_name: String representation of the module (e.g., "my_package.module_name")
    :return: Type of the loaded class
    """
    loaded_module = import_module(module_name)
    if not hasattr(loaded_module, class_name):
        error = f"Cannot load class '{class_name}' from module '{loaded_module.__name__}'."
        raise ImportError(error)

    return getattr(loaded_module, class_name)


def determine_pytorch_device() -> torch.device:
    """Determine which PyTorch device to use."""
    if torch.cuda.is_available():  # Use CUDA on Linux if available
        return torch.device("cuda")
    if platform.system() == "Darwin":  # Use Metal on macOS
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")  # Otherwise, fallback to CPU
