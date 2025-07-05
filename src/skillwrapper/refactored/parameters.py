"""Define classes to represent and manage object-typed discrete parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from skillwrapper.refactored.utils import is_camelcase


@dataclass(frozen=True)
class DiscreteParameter:
    """An object-typed discrete parameter (e.g., of a skill, predicate, or operator)."""

    name: str  # Name of the lifted parameter
    object_type: str  # Object type expected by the parameter
    semantics: str | None = None  # Optional NL description of the parameter's meaning

    def __post_init__(self) -> None:
        """Validate expected properties of any DiscreteParameter instance."""
        if not is_camelcase(self.object_type):
            raise ValueError(f"Discrete parameter type '{self.object_type}' must be CamelCase.")

    def __str__(self) -> str:
        """Create a readable string representation of the discrete parameter."""
        semantics_str = f": {self.semantics}" if self.semantics else ""
        return f"{self.name} (Type {self.object_type}){semantics_str}"

    @classmethod
    def from_yaml(cls, param_name: str, param_data: dict[str, Any]) -> DiscreteParameter:
        """Import a DiscreteParameter instance from YAML data."""
        return DiscreteParameter(param_name, param_data["type"], param_data.get("semantics"))

    @classmethod
    def tuple_from_yaml(cls, params_data: dict[str, Any]) -> tuple[DiscreteParameter, ...]:
        """Import a tuple of DiscreteParameter instances from a dictionary of YAML data."""
        return tuple(DiscreteParameter.from_yaml(name, data) for name, data in params_data.items())

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert the discrete parameter into a dictionary of data to be exported to YAML."""
        yaml_data = {"type": self.object_type}
        if self.semantics is not None:
            yaml_data["semantics"] = self.semantics
        return {self.name: yaml_data}


Bindings = dict[str, str]  # Map from parameter names to their bound concrete objects

# Represents any sequence of parameters or their bound arguments
ParametersT = Sequence[DiscreteParameter] | Sequence[str]
