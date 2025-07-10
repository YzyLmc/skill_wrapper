"""Define a class and type alias to represent object-typed discrete parameters."""

from __future__ import annotations

from dataclasses import dataclass

from skillwrapper.refactored.utils import YAMLData, is_camel_case


@dataclass(frozen=True)
class DiscreteParameter:
    """An object-typed discrete parameter (e.g., of a skill, predicate, or operator)."""

    name: str
    """Name of the lifted parameter."""

    object_type: str
    """Object type required of any object bound to the parameter (expected to be CamelCase)."""

    semantics: str | None = None
    """Optional natural language description of the parameter's meaning."""

    def __post_init__(self) -> None:
        """Validate expected properties of any DiscreteParameter instance."""
        if not is_camel_case(self.object_type):
            raise ValueError(f"Discrete parameter type '{self.object_type}' must be CamelCase.")

    def __str__(self) -> str:
        """Create a readable string representation of the discrete parameter."""
        semantics_str = f": {self.semantics}" if self.semantics else ""
        return f"{self.name} (Type {self.object_type}){semantics_str}"

    @classmethod
    def from_yaml(cls, param_name: str, param_data: YAMLData) -> DiscreteParameter:
        """Import a DiscreteParameter instance from YAML data."""
        return DiscreteParameter(param_name, param_data["type"], param_data.get("semantics"))

    @classmethod
    def tuple_from_yaml(cls, params_data: YAMLData) -> tuple[DiscreteParameter, ...]:
        """Import a tuple of DiscreteParameter instances from a dictionary of YAML data."""
        return tuple(DiscreteParameter.from_yaml(name, data) for name, data in params_data.items())

    def to_yaml_dict(self) -> YAMLData:
        """Convert the discrete parameter into a dictionary of data to be exported to YAML."""
        yaml_data = {"type": self.object_type}
        if self.semantics is not None:
            yaml_data["semantics"] = self.semantics
        return {self.name: yaml_data}


Bindings = dict[str, str]
"""A mapping from parameter names to their bound concrete objects."""
