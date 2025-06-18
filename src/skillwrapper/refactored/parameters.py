"""Define classes to represent and manage object-typed discrete parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscreteParameter:
    """An object-typed discrete parameter (e.g., of a skill, predicate, or operator)."""

    name: str  # Name of the lifted parameter
    object_type: str  # Object type expected by the parameter
    semantics: str | None  # Optional natural language description of the parameter's meaning


Bindings = dict[str, str]  # Map from parameter names to their bound concrete objects
