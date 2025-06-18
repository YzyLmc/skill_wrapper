"""Define classes to represent symbolic predicates, lifted and grounded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from skillwrapper.refactored.parameters import Bindings, DiscreteParameter
from skillwrapper.refactored.utils import StateT


@dataclass(frozen=True)
class Predicate(Generic[StateT]):
    """A symbolic predicate with object-typed parameters."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    semantics: str | None  # Optional natural language description of the predicate's meaning

    def __str__(self) -> str:
        """Return a readable string representation of the predicate."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    def ground_with(self, bindings: Bindings) -> PredicateInstance:
        """Ground the predicate using the given parameter bindings."""
        return PredicateInstance(self, bindings)

    def holds_in(self, state: StateT, bindings: Bindings) -> bool:
        """Evaluate whether the predicate holds in a state under the given parameter bindings.

        :param state: Low-level environment state in which the predicate is evaluated
        :param bindings: Parameter bindings used to ground the predicate
        :return: True if the grounded predicate holds, else False
        """
        raise NotImplementedError("Predicate.holds_in(...)")


@dataclass(frozen=True)
class PredicateInstance(Generic[StateT]):
    """A predicate grounded using particular concrete objects."""

    predicate: Predicate[StateT]
    bindings: Bindings

    def holds_in(self, state: StateT) -> bool:
        """Evaluate whether the predicate instance holds in the given state."""
        return self.predicate.holds_in(state, self.bindings)


AbstractState = set[PredicateInstance]  # Abstract state = Set of all true grounded predicates
