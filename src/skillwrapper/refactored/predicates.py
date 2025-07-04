"""Define classes to represent symbolic predicates, lifted and grounded."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Generic

from skillwrapper.refactored.environment import ConcreteObjects
from skillwrapper.refactored.parameters import Bindings, DiscreteParameter
from skillwrapper.refactored.utils import StateT


@dataclass(frozen=True)
class Predicate(Generic[StateT]):
    """A symbolic predicate with object-typed parameters."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    semantics: str | None = None  # Optional NL description of the predicate's meaning

    def __str__(self) -> str:
        """Return a readable string representation of the predicate."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    def as_pddl(self) -> str:
        """Return a PDDL string representation of the predicate."""
        type_map: dict[str, set[str]] = {}  # Map type names to predicate params of the type
        for param in self.parameters:
            if param.object_type not in type_map:
                type_map[param.object_type] = set()
            type_map[param.object_type].add(param.name)

        type_groups = [
            f"{' '.join(f'?{p}' for p in params)} - {type_name}"
            for type_name, params in type_map.items()
        ]

        return f"({self.name} {' '.join(g for g in type_groups)})"

    def ground_with(self, bindings: Bindings) -> PredicateInstance:
        """Ground the predicate using the given parameter bindings."""
        return PredicateInstance(self, bindings)

    def holds_in(self, state: StateT, bindings: Bindings) -> bool:
        """Evaluate whether the predicate holds in a state under the given parameter bindings.

        :param state: Low-level environment state in which the predicate is evaluated
        :param bindings: Parameter bindings used to ground the predicate
        :return: True if the grounded predicate holds, else False
        """
        raise NotImplementedError("Predicate.holds_in(...)")  # TODO

    def compute_all_groundings(self, objects: ConcreteObjects) -> set[PredicateInstance]:
        """Compute the set of all valid groundings of the predicate using the given objects.

        :param objects: Set of all concrete objects in the environment
        :return: Set of all valid instances of the predicate using the given objects
        """
        objs_per_param_type = (
            objects.get_all_objects_of_type(param.object_type) for param in self.parameters
        )  # Generator over [the set of all objects of the type] of each predicate parameter

        # Find all valid tuples of concrete args by taking a Cartesian product over the object sets
        all_valid_groundings = product(*objs_per_param_type)
        all_bindings = (
            {param.name: obj_name}
            for grounding in all_valid_groundings
            for param, obj_name in zip(self.parameters, grounding, strict=True)
        )

        return {PredicateInstance(self, bindings) for bindings in all_bindings}


@dataclass(frozen=True)
class PredicateInstance(Generic[StateT]):
    """A predicate grounded using particular concrete objects."""

    predicate: Predicate[StateT]
    bindings: Bindings

    def holds_in(self, state: StateT) -> bool:
        """Evaluate whether the predicate instance holds in the given state."""
        return self.predicate.holds_in(state, self.bindings)

    def __str__(self) -> str:
        """Return a readable string representation of the predicate instance."""
        args_string = ", ".join(self.bindings[p.name] for p in self.predicate.parameters)
        return f"{self.predicate.name}({args_string})"
