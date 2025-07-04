"""Define classes to represent symbolic predicates, lifted and grounded."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import Generic

from skillwrapper.refactored.environment import ConcreteObjects
from skillwrapper.refactored.parameters import Bindings, DiscreteParameter
from skillwrapper.refactored.pddl import PDDLable
from skillwrapper.refactored.utils import StateT


@dataclass(frozen=True)
class Predicate(PDDLable, Generic[StateT]):
    """A symbolic predicate with object-typed parameters."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    semantics: str | None = None  # Optional NL description of the predicate's meaning

    def __str__(self) -> str:
        """Return a readable string representation of the predicate."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    @classmethod
    def from_pddl(cls, pddl: str) -> Predicate:
        """Construct a Predicate instance from a string of PDDL.

        :param pddl: PDDL string representation of a predicate
        :return: Constructed Predicate instance
        """
        match = re.match(r"^\((\S+)(.*)\)$", pddl.strip())
        if not match:
            raise ValueError(f"Could not parse Predicate from PDDL string: '{pddl}'")

        name = match.group(1).strip()
        params_string = match.group(2)

        # Process the parameters string to identify the parameters and their types
        parameters: list[DiscreteParameter] = []  # Completely finalized parameters
        awaiting_type: list[str] = []  # Parameter names waiting for their type to be specified
        next_token_is_type = False  # Indicates that the next token will be a parameter type

        for token in params_string.split():
            if next_token_is_type:
                type_name = token
                parameters.extend(DiscreteParameter(param, type_name) for param in awaiting_type)

                next_token_is_type = False
                awaiting_type = []

            elif token == "-":
                next_token_is_type = True
            else:
                awaiting_type.append(token)

        if awaiting_type:
            error = f"Predicate '{name}' didn't define a type for parameters: {awaiting_type}."
            raise ValueError(error)

        return Predicate(name, tuple(parameters))

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate."""
        types_to_params: dict[str, list[str]] = {}  # Map type names to all such predicate params
        for param in self.parameters:
            if param.object_type not in types_to_params:
                types_to_params[param.object_type] = []
            types_to_params[param.object_type].append(param.name)

        type_groups = []
        for type_name, relevant_params in types_to_params.items():
            pddl_params = " ".join(relevant_params)
            type_groups.append(f"{pddl_params} - {type_name}")

        params_string = " " + " ".join(type_groups) if type_groups else ""
        return f"({self.name}{params_string})"

    def ground_with(self, bindings: Bindings) -> PredicateInstance:
        """Ground the predicate using the given parameter bindings."""
        return PredicateInstance(self, bindings)

    def compute_all_groundings(self, objects: ConcreteObjects) -> set[PredicateInstance]:
        """Compute the set of all valid groundings of the predicate using the given objects.

        :param objects: Set of concrete objects used to ground the predicate
        :return: Set of all valid instances of the predicate using the given objects
        """
        objs_per_param_type = (
            objects.get_all_objects_of_type(param.object_type) for param in self.parameters
        )  # Generator over [the set of all objects of the type] of each predicate parameter

        # Find all valid tuples of concrete args by taking a Cartesian product
        all_valid_groundings = product(*objs_per_param_type)

        all_grounded_predicates: set[PredicateInstance] = set()
        for grounding in all_valid_groundings:
            bindings = {p.name: obj for p, obj in zip(self.parameters, grounding, strict=True)}
            all_grounded_predicates.add(PredicateInstance(self, bindings))

        return all_grounded_predicates

    def holds_in(self, state: StateT, bindings: Bindings) -> bool:
        """Evaluate whether the predicate holds in a state under the given bindings.

        :param state: Low-level environment state in which the predicate is evaluated
        :param bindings: Parameter bindings used to ground the predicate
        :return: True if the grounded predicate holds, else False
        """
        raise NotImplementedError("Predicate.holds_in(...)")  # TODO


@dataclass(frozen=True)
class PredicateInstance(Generic[StateT]):
    """A predicate grounded using particular concrete objects."""

    predicate: Predicate[StateT]
    bindings: Bindings

    def __hash__(self) -> int:
        """Compute the hash value of the predicate instance based on its string representation."""
        return hash(str(self))

    def __str__(self) -> str:
        """Return a readable string representation of the predicate instance."""
        args_string = ", ".join(self.bindings[p.name] for p in self.predicate.parameters)
        return f"{self.predicate.name}({args_string})"

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate instance."""
        args_string = " ".join(self.bindings[p.name] for p in self.predicate.parameters)
        return f"({self.predicate.name} {args_string})"

    def holds_in(self, state: StateT) -> bool:
        """Evaluate whether the predicate instance holds in the given state."""
        return self.predicate.holds_in(state, self.bindings)


@dataclass(frozen=True)
class PositiveNegativePredicates:
    """A collection containing positive and negative predicates."""

    positive: set[Predicate]
    negative: set[Predicate]

    @classmethod
    def from_pddl(cls, pddl: str) -> PositiveNegativePredicates:
        """Construct a PositiveNegativePredicates instance from a string of PDDL.

        The PDDL string is assumed to contain a sequence of predicates, as in:
            "(OnTop ?x ?y) (not (Under ?a ?b)) (Open ?c)..."

        :param pddl: PDDL string representation of positive and negative predicates
        :return: Constructed PositiveNegativePredicates instance
        """
        positive = set()
        negative = set()

        # Parse through the predicates by finding balanced parentheses
        i = 0
        while i < len(pddl):
            if pddl[i] != "(":
                i += 1
                continue

            # Otherwise, the current character (index i) is an open parenthesis: "("
            close_parens_needed = 1
            j = i + 1
            while j < len(pddl) and close_parens_needed > 0:
                if pddl[j] == "(":
                    close_parens_needed += 1
                elif pddl[j] == ")":
                    close_parens_needed -= 1
                j += 1

            if not close_parens_needed:
                predicate_string = pddl[i:j]

                # Check if the predicate is negated
                not_match = re.match(r"^\(\s*not\s+(.+)\)$", predicate_string, re.DOTALL)
                if not_match:
                    inner_content = not_match.group(1).strip()
                    negative.add(Predicate.from_pddl(inner_content))
                else:
                    positive.add(Predicate.from_pddl(predicate_string))
            else:
                raise ValueError(f"Unmatched parentheses when parsing PDDL predicates: {pddl}")

        return PositiveNegativePredicates(positive, negative)
