"""Define classes to represent abstract symbolic actions, both lifted and grounded."""

from __future__ import annotations

import re
from dataclasses import dataclass

from skillwrapper.refactored.abstract_states import AbstractState
from skillwrapper.refactored.parameters import Bindings, DiscreteParameter
from skillwrapper.refactored.predicates import (
    PositiveNegativePredicates,
    Predicate,
    PredicateInstance,
)


@dataclass(frozen=True)
class GroundedPreconditions:
    """A collection of grounded preconditions (i.e., required true/false predicate instances)."""

    positive: set[PredicateInstance]  # Grounded predicates that must be true
    negative: set[PredicateInstance]  # Grounded predicates that must be false

    def satisfied_in(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the grounded preconditions are satisfied in an abstract state."""
        if any((pos_precondition not in abstract_state) for pos_precondition in self.positive):
            return False
        return all((neg_precondition not in abstract_state) for neg_precondition in self.negative)


@dataclass(frozen=True)
class Preconditions:
    """A collection of predicates defining positive and negative preconditions."""

    positive: set[Predicate]  # Predicates that must hold true to apply the operator
    negative: set[Predicate]  # Predicates that must be false to apply the operator

    @classmethod
    def from_pddl(cls, pddl: str, predicates: dict[str, Predicate]) -> Preconditions:
        """Construct a Preconditions instance from a string of PDDL.

        :param pddl: PDDL string representation of preconditions
        :param predicates: Collection of predicates available in the PDDL domain
        :return: Constructed Preconditions instance
        """
        match = re.match(r":precondition\s*\(\s*and(.*)\)", pddl.strip(), re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse Preconditions from PDDL string: '{pddl}'")

        predicates_string = match.group(1).strip()
        parsed_predicates = PositiveNegativePredicates.from_pddl(predicates_string, predicates)

        return Preconditions(parsed_predicates.positive, parsed_predicates.negative)

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the preconditions."""
        positive_pre = "\n\t".join(sorted(str(pre) for pre in self.positive))
        negative_pre = "\n\t".join(sorted(f"(not {pre})" for pre in self.negative))

        return f":precondition (and\n\t{positive_pre}\n\t{negative_pre}\n)"

    def ground_with(self, bindings: Bindings) -> GroundedPreconditions:
        """Ground the preconditions using the given parameter bindings."""
        return GroundedPreconditions(
            positive={p.ground_with(bindings) for p in self.positive},
            negative={p.ground_with(bindings) for p in self.negative},
        )


@dataclass(frozen=True)
class GroundedEffects:
    """A collection of grounded effects (i.e., predicate instances added/removed by an action)."""

    add: set[PredicateInstance]  # Grounded predicates added to the abstract state
    delete: set[PredicateInstance]  # Grounded predicates removed from the abstract state

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the grounded effects to the given abstract state."""
        return AbstractState(abstract_state.facts.difference(self.delete).union(self.add))


@dataclass(frozen=True)
class Effects:
    """A collection of predicates defining add and delete effects of an operator."""

    add: set[Predicate]  # Predicates added to the abstract state by the operator
    delete: set[Predicate]  # Predicates deleted from the abstract state by the operator

    @classmethod
    def from_pddl(cls, pddl: str, predicates: dict[str, Predicate]) -> Effects:
        """Construct an Effects instance from a string of PDDL.

        :param pddl: PDDL string representation of an effects set
        :param predicates: Collection of predicates available in the PDDL domain
        :return: Constructed Effects instance
        """
        match = re.match(r":effect\s*\(\s*and(.*)\)", pddl.strip(), re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse Effects from PDDL string: '{pddl}'")

        predicates_string = match.group(1).strip()
        parsed_predicates = PositiveNegativePredicates.from_pddl(predicates_string, predicates)

        return Effects(add=parsed_predicates.positive, delete=parsed_predicates.negative)

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the effects."""
        add_eff = "\n\t".join(sorted(str(eff) for eff in self.add))
        del_eff = "\n\t".join(sorted(f"(not {eff})" for eff in self.delete))

        return f":effect (and\n\t{add_eff}\n\t{del_eff}\n)"

    def ground_with(self, bindings: Bindings) -> GroundedEffects:
        """Ground the effects using the given parameter bindings."""
        return GroundedEffects(
            add={p.ground_with(bindings) for p in self.add},
            delete={p.ground_with(bindings) for p in self.delete},
        )


@dataclass(frozen=True)
class Operator:
    """A lifted abstract action defining an abstract transition model for a skill."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    preconditions: Preconditions  # Positive and negative preconditions for applying the operator
    effects: Effects  # Effects added to and deleted from the abstract state by the operator

    def ground_with(self, bindings: Bindings) -> OperatorInstance:
        """Ground the operator using the given parameter bindings."""
        return OperatorInstance(self, bindings)


class OperatorInstance:
    """An operator grounded with concrete objects."""

    def __init__(self, operator: Operator, bindings: Bindings) -> None:
        """Initialize the operator instance with an operator and parameter bindings."""
        self.operator = operator
        self.bindings = bindings

        # Ground the operator instance's preconditions and effects
        self.ground_preconditions = self.operator.preconditions.ground_with(self.bindings)
        self.ground_effects = self.operator.effects.ground_with(self.bindings)

    @property
    def grounded_signature(self) -> str:
        """Return a readable representation of the operator instance's signature."""
        ordered_args = ", ".join(self.bindings[p.name] for p in self.operator.parameters)
        return f"{self.operator.name}({ordered_args})"

    def is_applicable(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the operator instance is applicable in an abstract state."""
        return self.ground_preconditions.satisfied_in(abstract_state)

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the operator instance to transition from the given abstract state."""
        if not self.is_applicable(abstract_state):
            error = f"Cannot apply {self.grounded_signature} in abstract state: {abstract_state}"
            raise ValueError(error)

        return self.ground_effects.apply(abstract_state)
