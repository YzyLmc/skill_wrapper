"""Define classes to represent symbolic abstract states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from skillwrapper.refactored.environment import ConcreteObjects
from skillwrapper.refactored.predicates import Predicate, PredicateInstance
from skillwrapper.refactored.utils import StateT


@dataclass(frozen=True)
class AbstractState:
    """An abstract state is the set of true predicate instances (i.e., facts) in a state."""

    facts: set[PredicateInstance]  # Set of predicate instances that hold true in the state

    @classmethod  # TODO: Replace predicatestate_to_pddlstate with this!
    def from_partially_evaluated(cls, partial: PartiallyEvaluatedAbstractState) -> AbstractState:
        """Construct an AbstractState instance based on a partially evaluated abstract state."""
        return AbstractState({fact for fact, value in partial.fact_values.items() if value})

    def __contains__(self, predicate_instance: PredicateInstance) -> bool:
        """Evaluate whether a given predicate instance is in the abstract state."""
        return predicate_instance in self.facts

    def __str__(self) -> str:
        """Create a readable string representation of the abstract state."""
        sorted_facts = "\n\t".join(sorted(str(fact) for fact in self.facts))
        return f"AbstractState(\n{sorted_facts}\n)"


class AbstractStateSpace(Generic[StateT]):
    """An abstract state space specifies all possible predicate instances in any abstract state."""

    def __init__(self, predicates: set[Predicate], objects: ConcreteObjects) -> None:
        """Initialize the abstract state space using all valid groundings of the given predicates.

        :param predicates: Set of predicate defining possible abstract relations between objects
        :param objects: Concrete objects in the current environment
        """
        self.possible_facts: set[PredicateInstance] = set()
        for predicate in predicates:
            all_instances = predicate.compute_all_groundings(objects)
            self.possible_facts.update(all_instances)

    def abstract(self, state: StateT) -> AbstractState:
        """Compute the abstract state for the given low-level state.

        :param state: Low-level state of the environment
        :return: Computed abstract state (i.e., all facts that are true in the low-level state)
        """
        return AbstractState({fact for fact in self.possible_facts if fact.holds_in(state)})


class PartiallyEvaluatedAbstractState(Generic[StateT]):
    """An abstract state containing potentially yet-to-be-evaluated predicate instances."""

    def __init__(self, abstract_state_space: AbstractStateSpace[StateT]) -> None:
        """Initialize the partially evaluated abstract state based on an abstract state space."""
        self.fact_values: dict[PredicateInstance, bool | None] = {}  # True, False, None (unknown)
        for fact in abstract_state_space.possible_facts:
            self.fact_values[fact] = None

    def set_fact_true(self, fact: PredicateInstance) -> None:
        """Set the Boolean value of the specified fact as true."""
        if fact not in self.fact_values:
            raise KeyError(f"Cannot set value of unrecognized fact: {fact}")
        self.fact_values[fact] = True

    def set_fact_false(self, fact: PredicateInstance) -> None:
        """Set the Boolean value of the specified fact as false."""
        if fact not in self.fact_values:
            raise KeyError(f"Cannot set value of unrecognized fact: {fact}")
        self.fact_values[fact] = False

    def add_possible_facts(self, new_facts: set[PredicateInstance]) -> None:
        """Add new possible facts to the partially evaluated abstract state if they don't exist."""
        for fact in new_facts:
            if fact not in self.fact_values:
                self.fact_values[fact] = None  # Initialize the fact's value as unknown

    def get_unevaluated_facts(self) -> set[PredicateInstance]:
        """Retrieve any unevaluated facts in the partially evaluated abstract state."""
        return {fact for fact, value in self.fact_values.items() if value is None}

    def prune_to_facts(self, keep_facts: set[PredicateInstance]) -> None:
        """Prune the partially evaluated abstract state to only include facts in the given set."""
        new_fact_values: dict[PredicateInstance, bool | None] = {}
        for fact in keep_facts:
            if fact in self.fact_values:
                new_fact_values[fact] = self.fact_values[fact]

        self.fact_values = new_fact_values
