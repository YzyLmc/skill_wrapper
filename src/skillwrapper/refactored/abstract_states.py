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

    facts: set[PredicateInstance]  # Set of predicate instances that hold true in a state

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

        :param predicates: Set of predicates defining possible abstract relations
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
