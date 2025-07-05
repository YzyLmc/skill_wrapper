"""Define a class to manage the evaluation of an evolving set of grounded predicates."""

from collections.abc import Iterable
from typing import Generic

from skillwrapper.refactored.predicates import PredicateInstance
from skillwrapper.refactored.utils import StateT


class PredicateEvaluations(Generic[StateT]):
    """Manage the evaluation of an evolving set of grounded predicates on a low-level state."""

    def __init__(self, state: StateT) -> None:
        """Initialize the predicate evaluator with a low-level environment state.

        :param state: Low-level environment state on which predicates will be evaluated
        """
        self.low_level_state = state  # Low-level state on which predicates are evaluated

        # Map grounded predicates to their value on the low-level state, or None if unevaluated
        self.evaluations: dict[PredicateInstance, bool | None] = {}

    def get_unevaluated(self) -> set[PredicateInstance]:
        """Retrieve the collection of grounded predicates that remain unevaluated."""
        return {predicate for predicate, value in self.evaluations.items() if value is None}

    def add_grounded_predicates(self, predicates: Iterable[PredicateInstance]) -> None:
        """Add the given grounded predicates to the collection of evaluated predicates.

        :param predicates: Collection of predicate instances to be added
        """
        for p_instance in predicates:
            if p_instance not in self.evaluations:
                self.evaluations[p_instance] = None  # Predicates default to unevaluated

    def update_value(self, grounded_predicate: PredicateInstance, value: bool) -> None:
        """Update the evaluation of a grounded predicate to the given value.

        :param grounded_predicate: Predicate instance whose evaluation is updated
        :param value: Boolean value to which the predicate's evaluation is set
        :raises KeyError: If the predicate isn't defined in the set of evaluated predicates
        """
        if grounded_predicate not in self.evaluations:
            raise KeyError(f"Cannot update value of unrecognized predicate: {grounded_predicate}")
        self.evaluations[grounded_predicate] = value


# TODO: yaml.add_representer and yaml.add_constructor? Export PredicateState to YAML
# TODO: Do we care about filter_pred_list? It took a list of only the predicates to be kept
