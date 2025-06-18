"""Define classes to represent abstract symbolic actions, both lifted and grounded."""

from dataclasses import dataclass
from typing import Generic

from skillwrapper.refactored.parameters import Bindings, DiscreteParameter
from skillwrapper.refactored.predicates import AbstractState, Predicate
from skillwrapper.refactored.utils import StateT


@dataclass(frozen=True)
class Operator(Generic[StateT]):
    """A lifted abstract action defining an abstract transition model for a skill."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    positive_preconditions: set[Predicate]  # Must be true to apply the operator
    negative_preconditions: set[Predicate]  # Must be false to apply the operator
    add_effects: set[Predicate]  # Predicates made true by the operator
    delete_effects: set[Predicate]  # Predicates made false by the operator


class OperatorInstance(Generic[StateT]):
    """An operator grounded with concrete objects."""

    def __init__(self, operator: Operator, bindings: Bindings) -> None:
        """Initialize the operator instance with an operator and parameter bindings."""
        self.operator = operator
        self.bindings = bindings

        # Ground the operator instance's preconditions and effects
        self.ground_pos_pre = {
            pre.ground_with(self.bindings) for pre in self.operator.positive_preconditions
        }
        self.ground_neg_pre = {
            pre.ground_with(self.bindings) for pre in self.operator.negative_preconditions
        }
        self.ground_add_eff = {eff.ground_with(self.bindings) for eff in self.operator.add_effects}
        self.ground_delete_eff = {
            eff.ground_with(self.bindings) for eff in self.operator.delete_effects
        }

    @property
    def grounded_signature(self) -> str:
        """Return a string representation of the operator instance's signature."""
        ordered_args = ", ".join(self.bindings[p.name] for p in self.operator.parameters)
        return f"{self.operator.name}({ordered_args})"

    def is_applicable(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the operator instance is applicable in an abstract state."""
        if any((pos_pre not in abstract_state) for pos_pre in self.ground_pos_pre):
            return False
        return all((neg_pre not in abstract_state) for neg_pre in self.ground_neg_pre)

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the grounded operator to update the given abstract state."""
        if not self.is_applicable(abstract_state):
            error = f"Cannot apply {self.grounded_signature} in abstract state: {abstract_state}"
            raise ValueError(error)

        return abstract_state.difference(self.ground_delete_eff).union(self.ground_add_eff)
