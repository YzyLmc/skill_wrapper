"""Define classes to represent abstract symbolic actions, both lifted and grounded."""

from __future__ import annotations

from dataclasses import dataclass

from skillwrapper.refactored.abstract_states import AbstractState
from skillwrapper.refactored.parameters import Bindings, DiscreteParameter
from skillwrapper.refactored.predicates import Predicate, PredicateInstance


@dataclass
class GroundedPreconditions:
    """A collection of grounded preconditions (i.e., required true/false predicate instances)."""

    true_set: set[PredicateInstance]  # Grounded predicates that must be true
    false_set: set[PredicateInstance]  # Grounded predicates that must be false

    def satisfied_in(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the grounded preconditions are satisfied in an abstract state."""
        if any((pos_pre not in abstract_state) for pos_pre in self.true_set):
            return False
        return all((neg_pre not in abstract_state) for neg_pre in self.false_set)


@dataclass
class Preconditions:
    """A collection of predicates defining positive and negative preconditions."""

    true_set: set[Predicate]  # Predicates that must hold true to apply the operator
    false_set: set[Predicate]  # Predicates that must be false to apply the operator

    def ground_with(self, bindings: Bindings) -> GroundedPreconditions:
        """Ground the preconditions using the given parameter bindings."""
        true_set = {p.ground_with(bindings) for p in self.true_set}
        false_set = {p.ground_with(bindings) for p in self.false_set}
        return GroundedPreconditions(true_set, false_set)

    def as_pddl(self) -> str:
        """Return a PDDL string representation of the preconditions."""
        positive_pre = "\n".join(f"\t{pre}" for pre in sorted(self.true_set))
        negative_pre = "\n".join(f"\t(not {pre})" for pre in sorted(self.false_set))
        return f":precondition (and\n{positive_pre}\n{negative_pre}\n)"


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

    add_effects: set[Predicate]
    delete_effects: set[Predicate]

    def ground_with(self, bindings: Bindings) -> GroundedEffects:
        """Ground the effects using the given parameter bindings."""
        add_effects = {p.ground_with(bindings) for p in self.add_effects}
        delete_effects = {p.ground_with(bindings) for p in self.delete_effects}
        return GroundedEffects(add_effects, delete_effects)

    def as_pddl(self) -> str:
        """Return a PDDL string representation of the effects."""
        add_eff = "\n".join(f"\t{eff}" for eff in sorted(self.add_effects))
        delete_eff = "\n".join(f"\t(not {eff})" for eff in sorted(self.delete_effects))
        return f":effect (and\n{add_eff}\n{delete_eff}\n)"


@dataclass(frozen=True)
class Operator:
    """A lifted abstract action defining an abstract transition model for a skill."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    preconditions: Preconditions  # Predicates that must be true/false to apply the operator
    effects: Effects  # Predicates made true/false by applying the operator

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
        self.ground_pre = self.operator.preconditions.ground_with(self.bindings)
        self.ground_eff = self.operator.effects.ground_with(self.bindings)

        self.always_changed = self.get_always_changed()

    @property
    def grounded_signature(self) -> str:
        """Return a readable representation of the operator instance's signature."""
        ordered_args = ", ".join(self.bindings[p.name] for p in self.operator.parameters)
        return f"{self.operator.name}({ordered_args})"

    def get_always_changed(self) -> set[PredicateInstance]:
        """Retrieve all predicate instances always changed by this operator instance."""
        made_false = {p for p in self.ground_pre.true_set if p in self.ground_eff.delete}
        made_true = {p for p in self.ground_pre.false_set if p in self.ground_eff.add}
        return made_false.union(made_true)

    def is_applicable(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the operator instance is applicable in an abstract state."""
        return self.ground_pre.satisfied_in(abstract_state)

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the operator instance to update the given abstract state."""
        if not self.is_applicable(abstract_state):
            error = f"Cannot apply {self.grounded_signature} in abstract state: {abstract_state}"
            raise ValueError(error)

        return self.ground_eff.apply(abstract_state)
