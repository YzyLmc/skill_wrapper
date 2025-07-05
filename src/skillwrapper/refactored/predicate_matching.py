"""Define functions to support matching between abstract states and predicates."""

from skillwrapper.refactored.abstract_states import AbstractState
from skillwrapper.refactored.predicates import Predicate, PredicateInstance
from skillwrapper.refactored.unification import UnifierBindings, unify


def exemplifies(
    abstract_state: AbstractState,
    positive: set[Predicate],
    negative: set[Predicate],
) -> bool:
    """Evaluate whether the given abstract state exemplifies the given conditions.

    We say that an abstract state (i.e., a set of grounded predicates) "exemplifies" some set
        of predicates (i.e., required conditions) if there exists a consistent grounding of
        the predicates such that all resulting grounded conditions appear in the abstract state.

    :param abstract_state: Set of grounded predicates representing "known true" conditions
    :param positive: Positive conditions to be satisfied in the exemplification
    :param negative: Negative conditions to be avoided in the exemplification
    :return: True if the abstract state exemplifies the predicates, else False
    """
    # Map from predicates (as strings) to the sets of corresponding facts in the abstract state
    predicate_to_facts: dict[str, set[PredicateInstance]] = {}
    for fact in abstract_state.facts:
        predicate = str(fact.predicate)
        if predicate not in predicate_to_facts:
            predicate_to_facts[predicate] = set()
        predicate_to_facts[predicate].add(fact)

    # Check that all positive predicates exist at least once in the abstract state
    for predicate in positive:
        if str(predicate) not in predicate_to_facts:
            return False

    bindings = find_consistent_bindings(predicate_to_facts, list(positive), 0, list(negative), {})

    return bindings is not None


def find_consistent_bindings(
    predicate_to_facts: dict[str, set[PredicateInstance]],
    positive: list[Predicate],
    positive_idx: int,
    negative: list[Predicate],
    bindings: UnifierBindings,
) -> UnifierBindings | None:
    """Find consistent bindings to satisfy the given conditions with the given facts.

    :param predicate_to_facts: Map from full-predicate strings to sets of relevant facts
    :param positive: List of predicates to be satisfied under the binding
    :param positive_idx: Index of the current positive condition to be satisfied
    :param negative: List of predicates to be avoided under the binding
    :param bindings: Map from parameter names to bound expressions
    :return: Consistent bindings for the predicates, or None if no valid binding exists
    """
    # Check for any negative conditions that are fully grounded under the current bindings
    for neg_predicate in negative:
        if str(neg_predicate) not in predicate_to_facts:
            continue  # Skip negative predicates that never appear as any fact

        if is_fully_grounded(neg_predicate, bindings):
            neg_grounded = PredicateInstance(
                neg_predicate,
                bindings={p.name: bindings[p.name] for p in neg_predicate.parameters},
            )

            # Check if this grounded negative condition matches any fact
            if neg_grounded in predicate_to_facts[str(neg_predicate)]:
                return None

    if positive_idx >= len(positive):  # Base case: all positive conditions have been satisfied
        return bindings

    pos_predicate = positive[positive_idx]

    # Try to unify this positive condition with each fact of the same predicate
    for fact in predicate_to_facts[str(pos_predicate)]:
        new_bindings = unify(pos_predicate, fact, bindings.copy())

        if new_bindings is not None:
            return find_consistent_bindings(
                predicate_to_facts,
                positive,
                positive_idx + 1,
                negative,
                new_bindings,
            )

    return None  # If we couldn't unify with any fact, we'll have to backtrack


def is_fully_grounded(predicate: Predicate, bindings: UnifierBindings) -> bool:
    """Check if all parameters of the predicate have been bound."""
    return all(param.name in bindings for param in predicate.parameters)
