"""Implement the unification algorithm for first-order logic expressions.

Reference: Figure 9.1, Section 9.2, pg. 285 of AIMA (4th Ed.) by Stuart Russell and Peter Norvig
"""

from collections.abc import Sequence
from typing import Any

from skillwrapper.refactored.abstract_states import AbstractState
from skillwrapper.refactored.parameters import DiscreteParameter, ParametersT
from skillwrapper.refactored.predicates import (
    Predicate,
    PredicateInstance,
    PredicateSequence,
    PredicateT,
    get_parameters,
)

# Type representing all possible input types for unify()
Unifiable = DiscreteParameter | str | ParametersT | PredicateT | PredicateSequence

UnifierBindings = dict[str, Any]  # Maps from parameter names to bound FOL expressions


def unify(x: Unifiable, y: Unifiable, bindings: UnifierBindings | None) -> UnifierBindings | None:
    """Find parameter bindings to unify the given first-order logic (FOL) expressions.

    Reference: Figure 9.1, Section 9.2, pg. 285 of AIMA (4th Ed.) by Russell and Norvig.

    :param x: One of the two FOL expressions to be unified
    :param y: The other FOL expression to be unified
    :param bindings: Current set of parameter-expression bindings (or None upon failure)
    :return: Bindings that unify the expressions, else None if unification fails
    """
    if bindings is None:
        return None

    if type(x) is type(y) and x == y:
        return bindings

    if isinstance(x, DiscreteParameter):
        return unify_parameter(x, y, bindings)
    if isinstance(y, DiscreteParameter):
        return unify_parameter(y, x, bindings)

    if isinstance(x, PredicateT) and isinstance(y, PredicateT):
        return unify_predicates(x, y, bindings)

    if isinstance(x, Sequence) and isinstance(y, Sequence):
        return unify_sequences(x, y, bindings)

    raise RuntimeError(
        f"Error: Reached unhandled case during unification."
        f"\n\tFirst FOL expression: {x}\n\tSecond FOL expression: {y}",
    )


def unify_parameter(
    param: DiscreteParameter,
    x: Unifiable,
    bindings: UnifierBindings | None,
) -> UnifierBindings | None:
    """Find a parameter binding to unify with the given first-order logic (FOL) expression.

    :param param: Parameter for which a binding is found
    :param x: First-order logic expression to unify with
    :param bindings: Current set of parameter-object bindings (or None upon failure)
    :return: Updated bindings to unify with the FOL expression, else None if unification fails
    """
    if bindings is None:
        return None

    if param.name in bindings:
        bound_value = bindings[param.name]
        return unify(bound_value, x, bindings)

    if isinstance(x, DiscreteParameter) and x.name in bindings:
        bound_value = bindings[x.name]
        return unify(param, bound_value, bindings)

    if occurs_in(param, x):
        return None  # If the parameter occurs inside the expression, unification is impossible

    bindings[param.name] = x
    return bindings


def occurs_in(param: DiscreteParameter, x: Unifiable) -> bool:
    """Check whether the given parameter occurs in a first-order logic expression.

    :param param: Parameter searched for within the expression
    :param x: First-order logic expression searched for the parameter
    :return: True if the parameter occurs in the expression, else False
    """
    if isinstance(x, DiscreteParameter) and param == x:
        raise RuntimeError(f"occurs_in() received identical DiscreteParameters: {param} {x}")

    if isinstance(x, Predicate):
        return param in x.parameters

    if isinstance(x, (PredicateInstance, str)):
        return False

    if isinstance(x, ParametersT):
        return any((isinstance(x_param, DiscreteParameter) and param == x_param) for x_param in x)

    if isinstance(x, Sequence):  # Handle the PredicateSequence case
        return any(occurs_in(param, item) for item in x)

    raise RuntimeError(f"Unexpected FOL expression type: {x} (type {type(x)})")


def unify_predicates(
    x: PredicateT,
    y: PredicateT,
    bindings: UnifierBindings | None,
) -> UnifierBindings | None:
    """Find parameter bindings to unify the two given predicates.

    :param x: One predicate-like expression to unify
    :param y: Another predicate-like expression to unify
    :param bindings: Current set of parameter-expression bindings (or None upon failure)
    :return: Updated bindings to unify the predicates, else None if unification fails
    """
    if bindings is None or x.name != y.name:  # Can't unify predicates with differing names
        return None

    # If the predicates have the same name, all that's left to unify are their parameters
    return unify_sequences(get_parameters(x), get_parameters(y), bindings)


def unify_sequences(
    x: Sequence[Unifiable],
    y: Sequence[Unifiable],
    bindings: UnifierBindings | None,
) -> UnifierBindings | None:
    """Find parameter bindings to unify the given sequences.

    :param x: Sequence of elements to be unified
    :param y: Another sequence of elements to be unified
    :param bindings: Current set of parameter-expression bindings (or None upon failure)
    :return: Updated bindings to unify the sequences, else None if unification fails
    """
    if len(x) != len(y):  # Impossible to unify two sequences with differing lengths
        return None

    if len(x) > 1:
        (x_first, *x_rest) = x
        (y_first, *y_rest) = y
        return unify_sequences(x_rest, y_rest, unify(x_first, y_first, bindings))

    (x_only,) = x
    (y_only,) = y
    return unify(x_only, y_only, bindings)
