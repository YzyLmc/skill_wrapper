"""Unit tests for classes related to operators."""

import pytest

from skillwrapper.refactored.operators import Effects, Preconditions
from skillwrapper.refactored.parameters import DiscreteParameter
from skillwrapper.refactored.predicates import Predicate


@pytest.fixture
def available_predicates() -> dict[str, Predicate]:
    """Define a mapping of predicates available in an example PDDL domain."""
    holding = Predicate("Holding", parameters=(DiscreteParameter("?obj", "Pickable"),))
    surface_clear = Predicate(
        "SurfaceClear",
        parameters=(DiscreteParameter("?surface", "Surface"),),
    )
    dirty_surface = Predicate(
        "DirtySurface",
        parameters=(DiscreteParameter("?surface", "Surface"),),
    )
    hand_empty = Predicate("HandEmpty", parameters=())
    stacked = Predicate(
        "Stacked",
        parameters=(
            DiscreteParameter("?on_top", "Stackable"),
            DiscreteParameter("?on_bottom", "Stackable"),
        ),
    )

    predicates = [holding, surface_clear, dirty_surface, hand_empty, stacked]
    return {p.name: p for p in predicates}


def test_preconditions_from_pddl(available_predicates: dict[str, Predicate]) -> None:
    """Verify that Preconditions instances can be constructed from PDDL strings."""
    # Arrange - Create a string representing PDDL preconditions
    preconditions_pddl = """:precondition (and
        (Holding ?obj)
        (SurfaceClear ?surface)
        (not (DirtySurface ?surface))
    )
    """

    # Act - Parse preconditions from the PDDL string
    preconditions = Preconditions.from_pddl(preconditions_pddl, available_predicates)

    # Assert - Verify that the parsed preconditions are correct
    assert len(preconditions.positive) == 2
    assert len(preconditions.negative) == 1
    assert available_predicates["Holding"] in preconditions.positive
    assert available_predicates["SurfaceClear"] in preconditions.positive
    assert available_predicates["DirtySurface"] in preconditions.negative


def test_effects_from_pddl(available_predicates: dict[str, Predicate]) -> None:
    """Verify that Effects instances can be constructed from PDDL strings."""
    # Arrange - Create a string representing PDDL effects
    effects_pddl = """:effect (and
        (HandEmpty)
        (not (Holding ?on_top))
        (Stacked ?on_top ?on_bottom)
    )
    """

    # Act - Parse effects from the PDDL string
    effects = Effects.from_pddl(effects_pddl, available_predicates)

    # Assert - Verify that the parsed effects are correct
    assert len(effects.add) == 2
    assert len(effects.delete) == 1
    assert available_predicates["HandEmpty"] in effects.add
    assert available_predicates["Stacked"] in effects.add

    expected_holding = Predicate("Holding", parameters=(DiscreteParameter("?on_top", "Pickable"),))
    assert expected_holding in effects.delete
