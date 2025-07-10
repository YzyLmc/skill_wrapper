"""Unit tests for the Predicate and PredicateInstance classes."""

import pytest

from skillwrapper.refactored.environment import ConcreteObjects
from skillwrapper.refactored.parameters import DiscreteParameter
from skillwrapper.refactored.predicates import Predicate, PredicateInstance


@pytest.fixture
def on_table() -> Predicate:
    """Define a predicate representing whether an object is on a table."""
    obj = DiscreteParameter("?obj", object_type="Pickable")
    table = DiscreteParameter("?table", object_type="Table")
    return Predicate("OnTable", parameters=(obj, table))


@pytest.fixture
def stacked() -> Predicate:
    """Define a predicate representing whether one stackable object is stacked on another."""
    on_top = DiscreteParameter("?on_top", object_type="Stackable")
    on_bottom = DiscreteParameter("?on_bottom", object_type="Stackable")
    return Predicate("Stacked", parameters=(on_top, on_bottom))


@pytest.fixture
def hand_empty() -> Predicate:
    """Define a predicate representing whether the robot's hand is empty."""
    return Predicate("HandEmpty", parameters=())


@pytest.fixture
def full() -> Predicate:
    """Define a predicate representing whether a container is full."""
    container = DiscreteParameter("?container", object_type="Fillable")
    return Predicate("Full", parameters=(container,))


@pytest.fixture
def dirty_surface() -> Predicate:
    """Define a predicate representing whether a surface is dirty."""
    surface = DiscreteParameter("?surface", object_type="Surface")
    return Predicate("DirtySurface", parameters=(surface,))


def test_predicate_from_pddl() -> None:
    """Verify that Predicate instances can be constructed from PDDL strings."""
    # Arrange - Create PDDL strings representing predicates
    hand_empty_pddl = "(HandEmpty)"
    holding_pddl = "(Holding ?obj - Pickable)"
    can_stack_pddl = "(CanStack ?on_top ?on_bottom - Stackable)"
    on_table_pddl = "(OnTable ?obj - Pickable ?table - Table)"
    surface_clear_pddl = "(SurfaceClear ?surface - Surface)"

    # Act - Construct a Predicate from each PDDL string
    hand_empty = Predicate.from_pddl(hand_empty_pddl)
    holding = Predicate.from_pddl(holding_pddl)
    can_stack = Predicate.from_pddl(can_stack_pddl)
    on_table = Predicate.from_pddl(on_table_pddl)
    surface_clear = Predicate.from_pddl(surface_clear_pddl)

    # Assert - Verify expected structure of the predicates
    assert hand_empty.name == "HandEmpty"
    assert len(hand_empty.parameters) == 0

    assert holding.name == "Holding"
    assert len(holding.parameters) == 1
    obj = holding.parameters[0]
    assert obj.name == "?obj"
    assert obj.object_type == "Pickable"

    assert can_stack.name == "CanStack"
    assert len(can_stack.parameters) == 2
    (on_top, on_bottom) = can_stack.parameters
    assert on_top.name == "?on_top"
    assert on_top.object_type == "Stackable"
    assert on_bottom.name == "?on_bottom"
    assert on_bottom.object_type == "Stackable"

    assert on_table.name == "OnTable"
    assert len(on_table.parameters) == 2
    (obj, table) = on_table.parameters
    assert obj.name == "?obj"
    assert obj.object_type == "Pickable"
    assert table.name == "?table"
    assert table.object_type == "Table"

    assert surface_clear.name == "SurfaceClear"
    assert len(surface_clear.parameters) == 1
    surface = surface_clear.parameters[0]
    assert surface.name == "?surface"
    assert surface.object_type == "Surface"


def test_predicate_to_pddl(
    on_table: Predicate,
    stacked: Predicate,
    hand_empty: Predicate,
    full: Predicate,
    dirty_surface: Predicate,
) -> None:
    """Verify that Predicate instances can be converted into PDDL strings."""
    # Arrange - Example predicates are created by fixtures

    # Act - Convert the predicates into PDDL string representations
    on_table_pddl = on_table.to_pddl()
    stacked_pddl = stacked.to_pddl()
    hand_empty_pddl = hand_empty.to_pddl()
    full_pddl = full.to_pddl()
    dirty_surface_pddl = dirty_surface.to_pddl()

    # Assert - Verify that the PDDL strings match what's expected
    assert on_table_pddl == "(OnTable ?obj - Pickable ?table - Table)"
    assert stacked_pddl == "(Stacked ?on_top ?on_bottom - Stackable)"
    assert hand_empty_pddl == "(HandEmpty)"
    assert full_pddl == "(Full ?container - Fillable)"
    assert dirty_surface_pddl == "(DirtySurface ?surface - Surface)"


def test_all_groundings(on_table: Predicate, stacked: Predicate) -> None:
    """Verify that all valid predicate groundings are correctly computed on some examples."""
    # Arrange - Define two example predicates (OnTable and Stacked) and a set of concrete objects
    objects = ConcreteObjects(
        {
            "table1": {"Table"},
            "table2": {"Table"},
            "jar": {"Pickable"},
            "plate1": {"Pickable", "Stackable"},
            "plate2": {"Pickable", "Stackable"},
            "shelf": {"Shelf"},
        },
    )

    # Act - Compute all valid groundings of the predicates
    on_table_instances = on_table.compute_all_groundings(objects)
    stacked_instances = stacked.compute_all_groundings(objects)

    # Assert - Verify that the resulting groundings match what's expected
    assert on_table_instances, "Expected groundings for predicate 'OnTable' but received none."
    assert stacked_instances, "Expected groundings for predicate 'Stacked' but received none."

    expected_on_table_instances = {
        PredicateInstance(on_table, bindings={"?obj": "jar", "?table": "table1"}),
        PredicateInstance(on_table, bindings={"?obj": "jar", "?table": "table2"}),
        PredicateInstance(on_table, bindings={"?obj": "plate1", "?table": "table1"}),
        PredicateInstance(on_table, bindings={"?obj": "plate1", "?table": "table2"}),
        PredicateInstance(on_table, bindings={"?obj": "plate2", "?table": "table1"}),
        PredicateInstance(on_table, bindings={"?obj": "plate2", "?table": "table2"}),
    }
    assert on_table_instances == expected_on_table_instances

    expected_stacked_instances = {
        PredicateInstance(stacked, bindings={"?on_top": "plate1", "?on_bottom": "plate1"}),
        PredicateInstance(stacked, bindings={"?on_top": "plate1", "?on_bottom": "plate2"}),
        PredicateInstance(stacked, bindings={"?on_top": "plate2", "?on_bottom": "plate1"}),
        PredicateInstance(stacked, bindings={"?on_top": "plate2", "?on_bottom": "plate2"}),
    }
    assert stacked_instances == expected_stacked_instances
