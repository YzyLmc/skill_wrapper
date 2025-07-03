"""Unit tests for the Predicate and PredicateInstance classes."""

from skillwrapper.refactored.predicates import Predicate


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
