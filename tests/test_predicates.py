"""Define unit tests to verify correctness of the Predicate and PredicateInstance classes."""

import pytest

from skillwrapper.refactored.environment import ConcreteObjects
from skillwrapper.refactored.parameters import DiscreteParameter
from skillwrapper.refactored.predicates import Predicate, PredicateInstance


@pytest.fixture
def on_table() -> Predicate:
    """Define a predicate representing whether an object is on a table."""
    obj = DiscreteParameter("obj", object_type="Pickable")
    table = DiscreteParameter("table", object_type="Table")
    return Predicate("OnTable", parameters=(obj, table))


@pytest.fixture
def stacked() -> Predicate:
    """Define a predicate representing whether one stackable object is stacked on another."""
    on_top = DiscreteParameter("on_top", object_type="Stackable")
    on_bottom = DiscreteParameter("on_bottom", object_type="Stackable")
    return Predicate("Stacked", parameters=(on_top, on_bottom))


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

    # Assert - Verify that the resulting grounding match what's expected
    assert on_table_instances, "Expected groundings for predicate 'OnTable' but received none."
    assert stacked_instances, "Expected groundings for predicate 'Stacked' but received none."

    expected_on_table_instances = {
        PredicateInstance(on_table, bindings={"obj": "jar", "table": "table1"}),
        PredicateInstance(on_table, bindings={"obj": "jar", "table": "table2"}),
        PredicateInstance(on_table, bindings={"obj": "plate1", "table": "table1"}),
        PredicateInstance(on_table, bindings={"obj": "plate1", "table": "table2"}),
        PredicateInstance(on_table, bindings={"obj": "plate2", "table": "table1"}),
        PredicateInstance(on_table, bindings={"obj": "plate2", "table": "table2"}),
    }
    assert on_table_instances == expected_on_table_instances

    expected_stacked_instances = {
        PredicateInstance(on_table, bindings={"on_top": "plate1", "on_bottom": "plate1"}),
        PredicateInstance(on_table, bindings={"on_top": "plate1", "on_bottom": "plate2"}),
        PredicateInstance(on_table, bindings={"on_top": "plate2", "on_bottom": "plate1"}),
        PredicateInstance(on_table, bindings={"on_top": "plate2", "on_bottom": "plate2"}),
    }
    assert stacked_instances == expected_stacked_instances
