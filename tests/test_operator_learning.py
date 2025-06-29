"""Define unit tests to verify the correctness of operator learning."""

import pytest

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.parameters import DiscreteParameter
from skillwrapper.refactored.predicates import Predicate
from skillwrapper.refactored.skills import Skill


@pytest.fixture
def pb_domain() -> Domain:
    """Define a Domain to represent a robot with skills to manipulate a jar of peanut butter."""
    picked_obj = DiscreteParameter("picked", object_type="Pickable")
    pick_left_skill = Skill("PickLeft", parameters=(picked_obj))

    scoopable = DiscreteParameter("container", object_type="Scoopable")  # Can be scooped
    scooper = DiscreteParameter("scooper", object_type="Scooper")  # Any utensil for scooping
    scoop_skill = Skill("Scoop", parameters=(scoopable, scooper))

    opened_obj = DiscreteParameter("opened", object_type="Openable")  # Can be opened
    open_skill = Skill("Open", parameters=(opened_obj))

    return Domain.from_skills("PeanutButterDomain", {pick_left_skill, scoop_skill, open_skill})


@pytest.fixture
def pb_predicates() -> set[Predicate]:
    """Define example predicates for a domain involving scooping from a jar of peanut butter."""
    pickable_obj = DiscreteParameter("obj", object_type="Pickable")
    clear_above = Predicate("ClearAbove", parameters=(pickable_obj))

    robot = DiscreteParameter("robot", object_type="Robot")
    holding = Predicate("Holding", parameters=(robot, pickable_obj))

    openable_obj = DiscreteParameter("obj", object_type="Openable")
    lid_removed = Predicate("LidRemoved", parameters=(openable_obj))

    return {clear_above, holding, lid_removed}


# TODO: Objects: PBJar, Knife, Bread, Cup, Table, Shelf, Robot
# TODO: Could create predicates: IsOpen(obj: Openable)

# @pytest
# def test_operator_learning1() -> None:
#     """TODO: Document what this test checks.

#     This test was adapted from RCR_bridge.py to use refactored data structures.
#     """
#     clear_above = Predicate("ClearAbove")
