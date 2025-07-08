"""Define functions to implement the operator learning algorithm."""

from dataclasses import dataclass
from typing import Any  # TODO: Remove Any

from skillwrapper.refactored.operators import Operator
from skillwrapper.refactored.parameters import DiscreteParameter
from skillwrapper.refactored.skills import Skill, SkillInstance
from skillwrapper.refactored.transition_data import SuccessfulAbstractTransition


class ObjectRoles:
    """Maps object names to their potential parameter roles during operator learning."""

    def __init__(self, skill_instance: SkillInstance) -> None:
        """Initialize the fixed object roles imposed by the given skill instance.

        :param skill_instance: Skill instance common across the transitions in the partition
        """
        self.fixed_roles: dict[str, DiscreteParameter] = {
            skill_instance.bindings[p.name]: p for p in skill_instance.skill.parameters
        }

        self.additional_objects: set[str] = set()
        """Set of non-skill-parameter object names from predicate instances in the partition."""


def compute_operators(skill: Skill, partitions: Any) -> set[Operator]:
    """Calculate operators for one skill using partitions by termination set.

    Note: Adapted from create_operators_from_partitions(), which was in invent_predicate.py.

    :param skill: Skill for which operators are computed
    :param partitions: Map from partition IDs to sets of relevant abstract transitions
    :return: Set of calculated operators for the skill
    """
    # TODO: The original code looped over grounded skills, and if one corresponded to this skill,
    #   created one operator for each partition corresponding to the grounded skill.
    # Should each partition only contain transitions with the exact same skill instance?

    return {
        compute_one_operator(skill, transitions_in_partition)
        for partition_id, transitions_in_partition in partitions.items()
    }  # TODO: Do we care about the partition IDs at all once they've been computed? I doubt it


def compute_one_operator(partition: list[SuccessfulAbstractTransition]) -> Operator:
    """Compute a symbolic operator using one partition of successful abstract transitions.

    Note: This function is adapted from the following previous functions:
        - create_one_operator_from_one_partition() from invent_predicate.py
        - operator_from_transitions() from RCR_bridge.py

    :param partition: Collection of successful abstract transitions used to compute the operator
    :return: Constructed Operator instance
    """
    return None  # TODO


def compute_one_operator(dataset: SuccessfulAbstractDataset) -> Operator:
    """Compute a symbolic operator for a skill using one partition of abstract transitions.

    Note: Adapted from create_one_operator_from_one_partition() which was in invent_predicate.py.
        - operator_from_transitions() was also subsumed, although that seems to have
            mostly been conversion code into the type system of the RCR codebase.

    :param dataset: Dataset of abstract transitions corresponding to successful skill executions
    :return: Computed Operator instance for the partition
    """
    # TODO: Skipping bridge.unify_obj_type(transitions, grounded_skill, type_dict)... because
    #   it's unclear why we need to adjust the types for type-parameterized predicates?
    # This results in the "unified transitions" list, which I'll take as the input dataset here

    # TODO: We finally call get_action_from_cluster() which takes in the transitions and obj2pid,
    #   which: For all objects in the skill parameters, if not have one, add the object to obj2pid
    #   with a steadily increasing integer. Also save a mapping for "PID to type"
    # i.e., create potential operator parameters for the concrete objects in the skill instance,
    #   then for any other objects used in this partition, assign them new parameters? I'm not
    #   sure if this logic is sound: couldn't a single concrete object play different roles in
    #   different transitions, even for the same skill, say we were near one box and later another?

    return None  # TODO


# def compute_one_operator(skill: Skill, partition: Any) -> Operator:
#     """Compute a symbolic operator for a skill using one partition of abstract transitions.
#
#     :param skill: Skill corresponding to the computed operator
#     :param partition: Set of abstract transitions partitioned based on termination and effects
#     :return: Computed Operator instance
#     """
#     # TODO: Calls bridge.unify_obj_type(transitions, grounded_skill, type_dict)...
#     # Creates unified_transitions by iterating over transitions and their before/after states...
#     #   Create a PredicateState as follows:
#     #       For each grounded predicate in the state, populate a types_list either with an
#     #           object's type if defined in obj2type (which came from bridge.unify_obj_type()),
#     #           or as defined in the predicate instance. Use types_list to create a Predicate.
#     #   Add the constructed PredicateState to a so-called "unified transition"
#     #   Add the constructed "unified transition" to the list of unified transitions
#     #
#     #   Finally, return bridge.operator_from_transitions(
#     #       Pass in the unified transitions, skill instance, type_dict, and obj2type,
#     #   ) plus the bridge's "get_pid_to_type()" and the constructed obj2type

# unify_obj_type:
# Finds the common lowest hierarchy(?) of type of each parameter in PRE/EFF of the skill. Does so
#   by precomputing the PRE and EFF and determines the lowest hierarchy type for each parameter.
#   Output: Original PDDL transitions with object types replaced by "common lowest hierarchy type"
#
# All predicate parameters are already typed, right? So I don't understand how this is relevant...
