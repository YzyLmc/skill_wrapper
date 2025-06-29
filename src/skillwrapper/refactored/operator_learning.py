"""Define data structures and functions to learn operators from abstract transitions."""

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.operators import Operator
from skillwrapper.refactored.skills import Skill
from skillwrapper.refactored.transition_data import AbstractDataset, AbstractTransition


def learn_operators(dataset: AbstractDataset, domain: Domain) -> set[Operator]:
    """Learn operators from the given dataset of abstracted skill executions in the given domain.

    :param dataset: Collection of abstracted skill execution traces
    :param domain: Domain specifying the available skills
    :return: Set of learned operators
    """
    learned_operators: set[Operator] = set()
    for skill in domain.skills.values():
        relevant_transitions = dataset.get_transitions_for_skill(skill)
        new_operators = learn_operators_for_skill(skill, relevant_transitions)
        learned_operators.update(new_operators)  # TODO: Map-reduce over individual cores

    return learned_operators


def learn_operators_for_skill(skill: Skill, transitions: set[AbstractTransition]) -> set[Operator]:
    """Learn operators for a particular skill given a dataset of its abstract transitions.

    :param skill: Skill involved in the given abstract transitions
    :param transitions: Collection of abstract transitions involving the skill
    :return: Set of operators learned for the skill
    """
    return set()  # TODO
