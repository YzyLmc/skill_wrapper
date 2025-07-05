"""Define functions implementing the predicate invention algorithm."""

from dataclasses import dataclass

from skillwrapper.refactored.abstract_model import AbstractModel
from skillwrapper.refactored.abstract_states import AbstractState
from skillwrapper.refactored.gpt4_client import GPT4Client
from skillwrapper.refactored.operators import Operator, Preconditions
from skillwrapper.refactored.predicates import Predicate, PredicateInstance
from skillwrapper.refactored.skills import Skill
from skillwrapper.refactored.transition_data import AbstractDataset, AbstractTransition
from skillwrapper.refactored.unification import find_consistent_bindings

@dataclass(frozen=True)
class ContrastivePair:
    """A pair of contradictory abstract transitions to be used in predicate invention."""

    transitions: tuple[AbstractTransition, AbstractTransition]  # Their order has no meaning


OperatorsMap = dict[Skill, set[Operator]]  # TODO: Put somewhere reasonable


def in_alpha(abstract_state: AbstractState, operators: set[Operator]) -> bool:
    """Evaluate whether an abstract state is in the alpha set of a skill's operators.

    By "alpha set" we mean the set of (abstract) states satisfying the preconditions of any
        of the operators for a particular skill, permitting any grounding for the predicates.

    :param abstract_state: Consists of the set of grounded predicates that held in some state
    :param operators: Set of all operators corresponding to a particular skill
    :return: True if the abstract state is in the alpha set, otherwise False.
    """
    return any(exemplifies(abstract_state, op.preconditions) for op in operators)

    return False  # TODO

def invent_predicate(skill: Skill, model: AbstractModel, dataset: AbstractDataset) -> Predicate:
    """Invent a predicate for the given skill using the current abstract model and dataset.

    :param skill: A particular skill to invent a predicate for
    :param model: Abstract model defining the existing predicates and operators
    :param dataset: Collection of abstract skill execution traces
    :return: Invented predicate (TODO: Do we always?)
    """
    # TODO: Assume that we've already synchronized the abstract state with all current predicates


def find_mismatches(
    skill: Skill,
    model: AbstractModel,
    dataset: AbstractDataset,
) -> set[ContrastivePair]:
    """Identify contradictory pairs of states, where ... TODO."""
    relevant_transitions = dataset.get_abstract_transitions_for_skill(skill)

    for operator in model.operators[skill]:
        # TODO: Generate all possible groundings... of the operator? Fix this skill's parameters
        # TODO: If... something in alpha? Save whether the step was in alpha

    return set()  # TODO
