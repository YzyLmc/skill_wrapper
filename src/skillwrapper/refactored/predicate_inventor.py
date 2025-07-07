"""Define functions implementing the predicate invention algorithm."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from skillwrapper.refactored.abstract_model import AbstractModel
from skillwrapper.refactored.abstract_states import AbstractState
from skillwrapper.refactored.gpt4_client import GPT4Client
from skillwrapper.refactored.operators import Operator
from skillwrapper.refactored.predicate_matching import exemplifies
from skillwrapper.refactored.predicates import Predicate
from skillwrapper.refactored.skills import Skill
from skillwrapper.refactored.transition_data import (
    AbstractDataset,
    AbstractStateDelta,
    AbstractTransition,
)


class ContrastivePairType(Enum):
    """Enumerates types of contrastive pairs (i.e., those based on preconditions vs effects)."""

    PRE = "Preconditions"
    EFF = "Effects"


@dataclass(frozen=True)
class ContrastivePair:
    """A pair of contradictory abstract transitions to be used in predicate invention."""

    skill: Skill  # Skill common across the two abstract transitions
    transitions: tuple[AbstractTransition, AbstractTransition]  # Their order has no meaning
    pair_type: ContrastivePairType
    # TODO: This may need to instead have low-level transitions?

    def get_images(self) -> list[str | Path]:
        """Retrieve the images illustrating the contrast in the contrastive pair."""
        return []  # TODO: Convert to low-level transitions and return image paths


def find_contrastive_pairs(
    skill: Skill,
    model: AbstractModel,
    dataset: AbstractDataset,
) -> set[ContrastivePair]:
    """Find all contrastive pairs for a under the given abstract model.

    Note: Replaces detect_mismatch() which was previously in invent_predicate.py.

    :param skill: Skill for which contradictory transition pairs will be found
    :param model: Symbolic abstract model of the skill's dynamics
    :param dataset: Dataset of observed abstract transitions
    :return: Set of identified contrastive pairs
    """
    operators = model.operators[skill]

    transitions_list = list(dataset.get_abstract_transitions_for_skill(skill))
    transitions_in_alpha = [in_alpha(t.abstract_before, operators) for t in transitions_list]
    transitions_and_alphas = list(zip(transitions_list, transitions_in_alpha, strict=True))

    # Look for contrastive pairs with matching alpha set membership but differing execution success
    contrastive_pairs: set[ContrastivePair] = set()
    for i, (transition_i, in_alpha_i) in enumerate(transitions_and_alphas):
        for transition_j, in_alpha_j in transitions_and_alphas[i + 1 :]:
            if in_alpha_i == in_alpha_j and transition_i.success != transition_j.success:
                contrastive_pairs.add(
                    ContrastivePair(skill, (transition_i, transition_j), ContrastivePairType.PRE),
                )

    return contrastive_pairs


def in_alpha(abstract_state: AbstractState, operators: set[Operator]) -> bool:
    """Evaluate whether an abstract state is in the "alpha set" of a skill's operators.

    By "alpha set" we mean the set of abstract states satisfying the preconditions of any of the
        operators for a particular skill, under any consistent grounding of the preconditions.

    Note: Replaces in_alpha() which was previously in invent_predicate.py.
        - TODO: That function seemed to also support checking for effects?

    :param abstract_state: Set of facts (i.e., grounded predicates) that hold in some state
    :param operators: Set of operators for some particular skill
    :return: True if the abstract state is in the alpha set, otherwise False.
    """
    if not operators:
        return True

    for operator in operators:
        positive_pre = operator.preconditions.positive
        negative_pre = operator.preconditions.negative
        if exemplifies(abstract_state, positive=positive_pre, negative=negative_pre):
            return True

    return False


def invent_predicates(
    skill: Skill,
    model: AbstractModel,
    dataset: AbstractDataset,
    max_attempts: int = 3,
) -> set[Predicate]:
    """Invent predicates for the given skill using the current abstract model and dataset.

    :param skill: Skill whose transitions will be used to trigger predicate invention
    :param model: Abstract model defining the existing predicates and operators
    :param dataset: Collection of abstract skill execution traces
    :param max_attempts: Maximum number of attempts to invent a predicate for any contrastive pair
    :return: Set of invented predicates
    """
    # TODO: Assume that we've already synchronized the abstract state with all current predicates
    contrastive_pairs = find_contrastive_pairs(skill, model, dataset)
    if contrastive_pairs:
        for _ in range(max_attempts):
            pass

    return set()  # TODO


class PredicateInventor:
    """Invents predicates using a VLM based on observed skill execution transitions."""

    def __init__(self, current_model: AbstractModel) -> None:
        """Initialize the predicate inventor.

        :param current_model: Current abstract model specifying learned predicates and operators
        """
        self.gpt_client = GPT4Client()
        self.prompt_template = ""  # TODO: Populate from domain YAML

        self.current_model = current_model
        self.rejected_predicates: dict[Skill, set[Predicate]] = {}  # Rejected predicates per skill

    @property
    def predicates(self) -> set[Predicate]:
        """Retrieve the current set of predicates in the learned abstract model."""
        return self.current_model.predicates

    def create_prompt(self, pair: ContrastivePair) -> str:
        """Create a text prompt for predicate invention based on the given contrastive pair.

        :param pair: Contrastive pair of skill transitions that triggered predicate invention
        :return: Text of the constructed prompt (to be paired with relevant images)
        """
        for predicate in self.predicates:
            if predicate.semantics is None:
                raise ValueError(f"Predicate {predicate} has undefined semantics.")

        transition_a, transition_b = pair.transitions
        predicates_to_avoid = self.rejected_predicates.get(pair.skill, set())

        return (
            self.prompt_template.replace("[SKILL]", str(pair.skill))
            .replace("[SKILL_INSTANCE_A]", str(transition_a.skill_instance))
            .replace("[SKILL_INSTANCE_B]", str(transition_b.skill_instance))
            .replace("[SUCCESS_A]", "succeeded" if bool(transition_a.success) else "failed")
            .replace("[SUCCESS_B]", "succeeded" if bool(transition_b.success) else "failed")
            .replace("[PREDICATES]", "\n".join(f"{p}: {p.semantics}" for p in self.predicates))
            .replace("[REJECTED_PREDICATES]", ", ".join(str(p) for p in predicates_to_avoid))
        )

    def generate_predicate(self, pair: ContrastivePair) -> Predicate:
        """Use the VLM to propose a new predicate based on the given contrastive pair.

        Note: Replaces generate_pred() which was previously in invent_predicate.py.

        :param pair: Contrastive pair of skill transitions triggering predicate invention
        :return: Invented predicate based on the given data
        """
        text_prompt = self.create_prompt(pair)
        prompt_images = pair.get_images()

        # TODO: Why did precondition pairs need 2 images while effect pairs need 4?
        response = self.gpt_client.generate_multimodal(text_prompt, prompt_images)
        response_pieces = [s.strip() for s in response.split(":")]
        if len(response_pieces) != 2:
            raise RuntimeError(f"Invalid predicate invention response from VLM:\n{response}")

        predicate_string, semantics = response_pieces

        return Predicate.from_string(predicate_string, semantics)

    def invent_one_predicate(self, pair: ContrastivePair) -> Predicate | None:
        """Invent a predicate to resolve the given contrastive pair of observed transitions.

        :param pair: Contrastive pair of skill transitions triggering predicate invention
        :return: Invented predicate based on the contrastive pair, or None if invention fails
        """
        new_predicate = self.generate_predicate(pair)

        if new_predicate in self.predicates:
            print(f"Predicate {new_predicate} is already in the predicate set.")
            return None
        if new_predicate in self.rejected_predicates[pair.skill]:
            print(f"The VLM generated a predicate we've already rejected: {new_predicate}")
            return None

        hypothetical_predicate_set = self.current_model.predicates.copy()
        hypothetical_predicate_set.add(new_predicate)

        # TODO: Update the abstract state evaluation for the new predicate

        # TODO: ...

        return None  # TODO

    def partition(
        self,
        transitions: list[AbstractTransition],
    ) -> dict[tuple[int, int], set[AbstractTransition]]:
        """Partition a dataset of abstract transitions by termination states and effects.

        Only successful transitions will be used for partitioning.

        Note: Replaces partition_by_termination_n_eff() which was in invent_predicate.py.

        :param transitions: Collection of abstract skill execution transitions
        :return: Map from partition ID tuples to sets of abstract transitions in each partition
        """
        successful_transitions = [t for t in transitions if t.success]

        # Map abstract termination states to the indices of all corresponding transitions
        termination_partitions: dict[AbstractState, set[int]] = {}
        effects_partitions: dict[AbstractStateDelta, set[int]] = {}

        for idx, transition in enumerate(successful_transitions):
            assert transition.abstract_after is not None, "Expected an 'after' abstract state"

            # Partition based on termination sets using the 'after' state of each transition
            if transition.abstract_after not in termination_partitions:
                termination_partitions[transition.abstract_after] = set()
            termination_partitions[transition.abstract_after].add(idx)

            # Partition based on effects using the sets of predicates changed by each transition
            abstract_delta = transition.compute_abstract_delta()

            if abstract_delta not in effects_partitions:  # TODO: Was this meant to be the delta?
                effects_partitions[abstract_delta] = set()
            effects_partitions[abstract_delta].add(idx)

        ### Compute a map from skill instances to combined termination-effects partitions ###

        # Map from successful transition indices to tuples of (term, eff) partition indices
        transition_idx_to_partitions: dict[int, list[int]] = {}
        for idx in range(len(successful_transitions)):
            transition_idx_to_partitions[idx] = [-1, -1]  # All -1s should be replaced

        for term_idx, termination_state in enumerate(termination_partitions.keys()):
            for transition_idx in termination_partitions[termination_state]:
                transition_idx_to_partitions[transition_idx][0] = term_idx

        for eff_idx, abstract_delta in enumerate(effects_partitions.keys()):
            for transition_idx in effects_partitions[abstract_delta]:
                transition_idx_to_partitions[transition_idx][1] = eff_idx

        # Output a map from (termination partition index, effect partition index) to transitions
        full_partitions: dict[tuple[int, int], set[AbstractTransition]] = {}
        for transition_idx, partitions in transition_idx_to_partitions.items():
            assert len(partitions) == 2, "Expected exactly two partitions"
            assert all(x >= 0 for x in partitions), "All partitions should have valid indices"

            term_idx, eff_idx = partitions
            partitions_tuple = (term_idx, eff_idx)
            transition = successful_transitions[transition_idx]

            if partitions_tuple not in full_partitions:
                full_partitions[partitions_tuple] = set()
            full_partitions[partitions_tuple].add(transition)

        return full_partitions  # TODO:


# TODO: Create a EvaluatedPredicates dataclass to manage partially evaluated predicate sets, to
#   allow converting to AbstractStates while keep the actual evaluation process separate?

# (TODO) Score by partitions (to do so, we'll calculate the hypothetical operators per partition):
#   1. Create a map from SkillInstances to all relevant *successful* (abstract?) transitions
#   2. Create a map from (SkillInstances to?) partition ID tuples to relevant transitions(?)
#   3. Create operators using the calculated partitions
