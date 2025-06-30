"""Represent skills, environments, and domains and handle their import/export from YAML."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from skillwrapper.utils import determine_pytorch_device


### Environment Layer - Defines the initial state and objects in a scenario ###
@dataclass(frozen=True)
class AnnotatedImage:
    """An image of the environment with an (optional) associated natural language description."""

    image_path: Path  # Filepath to the image
    description: str | None  # Optional description of the photo of the environment


class EgocentricImageState:
    """An environment state represented as a collection of egocentric images."""

    def __init__(self, initial_images: dict[str, AnnotatedImage]) -> None:
        """Initialize the egocentric image-based state."""
        self.latest_images = initial_images  # Map from location names to relevant images/NL

    @classmethod
    def from_yaml(cls, yaml_data: dict[str, Any]) -> EgocentricImageState:
        """Import an EgocentricImageState instance from YAML data.

        :param yaml_data: Dictionary of data describing an egocentric image-based state
        :return: Constructed EgocentricImageState instance
        """
        locations: dict[str, AnnotatedImage] = {}  # Maps each location name to its image
        for location_name, image_data in yaml_data.items():
            image_path = Path(image_data.get("image_path", "NO PATH SPECIFIED"))
            if not image_path.exists():
                error = f"Location {location_name} had invalid image path: {image_path}"
                raise FileNotFoundError(error)

            locations[location_name] = AnnotatedImage(image_path, image_data.get("description"))

        return EgocentricImageState(locations)


### "RCR Bridge" Layer - No clue what that's supposed to mean (SkillWrapper doesn't use RCRs) ###

# TODO: Decipher the generate_possible_groundings() function

### Skill Sequence Proposal Layer ###


@dataclass(frozen=True)
class SkillTransition(Generic[StateT]):
    """An observed transition resulting from executing a skill instance in an environment."""

    state_before: StateT  # State from which the skill execution was attempted
    skill_instance: SkillInstance  # Concrete skill that was (possibly) executed
    success: bool  # Was the skill successfully executed?
    state_after: StateT | None  # State after the skill executed, if the skill succeeded

    def __post_init__(self) -> None:
        """Verify that the constructed transition is valid."""
        if self.success and self.state_after is None:
            raise ValueError("A successful skill transition must include an 'after' state.")

    @classmethod
    def from_yaml(
        cls,
        state_type: type[StateT],
        yaml_data: dict[str, Any],
        domain: Domain,
        env: Environment,
    ) -> SkillTransition:
        """Load a SkillTransition instance from data loaded from YAML."""
        for key in ["state_before", "skill_instance", "success"]:
            if key not in yaml_data:
                raise KeyError(f"SkillTransition.from_yaml() requires the YAML key: '{key}'")

        state_before = state_type.from_yaml(yaml_data["state_before"])
        skill_instance = SkillInstance.from_string(yaml_data["skill_instance"], domain, env)
        success = bool(yaml_data["success"])
        state_after = state_type.from_yaml(yaml_data["state_after"]) if success else None

        return SkillTransition(state_before, skill_instance, success, state_after)


SkillExecutionTrace = list[SkillTransition]  # A sequence of attempted skill executions
Dataset = list[SkillExecutionTrace]  # A collection of skill execution traces


@dataclass(frozen=True)
class Prompts:
    """A pair of prompts for an LLM specifying a context and a repeatable task."""

    system_prompt: str
    task_prompt: str


class SkillSequenceProposer:
    """Proposes exploratory skill sequences using a vision-language model."""

    def __init__(
        self,
        domain: Domain,
        env: Environment,
        prompt_path: Path,
        predicates: list[Predicate] | None,
        dataset: Dataset | None,
    ) -> None:
        """Initialize the skill sequence proposer.

        :param domain: SkillWrapper domain specifying skills and object types
        :param env: SkillWrapper environment specifying objects and the initial state
        :param prompt_path: YAML filepath specifying prompts for the VLM
        :param predicates: List of predicates already learned (or None if first iteration)
        :param dataset: Existing dataset of skill execution traces (or None if first iteration)
        """
        self.domain = domain
        self.env = env
        self.skill_to_idx = {
            skill.name: idx for idx, skill in enumerate(self.domain.skills.values())
        }

        # Map from predicate names to a description of their semantics
        self.predicate_semantics = (
            predicate_list_to_semantics_dict(predicates) if predicates else {}
        )

        # Initialize frequency counts for all skill instance pairs
        if dataset is None:
            dataset = []  # Empty dataset => All skill counts will remain zero
        self.skill_pairs_matrix = self._compute_skill_pair_matrix(dataset)
        self.total_skills_executed = sum(len(trace) for trace in dataset)

        ### Parameters kept from original implementation ###
        self.curr_shannon_entropy = 0.0
        self.generation_args = {
            "temperature": 0.6,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.35,
            "top_p": 1.0,
            "max_tokens": 550,
            "engine": "gpt-4o",
            "stop": "",
        }

        self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.device = determine_pytorch_device()

        # TODO: Use prompt_path
        # TODO: OBJECT_IN_SCENE changed to OBJECTS_IN_SCENE
        self.system_prompt = "None"  # TODO
        self.task_prompt = "None"  # TODO
        self.env_description = "None"  # TODO

        # Embedding model is used to ground LLM output to groundable/executable skills and objects
        self.embedding_model = SentenceTransformer("stsb-roberta-large").to(self.device)

        self.skill_embeddings = self.embedding_model.encode(
            [skill.name for skill in self.domain.skills.values()],
            batch_size=32,
            convert_to_tensor=True,
            device=self.device,
        )
        self.object_name_embeddings = self.embedding_model.encode(
            list(self.env.objects.object_names),
            batch_size=32,
            convert_to_tensor=True,
            device=self.device,
        )

        self.h = 1  # KDE parameter
        # scaling parameters for pareto-optimal task selection
        self.k = 10  # set period after how many skill executions to switch mode
        # all alphas are in the range [1,3]
        self.chainability_alpha = lambda _: 1
        self.entropy_gain_alpha = lambda x: np.cos((np.pi / self.k) * x) + 2

    def create_llm_prompt(self) -> Prompts:
        """Create prompts for the LLM to propose skill sequences."""
        skill_prompts = []
        for skill_name, skill in self.domain.skills.items():
            param_descriptions = [
                f"{param.name} (Type {param.object_type}): {param.semantics}"
                for param in skill.parameters
            ]
            skill_prompt = f"{skill_name}\n" + "\n".join(param_descriptions)
            skill_prompts.append(skill_prompt)

        objects_with_types = [
            f"{obj_name}: {list(types)}" for obj_name, types in self.env.objects.objects.items()
        ]

        least_explored_skills = self.get_least_explored_skills()
        task_prompt = copy.copy(self.task_prompt)
        task_prompt = (
            task_prompt.replace("[SKILL_PROMPT]", "\n\n".join(skill_prompts))
            .replace("[OBJECTS_IN_SCENE]", "\n".join(objects_with_types))
            .replace("[ENV_DESCRIPTION]", self.env_description)
            .replace("[LEAST_EXPLORED_SKILLS]", ", ".join(least_explored_skills))
        )

        return Prompts(system_prompt=self.system_prompt, task_prompt=task_prompt)

    ### COVERAGE: Functions for entropy computation and determining least explored tasks ###

    def compute_entropy_for_sequence(self, skill_sequence: list[Skill]) -> tuple[float, np.ndarray]:
        """Compute the Shannon entropy for the given skill sequence.

        :return: Tuple of (entropy value after executing the sequence, updated skill pairs matrix)
        """
        new_skill_pairs_matrix = copy.deepcopy(self.skill_pairs_matrix)
        p1 = 0
        p2 = min(1, len(skill_sequence))
        while p2 < len(skill_sequence):
            idx1 = self.skill_to_idx[skill_sequence[p1].name]
            idx2 = self.skill_to_idx[skill_sequence[p2].name]
            new_skill_pairs_matrix[idx1, idx2] += 1
            p1 = p2
            p2 += 1

        normalized_skill_pair_prob = (
            new_skill_pairs_matrix / np.sum(new_skill_pairs_matrix)
            if np.sum(new_skill_pairs_matrix) > 0
            else new_skill_pairs_matrix
        )
        log_skill_pair_prob = np.where(
            normalized_skill_pair_prob > 0.0,
            np.log(normalized_skill_pair_prob),
            0.0,
        )
        new_shannon_entropy = np.sum(-normalized_skill_pair_prob * log_skill_pair_prob)
        return new_shannon_entropy, new_skill_pairs_matrix

    def compute_shannon_entropy(
        self,
        skill_sequences: list[list[Skill]],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Compute the Shannon entropy for a collection of proposed skill sequences."""
        normalized_skill_pair_prob = (
            self.skill_pairs_matrix / np.sum(self.skill_pairs_matrix)
            if np.sum(self.skill_pairs_matrix) > 0
            else self.skill_pairs_matrix
        )
        log_skill_pair_prob = np.where(
            normalized_skill_pair_prob > 0,
            np.log(normalized_skill_pair_prob),
            0,
        )
        curr_shannon_entropy = np.sum(-normalized_skill_pair_prob * log_skill_pair_prob)

        skill_sequence_entropy_gains = []
        skill_sequence_skill_counts = []
        # measure entropy gain for each task
        for skill_sequence in skill_sequences:
            entropy, counts = self.compute_entropy_for_sequence(skill_sequence)

            # entropy gain is maximum of difference
            skill_sequence_entropy_gains.append(entropy - curr_shannon_entropy)
            skill_sequence_skill_counts.append(counts)

        return np.array(skill_sequence_entropy_gains), skill_sequence_skill_counts


"""Define a class to learn symbolic operators from observed skill transitions."""

from dataclasses import dataclass

from skillwrapper.refactored.environment import ConcreteObjects
from skillwrapper.refactored.operators import Operator
from skillwrapper.refactored.parameters import DiscreteParameter
from skillwrapper.refactored.predicates import PredicateInstance
from skillwrapper.refactored.skills import Skill
from skillwrapper.refactored.transition_data import AbstractDataset, AbstractTransition
from skillwrapper.refactored.utils import StateT


@dataclass
class TransitionMapping:
    """A mapping between objects and operator parameters for a single abstract transition."""

    transition: AbstractTransition
    object_name_to_param_idx: dict[str, int]  # Map object names to parameter indices
    param_idx_to_type: dict[int, str]  # Map parameter indices to their object types
    next_unused_param_idx: int = 0  # Index of the next unused operator parameter (TODO: Context?)


@dataclass(frozen=True)
class ExtractedEffects:
    """Grounded effects extracted from an abstract transition."""

    add_effects: set[PredicateInstance]  # Grounded predicates made true after the skill
    delete_effects: set[PredicateInstance]  # Grounded predicates made false after the skill


@dataclass(frozen=True)
class ParameterPosition:
    """A parameter's position in a predicate structure."""

    param_idx: int  # Index of the predicate parameter
    object_type: str  # Type of object associated with the parameter


class PredicateStructure:
    """Represents the parameter structure of a predicate instance."""

    def __init__(self, p_instance: PredicateInstance, mapping: TransitionMapping) -> None:
        """Initialize the predicate structure for the given predicate instance.

        :param p_instance: Predicate instance whose structure is analyzed
        :param mapping: Example mapping of skill parameters for an abstract transition
        """
        # TODO: Could we just use a tuple of types?

        param_structure: list[ParameterPosition] = []

        for param in p_instance.predicate.parameters:
            bound_object = p_instance.bindings[param.name]

            # Get or assign a parameter index for this trans
            if bound_object not in mapping.object_name_to_param_idx:  # Assign new index!
                param_idx = mapping.next_unused_param_idx
                mapping.object_name_to_param_idx[object] = param_idx
                mapping.param_idx_to_type[param_idx] = param.object_type
                mapping.next_unused_param_idx += 1
            else:
                param_idx = mapping.object_name_to_param_idx[bound_object]

            param_structure.append(ParameterPosition(param_idx, param.object_type))

        self.structure = tuple(param_structure)


@dataclass(frozen=True)
class StructuralPredicateKey:
    """A key for identifying predicates with analogous structure."""

    predicate_name: str
    param_structure: PredicateStructure


class OperatorLearner:
    """A system for learning operators given observed skill transitions."""

    def __init__(self, object_types: ConcreteObjects) -> None:
        """Initialize the operator learning system with a mapping of object types.

        :param object_types: Maps concrete object names to their object types
        """
        self.object_types = object_types

    def learn_operator(self, dataset: AbstractDataset, skill: Skill) -> Operator:
        """Learn an operator from a skill's successful abstract transitions in a dataset.

        :param dataset: Collection of abstracted skill execution traces
        :param skill: Relevant skill to learn an operator for
        :return: Operator learned from the abstract dataset
        """
        successful_transitions = [
            t for trace in dataset for t in trace if t.skill_name == skill.name and t.success
        ]  # Filter to only successful abstract transitions for the relevant skill
        if not successful_transitions:
            error = f"No successful transitions found to learn an operator for skill {skill.name}"
            raise ValueError(error)

        # Build a mapping between objects and operator parameters for all transitions
        transition_mappings: list[TransitionMapping] = []

        # Establish the minimal operator parameters based on the skill's parameters
        skill_param_positions = {
            idx: DiscreteParameter(f"?p{idx}", param.object_type, None)
            for idx, param in enumerate(skill.parameters)
        }

        # Build object mappings for each transition
        for transition in successful_transitions:
            mapping = TransitionMapping(transition, {}, {}, len(skill.parameters))

            # Map skill parameters first (ensures consistent parameter positions)
            for idx, param in enumerate(skill.parameters):
                bound_object = transition.skill_instance.bindings[param.name]
                mapping.object_name_to_param_idx[bound_object] = idx
                mapping.param_idx_to_type[idx] = param.object_type

            transition_mappings.append(mapping)

        # Extract all changed predicate instances from the abstract transitions
        all_changed_predicates = OperatorLearner.extract_changed_predicates(successful_transitions)

        # Find effects common to all transitions (as example grounded predicates)

        return None  # TODO

    @staticmethod
    def extract_changed_predicates(data: list[AbstractTransition]) -> set[PredicateInstance]:
        """Extract all predicate instances that changed in any of the given abstract transitions.

        :param data: Collection of abstract transitions corresponding to skill executions
        :return: Set of grounded predicates that changed in any given transition
        """
        changed = set()

        for t in data:
            assert t.abstract_after is not None, "Expected abstract 'after' state."

            # Include predicates that were added or deleted from the abstract state
            changed.update(t.abstract_before.symmetric_difference(t.abstract_after))

        return changed

    def extract_common_effects(self, transition_data: list[TransitionMapping]) -> ExtractedEffects:
        """Extract effects that are common across all given transitions.

        Look for effects with the same structure, even if their specific objects differ.

        :param transition_data: Collection of parameter mappings for abstract transitions
        :return: Example ground effects common across the given transition data
        """
        if not transition_data:
            return ExtractedEffects(set(), set())

        # Process first transition to get an example effect structure
        first = transition_data[0]
        assert first.transition.abstract_after is not None, "Expected an 'after' abstract state."
        first_add_effects = first.transition.abstract_after - first.transition.abstract_before
        first_delete_effects = first.transition.abstract_before - first.transition.abstract_after

        # Build a structural representation of the effects
        structural_add: dict[StructuralPredicateKey, PredicateInstance] = {}
        structural_delete: dict[StructuralPredicateKey, PredicateInstance] = {}

        for predicate_instance in first_add_effects:
            param_structure = PredicateStructure(predicate_instance, first)
            key = StructuralPredicateKey(predicate_instance.predicate.name, param_structure)
            structural_add[key] = predicate_instance

        for predicate_instance in first_delete_effects:
            param_structure = PredicateStructure(predicate_instance, first)
            key = StructuralPredicateKey(predicate_instance.predicate.name, param_structure)
            structural_delete[key] = predicate_instance

            # TODO: Continue! I should just read `get_action_from_cluster`

        return None  # TODO


def invent_predicates(predicates: set[Predicate], skills: set[Skill]) -> set[Predicate]:
    """Invent predicates for the given collection of skill executability traces.

    Implements Algorithm 3 of the SkillWrapper paper (Yang et al., 2024).

    :param predicates: Set of previously invented predicates
    :param skills: Set of available robot skills
    """
    # While exists (s_i, s_i') and (s_j, s_j') in the data s.t. current abstraction
    #   of s_i = abs(s_j) but the observed data of I_w(s_i) and I_w(s_j) differ...
    #   We need to invent a predicate to disambiguate these two cases!
    # if validate_precondition(P, w, D), then add this predicate
    # Check: How often is this actually a precondition of the skill? Can filter if inconsistent
    #
    # PRE: Whenever the skill succeeded, this predicate must have been always true
    # Always true, or always false, works (i.e., PRE and EFF can contain negative predicates)

    # While exists (s_i, s_i') and (s_j, s_j') in the data s.t. abstraction function shows
    #   same effects, meaning F(s_i') - F(s_i) = F(s_j') - F(s_j) but I_w(s_i) =/= I_w(s_j)...
    #   We invent a predicate to capture the unexpressed effect of the skill
    # Make sure the invented predicate differentiates these cases! And is always an effect
    #
    # EFF: Must have ended up true (or always false) after all successful executions
    #
    # We now do scoring over partitions: Add invented P to PRE or EFF only after we've clustered
    #   based on effects. i.e., we do scoring over partitions, not the entire skills.
    # Use the candidate predicate set when computing clustered effect sets, then determine
    #   whether the predicate should be added to any PRE or EFF.
    # A trace is samples of (s, w, s'). Effect is set of changed predicates

    return predicates
