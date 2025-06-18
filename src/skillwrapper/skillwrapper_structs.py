"""Represent skills, environments, and domains and handle their import/export from YAML."""

from __future__ import annotations

import copy
import inspect
import os
import re
from collections.abc import Callable, KeysView
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar, get_type_hints

import numpy as np
import yaml
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from skillwrapper.utils import determine_pytorch_device


### Meta-Domain Layer - Define domains based on Python method signatures ###
def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    # Insert underscore before uppercase letters that follow lowercase letters
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return s1.lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase."""
    chunks = name.split("_")
    return "".join(word.capitalize() for word in chunks)


def import_yaml_into_dict(yaml_path: Path, required_keys: set[str]) -> dict[str, Any]:
    """Import data from a YAML file into a Python dictionary.

    :param yaml_path: Filepath to a YAML file containing data to be imported
    :param required_keys: Keys verified to exist in the imported dictionary
    :return: Dictionary mapping YAML keys to corresponding imported data
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Cannot import data from nonexistent YAML file: {yaml_path}")

    try:
        with yaml_path.open() as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
    except yaml.YAMLError as err:
        raise RuntimeError(f"Could not load data from YAML file: {yaml_file}") from err

    for key in required_keys:
        if key not in yaml_data:
            raise KeyError(f"Required key '{key}' is missing from the YAML file: {yaml_file}")

    return yaml_data


def skill_fn(func: Callable) -> Callable:
    """Mark a function as implementing a skill."""
    func._is_skill = True
    return func


def parse_docstring_params(docstring: str) -> dict[str, str]:
    """Extract parameter semantics from a docstring.

    :param docstring: String containing the docstring of a skill function
    :return: Map from parameter names to their semantic descriptions
    """
    param_docs = {}
    param_pattern = r":param\s+(\w+):\s*([^\n]+)"

    for match in re.finditer(param_pattern, docstring):
        param_name = match.group(1)
        description = match.group(2)
        param_docs[param_name] = description

    return param_docs


def method_to_skill(method: Callable[[Any], Any]) -> Skill:
    """Convert a protocol method into a Skill definition.

    :param method: Method defining the parameter signature of a skill
    :return: Constructed Skill instance
    """
    skill_name = snake_to_camel(method.__name__)
    method_params = inspect.signature(method).parameters
    type_hints = get_type_hints(method)

    # Parse docstring for parameter descriptions
    docstring = inspect.getdoc(method) or ""
    param_docs = parse_docstring_params(docstring)

    parameters = []
    for param_name in method_params:
        if param_name == "self":
            continue  # Skip 'self' parameter

        # Get the parameter object type from the type hints
        param_type = type_hints.get(param_name, Any)
        object_type = param_type.__name__.capitalize() if hasattr(param_type, "__name__") else None
        if object_type is None:
            error = f"Skill '{skill_name}' didn't define a type for parameter '{param_name}'"
            raise ValueError(error)

        # Get parameter semantics from the method docstring
        semantics = param_docs.get(param_name)
        if semantics is None:
            error = f"Skill '{skill_name}' didn't define semantics for parameter '{param_name}'"
            raise ValueError(error)

        parameters.append(SkillParameter(param_name, object_type, semantics))

    return Skill(skill_name, tuple(parameters))


### Domain Model Layer - Defines the available skills and their parameters ###


@dataclass(frozen=True)
class SkillParameter:
    """An object-typed discrete parameter of a skill."""

    name: str
    object_type: str
    semantics: str  # English description of the parameter's meaning


Bindings = dict[str, str]  # Map from parameter names to their bound concrete objects

SkillsProtocol = Any  # Stands in for skill protocols for different domains


@dataclass(frozen=True)
class Skill:
    """A skill parameterized by object-typed arguments."""

    name: str
    parameters: tuple[SkillParameter, ...]

    @classmethod
    def from_yaml(cls, skill_name: str, yaml_data: dict[str, Any]) -> Skill:
        """Load a Skill instance from data imported from YAML."""
        assert "parameters" in yaml_data, f"Key 'parameters' missing from YAML data: {yaml_data}."

        skill_params = [
            SkillParameter(param_name, param_data["type"], param_data["semantics"])
            for param_name, param_data in yaml_data["parameters"].items()
        ]

        return Skill(skill_name, tuple(skill_params))  # Execution function registered separately

    def __str__(self) -> str:
        """Return a readable string representation of the skill."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    def to_yaml(self) -> dict[str, Any]:
        """Convert the Skill object into a dictionary of YAML data."""
        return {self.name: self.params_to_yaml()}

    def params_to_yaml(self) -> dict[str, Any]:
        """Convert the Skill parameters into a dictionary of YAML data under a `parameters` key."""
        return {
            "parameters": {
                param.name: {
                    "type": param.object_type,
                    "semantics": param.semantics,
                }
                for param in self.parameters
            },
        }

    def execute(self, executor: SkillsProtocol, bindings: dict[str, str]) -> None:
        """Execute this skill under the given object bindings.

        :param executor: Protocol defining an interface to skill execution
        :param bindings: Map from parameter names to bound object names
        """
        method_name = camel_to_snake(self.name)  # CamelCase skill name -> snake_case method name

        # Access the executor method dynamically
        if not hasattr(executor, method_name):
            raise NotImplementedError(f"Executor has no method: {method_name}")

        method = getattr(executor, method_name)
        args = [bindings[param.name] for param in self.parameters]
        method(*args)


ObjectTypeSet = set[Any]  # Allow object types to be expressed as strings or NewTypes


@dataclass(frozen=True)
class Domain:
    """A domain represents aspects of planning problems that are shared across environments."""

    skills: dict[str, Skill]  # Map from skill names to Skill instances
    object_types: set[str]  # Set of object types in the domain

    @staticmethod
    def extract_type_names(types: ObjectTypeSet) -> set[str]:
        """Convert a set of NewType objects or strings into a set of type names."""
        result = set()
        for t in types:
            type_name = t.__name__ if hasattr(t, "__name__") else str(t)
            result.add(type_name.capitalize())
        return result

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> Domain:
        """Import a Domain instance from a YAML file.

        :param yaml_path: Filepath to a YAML file containing skills and type data
        :return: Constructed Domain instance
        """
        yaml_data = import_yaml_into_dict(yaml_path, required_keys={"skills", "types"})

        skills = [Skill.from_yaml(name, data) for name, data in yaml_data["skills"].items()]
        skills_dict = {skill.name: skill for skill in skills}

        return Domain(skills_dict, yaml_data["types"])

    @classmethod
    def from_protocol(cls, object_types: ObjectTypeSet, protocol: type[Any]) -> Domain:
        """Extract a SkillWrapper domain from the methods of a Python protocol.

        :param object_types: Set of all object types in the domain
        :param protocol: Python protocol specifying the signatures of the domain's skills
        """
        skills: dict[str, Skill] = {}

        for method_name in dir(protocol):
            if method_name.startswith("_"):
                continue
            method = getattr(protocol, method_name)
            if hasattr(method, "_is_skill"):
                skill = method_to_skill(method)
                skills[skill.name] = skill

        type_names = Domain.extract_type_names(object_types)

        # Extract all object types used by skills
        used_types = set()
        for skill in skills.values():
            for param in skill.parameters:
                used_types.add(param.object_type)

        # Verify that all extracted types are used by at least one skill
        unused_types = type_names - used_types
        if unused_types:
            raise ValueError(
                f"Unused object types: {sorted(unused_types)}. "
                "These types are declared in the domain but not used by any skill.",
            )

        # Verify that all skills only use types defined for the domain
        undefined_types = used_types - type_names
        if undefined_types:
            raise ValueError(
                f"Skills use undefined object types: {sorted(undefined_types)}. "
                "Add these types to the `object_types` set or fix typos in skill signatures.",
            )

        # Verify that the skill set and object types sets are not empty
        if not skills:
            raise ValueError(f"No skills found in the protocol {protocol.__name__}.")

        if not type_names:
            raise ValueError(f"No object types specified for the domain {protocol.__name__}.")

        return Domain(skills, type_names)

    def export_to_yaml(self, output_path: Path) -> None:
        """Export the domain as YAML data to the specified filepath."""
        skills_data = {name: skill.params_to_yaml() for name, skill in self.skills.items()}
        types_data = list(self.object_types)

        yaml_data = {"skills": skills_data, "types": types_data}

        with output_path.open("w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False)


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


class ConcreteObjects:
    """A collection of concrete objects and their types."""

    def __init__(self, objects: dict[str, set[str]]) -> None:
        """Initialize the collection of concrete objects."""
        self.objects = objects

    @property
    def object_names(self) -> KeysView[str]:
        """Retrieve all object names in this collection."""
        return self.objects.keys()

    @property
    def all_object_types(self) -> set[str]:
        """Compute the set of all object types used in this collection."""
        all_types = set()
        for types_set in self.objects.values():
            all_types.update(types_set)
        return all_types

    def get_object_types(self, object_name: str) -> set[str]:
        """Retrieve the type(s) of the named object."""
        return self.objects[object_name]

    def __contains__(self, object_name: str) -> bool:
        """Evaluate whether the named object is in this collection."""
        return object_name in self.objects


@dataclass(frozen=True)
class Environment:
    """An environment represents problem aspects that vary across different scenes."""

    initial_state: EgocentricImageState
    objects: ConcreteObjects

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> Environment:
        """Import an Environment instance from a YAML file."""
        yaml_data = import_yaml_into_dict(
            yaml_path,
            required_keys={"initial-state", "object-types"},
        )

        initial_state = EgocentricImageState.from_yaml(yaml_data["initial-state"])
        objects_dict = {obj: set(types) for obj, types in yaml_data["object-types"].items()}

        return Environment(initial_state, ConcreteObjects(objects_dict))


### Skill Instantiation and Execution Layer ###


@dataclass
class SkillInstance:
    """A skill instantiated using particular concrete objects."""

    skill: Skill  # Specifies the skill instance's parameter signature
    bindings: Bindings  # Maps each skill parameter name to the name of its bound object

    @classmethod
    def from_string(cls, string: str, domain: Domain, env: Environment) -> SkillInstance:
        """Construct a SkillInstance from the given string.

        :param string: String description of the skill instance
        :param domain: Domain defining the available skills
        :param env: Environment defining valid objects and their types
        :return: Constructed SkillInstance instance
        """
        match = re.match(r"^(\w+)\(([^)]*)\)$", string.strip())
        if not match:
            raise ValueError(f"Could not parse SkillInstance string: '{string}'")

        skill_name = match.group(1)
        args_string = match.group(2).strip()

        args = [arg.strip() for arg in args_string.split(",")] if args_string else []

        if skill_name not in domain.skills:
            raise ValueError(f"Invalid skill name parsed from string: '{skill_name}'")

        skill = domain.skills[skill_name]
        if len(skill.parameters) != len(args):
            error = f"Skill '{skill_name}' expects {len(skill.parameters)} args, not {len(args)}"
            raise ValueError(error)

        bindings: Bindings = {}
        for idx, param in enumerate(skill.parameters):
            bound_object = args[idx]

            if bound_object not in env.objects:
                raise ValueError(f"Object '{bound_object}' not found in the environment")

            obj_types = env.objects.get_object_types(bound_object)

            if param.object_type not in obj_types:
                raise ValueError(
                    f"Parameter {param.name} expects type {param.object_type} "
                    f"but argument object {bound_object} has type(s) {obj_types}.",
                )
            bindings[param.name] = bound_object

        return SkillInstance(skill, bindings)

    def execute(self, executor: SkillsProtocol) -> None:
        """Execute this skill instance."""
        self.skill.execute(executor, self.bindings)


### Skill Abstractions Layer ###


@dataclass(frozen=True)
class PredicateParameter:
    """A typed parameter of a predicate."""

    name: str
    object_type: str
    semantics: str  # TODO: Does the LLM provide this?


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class Predicate(Generic[StateT]):
    """A symbolic predicate with object-typed parameters."""

    name: str
    parameters: tuple[PredicateParameter, ...]
    semantics: str  # TODO: Does the LLM provide this?

    def ground_with(self, bindings: Bindings) -> PredicateInstance:
        """Ground the predicate using the given parameter bindings."""
        return PredicateInstance(self, bindings)

    def __str__(self) -> str:
        """Return a readable string representation of the predicate."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"


@dataclass(frozen=True)
class PredicateInstance(Generic[StateT]):
    """A predicate grounded using particular concrete objects."""

    predicate: Predicate
    bindings: Bindings

    def holds_in(self, state: StateT) -> bool:
        """Evaluate whether the grounded predicate holds in the given state."""
        raise NotImplementedError("Need to implement: PredicateInstance.holds_in(state)")


AbstractState = set[PredicateInstance]  # Abstract state = Set of grounded predicates that are true


@dataclass(frozen=True)
class OperatorParameter:
    """A typed parameter of an operator."""

    name: str
    object_type: str


@dataclass(frozen=True)
class Operator(Generic[StateT]):
    """A lifted abstract action defining an abstract transition model for a skill."""

    name: str
    parameters: tuple[OperatorParameter, ...]
    preconditions: set[Predicate]  # Abstract conditions required to execute the operator
    add_effects: list[Predicate]  # Abstract conditions made true by executing the operator
    delete_effects: list[Predicate]  # Abstract conditions made false by executing the operator

    def is_applicable(self, bindings: Bindings, state: StateT) -> bool:
        """Evaluate whether the operator is applicable under the given bindings in a state."""
        return all(pre.ground_with(bindings).holds_in(state) for pre in self.preconditions)

    def apply(self, bindings: Bindings, abstract_state: AbstractState) -> AbstractState:
        """Apply the operator's effects, under the given bindings, to update the abstract state."""
        if not self.preconditions.issubset(abstract_state):
            raise ValueError(
                f"Operator {self.name} is not applicable in the given abstract state.",
            )

        add_effects = {eff.ground_with(bindings) for eff in self.add_effects}
        delete_effects = {eff.ground_with(bindings) for eff in self.delete_effects}

        return abstract_state.difference(delete_effects).union(add_effects)


@dataclass(frozen=True)
class OperatorInstance:
    """An operator grounded with concrete objects."""

    operator: Operator
    bindings: Bindings

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the grounded operator to update the given abstract state."""
        return self.operator.apply(self.bindings, abstract_state)


def predicate_list_to_semantics_dict(predicates: list[Predicate]) -> dict[str, str]:
    """Convert a list of predicates to a dictionary mapping predicate names to their semantics."""
    return {p.name: p.semantics for p in predicates}


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

    def _compute_skill_pair_matrix(self, dataset: Dataset) -> np.ndarray:
        """Count the number of skill bigrams from previously executed skill sequences.

        :param dataset: Collection of observed skill execution traces
        :return: NumPy array of skill pair counts
        """
        skill_pair_counts = np.zeros((len(self.domain.skills), len(self.domain.skills)))

        for execution_trace in dataset:
            prev_skill_name = None
            for transition in execution_trace:
                curr_skill_name = transition.skill_instance.skill.name

                if prev_skill_name is not None:
                    idx1 = self.skill_to_idx[prev_skill_name]
                    idx2 = self.skill_to_idx[curr_skill_name]
                    skill_pair_counts[idx1, idx2] += 1

                prev_skill_name = curr_skill_name

        return skill_pair_counts

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

    def get_least_explored_skills(self, max_pairs: int = 5) -> list[str]:
        """Find the least-explored consecutive pair(s) of skills.

        :param max_pairs: Maximum number of skill pairs to return (defaults to 5)
        :return: List of strings specifying the least-explored skill pairs
        """
        min_value = np.min(self.skill_pairs_matrix)  # Find minimum value in the matrix
        min_indices = np.argwhere(self.skill_pairs_matrix == min_value)  # All min-value indices
        if len(min_indices) > max_pairs:
            selected_indices = min_indices[
                np.random.choice(len(min_indices), size=max_pairs, replace=False)  # TODO: RNG
            ]
        else:
            selected_indices = min_indices

        least_explored_pairs = []
        skills_list = list(self.skill_to_idx.keys())
        for idx1, idx2 in selected_indices:
            skill1 = skills_list[idx1]
            skill2 = skills_list[idx2]
            least_explored_pairs.append(f"({skill1}, {skill2})")

        return least_explored_pairs

    ### CHAINABILITY ###
    def get_skill_sequence_executability(
        self,
        skill_sequence: list[SkillInstance],
        initial_state: AbstractState | None,
    ) -> list[bool]:
        """Compute whether each skill in the proposed sequence is executable.

        :param skill_sequence: Sequence of proposed skill instances
        :param initial_state: Initial abstract state, or None if no predicates are yet proposed
        :return: List of Booleans indicating whether each skill was executable
        """
        return []  # TODO

    # def get_skill_sequence_executability(
    #     self,
    #     skill_sequence: list[Skill],
    #     init_state: PredicateState | None,
    # ) -> list[bool]:
    #     """self.operator_dictionary :: {lifted_skill: [(LiftedPDDLAction, {pid: int: type: str})]}"""

    #     def apply_skill(
    #         grounded_skill,
    #         pddl_state: PDDLState,
    #         pid2type,
    #         type_dict,
    #     ) -> bool | PDDLState:
    #         """Check if there exist an operator that makes the skill executable.

    #         Returns:
    #             bool :: if the skill is executable
    #             pddl_state :: next state if executable, the original state otehrwise

    #         """
    #         for lifted_operator, pid2type in self.operator_dictionary[grounded_skill.lifted()]:
    #             possible_groundings = generate_possible_groundings(
    #                 pid2type,
    #                 type_dict,
    #                 fixed_grounding=grounded_skill.params,
    #             )
    #             for grounding in possible_groundings:
    #                 param_name2param_object = {
    #                     str(param): param.get_grounded_parameter(
    #                         grounding[int(str(param).split("_p")[-1])],
    #                     )
    #                     for param in lifted_operator.parameters
    #                     if not str(param).startswith("_")
    #                 } | {"_p1": Parameter(None, "", None)}
    #                 grounded_operator: LiftedPDDLAction = lifted_operator.get_grounded_action(
    #                     param_name2param_object,
    #                     0,
    #                 )
    #                 if grounded_operator.check_applicability(pddl_state):
    #                     return True, grounded_operator.apply(pddl_state)
    #         return False, pddl_state

    #     if (
    #         init_state is None
    #     ):  # No predicate has been proposed yet. Chainability are always the same, i.e, always chainable
    #         return [True] * len(skill_sequence)

    #     executable_list = []
    #     bridge = RCR_bridge()
    #     pddl_state = bridge.predicatestate_to_pddlstate(init_state)
    #     for grounded_skill in skill_sequence:
    #         executable, pddl_state = apply_skill(grounded_skill, pddl_state)
    #         executable_list.append(executable)
    #     return executable_list
