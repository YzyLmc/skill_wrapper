"""Represent skills, environments, and domains and handle their import/export from YAML."""

from __future__ import annotations

import re
from collections.abc import Callable, KeysView
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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


### Domain Model Layer - Defines the available skills and their parameters ###


@dataclass(frozen=True)
class SkillParameter:
    """An object-typed discrete parameter of a skill."""

    name: str
    object_type: str
    semantics: str  # English description of the parameter's meaning


ExecutorProtocol = Any  # TODO: Implement next
Bindings = dict[str, str]  # Map from skill parameter names to their bound concrete objects


@dataclass(frozen=True)
class Skill:
    """A skill parameterized by object-typed arguments."""

    name: str
    parameters: tuple[SkillParameter, ...]
    execute_fn: Callable[[ExecutorProtocol, Bindings], None] | None = field(default=None)

    @classmethod
    def from_yaml(cls, skill_name: str, yaml_data: dict[str, Any]) -> Skill:
        """Load a Skill instance from data imported from YAML."""
        assert "parameters" in yaml_data, f"Key 'parameters' missing from YAML data: {yaml_data}."

        skill_params = [
            SkillParameter(param_name, param_data["type"], param_data["semantics"])
            for param_name, param_data in yaml_data["parameters"].items()
        ]

        return Skill(skill_name, tuple(skill_params))  # Execution function registered separately

    def to_yaml(self) -> dict[str, Any]:
        """Convert the Skill object into a dictionary of YAML data."""
        return {
            self.name: {
                "parameters": {
                    param.name: {
                        "type": param.object_type,
                        "semantics": param.semantics,
                    }
                    for param in self.parameters
                },
            },
        }

    def execute(self, executor: ExecutorProtocol, bindings: dict[str, str]) -> None:
        """Execute this skill under the given object bindings.

        :param executor: Protocol defining an interface to robot execution
        :param bindings: Map from parameter names to bound object names
        """
        if self.execute_fn is None:
            raise NotImplementedError(f"No execution function for skill: {self.name}")
        self.execute_fn(executor, bindings)


@dataclass(frozen=True)
class Domain:
    """A domain represents aspects of planning problems that are shared across environments."""

    skills: dict[str, Skill]  # Map from skill names to Skill instances
    object_types: set[str]  # Set of object types in the domain

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
        self._objects = objects

    @property
    def object_names(self) -> KeysView[str]:
        """Retrieve all object names in this collection."""
        return self._objects.keys()

    @property
    def all_object_types(self) -> set[str]:
        """Compute the set of all object types used in this collection."""
        all_types = set()
        for types_set in self._objects.values():
            all_types.update(types_set)
        return all_types

    def get_object_types(self, object_name: str) -> set[str]:
        """Retrieve the type(s) of the named object."""
        return self._objects[object_name]

    def __contains__(self, object_name: str) -> bool:
        """Evaluate whether the named object is in this collection."""
        return object_name in self._objects


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

    def execute(self, executor: ExecutorProtocol) -> None:
        """Execute this skill instance."""
        self.skill.execute(executor, self.bindings)
