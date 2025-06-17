"""Represent skills, environments, and domains and handle their import/export from YAML."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, KeysView
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


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


Bindings = dict[str, str]  # Map from skill parameter names to their bound concrete objects

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

    def execute(self, executor: SkillsProtocol) -> None:
        """Execute this skill instance."""
        self.skill.execute(executor, self.bindings)
