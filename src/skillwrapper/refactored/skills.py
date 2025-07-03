"""Define classes to represent object-parameterized skills and their instantiations."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from skillwrapper.refactored.parameters import Bindings, DiscreteParameter
from skillwrapper.refactored.utils import camel_to_snake, parse_docstring_params, snake_to_camel

if TYPE_CHECKING:
    from collections.abc import Callable

    from skillwrapper.refactored.domain import Domain
    from skillwrapper.refactored.environment import Environment


SkillsProtocol = Any  # Stands in for skill protocols for different domains


def skill_fn(func: Callable) -> Callable:
    """Mark a function as implementing a skill."""
    func._is_skill = True
    return func


@dataclass(frozen=True)
class Skill:
    """A skill parameterized by object-typed arguments."""

    name: str
    parameters: tuple[DiscreteParameter, ...]

    @classmethod
    def from_yaml(cls, skill_name: str, yaml_data: dict[str, Any]) -> Skill:
        """Load a Skill instance from data imported from YAML."""
        assert "parameters" in yaml_data, f"Key 'parameters' missing from YAML data: {yaml_data}."

        return Skill(skill_name, DiscreteParameter.tuple_from_yaml(yaml_data["parameters"]))

    def __str__(self) -> str:
        """Return a readable string representation of the skill."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    def to_yaml(self) -> dict[str, Any]:
        """Convert the Skill object into a dictionary of YAML data."""
        return {self.name: self.params_to_yaml()}

    def params_to_yaml(self) -> dict[str, Any]:
        """Convert the Skill parameters into a dictionary of YAML data under a `parameters` key."""
        params_dict: dict[str, Any] = {"parameters": {}}
        for param in self.parameters:
            params_dict["parameters"].update(param.to_yaml_dict())

        return params_dict

    def execute(self, executor: SkillsProtocol, bindings: Bindings) -> None:
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
        param_type = type_hints.get(param_name)
        if param_type is None:
            error = f"Skill '{skill_name}' didn't define a type for parameter '{param_name}'"
            raise ValueError(error)

        object_type = param_type.__name__.capitalize()

        # Get parameter semantics from the method docstring (required for docstring-defined params)
        semantics = param_docs.get(param_name)
        if semantics is None:
            error = f"Skill '{skill_name}' didn't define semantics for parameter '{param_name}'"
            raise ValueError(error)

        parameters.append(DiscreteParameter(param_name, object_type, semantics))

    return Skill(skill_name, tuple(parameters))


@dataclass
class SkillInstance:
    """A skill instantiated using particular concrete objects."""

    skill: Skill  # Specifies the skill instance's parameter signature
    bindings: Bindings  # Maps each skill parameter name to the name of its bound object

    def __str__(self) -> str:
        """Return a readable string representation of the skill instance."""
        args_string = ", ".join(self.bindings[p.name] for p in self.skill.parameters)
        return f"{self.skill.name}({args_string})"

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
            error = f"Skill '{skill_name}' expects {len(skill.parameters)} args, not {len(args)}."
            raise ValueError(error)

        bindings: Bindings = {}
        for bound_object, param in zip(args, skill.parameters, strict=True):
            if bound_object not in env.objects:
                raise ValueError(f"Object '{bound_object}' not found in the environment.")

            obj_types = env.objects.get_types_of_object(bound_object)

            if param.object_type not in obj_types:
                error = (
                    f"Cannot parse skill instance from '{string}' because skill parameter "
                    f"{param.name} expects type {param.object_type} but the provided "
                    f"argument object {bound_object} only has type(s) {obj_types}."
                )
                raise ValueError(error)
            bindings[param.name] = bound_object

        return SkillInstance(skill, bindings)

    def execute(self, executor: SkillsProtocol) -> None:
        """Execute this skill instance."""
        self.skill.execute(executor, self.bindings)
