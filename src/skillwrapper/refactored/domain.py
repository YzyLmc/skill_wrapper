"""Define a class representing aspects of problems shared across environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from skillwrapper.refactored.skills import Skill, SkillsProtocol, method_to_skill
from skillwrapper.refactored.utils import import_yaml_into_dict

if TYPE_CHECKING:
    from pathlib import Path

ObjectTypeSet = set[Any]  # Allow object types to be expressed as strings or NewTypes


def extract_type_names(types: ObjectTypeSet) -> set[str]:
    """Convert a set of NewType objects or strings into a set of type names."""
    result = set()
    for t in types:
        type_name = t.__name__ if hasattr(t, "__name__") else str(t)
        result.add(type_name.capitalize())
    return result


@dataclass(frozen=True)
class Domain:
    """A domain represents aspects of planning problems that are shared across environments."""

    name: str
    skills: dict[str, Skill]  # Map from skill names to Skill instances
    object_types: set[str]  # Set of object types in the domain

    def __post_init__(self) -> None:
        """Verify that the Domain is valid with respect to the following properties.

        Valid domains must:
            - Define at least one skill and one object type
            - Use all defined object types in at least one skill
            - Define all object types used by any skill parameter
        """
        # Verify that the domain contains at least one skill and one object type
        if not self.skills:
            raise ValueError(f"Domain '{self.name}' doesn't define any skills.")

        if not self.object_types:
            raise ValueError(f"Domain '{self.name}' doesn't specify any object types.")

        # Compute which object types are used by some skill in the domain
        used_types: set[str] = set()
        for skill in self.skills.values():
            for param in skill.parameters:
                used_types.add(param.object_type)

        # Verify that all defined object types are used by at least one skill
        unused_types = sorted(self.object_types - used_types)
        if unused_types:
            raise ValueError(
                f"Domain '{self.name}' defines unused object types: {unused_types}.\n"
                "These types are declared in the domain but not used by any skill.",
            )

        # Verify that all skills only use types defined in the domain
        undefined_types = sorted(used_types - self.object_types)
        if undefined_types:
            raise ValueError(
                f"Skills in domain '{self.name}' use undefined object types: {undefined_types}.\n"
                "Add these types to the `object_types` set or fix typos in skill signatures.",
            )

    @classmethod
    def from_skills(cls, name: str, skills: set[Skill]) -> Domain:
        """Construct a Domain instance from the given set of skills.

        :param name: Name of the domain
        :param skills: Skills for the domain defining the available object types
        :return: Constructed Domain instance
        """
        skill_dict = {skill.name: skill for skill in skills}
        object_types = {p.object_type for skill in skills for p in skill.parameters}
        return Domain(name, skill_dict, object_types)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> Domain:
        """Import a Domain instance from a YAML file.

        :param yaml_path: Filepath to a YAML file containing skills and type data
        :return: Constructed Domain instance
        """
        yaml_data = import_yaml_into_dict(yaml_path, required_keys={"skills", "types"})

        skills = [Skill.from_yaml(name, data) for name, data in yaml_data["skills"].items()]
        skills_dict = {skill.name: skill for skill in skills}

        domain_name = yaml_path.stem

        return Domain(domain_name, skills_dict, set(yaml_data["types"]))

    @classmethod
    def from_protocol(cls, object_types: ObjectTypeSet, protocol: SkillsProtocol) -> Domain:
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
            else:
                raise ValueError(f"Skill protocol method {method_name} not tagged with @skill_fn")

        type_names = extract_type_names(object_types)

        domain_name = protocol.__name__

        return Domain(domain_name, skills, type_names)

    def export_to_yaml(self, output_path: Path) -> None:
        """Export the domain as YAML data to the specified filepath."""
        skills_data = {name: skill.params_to_yaml() for name, skill in self.skills.items()}
        types_data = sorted(self.object_types)

        yaml_data = {"skills": skills_data, "types": types_data}

        with output_path.open("w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False)
