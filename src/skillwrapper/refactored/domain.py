"""Define a class representing aspects of problems shared across environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from skillwrapper.refactored.skills import Skill, SkillsProtocol, method_to_skill
from skillwrapper.refactored.utils import import_yaml_into_dict

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

        type_names = extract_type_names(object_types)

        # Extract all object types used by skills
        used_types: set[str] = set()
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

        # Verify that all skills only use types defined in the domain
        undefined_types = used_types - type_names
        if undefined_types:
            raise ValueError(
                f"Skills use undefined object types: {sorted(undefined_types)}. "
                "Add these types to the `object_types` set or fix typos in skill signatures.",
            )

        # Verify that the domain contains at least one skill and one object type
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
