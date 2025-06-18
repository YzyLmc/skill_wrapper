"""Define a class to represent aspects that vary across problems within a domain."""

from __future__ import annotations

from collections.abc import KeysView
from dataclasses import dataclass
from pathlib import Path
from typing import Generic

from skillwrapper.refactored.utils import StateT, import_yaml_into_dict


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
class Environment(Generic[StateT]):
    """An environment represents problem aspects that vary across different scenes."""

    initial_state: StateT  # Any YAML-importable state type
    objects: ConcreteObjects

    @classmethod
    def from_yaml(cls, yaml_path: Path, state_type: type[StateT]) -> Environment:
        """Import an Environment instance from a YAML file."""
        yaml_data = import_yaml_into_dict(
            yaml_path,
            required_keys={"initial-state", "object-types"},
        )

        initial_state = state_type.from_yaml(yaml_data["initial-state"])
        objects_dict = {obj: set(types) for obj, types in yaml_data["object-types"].items()}

        return Environment(initial_state, ConcreteObjects(objects_dict))
