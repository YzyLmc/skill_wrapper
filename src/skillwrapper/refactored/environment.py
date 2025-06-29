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
        self.objects = objects  # Maps object names to their set of types
        self.types_to_object_names = ConcreteObjects.compute_types_to_object_names(self.objects)

    def __contains__(self, object_name: str) -> bool:
        """Evaluate whether the named object is in this collection."""
        return object_name in self.objects

    @property
    def object_names(self) -> KeysView[str]:
        """Retrieve all object names in this collection."""
        return self.objects.keys()

    @property
    def all_object_types(self) -> KeysView[str]:
        """Retrieve all object types used in this collection."""
        return self.types_to_object_names.keys()

    def get_types_of_object(self, object_name: str) -> set[str]:
        """Retrieve the type(s) of the named object."""
        return self.objects[object_name]

    def get_all_objects_of_type(self, obj_type: str) -> set[str]:
        """Retrieve the names of all objects of the given object type.

        :raises KeyError: If the given object type is unknown
        """
        if obj_type not in self.types_to_object_names:
            raise KeyError(f"Unknown object type: '{obj_type}'")

        return self.types_to_object_names[obj_type]

    @staticmethod
    def compute_types_to_object_names(objects: dict[str, set[str]]) -> dict[str, set[str]]:
        """Construct a map from each object type to the names of all objects of that type.

        :param objects: Map from object names to their set of types
        :return: Map from each object type to the names of all concrete objects of that type
        """
        types_to_object_names: dict[str, set[str]] = {}

        for object_name, object_types in objects.items():
            for obj_type in object_types:
                if obj_type not in types_to_object_names:
                    types_to_object_names[obj_type] = set()
                types_to_object_names[obj_type].add(object_name)

        return types_to_object_names


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
