"""Define the domain (i.e., available skills and object types) for the Franka robot."""

from pathlib import Path
from typing import NewType, Protocol

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.environment import Environment
from skillwrapper.refactored.skills import skill_fn

### Define all object types in the domain ###

Pickable = NewType("Pickable", str)
Pourable = NewType("Pourable", str)
Fillable = NewType("Fillable", str)
Sponge = NewType("Sponge", str)
Stackable = NewType("Stackable", str)
Surface = NewType("Surface", str)
Location = NewType("Location", str)


### Define a protocol specifying the structure of all skills in the domain ###


class FrankaSkillsProtocol(Protocol):
    """Protocol defining the interface for the Franka robot's skills."""

    @skill_fn
    def pick(self, picked: Pickable) -> None:
        """Pick up an object.

        :param picked: Object to be picked up
        """

    @skill_fn
    def place(self, placed: Pickable, location: Location) -> None:
        """Place an object at a specified location.

        :param placed: Object placed at a location
        :param location: Location to place the object
        """

    @skill_fn
    def pour(self, pour_from: Pourable, pour_into: Fillable) -> None:
        """Pour from one container into another.

        :param pour_from: Container to pour liquid from
        :param pour_into: Container to fill
        """

    @skill_fn
    def stack(self, on_top: Stackable, on_bottom: Stackable) -> None:
        """Stack two objects of the same shape (e.g., two bowls or two plates).

        :param on_top: Object stacked on top of the other
        :param on_bottom: Object on the bottom of the stacked pair
        """

    @skill_fn
    def wipe(self, sponge: Sponge, surface: Surface) -> None:
        """Wipe a dirty surface using a sponge.

        :param sponge: Sponge used to wipe the surface
        :param surface: Dirty surface to be wiped
        """


class FrankaSkillsDummyExecutor(FrankaSkillsProtocol):
    """Dummy implementation of the protocol interface for the Franka robot's skills."""

    @skill_fn
    def pick(self, picked: Pickable) -> None:
        """Pick up an object.

        :param picked: Object to be picked up
        """
        print(f"Picking object '{picked}'...")

    @skill_fn
    def place(self, placed: Pickable, location: Location) -> None:
        """Place an object at a specified location.

        :param placed: Object placed at a location
        :param location: Location to place the object
        """
        print(f"Placing object '{placed}' at location '{location}'...")

    @skill_fn
    def pour(self, pour_from: Pourable, pour_into: Fillable) -> None:
        """Pour from one container into another.

        :param pour_from: Container to pour liquid from
        :param pour_into: Container to fill
        """
        print(f"Pouring from '{pour_from}' into '{pour_into}'...")

    @skill_fn
    def stack(self, on_top: Stackable, on_bottom: Stackable) -> None:
        """Stack two objects of the same shape (e.g., two bowls or two plates).

        :param on_top: Object stacked on top of the other
        :param on_bottom: Object on the bottom of the stacked pair
        """
        print(f"Stacking object '{on_top}' onto '{on_bottom}'...")

    @skill_fn
    def wipe(self, sponge: Sponge, surface: Surface) -> None:
        """Wipe a dirty surface using a sponge.

        :param sponge: Sponge used to wipe the surface
        :param surface: Dirty surface to be wiped
        """
        print(f"Wiping surface '{surface}' using sponge '{sponge}'...")


def main() -> None:
    """Construct the skills in the Franka domain and export them to YAML."""
    object_types = {Pickable, Pourable, Fillable, Sponge, Stackable, Surface, Location}

    domain = Domain.from_protocol(object_types, FrankaSkillsProtocol)
    print(f"Generated {len(domain.skills)} skills from protocol:")
    for skill in domain.skills.values():
        print(f"  {skill}")

    # Export the Franka domain to YAML
    output_path = Path("domains/franka/exported_domain.yaml")
    domain.export_to_yaml(output_path)
    print(f"\nExported domain to {output_path}")

    # Import an example Franka environment from YAML
    example_env_yaml = Path("domains/franka/envs/env1.yaml")
    env = Environment.from_yaml(example_env_yaml)
    print(f"Imported environment from YAML path: {example_env_yaml}")

    for obj_name in env.objects.object_names:
        obj_types = env.objects.get_types_of_object(obj_name)
        print(f"  Object '{obj_name}' has types:\t{obj_types}")


if __name__ == "__main__":
    main()
