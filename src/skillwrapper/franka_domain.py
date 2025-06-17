"""Define the domain (i.e., available skills and object types) for the Franka robot."""

from pathlib import Path
from typing import NewType, Protocol

from skillwrapper.skillwrapper_structs import Domain, Environment, skill_fn

### Define all object types in the domain ###

Robot = NewType("Robot", str)
Pickable = NewType("Pickable", str)
Pourable = NewType("Pourable", str)
Container = NewType("Container", str)
Sponge = NewType("Sponge", str)
Plate = NewType("Plate", str)
Bowl = NewType("Bowl", str)


### Define a protocol specifying the structure of all skills in the domain ###
class FrankaSkillsProtocol(Protocol):
    """Protocol defining the interface for the Franka robot's skills."""

    @skill_fn
    def pick(self, robot: Robot, picked: Pickable) -> None:
        """Pick up an object.

        :param robot: Robot executing the skill
        :param picked: Object to be picked up
        """

    @skill_fn
    def place_back(self, robot: Robot, placed: Pickable) -> None:
        """Place back an object where it was picked up from.

        :param robot: Robot executing the skill
        :param placed: Object placed back into its original position
        """

    @skill_fn
    def pour(self, robot: Robot, pour_from: Pourable, pour_into: Container) -> None:
        """Pour from one container into another.

        :param robot: Robot executing the skill
        :param pour_from: Container to pour liquid from
        :param pour_into: Container to pour liquid into
        """

    @skill_fn
    def stack(self, robot: Robot, bowl: Bowl, plate: Plate) -> None:
        """Stack a bowl onto a plate.

        :param robot: Robot executing the skill
        :param bowl: Smaller bowl to be stacked onto the plate
        :param plate: Plate that the bowl will be stacked on
        """

    @skill_fn
    def wipe(self, robot: Robot, sponge: Sponge, plate: Plate) -> None:
        """Wipe a dirty plate using a sponge.

        :param robot: Robot executing the skill
        :param sponge: Sponge used to wipe the plate
        :param plate: Dirty plate to be wiped
        """


def main() -> None:
    """Construct the skills in the Franka domain and export them to YAML."""
    object_types = {Robot, Pickable, Pourable, Container, Sponge, Plate, Bowl}

    domain = Domain.from_protocol(object_types, FrankaSkillsProtocol)
    print(f"Generated {len(domain.skills)} skills from protocol:")
    for skill in domain.skills.values():
        print(f"  {skill}")

    # Export the Franka domain to YAML
    output_path = Path("task_config/franka/exported_domain.yaml")
    domain.export_to_yaml(output_path)
    print(f"\nExported domain to {output_path}")

    # Import an example Franka environment from YAML
    example_env_yaml = Path("task_config/franka/envs/env1.yaml")
    env = Environment.from_yaml(example_env_yaml)
    print(f"Imported environment from YAML path: {example_env_yaml}")

    for obj_name in env.objects.object_names:
        obj_types = env.objects.get_object_types(obj_name)
        print(f"  Object '{obj_name}' has types:\t{obj_types}")


if __name__ == "__main__":
    main()
