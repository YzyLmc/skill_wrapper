"""Define a robot interface for the Franka robot."""

import click

from skillwrapper.refactored.egocentric_image_state import EgocentricImageState
from skillwrapper.refactored.franka_skills_protocol import FrankaSkillsProtocol
from skillwrapper.refactored.robot_interface import RobotInterface, SkillExecutionResult
from skillwrapper.refactored.skills import SkillInstance


class FrankaDummyInterface(RobotInterface[EgocentricImageState], FrankaSkillsProtocol):
    """An interface for executing skills on and obtaining the state from the Franka robot."""

    def __init__(self) -> None:
        """Initialize the interface for Franka."""

    def get_current_state(self) -> EgocentricImageState:
        """Return a dummy egocentric image-based state from Franka."""
        return EgocentricImageState(initial_images={})

    def ask_for_skill_success(self, skill_instance: SkillInstance) -> bool:
        """Ask a human whether or not the given skill instance would succeed in the current state.

        :param skill_instance: To-be-executed skill instantiated using particular objects
        :return: True if the skill would succeed, else False
        """
        return click.confirm(
            text=f"Would the skill {skill_instance} succeed in the current state?",
            default=None,
        )


# class FrankaSkillsDummyExecutor(FrankaSkillsProtocol):
#     """Dummy implementation of the protocol interface for the Franka robot's skills."""

#     @skill_fn
#     def pick(self, picked: Pickable) -> SkillExecutionResult:
#         """Pick up an object.

#         :param picked: Object to be picked up
#         """
#         print(f"Picking object '{picked}'...")

#     @skill_fn
#     def place(self, placed: Pickable, location: Location) -> SkillExecutionResult:
#         """Place an object at a specified location.

#         :param placed: Object placed at a location
#         :param location: Location to place the object
#         """
#         print(f"Placing object '{placed}' at location '{location}'...")

#     @skill_fn
#     def pour(self, pour_from: Pourable, pour_into: Fillable) -> SkillExecutionResult:
#         """Pour from one container into another.

#         :param pour_from: Container to pour liquid from
#         :param pour_into: Container to fill
#         """
#         print(f"Pouring from '{pour_from}' into '{pour_into}'...")

#     @skill_fn
#     def stack(self, on_top: Stackable, on_bottom: Stackable) -> SkillExecutionResult:
#         """Stack two objects of the same shape (e.g., two bowls or two plates).

#         :param on_top: Object stacked on top of the other
#         :param on_bottom: Object on the bottom of the stacked pair
#         """
#         print(f"Stacking object '{on_top}' onto '{on_bottom}'...")

#     @skill_fn
#     def wipe(self, sponge: Sponge, surface: Surface) -> SkillExecutionResult:
#         """Wipe a dirty surface using a sponge.

#         :param sponge: Sponge used to wipe the surface
#         :param surface: Dirty surface to be wiped
#         """
#         print(f"Wiping surface '{surface}' using sponge '{sponge}'...")
