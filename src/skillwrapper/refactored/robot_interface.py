"""Define a protocol to interface with robots from SkillWrapper."""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from skillwrapper.refactored.skills import SkillsProtocol

StateT_co = TypeVar("StateT_co", covariant=True)


@dataclass(frozen=True)
class SkillExecutionResult(Generic[StateT_co]):
    """The result of executing a skill in the environment."""

    success: bool  # Did the skill succeed?
    state: StateT_co  # Resulting state after the skill execution


class RobotInterface(Protocol[StateT_co], SkillsProtocol):
    """An interface to request the current state from a robot."""

    def get_current_state(self) -> StateT_co:
        """Request the current state from the robot."""
        ...
