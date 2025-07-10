"""Define functions to implement an operator learning algorithm."""

from dataclasses import dataclass

from skillwrapper.refactored.skills import SkillInstance
from skillwrapper.refactored.transition_data import SuccessfulAbstractTransition


@dataclass(frozen=True)
class Partition:
    """A collection of successful skill transitions sharing a common abstract termination set."""

    skill_instance: SkillInstance  # Skill instance used in every transition in the partition
    transitions: list[SuccessfulAbstractTransition]
