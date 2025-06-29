"""Define classes to represent observed state transitions used for abstraction learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

from skillwrapper.refactored.abstract_states import AbstractState, AbstractStateSpace
from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.environment import Environment
from skillwrapper.refactored.skills import Skill, SkillInstance
from skillwrapper.refactored.utils import StateT


@dataclass(frozen=True)
class SkillTransition(Generic[StateT]):
    """An observed transition resulting from executing a skill instance in an environment."""

    state_before: StateT  # State from which the skill execution was attempted
    skill_instance: SkillInstance  # Concrete skill that was (possibly) executed
    success: bool  # Was the skill successfully executed?
    state_after: StateT | None  # State after the skill executed, if the skill succeeded

    def __post_init__(self) -> None:
        """Verify that the constructed transition is valid."""
        if self.success and self.state_after is None:
            raise ValueError("A successful skill transition must include an 'after' state.")

    @classmethod
    def from_yaml(
        cls,
        state_type: type[StateT],
        yaml_data: dict[str, Any],
        domain: Domain,
        env: Environment,
    ) -> SkillTransition:
        """Load a SkillTransition instance from data loaded from YAML."""
        for key in ["state_before", "skill_instance", "success"]:
            if key not in yaml_data:
                raise KeyError(f"SkillTransition.from_yaml() requires the YAML key: '{key}'")

        state_before = state_type.from_yaml(yaml_data["state_before"])
        skill_instance = SkillInstance.from_string(yaml_data["skill_instance"], domain, env)
        success = bool(yaml_data["success"])

        if success and "state_after" not in yaml_data:
            error = f"YAML key 'state_after' is required if 'success' was true: {yaml_data}"
            raise KeyError(error)
        state_after = state_type.from_yaml(yaml_data["state_after"]) if success else None

        return SkillTransition(state_before, skill_instance, success, state_after)

    @property
    def skill_name(self) -> str:
        """Retrieve the name of the skill used in the transition."""
        return self.skill_instance.skill.name

    def make_abstract(self, abstract_space: AbstractStateSpace[StateT]) -> AbstractTransition:
        """Convert the transition between low-level states into an abstract transition.

        :param abstract_space: Abstract state space defining the space of possible facts
        :return: Constructed abstract transition
        """
        abstract_before = abstract_space.abstract(self.state_before)
        abstract_after = (
            None if self.state_after is None else abstract_space.abstract(self.state_after)
        )

        return AbstractTransition(
            abstract_before,
            self.skill_instance,
            self.success,
            abstract_after,
        )


SkillExecutionTrace = list[SkillTransition[StateT]]  # A sequence of attempted skill executions
Dataset = list[SkillExecutionTrace[StateT]]  # A collection of skill execution traces


@dataclass(frozen=True)
class AbstractTransition:
    """A transition representing the change in abstract state due to a skill execution."""

    abstract_before: AbstractState
    skill_instance: SkillInstance  # Concrete skill that was (possibly) executed
    success: bool  # Was the skill successfully executed?
    abstract_after: AbstractState | None

    def __post_init__(self) -> None:
        """Verify that the constructed abstract transition is valid."""
        if self.success and self.abstract_after is None:
            raise ValueError("A successful abstract transition must include an 'after' state.")

    @property
    def skill_name(self) -> str:
        """Retrieve the name of the skill used in the abstract transition."""
        return self.skill_instance.skill.name


AbstractExecutionTrace = list[AbstractTransition]  # A sequence of abstracted skill transitions


@dataclass(frozen=True)
class AbstractDataset:
    """An abstract dataset is a collection of abstracted skill execution traces."""

    abstract_traces: list[AbstractExecutionTrace]

    def get_transitions_for_skill(self, skill: Skill) -> set[AbstractTransition]:
        """Extract only the abstract transitions involving the given skill.

        :param skill: The skill involved in the extracted abstract transitions
        :return: Set of extracted abstract transitions involving the skill
        """
        return {
            transition
            for trace in self.abstract_traces
            for transition in trace
            if transition.skill_instance.skill == skill
        }
