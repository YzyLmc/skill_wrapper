"""Define classes to represent observed state transitions used for abstraction learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.environment import Environment
from skillwrapper.refactored.skills import SkillInstance
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


SkillExecutionTrace = list[SkillTransition[StateT]]  # A sequence of attempted skill executions
Dataset = list[SkillExecutionTrace[StateT]]  # A collection of skill execution traces
