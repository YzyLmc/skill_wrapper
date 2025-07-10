"""Define a class to represent the abstract model of skills learned by SkillWrapper."""

from dataclasses import dataclass

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.operators import Operator
from skillwrapper.refactored.predicates import Predicate
from skillwrapper.refactored.skills import Skill


@dataclass(frozen=True)
class AbstractModel:
    """An abstract model of the skills in a robot domain."""

    domain: Domain  # Domain defining the skills being modeled
    predicates: set[Predicate]
    operators: dict[Skill, set[Operator]]  # Map from skills to their corresponding operators
