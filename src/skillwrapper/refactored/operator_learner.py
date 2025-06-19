"""Define a class to learn symbolic operators from observed skill transitions."""

from skillwrapper.refactored.environment import ConcreteObjects
from skillwrapper.refactored.operators import Operator
from skillwrapper.refactored.predicates import PredicateInstance
from skillwrapper.refactored.skills import Skill
from skillwrapper.refactored.transition_data import AbstractDataset, AbstractTransition
from skillwrapper.refactored.utils import StateT


class OperatorLearner:
    """A system for learning operators given observed skill transitions."""

    def __init__(self, object_types: ConcreteObjects) -> None:
        """Initialize the operator learning system with a mapping of object types.

        :param object_types: Maps concrete object names to their object types
        """
        self.object_types = object_types



    def learn_operator(self, dataset: AbstractDataset, skill: Skill) -> Operator:
        """Learn an operator from a skill's successful abstract transitions in a dataset.

        :param dataset: Collection of abstracted skill execution traces
        :param skill: Relevant skill to learn an operator for
        :return: Operator learned from the abstract dataset
        """
        successful_transitions = [
            t for trace in dataset for t in trace if t.skill_name == skill.name and t.success
        ]  # Filter to only successful abstract transitions for the relevant skill
        if not successful_transitions:
            error = f"No successful transitions found to learn an operator for skill {skill.name}"
            raise ValueError(error)
        
        # Initialize variables to track which parameters we've created for this operator
        next_unused_param_idx: int = 0 # Index of the next operator parameter to be created
        
        # Map concrete object names (from skill instances) to their operator parameter indices
        object_to_operator_param_idx: dict[str, int] = {}
        param_idx_to_type: dict[int, str] = {}  # Map a parameter index to its object type

        # Assign parameter IDs to skill arguments first
        for idx, skill_param in enumerate(skill.parameters):


        return None  # TODO

    @staticmethod
    def extract_changed_predicates(data: list[AbstractTransition]) -> set[PredicateInstance]:
        """Extract all predicate instances that changed in any of the given abstract transitions.

        :param data: Collection of abstract transitions corresponding to skill executions
        :return: Set of grounded predicates that changed in any given transition
        """
        changed = set()

        for t in data:
            assert t.abstract_after is not None, "Expected abstract 'after' state."

            # Include predicates that were added or deleted from the abstract state
            changed.update(t.abstract_before.symmetric_difference(t.abstract_after))

        return changed
