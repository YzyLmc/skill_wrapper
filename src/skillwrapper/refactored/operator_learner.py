"""Define a class to learn symbolic operators from observed skill transitions."""

from skillwrapper.refactored.environment import ConcreteObjects
from skillwrapper.refactored.transition_data import Dataset
from skillwrapper.refactored.utils import StateT

class OperatorLearner:
    """Learns operators given observed skill transitions."""

    def __init__(self, object_types: ConcreteObjects) -> None:
        """Initialize the operator learning system with a mapping of object types.

        :param object_types: Maps concrete object names to their object types
        """
        self.object_types = object_types

    def learn_operator(self, transitions: SuccessfulSkillTransitions[StateT])


self.successful_transitions = [
            transition for trace in dataset for transition in trace if transition.success
        ]