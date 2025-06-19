"""Define a class to learn symbolic operators from observed skill transitions."""

from dataclasses import dataclass

from skillwrapper.refactored.environment import ConcreteObjects
from skillwrapper.refactored.operators import Operator
from skillwrapper.refactored.parameters import DiscreteParameter
from skillwrapper.refactored.predicates import PredicateInstance
from skillwrapper.refactored.skills import Skill
from skillwrapper.refactored.transition_data import AbstractDataset, AbstractTransition
from skillwrapper.refactored.utils import StateT


@dataclass
class TransitionMapping:
    """A mapping between objects and operator parameters for a single abstract transition."""

    transition: AbstractTransition
    object_name_to_param_idx: dict[str, int]  # Map object names to parameter indices
    param_idx_to_type: dict[int, str]  # Map parameter indices to their object types
    next_unused_param_idx: int = 0  # Index of the next unused operator parameter (TODO: Context?)


@dataclass(frozen=True)
class ExtractedEffects:
    """Grounded effects extracted from an abstract transition."""

    add_effects: set[PredicateInstance]  # Grounded predicates made true after the skill
    delete_effects: set[PredicateInstance]  # Grounded predicates made false after the skill


@dataclass(frozen=True)
class ParameterPosition:
    """A parameter's position in a predicate structure."""

    param_idx: int  # Index of the predicate parameter
    object_type: str  # Type of object associated with the parameter


class PredicateStructure:
    """Represents the parameter structure of a predicate instance."""

    def __init__(self, p_instance: PredicateInstance, mapping: TransitionMapping) -> None:
        """Initialize the predicate structure for the given predicate instance.

        :param p_instance: Predicate instance whose structure is analyzed
        :param mapping: Example mapping of skill parameters for an abstract transition
        """
        # TODO: Could we just use a tuple of types?

        param_structure: list[ParameterPosition] = []

        for param in p_instance.predicate.parameters:
            bound_object = p_instance.bindings[param.name]

            # Get or assign a parameter index for this trans
            if bound_object not in mapping.object_name_to_param_idx:  # Assign new index!
                param_idx = mapping.next_unused_param_idx
                mapping.object_name_to_param_idx[object] = param_idx
                mapping.param_idx_to_type[param_idx] = param.object_type
                mapping.next_unused_param_idx += 1
            else:
                param_idx = mapping.object_name_to_param_idx[bound_object]

            param_structure.append(ParameterPosition(param_idx, param.object_type))

        self.structure = tuple(param_structure)


@dataclass(frozen=True)
class StructuralPredicateKey:
    """A key for identifying predicates with analogous structure."""

    predicate_name: str
    param_structure: PredicateStructure


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

        # Build a mapping between objects and operator parameters for all transitions
        transition_mappings: list[TransitionMapping] = []

        # Establish the minimal operator parameters based on the skill's parameters
        skill_param_positions = {
            idx: DiscreteParameter(f"?p{idx}", param.object_type, None)
            for idx, param in enumerate(skill.parameters)
        }

        # Build object mappings for each transition
        for transition in successful_transitions:
            mapping = TransitionMapping(transition, {}, {}, len(skill.parameters))

            # Map skill parameters first (ensures consistent parameter positions)
            for idx, param in enumerate(skill.parameters):
                bound_object = transition.skill_instance.bindings[param.name]
                mapping.object_name_to_param_idx[bound_object] = idx
                mapping.param_idx_to_type[idx] = param.object_type

            transition_mappings.append(mapping)

        # Extract all changed predicate instances from the abstract transitions
        all_changed_predicates = OperatorLearner.extract_changed_predicates(successful_transitions)

        # Find effects common to all transitions (as example grounded predicates)

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

    def extract_common_effects(self, transition_data: list[TransitionMapping]) -> ExtractedEffects:
        """Extract effects that are common across all given transitions.

        Look for effects with the same structure, even if their specific objects differ.

        :param transition_data: Collection of parameter mappings for abstract transitions
        :return: Example ground effects common across the given transition data
        """
        if not transition_data:
            return ExtractedEffects(set(), set())

        # Process first transition to get an example effect structure
        first = transition_data[0]
        assert first.transition.abstract_after is not None, "Expected an 'after' abstract state."
        first_add_effects = first.transition.abstract_after - first.transition.abstract_before
        first_delete_effects = first.transition.abstract_before - first.transition.abstract_after

        # Build a structural representation of the effects
        structural_add: dict[StructuralPredicateKey, PredicateInstance] = {}
        structural_delete: dict[StructuralPredicateKey, PredicateInstance] = {}

        for predicate_instance in first_add_effects:
            param_structure = PredicateStructure(predicate_instance, first)
            key = StructuralPredicateKey(predicate_instance.predicate.name, param_structure)
            structural_add[key] = predicate_instance

        for predicate_instance in first_delete_effects:
            param_structure = PredicateStructure(predicate_instance, first)
            key = StructuralPredicateKey(predicate_instance.predicate.name, param_structure)
            structural_delete[key] = predicate_instance

            # TODO: Continue! I should just read `get_action_from_cluster`

        return None  # TODO
