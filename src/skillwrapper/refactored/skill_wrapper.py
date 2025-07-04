"""Define a central class to organize running the SkillWrapper algorithm."""

from pathlib import Path

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.environment import Environment
from skillwrapper.refactored.predicates import Predicate
from skillwrapper.refactored.skill_sequence_proposer import SkillSequence, SkillSequenceProposer
from skillwrapper.refactored.skills import SkillsProtocol
from skillwrapper.refactored.transition_data import Dataset
from skillwrapper.refactored.utils import (
    camel_to_snake,
    import_yaml_into_dict,
    load_class_from_module,
    snake_to_camel,
)


class SkillWrapper:
    """Central manager for running the SkillWrapper algorithm."""

    def __init__(self, domain_yaml: Path, env_yaml: Path) -> None:
        """Initialize the SkillWrapper algorithm using a domain imported from YAML.

        :param domain_yaml: Path to a YAML file specifying problem aspects shared across settings
        :param env_yaml: Path to a YAML file specifying details of the current environment
        :param skill_executor: Protocol defining an interface to execute the domain's skills
        """
        self.domain = Domain.from_yaml(domain_yaml)
        self.env = Environment.from_yaml(env_yaml)
        self.skill_executor = self.load_executor(domain_yaml)

        self.validate_configuration()

        # Import the skill sequence proposal prompt from the domain YAML file
        sspp_required_key = {"skill-sequence-proposal-prompt"}
        sspp_yaml_data = import_yaml_into_dict(domain_yaml, required_keys=sspp_required_key)
        ssp_prompt: str = sspp_yaml_data["skill-sequence-proposal-prompt"]

        self.skill_sequence_proposer = SkillSequenceProposer(self.domain, ssp_prompt)

        # Member variables tracking the state of the abstractions learned by SkillWrapper
        self.predicates: set[Predicate] = set()
        self.dataset: Dataset = []

    def load_executor(self, domain_yaml: Path) -> SkillsProtocol:
        """Load the skill executor protocol based on the domain's YAML file.

        :param domain_yaml: Path to a YAML file specifying where to import the executor from
        :return: Constructed SkillsProtocol instance
        """
        yaml_data = import_yaml_into_dict(domain_yaml, required_keys={"skill-executor"})
        exec_data = yaml_data["skill-executor"]
        assert "module" in exec_data, "Expected skill executor YAML data to specify a module."
        assert "class" in exec_data, "Expected skill executor YAML data to specify a class."

        executor_class = load_class_from_module(exec_data["class"], exec_data["module"])
        return executor_class()

    def validate_configuration(self) -> None:
        """Verify that the loaded domain, environment, and skill executor are consistent."""
        # Check that all types used in the environment exist in the domain
        for type_name in self.env.objects.all_object_types:
            if type_name not in self.domain.object_types:
                error = f"Environment uses type '{type_name}' but domain doesn't define this type."
                raise RuntimeError(error)

        # Check that all skills defined in the domain have a corresponding executor method
        for skill_name in self.domain.skills:
            skill_method_name = camel_to_snake(skill_name)
            if not hasattr(self.skill_executor, skill_method_name):
                raise RuntimeError(f"Skill executor doesn't have a method for skill {skill_name}.")

        # Check that all executor methods have a corresponding skill in the domain
        for method_name in dir(self.skill_executor):
            if method_name.startswith("_"):
                continue
            skill_name = snake_to_camel(method_name)
            if skill_name not in self.domain.skills:
                error = f"Skill executor method '{method_name}' doesn't correspond to a skill."
                raise RuntimeError(error)

    def propose_skill_sequence(self) -> SkillSequence:
        """Propose an exploratory skill sequence by querying a VLM.

        :return: Proposed sequence of skill instances to be executed in the current environment
        """
        return self.skill_sequence_proposer.propose_skill_sequence(self.dataset, self.env)

    def execute_skills(self, skill_sequence: SkillSequence) -> None:
        """Execute the given skill sequence in the environment.

        :param skill_sequence: Proposed exploratory sequence of skills to execute
        """
        for skill_instance in skill_sequence:
            skill_instance.execute(self.skill_executor)  # TODO: What data should be returned?

    def change_environment(self, env_yaml: Path) -> None:
        """Change the current environment by loading from the given YAML file.

        :param env_yaml: Path to a YAML file specifying environment details
        """
        # TODO

    def propose_and_execute(self) -> None:
        """Propose and execute an exploratory skill sequence to collect transition data."""
        skill_sequence = self.propose_skill_sequence()
        self.execute_skills(skill_sequence)

    def invent_predicates(self) -> None:
        """Invent new predicates based on the current collected transition data."""
        # TODO

    def learn_operators(self) -> None:
        """Learn operators based on the current dataset and invented predicates."""
        # TODO

    def run_complete_loop(self) -> None:
        """Run a complete loop of the SkillWrapper algorithm."""
        # TODO

    def save_progress(self) -> None:
        """Save the current progress to file."""
        # TODO

    def load_progress(self) -> None:
        """Load previously collected data and learned abstractions from file."""
        # TODO

    def print_status(self) -> None:
        """Print the current status of the SkillWrapper algorithm."""
        # TODO
