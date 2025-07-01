"""Define a central class to organize running the SkillWrapper algorithm."""

from pathlib import Path

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.environment import Environment
from skillwrapper.refactored.predicates import Predicate
from skillwrapper.refactored.skill_sequence_proposer import SkillSequence, SkillSequenceProposer
from skillwrapper.refactored.skills import SkillsProtocol
from skillwrapper.refactored.transition_data import Dataset
from skillwrapper.refactored.utils import import_yaml_into_dict


class SkillWrapper:
    """Central manager for running the SkillWrapper algorithm."""

    def __init__(self, domain_yaml: Path, env_yaml: Path, skill_executor: SkillsProtocol) -> None:
        """Initialize the SkillWrapper algorithm using a domain imported from YAML.

        :param domain_yaml: Path to a YAML file specifying problem aspects shared across settings
        :param env_yaml: Path to a YAML file specifying details of the current environment
        :param skill_executor: Protocol defining an interface to execute the domain's skills
        """
        self.domain = Domain.from_yaml(domain_yaml)
        self.env = Environment.from_yaml(env_yaml)  # TODO: Provide state_type
        self.skill_executor = skill_executor

        # Import the skill sequence proposal prompt from the domain YAML file
        sspp_required_key = {"skill-sequence-proposal-prompt"}
        sspp_yaml_data = import_yaml_into_dict(domain_yaml, required_keys=sspp_required_key)
        ssp_prompt: str = sspp_yaml_data["skill-sequence-proposal-prompt"]

        self.skill_sequence_proposer = SkillSequenceProposer(self.domain, ssp_prompt)

        # Member variables tracking the state of the abstractions learned by SkillWrapper
        self.predicates: set[Predicate] = set()
        self.dataset: Dataset = []

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
        # TODO

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
