"""Define a central class to organize running the SkillWrapper algorithm."""

from pathlib import Path

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.environment import Environment


class SkillWrapper:
    """Central manager for running the SkillWrapper algorithm."""

    def __init__(self, domain_yaml: Path, env_yaml: Path) -> None:
        """Initialize the SkillWrapper algorithm by importing a domain and environment from YAML.

        :param domain_yaml: Path to a YAML file specifying problem aspects shared across settings
        :param env_yaml: Path to a YAML file specifying details of the current environment
        """
        self.domain = Domain.from_yaml(domain_yaml)
        self.env = Environment.from_yaml(env_yaml)

        self.validate_configuration()

    def validate_configuration(self) -> None:
        """Verify that the loaded domain and environment are consistent.

        :raises ValueError: If the environment uses an object type not defined by the domain
        """
        # Check that all types used in the environment exist in the domain
        for type_name in self.env.objects.all_object_types:
            if type_name not in self.domain.object_types:
                error = f"Environment uses type '{type_name}' but the domain doesn't define it."
                raise ValueError(error)

    def change_environment(self, env_yaml: Path) -> None:
        """Change the current environment to the environment loaded from the given YAML file."""
        self.env = Environment.from_yaml(env_yaml)

    def propose_and_execute_skills(self) -> None:
        """TODO"""

    def invent_predicates(self) -> None:
        """TODO"""

    def learn_operators(self) -> None:
        """TODO"""

    def run_complete_loop(self) -> None:
        """TODO"""

    def save_progress(self) -> None:
        """TODO"""

    def load_progress(self) -> None:
        """TODO"""

    def print_status(self) -> None:
        """TODO"""
