"""Define a class to manage proposing skill sequences using VLMs."""

from __future__ import annotations

import numpy as np

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.environment import Environment
from skillwrapper.refactored.predicates import Predicate
from skillwrapper.refactored.transition_data import Dataset


class SkillSequenceProposer:
    """Proposes exploratory skill sequences using a vision-language model."""

    def __init__(
        self,
        domain: Domain,
        env: Environment,
        predicates: list[Predicate],
        dataset: Dataset | None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize the skill sequence proposer.

        :param domain: SkillWrapper domain specifying skills and object types
        :param env: SkillWrapper environment specifying objects and the initial state
        :param predicates: List of predicates already learned (may be empty)
        :param dataset: Dataset of skill execution traces (or None if first iteration)
        :param rng: Random number generator seed, or None (defaults to None)
        """
        self.domain = domain
        self.env = env

        self.skill_to_idx: dict[str, int] = {
            skill.name: idx for idx, skill in enumerate(self.domain.skills.values())
        }

        self.predicate_semantics: dict[str, str] = {}  # Map predicate names to their semantics
        for p in predicates:
            assert p.semantics is not None, f"Predicate {p.name} doesn't have semantics!"
            self.predicate_semantics[p.name] = p.semantics

        # Initialize frequency counts for all skill instance pairs
        if dataset is None:
            dataset = []  # Empty dataset => All skill counts will remain zero
        self.skill_pairs_matrix = self.compute_skill_pair_matrix(dataset, self.skill_to_idx)
        self.total_dataset_size = sum(len(trace) for trace in dataset)

        self.rng = np.random.default_rng() if rng is None else np.random.default_rng(rng)

    @staticmethod
    def compute_skill_pair_matrix(dataset: Dataset, skill_to_idx: dict[str, int]) -> np.ndarray:
        """Count the number of skill bigrams from previously executed skill sequences.

        :param dataset: Collection of observed skill execution traces
        :param skill_to_idx: Map from skill names to corresponding indices
        :return: NumPy array of skill pair counts, indexed by (previous skill, current skill)
        """
        skill_pair_counts = np.zeros((len(skill_to_idx), len(skill_to_idx)))

        for execution_trace in dataset:
            prev_skill_name = None
            for transition in execution_trace:
                curr_skill_name = transition.skill_instance.skill.name

                if prev_skill_name is not None:
                    prev_skill_idx = skill_to_idx[prev_skill_name]
                    curr_skill_idx = skill_to_idx[curr_skill_name]
                    skill_pair_counts[prev_skill_idx, curr_skill_idx] += 1

                prev_skill_name = curr_skill_name

        return skill_pair_counts

    def get_least_explored_skill_pairs(self, max_pairs: int = 5) -> list[str]:
        """Find the least-explored consecutive pair(s) of skills.

        :param max_pairs: Maximum number of skill pairs to return (defaults to 5)
        :return: List of strings specifying the least-explored skill pairs
        """
        min_value = np.min(self.skill_pairs_matrix)
        min_indices = np.argwhere(self.skill_pairs_matrix == min_value)  # All min-value indices
        if len(min_indices) > max_pairs:
            min_indices = self.rng.choice(min_indices, size=max_pairs, replace=False)

        least_explored_pairs = []
        skills_list = list(self.skill_to_idx.keys())
        for idx_before, idx_after in min_indices:
            skill_before = skills_list[idx_before]
            skill_after = skills_list[idx_after]
            least_explored_pairs.append(f"({skill_before}, {skill_after})")

        return least_explored_pairs
