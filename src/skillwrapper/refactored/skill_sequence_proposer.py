"""Define a class to manage proposing skill sequences using VLMs."""

from __future__ import annotations

import re

import numpy as np
from sentence_transformers import SentenceTransformer

from skillwrapper.refactored.domain import Domain
from skillwrapper.refactored.environment import ConcreteObjects, Environment
from skillwrapper.refactored.gpt4_client import GPT4Client
from skillwrapper.refactored.skills import SkillInstance
from skillwrapper.refactored.transition_data import Dataset
from skillwrapper.refactored.utils import determine_pytorch_device

SkillSequence = list[SkillInstance]  # A sequence of instantiated skills to be executed


class SkillSequenceProposer:
    """Proposes exploratory skill sequences using a vision-language model."""

    def __init__(self, domain: Domain, prompt: str, rng: np.random.Generator | None = None) -> None:
        """Initialize the skill sequence proposer.

        :param domain: SkillWrapper domain specifying skills and object types
        :param prompt: Prompt provided to the LLM when proposing skill sequences
        :param rng: Random number generator seed, or None (defaults to None)
        """
        self.domain = domain
        self.prompt = prompt

        self.skill_names: list[str] = list(self.domain.skills)
        self.skill_to_idx = {name: idx for idx, name in enumerate(self.skill_names)}

        self.rng = np.random.default_rng() if rng is None else np.random.default_rng(rng)
        self.gpt_client = GPT4Client()

        self.device = str(determine_pytorch_device())
        self.embedding_model = SentenceTransformer("stsb-roberta-large").to(self.device)

        # Skill embeddings are a NumPy array with shape (# skills, embedding dimension)
        self.skill_embeddings = self.embedding_model.encode(
            [str(skill) for skill in self.domain.skills.values()],
            device=self.device,
            normalize_embeddings=True,
        )

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

    def create_prompt(self, dataset: Dataset, objects: ConcreteObjects) -> str:
        """Create an LLM prompt for skill sequence proposal.

        :param dataset: Dataset of transition data to inform skill sequence proposal
        :param objects: Collection of concrete objects in the environment
        """
        skill_prompts = []
        for skill_name, skill in self.domain.skills.items():
            param_descriptions = [str(param) for param in skill.parameters]
            skill_prompt = f"{skill_name}\n\t" + "\n\t".join(param_descriptions)
            skill_prompts.append(skill_prompt)

        objects_with_types = str(objects)
        least_explored_skills = self.get_least_explored_skill_pairs(dataset)

        return (
            self.prompt.replace("[SKILLS_PROMPT]", "\n\n".join(skill_prompts))
            .replace(
                "[OBJECTS_IN_SCENE]",
                objects_with_types,
            )
            .replace("[LEAST_EXPLORED_SKILLS]", ", ".join(least_explored_skills))
        )

    def get_least_explored_skill_pairs(self, dataset: Dataset, max_pairs: int = 5) -> list[str]:
        """Find the least-explored consecutive pair(s) of skills.

        :param dataset: Dataset of transition data used to identify least-explored skill pairs
        :param max_pairs: Maximum number of skill pairs to return (defaults to 5)
        :return: List of strings specifying the least-explored skill pairs
        """
        skill_pairs_matrix = self.compute_skill_pair_matrix(dataset, self.skill_to_idx)

        min_value = np.min(skill_pairs_matrix)
        min_indices = np.argwhere(skill_pairs_matrix == min_value)  # All min-value indices
        if len(min_indices) > max_pairs:
            min_indices = self.rng.choice(min_indices, size=max_pairs, replace=False)

        least_explored_pairs = []
        for idx_before, idx_after in min_indices:
            skill_before = self.skill_names[idx_before]
            skill_after = self.skill_names[idx_after]
            least_explored_pairs.append(f"({skill_before}, {skill_after})")

        return least_explored_pairs

    def find_most_similar_skill(self, skill_name: str) -> str:
        """Find the most similar skill in the domain based on semantic embedding similarity.

        :param skill_name: Name of a new or unknown skill
        :return: Name of the skill in the domain most similar to the given skill name
        """
        new_embedding = self.embedding_model.encode(
            skill_name,
            device=self.device,
            normalize_embeddings=True,
        )

        cosine_similarities = np.dot(self.skill_embeddings, new_embedding)
        print(f"Computed similarities have shape: {cosine_similarities.shape}")

        most_similar_idx = np.argmax(cosine_similarities)
        return self.skill_names[most_similar_idx]

    def find_most_similar_object(self, new_object: str, candidate_objects: list[str]) -> str:
        """Find the most similar of the given objects based on semantic embedding similarity.

        :param new_object: Name of a new or unknown object
        :param candidate_objects: List of objects considered for similarity
        :return: Name of the candidate object most similar to the new object name
        """
        new_embedding = self.embedding_model.encode(
            new_object,
            device=self.device,
            normalize_embeddings=True,
        )

        candidate_embeddings = self.embedding_model.encode(
            candidate_objects,
            device=self.device,
            normalize_embeddings=True,
        )

        cosine_similarities = np.dot(candidate_embeddings, new_embedding)
        print(f"Computed similarities have shape: {cosine_similarities.shape}")

        most_similar_idx = np.argmax(cosine_similarities)
        return candidate_objects[most_similar_idx]

    def construct_skill_sequences(self, llm_response: str, env: Environment) -> list[SkillSequence]:
        """Construct proposed skill sequences based on the text response from an LLM.

        :param llm_response: String response to a skill sequence proposal prompt
        :param env: Current environment specifying concrete objects and an initial state
        :return: List of sequences of skill instances (i.e., skills grounded with concrete objects)
        """
        curr_sequence_name = None
        skill_sequences: dict[str, list[SkillInstance]] = {}  # Maps sequence names to sequences

        for line in llm_response.split("\n"):
            if "Skill Sequence" in line:
                skill_sequences[line] = []
                curr_sequence_name = line
                continue

            # Try to parse a SkillInstance, replacing invalid skill/argument names as necessary
            match = re.match(r"^(\w+)\(([^)]*)\)", line.strip())
            if match and curr_sequence_name is not None:
                skill_name = match.group(1)
                if skill_name not in self.skill_names:
                    skill_name = self.find_most_similar_skill(skill_name)

                chosen_skill = self.domain.skills[skill_name]

                args_string = match.group(2).strip()
                args = [arg.strip() for arg in args_string.split(",")] if args_string else []

                accepted_args: list[str] = []
                for idx, argument in enumerate(args):
                    obj_name = argument

                    if obj_name not in env.objects.object_names:
                        obj_name = self.find_most_similar_object(
                            obj_name,
                            list(env.objects.object_names),
                        )

                    expected_type = chosen_skill.parameters[idx].object_type
                    if expected_type not in env.objects.get_types_of_object(obj_name):
                        candidates = list(env.objects.get_all_objects_of_type(expected_type))
                        obj_name = self.find_most_similar_object(obj_name, candidates)

                    accepted_args.append(obj_name)

                new_args_string = ", ".join(accepted_args)
                new_skill_string = f"{skill_name}({new_args_string})"

                parsed_skill = SkillInstance.from_string(new_skill_string, self.domain, env)
                skill_sequences[curr_sequence_name].append(parsed_skill)

        return list(skill_sequences.values())

    def propose_skill_sequence(self, dataset: Dataset, env: Environment) -> list[SkillInstance]:
        """Propose a sequence of skill instances to be executed in the given environment.

        :param dataset: Dataset of skill execution traces (empty on first iteration)
        :param env: Environment specifying concrete objects and the initial state
        :return: List of proposed skill instances
        """
        updated_prompt = self.create_prompt(dataset, env.objects)
        print(f"Updated prompt for LLM:\n\n{updated_prompt}")

        llm_response = self.gpt_client.generate(prompt=updated_prompt)
        print(f"LLM response:\n\n{llm_response}")

        proposed_sequences = self.construct_skill_sequences(llm_response, env)
        for i, sequence in enumerate(proposed_sequences):
            sequence_str = "\n\t".join(str(skill_instance) for skill_instance in sequence)
            print(f"Proposed Skill Sequence {i}:\n\t{sequence_str}\n")

        return proposed_sequences[0]  # TODO: Replace skill sequence selection heuristics
