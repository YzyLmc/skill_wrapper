"""Run the SkillWrapper algorithm using refactored data structures."""

import argparse
from pathlib import Path

from skillwrapper.skillwrapper_structs import (
    Domain,
    Environment,
    Prompts,
    SkillSequenceProposer,
    import_yaml_into_dict,
)
from skillwrapper.utils import GPT4, setup_logging


def main(args: argparse.Namespace) -> None:
    """Run the SkillWrapper algorithm using the given arguments."""
    logs_dir: Path = args.logs_dir
    domain_yaml: Path = args.domain_yaml
    env_yaml: Path = args.env_yaml

    domain = Domain.from_yaml(domain_yaml)
    domain_name = domain_yaml.stem
    env = Environment.from_yaml(env_yaml)
    env_name = env_yaml.stem

    sspp_keys = {"skill-sequence-proposal-prompts"}
    sspp_yaml_data = import_yaml_into_dict(domain_yaml, required_keys=sspp_keys)
    ssp_prompts = Prompts.from_yaml(sspp_yaml_data)

    logs_save_path = setup_logging(logs_dir, domain_name, env_name)

    model = GPT4(engine=args.model)

    skill_sequence_proposer = SkillSequenceProposer(domain, env, ssp_prompts)

    # TODO: Up next is to load "results" from a previous iteration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_yaml", type=Path, help="YAML file configuring the domain")
    parser.add_argument("env_yaml", type=Path, help="YAML file configuring the environment")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-4o-2024-08-06", "gpt-4o-2024-11-20"],
        default="gpt-4o-2024-11-20",
        help="OpenAI snapshot to use for GPT-4o",
    )
    parser.add_argument("--logs_dir", type=Path, default="logs", help="Directory for log files")

    args = parser.parse_args()

    main(args)
