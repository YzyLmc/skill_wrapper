"Random exploration baseline that randomly picks a skill and popultates the type-matched arguments randomly."
import random
import sys
sys.path.append(f".") # if you run this script from the root directory
import argparse

from src.utils import load_from_file, save_to_file, get_save_fpath
from src.data_structure import Skill

def main():
    task_config_fpath = f"task_config/{args.env}.yaml"
    task_config = load_from_file(task_config_fpath)

    skills = task_config['skills']
    objects = task_config['objects']

    for trial in range(args.num_trials):
        skill_sequence = []
        for step in range(args.num_steps):
            sampled_skill_name: str = random.choice(list(skills.keys()))
            lifted_skill: Skill = skills[sampled_skill_name]
            param_types = lifted_skill.types
            params = []
            for arg_type in param_types:
                candidates = [obj for obj, obj_meta in objects.items() if arg_type in obj_meta['types']]
                if not candidates:
                    raise ValueError(f"No object of type {arg_type} found for skill {sampled_skill_name}.")
                while True:
                    chosen_obj = random.choice(candidates)
                    if chosen_obj not in params:
                        break
                params.append(chosen_obj)
            
            grounded_skill = lifted_skill.ground_with(params)

            skill_sequence.append(grounded_skill)
        save_path = get_save_fpath(f"results/random_explore/{args.env}/skill_sequences", "skill_sequences", "yaml")
        save_to_file(skill_sequence, save_path)
        print(f"Saved random skill sequence to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["dorfl", "spot", "franka", "burger"], default="burger", help="Environment to run the random exploration in.")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of random trials to run.")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of random steps to take.")

    args = parser.parse_args()
    main()
    