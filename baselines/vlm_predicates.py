# Ablation baseline: VLM propose set of predicates and we use the same operator construction method
import sys
import argparse

sys.path.append(".")
from src.utils import GPT4, load_from_file, save_to_file, load_results
from src.invent_predicate import calculate_operators_for_all_skill, update_empty_predicates
from src.data_structure import Skill

def get_batch_pred(model,  args, prompt_fpath="prompts/evaluate_pred.yaml"):
    """
    Get a batch of predicates all at once from foundation model
    """
    def construct_prompt(prompt: str, lifted_skills: list[Skill]):
        placeholders = ["[SKILL_LIST]", "[OBJECT_LIST]"]
        while any([p in prompt for p in placeholders]):
            prompt = prompt.replace()
    prompt = load_from_file(prompt_fpath)[args.env]

def main():
    # load task configurations
    task_config = load_from_file(args.task_config_fpath)
    args.env = task_config["env"]
    type_dict = {obj: obj_meta['types'] for obj, obj_meta in task_config['objects'].items()}
    # load previous executed tasks
    # everything should be empty except tasks
    tasks, skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = load_results(args.load_fpath, task_config)
    # prompt foundation model for predicates
    # save the generated predicate to yaml file
    # evaluate truth values
    update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, args)
    # calculate operators using all images
    calculate_operators_for_all_skill()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config_fpath", type=str, default="task_config/dorfl.yaml", help="yaml file that store meta data of the env")
    parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", 'gpt-4o-2024-11-20'], default='gpt-4o-2024-11-20')
    parser.add_argument("--load_fpath", type=str, help="provide the log file to restore from a previous checkpoint. must specify if continue learning is true")
    args = parser.parse_args()

    main()