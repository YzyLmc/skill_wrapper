"""
For non-vila baselines:
Take in a set of problems, read the image or text of init state and goal state, and evaluate each predicate.
Save the truth value in yaml files.

Example command:
    python eval/eval_preds.py --baseline expert_operators --dataset test
"""
import argparse
import logging
import os
import sys
sys.path.append(f".")
from src.invent_predicate import eval_pred, possible_grounded_preds
from src.data_structure import PredicateState, Predicate
from src.utils import save_to_file, load_from_file, GPT4, setup_logging, raw_prompt

def eval_all_predicates(model: GPT4, lifted_pred_list: list[Predicate], type_dict: dict[str, list[str]], img_fpath: str, env, input_modality="image", batched=False) -> PredicateState:
    # find all possible groundings of predicates
    grounded_preds = possible_grounded_preds(lifted_pred_list, type_dict)
    predicate_state = PredicateState(grounded_preds)
    if batched:
        # get raw response first
        prompt_1 = load_from_file("prompts/evaluate_pred.yaml")["burger_step_1"]
        object_str = "\n".join([f"{obj}: {types}" for obj, types in type_dict.items()])
        pred_str = "\n".join([f"{str(pred)} : {pred.semantic}" for pred in lifted_pred_list])
        response_1 = raw_prompt(prompt_1.replace("[OBJECTS]", object_str).replace("[PREDICATES]", pred_str), [img_fpath])
        # summarize and correct
        prompt_2 = load_from_file("prompts/evaluate_pred.yaml")["burger_step_2"]
        object_names = ", ".join(list(type_dict.keys()))
        response_2 = raw_prompt(prompt_2.replace("[OBJECT_NAMES]", object_names).replace("[RESPONSE]", response_1))
        # parse the summarized response
        response_lines = response_2.strip().split("\n")
        true_grounded_preds = []
        for line in response_lines:
            if not line.strip():
                continue
            grounded_pred = Predicate.from_string(line.strip())
            try:
                grounded_pred.types = [p.types for p in lifted_pred_list if p.name == grounded_pred.name][0]  # assign types
            except:
                breakpoint()
            logging.info(f'Evaluating predicate {grounded_pred} to be True')
            true_grounded_preds.append(grounded_pred)
            try:
                predicate_state.set_pred_value(grounded_pred, True)
            except:
                breakpoint()

        # set the rest to False
        for grounded_pred in grounded_preds:
            if grounded_pred not in true_grounded_preds:
                predicate_state.set_pred_value(grounded_pred, False)
                logging.info(f'Evaluating predicate {grounded_pred} to be False')
    else:
        for i, grounded_pred in enumerate(grounded_preds):
            truth_value = eval_pred(img_fpath, grounded_pred, model, env, input_modality, log=True)
            predicate_state.set_pred_value(grounded_pred, truth_value)
            logging.info(f'Evaluating predicate {grounded_pred} to be {truth_value}')
            logging.info(f'{i+1}/{len(grounded_preds)} is done')
    return predicate_state

def main():
    # setup logging
    logging_dir = f"results/{args.baseline}/{args.env}/"
    logging_fpath = os.path.join(logging_dir, "eval_preds_log")
    setup_logging(logging_fpath, args.env)

    # init model
    model = GPT4(engine=args.model)

    if not args.iter_idx:
        # all iterations under the runs folder
        iters = os.listdir(f"results/{args.baseline}/{args.env}/runs/{args.run_idx}/")
        # largest iteration number
        args.iter_idx = max([int(i) for i in iters if i.isdigit()])

    # load predicates
    pred_fpath = f"results/{args.baseline}/{args.env}/runs/{args.run_idx}/{args.iter_idx}/predicates/predicates.yaml"
    lifted_pred_list = load_from_file(pred_fpath)
    logging.info(f"Loaded predicates from {pred_fpath}")

    # load typed dict
    task_config_fpath = f"task_config/{args.env}.yaml"
    task_config = load_from_file(task_config_fpath)
    type_dict = {obj: obj_meta['types'] for obj, obj_meta in task_config['objects'].items()}

    # loop through all problems under a dataset
    problem_dir = f"eval/data/{args.env}/{args.dataset}/problems/"
    for root, dirs, files in os.walk(problem_dir):
        for d in dirs:
            save_dir = f"results/{args.baseline}/{args.env}/pred_state/{args.dataset}/{d}"
            if os.path.exists(save_dir):
                logging.info(f"Problem {d} already has predicate states at {save_dir}, skipping...")
                continue
            os.makedirs(save_dir, exist_ok=True)
            logging.info(f"Processing problem {d} in {root}...")
            if args.input_modality == "image":
                init_state_fpath = os.path.join(root, d, "init_state.jpg")
                goal_state_fpath = os.path.join(root, d, "goal_state.jpg")
            elif args.input_modality == "text":
                init_state_fpath = os.path.join(root, d, "init_state.txt")
                goal_state_fpath = os.path.join(root, d, "goal_state.txt")
                raise NotImplementedError("Text input modality is not implemented yet.")

            init_pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, init_state_fpath, args.env, args.input_modality, batched=True)
            goal_pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, goal_state_fpath, args.env, args.input_modality, batched=True)

            
            save_to_file(init_pred_state, f"{save_dir}/init_state_{args.input_modality}.yaml")
            save_to_file(goal_pred_state, f"{save_dir}/goal_state_{args.input_modality}.yaml")
            logging.info(f"Saved predicate states to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", 'gpt-4o-2024-11-20', 'o3', 'gpt-5'], default='gpt-5')
    parser.add_argument("--run_idx", type=int, default=0, help="index of the run that produce the best operators.")
    parser.add_argument("--iter_idx", type=int, help="index of iter run the full refinement and proposal loop.")
    parser.add_argument("--baseline", type=str, choices=["fm_invent", "oracle_predicates", "expert_operators", "random_explore", "skillwrapper"], help="the name of the baseline")
    parser.add_argument("--dataset", type=str, choices=["test", "test1", "suc", "imp", "in_domain", "easy_0", "easy_1", "easy_2", "easy_3", "hard_0", "hard_1", "hard_2", "hard_3", "impossible_0", "impossible_1"], help="the name of the dataset")
    parser.add_argument("--env", type=str, choices=["dorfl", "franka", "spot", "burger"], default="burger", help="the name of the environment")

    parser.add_argument("--input_modality", type=str, choices=["image", "text"], default="image", help="the input modality of the state")

    args = parser.parse_args()
    
    main()
    
    