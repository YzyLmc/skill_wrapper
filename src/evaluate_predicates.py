# given a list of grounded predicates, evaluate each of them and save it as a yaml file
import argparse

from invent_predicate import eval_pred, possible_grounded_preds
from data_structure import PredicateState
from utils import save_to_file, load_from_file, GPT4

def eval_all_predicates(model, lifted_pred_list, type_dict, args):
    # find all possible groundings of predicates
    grounded_preds = possible_grounded_preds(lifted_pred_list, type_dict)
    predicate_state = PredicateState(grounded_preds)
    for i, grounded_pred in enumerate(grounded_preds):
        truth_value = eval_pred(args.img_fpath, grounded_pred, model, args=args)
        predicate_state.set_pred_value(grounded_pred, truth_value)
        print(f'Evaluating predicate {grounded_pred} to be {truth_value}')
        print(f'{i+1}/{len(grounded_preds)} is done')
    save_to_file(predicate_state, f"{args.save_dir}/truth_value.yaml")

def main():
    task_config = load_from_file(args.task_config_fpath)
    args.env = task_config["env"]
    type_dict = {obj: obj_meta['types'] for obj, obj_meta in task_config['objects'].items()}
    lifted_pred_list = load_from_file(args.predicate_fpath)
    model = GPT4(engine=args.model)

    eval_all_predicates(model, lifted_pred_list, type_dict, args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config_fpath", type=str, default="task_config/dorfl.yaml", help="yaml file that store meta data of the env")
    parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", 'gpt-4o-2024-11-20'], default='gpt-4o-2024-11-20')
    parser.add_argument("--predicate_fpath", type=str, help="provide the file path of lifted predicates yaml file")
    parser.add_argument("--img_fpath", type=str, help="provide the file path of image to evaluate")
    parser.add_argument("--save_dir", type=str, help="directory to save log files")
    args = parser.parse_args()
    
    main()
    
    