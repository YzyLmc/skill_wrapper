"""
Use priviledged predicates from burger simulator to learn operators.
Propose skill sequence as SkillWrapper, get predicate state directly without classification.

Example command:
    python baselines/oracle_predicates.py ++invent_pred_only=True
"""

# read plan from results, run the plans and save the predicate states in yaml files under results/transitions

# Call operator learning function in operator_learning.py to learn operators
import argparse
import logging
from collections import defaultdict
import os
import sys
sys.path.append(f".") 
import hydra
from omegaconf import DictConfig, OmegaConf
from robotouille.run_skill_sequence import exec_and_record


from src.utils import save_to_file, load_from_file, setup_logging, get_save_fpath, save_results, load_results, init_new_iter, clean_logging, GPT4
from src.invent_predicate import calculate_operators_for_all_skill, filter_predicates, calculate_operators_for_all_skill
from src.skill_sequence_proposing import SkillSequenceProposing
from src.data_structure import Skill, PredicateState, Predicate

def propose_and_execute(skill_sequence_proposing: SkillSequenceProposing, tasks, lifted_pred_list, skill2operator, save_dir, cfg):
    """
    Propose a skill sequence and execute the skill sequence
    """
    t = 0
    while t < 10:
        chosen_skill_sequence = skill_sequence_proposing.run_skill_sequence_proposing(lifted_pred_list, skill2operator, tasks)
        t += 1
        logging.info(f'Task: {[str(skill) for skill in chosen_skill_sequence]}')
        if len(chosen_skill_sequence) < 10:
            continue
        
        # save the plan
        save_path = get_save_fpath(f"{save_dir}/skill_sequences", "skill_sequence", "yaml")
        save_to_file(chosen_skill_sequence, save_path)

        # if in burger, execute the skill sequence and save the transitions in
        # results/oracle_predicates/burger/runs/{run_idx}/{iter_idx}/transitions/tasks.yaml
        if cfg.env == 'burger':
            kwargs = OmegaConf.to_container(cfg.game, resolve=True)
            environment_name = kwargs.pop('environment_name')
            exec_and_record(environment_name, chosen_skill_sequence, os.path.join(save_dir, "transitions"), oracle_state=True)
            
            break

        else:
            raise Exception(f"Execute the plan and collect data on {cfg.env} is not implemented yet.")
        

    task_fpath = os.path.join(save_dir, "transitions", "tasks.yaml")
    tasks = load_from_file(task_fpath)

    return tasks

@hydra.main(version_base=None, config_path="../hydra_conf", config_name="oracle_predicates_config")
def main(cfg: DictConfig):
    # init env
    task_config_fpath = f"task_config/{cfg.env}.yaml"
    task_config = load_from_file(task_config_fpath)
    type_dict = {obj: obj_meta['types'] for obj, obj_meta in task_config['objects'].items()}

    log_dir = f"results/oracle_predicates/{cfg.env}/log/"
    log_save_path = setup_logging(log_dir, cfg.env) # configure logging

    # init skill sequence proposing system
    skill_sequence_proposing = SkillSequenceProposing(task_config_fpath=task_config_fpath)
    
    # main loop
    iter_idx = cfg.iter_idx if cfg.iter_idx else 0
    for i in range(cfg.num_iter):
        if not cfg.invent_pred_only:

            # prepare folder structures, copy from previous iteration if exists
            load_dir = f"results/oracle_predicates/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}/"
            tasks, skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = load_results(load_dir, task_config)
            save_dir = init_new_iter(cfg.env, cfg.method, cfg.run_idx)

            # propose skill sequence
            tasks: list[Skill] = propose_and_execute(skill_sequence_proposing, tasks, lifted_pred_list, skill2operator, save_dir, cfg)

        else:
            load_dir = f"results/oracle_predicates/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}_partial/"
            assert os.path.exists(load_dir), "must provide tasks.yaml to start predicate invention."

        if not cfg.skill_seq_only:

            # load partial results copied from previous iteration
            if not cfg.invent_pred_only:
                load_dir = save_dir
            else:
                load_dir = f"results/oracle_predicates/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}_partial/"
            tasks, skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = load_results(load_dir, task_config)

            # copy and past load dir
            save_dir = f"results/oracle_predicates/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}"
            if not os.path.exists(save_dir):
                os.system(f"cp -r {load_dir} {save_dir}")

            # calculate operators for all skills using oracle predicates
            skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict)
            filtered_lifted_pred_list = filter_predicates(skill2operator, lifted_pred_list, grounded_predicate_truth_value_log, tasks, type_dict)
            skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, filtered_lifted_pred_list)

            # save results of the iteration by overwriting the copied folders
            # we don't save filtered predicate list because we don't want to lose any possible predicates in oracle predicate setting
            save_results(skill2operator, lifted_pred_list, grounded_predicate_truth_value_log, save_dir)

            # log results
            operator_string_lists = [[f"Skill:{str(lifted_skill)}\nOperator{str(operator_tuple[0])}\n" for operator_tuple in operator_tuples if operator_tuple] for lifted_skill, operator_tuples in skill2operator.items()]
            logging.info("Operators learned this round:")
            for operator_string_list in operator_string_lists: logging.info('\n'.join(operator_string_list))

        else:
            assert not cfg.invent_pred_only, "Either one of proposal and predicate invention must be called."

        logging.info(f"iteration #{i+1} is done")
        iter_idx += 1

        if cfg.step_by_step:
            logging.info(f"iteration #{i+1}/{cfg.num_iter} is done, run next iteration?")
            breakpoint()

    clean_logging(log_save_path)

if __name__ == "__main__":
    """
    Arguments & Default values:
    env: The name of the environment, one of ["dorfl", "spot", "franka", "burger"]
    model: The name of the GPT-4 model to use, one of ["gpt-4o-2024-08-06", 'gpt-4o-2024-11-20']

    run_idx: index of the run that produce the best operators.
    iter_idx: index of iter run the full refinement and proposal loop.

    num_iter: number of iterations to run
    max_retry_time: maximum time to generate predicate to distinguish two states.

    invent_pred_only: Read from existing data and invent predicates.
    skill_seq_only: Read from existing data and propose skill sequences.

    step_by_step: Whether to run in step-by-step mode.  

    game.environment_game: burger environment used for collecting transitions. default "easy/problems/0/problem"
    """
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--env", type=str, choices=["dorfl", "spot", "franka", "burger"], default="burger", help="the name of the environment")
    # parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", 'gpt-4o-2024-11-20'], default='gpt-4o-2024-11-20')

    # parser.add_argument("--run_idx", type=int, default=0, help="index of the run that produce the best operators.")
    # parser.add_argument("--iter_idx", type=int, help="index of iter run the full refinement and proposal loop.")

    # parser.add_argument("--num_iter", type=int, default=5, help="number of iterations to run")
    # parser.add_argument("--max_retry_time", type=int, default=3, help="maximum time to generate predicate to distinguish two states.")

    # parser.add_argument("--invent_pred_only", action="store_true", help="Read from existing data and invent predicates.")
    # parser.add_argument("--skill_seq_only", action="store_true", help="Read from existing data and invent predicates")

    # parser.add_argument("--step_by_step", action="store_true")

    # args = parser.parse_args()

    main()