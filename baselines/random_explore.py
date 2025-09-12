"""
Use priviledged predicates from burger simulator to learn operators.
Propose skill sequence as SkillWrapper, get predicate state directly without classification.

Example command:
    python baselines/random_explore.py ++invent_pred_only=True
"""

# read plan from results, run the plans and save the predicate states in yaml files under results/transitions

# Call operator learning function in operator_learning.py to learn operators
import argparse
import logging
from collections import defaultdict
import os
import sys
sys.path.append(f".") 

import random
import hydra
from omegaconf import DictConfig, OmegaConf
from robotouille.run_skill_sequence import exec_and_record


from src.utils import save_to_file, load_from_file, setup_logging, get_save_fpath, save_results, load_results, init_new_iter, clean_logging, GPT4
from src.invent_predicate import invent_predicates, calculate_operators_for_all_skill, filter_predicates, calculate_operators_for_all_skill
from src.skill_sequence_proposing import SkillSequenceProposing
from src.data_structure import Skill, PredicateState, Predicate


def propose_and_execute(cfg, save_dir, steps=10):
    task_config_fpath = f"task_config/{cfg.env}.yaml"
    task_config = load_from_file(task_config_fpath)

    skills = task_config['skills']
    objects = task_config['objects']

    skill_sequence = []
    for step in range(steps):
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
    save_path = get_save_fpath(f"{save_dir}/skill_sequences", "skill_sequence", "yaml")
    save_to_file(skill_sequence, save_path)
    print(f"Saved random skill sequence to {save_path}")

    # if in burger, execute the skill sequence and save the transitions in
    # results/random_explore/burger/runs/{run_idx}/{iter_idx}/transitions/tasks.yaml
    if cfg.env == 'burger':
        kwargs = OmegaConf.to_container(cfg.game, resolve=True)
        environment_name = kwargs.pop('environment_name')
        exec_and_record(environment_name, skill_sequence, os.path.join(save_dir, "transitions"))

    else:
        logging.warning(f"Execute the plan and collect data on {cfg.env} and save it as (or add to) {os.path.join(save_dir, 'transitions/tasks.yaml')}.")
        logging.warning(f"Then, continue from the breakpoint.")
        breakpoint()

    task_fpath = os.path.join(save_dir, "transitions", "tasks.yaml")
    tasks = load_from_file(task_fpath)

    return tasks

def invent_predicates_for_all_skill(model, lifted_pred_list, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, env: str, cfg):
    '''
    run one iteration of refinement and proposal
    pred_dict, skill2operator and skill2tasks are from refinement. 
    replay_buffer, grounded_predicate_dictionary, grounded_skill_dictionary are from task proposal.
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    '''
    for lifted_skill in skill2operator:
        skill2triedpred = defaultdict(list) # reset tried_predicate buffer after each skill
        skill2operator, lifted_pred_list, skill2triedpred, grounded_predicate_truth_value_log = invent_predicates(model, lifted_skill, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, env, skill2triedpred=skill2triedpred, max_t=cfg.max_retry_time)

    filtered_lifted_pred_list = filter_predicates(skill2operator, lifted_pred_list, grounded_predicate_truth_value_log, tasks, type_dict)
    skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, filtered_lifted_pred_list)

    return skill2operator, lifted_pred_list, grounded_predicate_truth_value_log

@hydra.main(version_base=None, config_path="../hydra_conf", config_name="random_explore_config")
def main(cfg: DictConfig):
    # init env
    task_config_fpath = f"task_config/{cfg.env}.yaml"
    task_config = load_from_file(task_config_fpath)
    type_dict = {obj: obj_meta['types'] for obj, obj_meta in task_config['objects'].items()}

    log_dir = f"results/random_explore/{cfg.env}/log/"
    log_save_path = setup_logging(log_dir, cfg.env) # configure logging

    model = GPT4(engine=cfg.model)

    # init skill sequence proposing system
    skill_sequence_proposing = SkillSequenceProposing(task_config_fpath=task_config_fpath)
    
    # main loop
    iter_idx = cfg.iter_idx if cfg.iter_idx else 0
    for i in range(cfg.num_iter):
        if not cfg.invent_pred_only:

            # prepare folder structures, copy from previous iteration if exists
            load_dir = f"results/random_explore/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}/"
            tasks, skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = load_results(load_dir, task_config)
            save_dir = init_new_iter(cfg.env, cfg.method, cfg.run_idx)

            # propose skill sequence
            tasks: list[Skill] = propose_and_execute(cfg, save_dir)

        else:
            load_dir = f"results/random_explore/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}_partial/"
            assert os.path.exists(load_dir), "must provide tasks.yaml to start predicate invention."

        if not cfg.skill_seq_only:

            # load partial results copied from previous iteration
            if not cfg.invent_pred_only:
                load_dir = save_dir
            else:
                load_dir = f"results/random_explore/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}_partial/"
            tasks, skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = load_results(load_dir, task_config)

            # copy and past load dir
            save_dir = f"results/random_explore/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}"
            if not os.path.exists(save_dir):
                os.system(f"cp -r {load_dir} {save_dir}")

            # invent predicates
            skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = invent_predicates_for_all_skill(model, lifted_pred_list, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, cfg.env, cfg)
 
            # save results of the iteration by overwriting the copied folders
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

    main()