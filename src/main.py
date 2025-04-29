'refinement and proposal loop'
'remember to change the for loop in update_tasks() if you have changed the input skill'
import argparse
import os
from copy import deepcopy
import logging
from collections import defaultdict
import re

from data_structure import Skill, yaml
from utils import GPT4, load_from_file, setup_logging, clean_logging, save_results
from skill_sequence_proposing import SkillSequenceProposing
from invent_predicate import invent_predicates
from ai2thor_task_exec import convert_task_to_code

def propose_and_execute(skill_sequence_proposing, tasks, lifted_pred_list, grounded_predicate_truth_value_log, skill2operator, args):
    """
    Propose a skill sequence and execute the skill sequence
    """
    t = 0
    task_success = False
    while t < 10 and not task_success:
        chosen_skill_sequence = skill_sequence_proposing.run_skill_sequence_proposing(tasks, lifted_pred_list, grounded_predicate_truth_value_log, skill2operator)
        t += 1
        logging.info(f'Task: {chosen_skill_sequence}')
        try:
            if len(chosen_skill_sequence) < 6:
                continue
            # this is an example way of executing tasks with a templated script
            generated_code = convert_task_to_code(chosen_skill_sequence)
            local_scope, global_scope = {}, {}
            exec(generated_code, global_scope, local_scope)
            task_success = True
        except:
            logging.info("Skill sequence execution failed.")
            pass

    if args.step_by_step:
            logging.info('Task done. You should check the images labels')
            breakpoint()

    # NOTE: The execution script should save a yaml file to 
    #   {args.save_dir}/tasks_{args.env}.yaml
    # in the format of:
    #   dict(task_name: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool)))
    # The file will be read and returned for predicate invention
    tasks = load_from_file(args.save_dir)
    return tasks

def invent_predicates_for_all_skill(model, lifted_pred_list, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, args):
    '''
    run one iteration of refinement and proposal
    pred_dict, skill2operator and skill2tasks are from refinement. 
    replay_buffer, grounded_predicate_dictionary, grounded_skill_dictionary are from task proposal.
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    '''
    for lifted_skill in skill2operator:
        skill2triedpred = {} # reset tried_predicate buffer after each skill
        skill2operator, lifted_pred_list, skill2triedpred, grounded_predicate_truth_value_log = invent_predicates(model, lifted_skill, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, skill2triedpred=skill2triedpred, max_t=args.max_retry_time)
    return skill2operator, lifted_pred_list, grounded_predicate_truth_value_log

def main():
    # init env
    env_config = load_from_file(args.env_config_fpath)

    log_save_path = setup_logging(args.save_dir, env_config["env"]) # configure logging

    # main loop
    if env_config["env"] in ["dorfl", "spot"]:
        model = GPT4(engine=args.model)
        if args.continue_learning:
            # TODO: continue learning not implemented for the new system
            # requires a load function to load grounded_predicate_truth_value_log, skill2operators, and lifted_pred_list
            raise NotImplementedError("Continue learning not implemented yet")
        
        else:   
            # start from scratch
            lifted_pred_list = []
            skill2operator = {}

        # init skill sequence proposing system
        skill_sequence_proposing = SkillSequenceProposing(env_config_fpath=args.env_config_fpath) # prompt not included but 
        
        if args.continue_learning:
            raise NotImplementedError("Continue learning not implemented yet")
            # start_num = str(max([int(key) for key in log_data.keys() if key.isdigit()]))
        else:
            start_num = "0"

        # TODO: read or init if start from scratch
        type_dict = {obj: obj_meta['types'] for obj, obj_meta in env_config['objects'].items()}
        tasks = {}
        grounded_predicate_truth_value_log = {}

        # main loop
        for i in range(int(start_num), args.num_iter):
            # propose skill sequence and execute
            tasks = propose_and_execute(skill_sequence_proposing, tasks, lifted_pred_list, grounded_predicate_truth_value_log, skill2operator, args)
            # invent predicates
            skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = invent_predicates_for_all_skill(model, lifted_pred_list, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, args)

            logging.info(f"iteration #{i+1} is done")
            operator_string_list = [f"Skill:{str(skill2operator)}\nOperator{str(operator_tuple[0])}\n" for lifted_skill, operator_tuple in skill2operator.items()]
            logging.info(f'Operators learned this round:\n{"\n".join(operator_string_list)}')
            save_results(skill2operator, lifted_pred_list, grounded_predicate_truth_value_log, args.save_dir)

            if args.step_by_step:
                logging.info(f"iteration #{i+1}/{args.num_iter} is done, run to next interation?")
                breakpoint()

        clean_logging(log_save_path)
    else:
        raise NotImplementedError(f"Env {env_config["env"]} has not been implemented.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config_fpath", type=str, default="task_config/test.yaml", help="yaml file that store meta data of the env")
    parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", "chatgpt-4o-latest"], default="gpt-4o-2024-08-06")
    parser.add_argument("--num_iter", type=int, default=5, help="num of iter run the full refinement and proposal loop.")
    parser.add_argument("--step_by_step", action="store_true")
    parser.add_argument("--max_retry_time", type=int, default=3, help="maximum time to generate predicate to distinguish two states.")
    parser.add_argument("--continue_learning", action='store_true')
    parser.add_argument("--load_fpath", type=str, help="provide the log file to restore from a previous checkpoint. must specify if continue learning is true")
    parser.add_argument("--save_dir", type=str, default='tasks/log', help="directory to save log files")
    parser.add_argument("--invent_pred_only", type=bool, action="store_true", help="Read from existing data and invent predicates.")
    args = parser.parse_args()

    main()