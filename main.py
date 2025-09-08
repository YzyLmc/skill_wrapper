"""
Main function for SkillWrapper. Because of the robot experiments, skill sequence proposal and predicate invention are separated into two parts.
"""
import argparse
import logging
from collections import defaultdict

from src.data_structure import Skill
from src.utils import GPT4, load_from_file, save_to_file, setup_logging, clean_logging, save_results, load_results, get_save_fpath
from src.skill_sequence_proposing import SkillSequenceProposing
from src.invent_predicate import invent_predicates, filter_predicates, calculate_operators_for_all_skill

def propose_and_execute(skill_sequence_proposing: SkillSequenceProposing, tasks, lifted_pred_list, skill2operator, args):
    """
    Propose a skill sequence and execute the skill sequence
    """
    t = 0
    task_success = False
    while t < 10 and not task_success:
        chosen_skill_sequence = skill_sequence_proposing.run_skill_sequence_proposing(lifted_pred_list, skill2operator, tasks)
        t += 1
        logging.info(f'Task: {[str(skill) for skill in chosen_skill_sequence]}')
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
            break
    # save tasks in json
    save_path = get_save_fpath("test_tasks", args.env, "json")
    save_to_file(chosen_skill_sequence, save_path)
    if args.step_by_step:
            logging.info('Task done. You should check the images labels')
            breakpoint()

    # NOTE: The execution script should save a yaml file to 
    #   {args.save_dir}/tasks_{args.env}.yaml
    # in the format of:
    #   dict(task_name: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool)))
    # The file will be read and returned for predicate invention
    tasks = load_from_file(save_path)
    return tasks

def invent_predicates_for_all_skill(model, lifted_pred_list, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, args):
    '''
    run one iteration of refinement and proposal
    pred_dict, skill2operator and skill2tasks are from refinement. 
    replay_buffer, grounded_predicate_dictionary, grounded_skill_dictionary are from task proposal.
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    '''
    for lifted_skill in skill2operator:
        skill2triedpred = defaultdict(list) # reset tried_predicate buffer after each skill
        skill2operator, lifted_pred_list, skill2triedpred, grounded_predicate_truth_value_log = invent_predicates(model, lifted_skill, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, skill2triedpred=skill2triedpred, max_t=args.max_retry_time, args=args)

    # filtered_lifted_pred_list = filter_predicates(skill2operator, lifted_pred_list, grounded_predicate_truth_value_log,tasks)
    # breakpoint()
    # skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks,filtered_lifted_pred_list)

    return skill2operator, lifted_pred_list, grounded_predicate_truth_value_log

def main():
    # init env
    task_config_fpath = f"task_config/{args.env}.yaml"
    task_config = load_from_file(task_config_fpath)
    type_dict = {obj: obj_meta['types'] for obj, obj_meta in task_config['objects'].items()}

    log_dir = f"results/skillwrapper/{args.env}/log/"
    log_save_path = setup_logging(log_dir, args.env) # configure logging

    model = GPT4(engine=args.model)

    # init skill sequence proposing system
    skill_sequence_proposing = SkillSequenceProposing(task_config_fpath=task_config_fpath)
    
    tasks, skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = load_results(args.load_fpath, task_config)
    
    # main loop
    for i in range(args.num_iter):
        if not args.invent_pred:
            # propose skill sequence
            tasks: list[Skill] = propose_and_execute(skill_sequence_proposing, tasks, lifted_pred_list, skill2operator, args)
        else:
            assert args.load_fpath is not None, "must provide tasks.yaml to start predicate invention."

        if not args.skill_seq:
            # invent predicates
            skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = invent_predicates_for_all_skill(model, lifted_pred_list, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, args)
        else:
            assert not args.invent_pred_only, "Either one of proposal and predicate invention must be called."

        save_results(skill2operator, lifted_pred_list, grounded_predicate_truth_value_log, args.save_dir)

        logging.info(f"iteration #{i+1} is done")

        # log results
        operator_string_lists = [[f"Skill:{str(lifted_skill)}\nOperator{str(operator_tuple[0])}\n" for operator_tuple in operator_tuples if operator_tuple] for lifted_skill, operator_tuples in skill2operator.items()]
        logging.info("Operators learned this round:")
        for operator_string_list in operator_string_lists: logging.info('\n'.join(operator_string_list))

        if args.step_by_step:
            logging.info(f"iteration #{i+1}/{args.num_iter} is done, run next iteration?")
            breakpoint()
    clean_logging(log_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, choices=["dorfl", "spot", "franka", "burger"], default="dorfl", help="the name of the environment")

    parser.add_argument("--model", type=str, choices=["gpt-4o-2024-08-06", 'gpt-4o-2024-11-20'], default='gpt-4o-2024-11-20')
    parser.add_argument("--num_iter", type=int, default=2, help="num of iter run the full refinement and proposal loop.")
    parser.add_argument("--max_retry_time", type=int, default=3, help="maximum time to generate predicate to distinguish two states.")
    parser.add_argument("--save_dir", type=str, help="directory to save learned operators files")
    parser.add_argument("--load_fpath", type=str, help="provide the log file to restore from a previous checkpoint. must specify if continue learning is true")

    parser.add_argument("--invent_pred", action="store_true", help="Read from existing data and invent predicates.")
    parser.add_argument("--skill_seq", action="store_true", help="Read from existing data and invent predicates")

    parser.add_argument("--step_by_step", action="store_true")

    args = parser.parse_args()

    main()