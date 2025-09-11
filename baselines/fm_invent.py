# Ablation baseline: VLM propose set of predicates and we use the same operator construction method
import logging
from collections import defaultdict
from copy import deepcopy
import os
import sys
sys.path.append(f".") 
import hydra
from omegaconf import DictConfig, OmegaConf
from robotouille.run_skill_sequence import exec_and_record

from src.data_structure import Skill, Predicate
from src.utils import GPT4, load_from_file, save_to_file, setup_logging, clean_logging, save_results, load_results, get_save_fpath, init_new_iter
from src.skill_sequence_proposing import SkillSequenceProposing
from src.invent_predicate import invent_predicates, filter_predicates, calculate_operators_for_all_skill, update_empty_predicates, score_by_partition


def get_batch_pred(model,  args, prompt_fpath="prompts/fm_invent.yaml") -> list[Predicate]:
    """
    Get a batch of predicates all at once from foundation model
    """
    def construct_prompt(prompt: str, lifted_skills: list[Skill]):
        placeholders = ["[SKILL_LIST]", "[OBJECT_LIST]"]
        while any([p in prompt for p in placeholders]):
            prompt = prompt.replace()
    prompt = load_from_file(prompt_fpath)[args.env]

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
        # results/fm_invent/burger/runs/{run_idx}/{iter_idx}/transitions/tasks.yaml
        if cfg.env == 'burger':
            kwargs = OmegaConf.to_container(cfg.game, resolve=True)
            environment_name = kwargs.pop('environment_name')
            exec_and_record(environment_name, chosen_skill_sequence, os.path.join(save_dir, "transitions"))

        else:
            logging.warning(f"Execute the plan and collect data on {cfg.env} and save it as (or add to) {os.path.join(save_dir, 'transitions/tasks.yaml')}.")
            logging.warning(f"Then, continue from the breakpoint.")
            breakpoint()

    task_fpath = os.path.join(save_dir, "transitions", "tasks.yaml")
    tasks = load_from_file(task_fpath)

    return tasks

def invent_predicates_for_all_skill(pred_pool: list[Predicate], model, lifted_pred_list, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, env: str, cfg):
    '''
    run one iteration of refinement and proposal
    pred_dict, skill2operator and skill2tasks are from refinement. 
    replay_buffer, grounded_predicate_dictionary, grounded_skill_dictionary are from task proposal.
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    '''
    for lifted_skill in skill2operator:
        for pred_type in {"precond", "eff"}:
            grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, env)
            for new_lifted_pred in pred_pool:
                hypothetical_pred_list = deepcopy(lifted_pred_list)
                hypothetical_pred_list.append(new_lifted_pred)
                hypothetical_grounded_predicate_truth_value_log = deepcopy(grounded_predicate_truth_value_log)
                hypothetical_grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, hypothetical_pred_list, type_dict, hypothetical_grounded_predicate_truth_value_log, env, skill=lifted_skill)
                add_new_pred = score_by_partition(lifted_skill, hypothetical_grounded_predicate_truth_value_log, tasks, pred_type, type_dict)
                if add_new_pred:
                    logging.info(f"Predicate {new_lifted_pred} added to predicate set by {pred_type} check")
                    lifted_pred_list.append(new_lifted_pred)
                    grounded_predicate_truth_value_log = hypothetical_grounded_predicate_truth_value_log
                    skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, lifted_pred_list)
                    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, env) # udpate for all skills
    
    skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict)

    return skill2operator, lifted_pred_list, grounded_predicate_truth_value_log

@hydra.main(version_base=None, config_path="../hydra_conf", config_name="fm_invent_config")
def main(cfg: DictConfig):
    # init env
    task_config_fpath = f"task_config/{cfg.env}.yaml"
    task_config = load_from_file(task_config_fpath)
    type_dict = {obj: obj_meta['types'] for obj, obj_meta in task_config['objects'].items()}

    log_dir = f"results/fm_invent/{cfg.env}/log/"
    log_save_path = setup_logging(log_dir, cfg.env) # configure logging

    model = GPT4(engine=cfg.model)

    # init skill sequence proposing system
    skill_sequence_proposing = SkillSequenceProposing(task_config_fpath=task_config_fpath)
    
    # main loop
    iter_idx = cfg.iter_idx if cfg.iter_idx else 0
    for i in range(cfg.num_iter):
        if not cfg.invent_pred_only:

            # prepare folder structures, copy from previous iteration if exists
            load_dir = f"results/fm_invent/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}/"
            tasks, skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = load_results(load_dir, task_config)
            save_dir = init_new_iter(cfg.env, cfg.method, cfg.run_idx)

            # propose skill sequence
            tasks: list[Skill] = propose_and_execute(skill_sequence_proposing, tasks, lifted_pred_list, skill2operator, save_dir, cfg)

        else:
            load_dir = f"results/fm_invent/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}_partial/"
            assert os.path.exists(load_dir), "must provide tasks.yaml to start predicate invention."

        if not cfg.skill_seq_only:

            # load partial results copied from previous iteration
            if not cfg.invent_pred_only:
                load_dir = save_dir
            else:
                load_dir = f"results/fm_invent/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}_partial/"
            tasks, skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = load_results(load_dir, task_config)

            # copy and past load dir
            save_dir = f"results/fm_invent/{cfg.env}/runs/{cfg.run_idx}/{iter_idx}"
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