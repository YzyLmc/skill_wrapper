
"""
VLM propose set of predicates and we use the same operator construction method

Example command:
    python baselines/fm_invent.py ++invent_pred_only=True
"""
import logging
from collections import defaultdict
from copy import deepcopy
import os
import sys
sys.path.append(f".") 
from datetime import datetime
import re

import hydra
from omegaconf import DictConfig, OmegaConf
from robotouille.run_skill_sequence import exec_and_record

from src.data_structure import Skill, Predicate
from src.utils import GPT4, load_from_file, save_to_file, setup_logging, clean_logging, save_results, load_results, get_save_fpath, init_new_iter
from src.skill_sequence_proposing import SkillSequenceProposing
from src.invent_predicate import invent_predicates, filter_predicates, calculate_operators_for_all_skill, update_empty_predicates, score_by_partition


def generate_pred_pool(tasks, task_config, lifted_pred_list: list[Predicate], model: GPT4, env, prompt_fpath='prompts/fm_invent.yaml') -> list[Predicate]:
    '''
    Find the latest skill sequence, build the prompt with transitions and task config, and corresponding images and generate a pool of predicates.
    '''
    def get_latest_time(times: list[str]) -> str:
        # Parse each string into datetime
        parsed_times = [
            datetime.strptime(t, "%Y-%m-%d-%H-%M-%S") for t in times
        ]
        # Find the maximum datetime
        latest_time = max(parsed_times)
        # Return it in the same string format
        return latest_time.strftime("%Y-%-m-%-d-%-H-%-M-%-S")
    
    def construct_prompt(prompt: str, transition, skills, objects, lifted_pred_list: list[Predicate]):
        """
        replace placeholders in the prompt
        pred_list :: list of lifted predicates
        """
        placeholders = ["[SKILL_SET]", "[OBJECT_SET]", "[TRANSITIONS]", "[PRED_LIST]"]
        while any([p in prompt for p in placeholders]):
            prompt = prompt.replace("[SKILL_SET]",  '\n'.join(skills))
            prompt = prompt.replace("[OBJECT_SET]",  '\n'.join(objects))
            prompt = prompt.replace("[TRANSITIONS]",  '\n'.join(transition))
            # construct predicate list from pred_dict
            if lifted_pred_list:
                pred_list_str = "\nAlso, you should avoid any synonyms or antonyms of the existing predicates. The existing predicates are:"
                pred_list_str += '\n'.join([f'{str(pred)}: {pred.semantic}' for pred in lifted_pred_list])
                pred_list_str += "\n"
                prompt = prompt.replace("[PRED_LIST]", pred_list_str)
            else:
                prompt = prompt.replace("[PRED_LIST]", "")
        return prompt
    
    # get latest skill sequence
    latest_sequence_key = get_latest_time(list(tasks.keys()))
    latest_sequence = tasks[latest_sequence_key]

    skills = task_config["skills"]
    skills_str = [str(skills[P]) for P in skills]
    skills_str: list[str] = [f"{sk+1}. {skills_str[sk]}" for sk in range(len(skills_str))]

    objects = task_config["objects"]
    objects_str: list[str] = [f"- {O}: {objects[O]['types']}" for O in objects]

    # transition description and images
    transition_str = [f"{i+1}: {latest_sequence[i]['skill']} (Success: {latest_sequence[i]['success']})" for i in range(1,len(latest_sequence))]
    transition_img_list = [latest_sequence[i]["image"] for i in range(len(latest_sequence))]

    prompt = load_from_file(prompt_fpath)[env]
    prompt = construct_prompt(prompt, transition_str, skills_str, objects_str, lifted_pred_list)
    logging.info('Generating predicate pool')
    # resp = model.generate(prompt)[0]
    resp = model.generate_multimodal(prompt, transition_img_list)[0]
    text = resp.split("Predicates:")[1]
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    new_pred_list = []
    i = 0 
    while i < len(lines):
        line = lines[i]
        
        # A predicate line should not start with "types:" or "semantics:"
        if not line.startswith(("types:", "semantics:")):
            name = line
            types = []
            semantic = ""
            
            # Look ahead for types and semantics
            if i + 1 < len(lines) and lines[i+1].startswith("types:"):
                types_line = lines[i+1]
                # extract inside braces { ... }
                types_match = re.search(r"\{(.*?)\}", types_line)
                if types_match:
                    types = [t.strip() for t in types_match.group(1).split(",")]
                i += 1
            
            if i + 1 < len(lines) and lines[i+1].startswith("semantics:"):
                semantic = lines[i+1].replace("semantics:", "").strip()
                i += 1
            
            new_pred_list.append(
                Predicate(name=name, types=types, semantic=semantic)
            )
        
        i += 1
    # parse the parameters from the output string into predicate parameters
    # e.g., "At(obj, loc)"" -> Predicate(name="At", types=["obj", "loc"])
    [print(p, p.semantic) for p in new_pred_list]
    breakpoint()
    return new_pred_list

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
            break

        else:
            logging.warning(f"Execute the plan and collect data on {cfg.env} and save it as (or add to) {os.path.join(save_dir, 'transitions/tasks.yaml')}.")
            logging.warning(f"Then, continue from the breakpoint.")
            breakpoint()
            break

    task_fpath = os.path.join(save_dir, "transitions", "tasks.yaml")
    tasks = load_from_file(task_fpath)

    return tasks

def invent_predicates_for_all_skill(pred_pool: list[Predicate], model, lifted_pred_list, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, env: str):
    '''
    run one iteration of refinement and proposal
    pred_dict, skill2operator and skill2tasks are from refinement. 
    replay_buffer, grounded_predicate_dictionary, grounded_skill_dictionary are from task proposal.
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    '''
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, env)

    for lifted_pred in pred_pool:
        # not predicate with same name
        dup = [pred for pred in lifted_pred_list if pred.name == lifted_pred.name]
        if dup:
            lifted_pred_list.append(lifted_pred)

    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, env)
    skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict)
    filtered_lifted_pred_list = filter_predicates(skill2operator, lifted_pred_list, grounded_predicate_truth_value_log, tasks, type_dict)
    skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, filtered_lifted_pred_list)
    
    return skill2operator, filtered_lifted_pred_list, grounded_predicate_truth_value_log

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
            pred_pool = generate_pred_pool(tasks, task_config, lifted_pred_list, model, cfg.env)
            skill2operator, lifted_pred_list, grounded_predicate_truth_value_log = invent_predicates_for_all_skill(pred_pool, model, lifted_pred_list, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, cfg.env)
 
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