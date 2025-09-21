'''
Get symbolic representation from skill semantic info and observation.
Data structures for logging:
    - read from data:
        - tasks :: dict(task_name: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool)))
            NOTE: step is int starting from 0. init state of the skill is at (step-1), next state is at step. step 0 has no skill
    - produced by skillwrapper
        - grounded_predicate_truth_value_log :: {task_name:{step:PredicateState}}
        - skill2operator :: {lifted_skill: [(LiftedPDDLAction, {pid: int: type: str}, {obj: str, type: str})]}
    '''
from collections import defaultdict
from copy import deepcopy
import itertools
from itertools import permutations, product
import logging
import random
from typing import Union
import sys
sys.path.append('.')

from src.utils import GPT4, load_from_file, raw_prompt
from src.data_structure import Skill, Predicate, PredicateState
from src.RCR_bridge import PDDLState, LiftedPDDLAction, Parameter, RCR_bridge, generate_possible_groundings, unify_transition

def possible_grounded_preds(lifted_pred_list: list[Predicate], type_dict: dict[str, list[str] ]) -> list[Predicate]:
    """
    Generate all possible grounded predicate using the combination of predicates and objects.

    Args:
        lifted_pred_list:: [Predicate]
        type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    Returns:
        grounded_predicates :: list of possible grounded predicates
    """
    # build inverse type_dict
    type_dict_inv = defaultdict(list)
    for param, type_ls in type_dict.items():
        for type  in type_ls:
            type_dict_inv[type].append(param)

    # generate all possible grounded predicates
    grounded_predicates = []
    for pred in lifted_pred_list:
        for params in itertools.product(*[type_dict_inv[p] for p in pred.types]):
            if not len(set(params)) == len(params):  # no duplicate params
                continue
            grounded_predicates.append(pred.ground_with(params, type_dict))
    return grounded_predicates

def calculate_pred_to_update(grounded_predicates: list[Predicate], grounded_skill: Skill) -> list[Predicate]:
    '''
    Given a skill and its parameters, find the set of predicates that need updates

    Args:
        grounded_skill :: skill :: grounded skill
        grounded_predicates::list:: list of grounded predicates, e.g., [{'name': 'At', 'params': ['Apple', 'Table']}]
    '''
    return [gp for gp in grounded_predicates if any([p in gp.params for p in grounded_skill.params]) or len(gp.types) == 0]

# # Not used
# # evaluate an execution using foundation model. Expected acc to be ~ 70%
# def eval_execution(model, skill, consecutive_pair, prompt_fpath='prompts/evalutate_task.txt'):
#     'Get successfulness of the execution given images and the skill name'
#     def construct_prompt(prompt, skill):
#         while "[SKILL]" in prompt:
#                 prompt = prompt.replace("[SKILL]", skill)
#         return prompt
#     prompt = load_from_file(prompt_fpath)
#     prompt = construct_prompt(prompt, skill)
#     return model.generate_multimodal(prompt, consecutive_pair)[0]

def eval_all_predicates(model: GPT4, lifted_pred_list: list[Predicate], type_dict: dict[str, list[str]], img_fpath: str, env, input_modality="image", batched=False) -> PredicateState:
    def get_response(lifted_pred_list, type_dict, img_fpath):
        prompt_1 = load_from_file("prompts/evaluate_pred.yaml")["burger_step_1"]
        object_str = "\n".join([f"{obj}: {types}" for obj, types in type_dict.items()])
        pred_str = "\n".join([f"{str(pred)} : {pred.semantic}" for pred in lifted_pred_list])
        response_1 = raw_prompt(prompt_1.replace("[OBJECTS]", object_str).replace("[PREDICATES]", pred_str), [img_fpath])
        # summarize and correct
        prompt_2 = load_from_file("prompts/evaluate_pred.yaml")["burger_step_2"]
        object_names = ", ".join(list(type_dict.keys()))
        pred_names = ", ".join([pred.name for pred in lifted_pred_list])
        response_2 = raw_prompt(prompt_2.replace("[OBJECT_NAMES]", object_names).replace("[PRED_NAMES]", pred_names).replace("[RESPONSE]", response_1))
        return response_2
    # find all possible groundings of predicates
    grounded_preds = possible_grounded_preds(lifted_pred_list, type_dict)
    predicate_state = PredicateState(grounded_preds)
    if batched:
        valid_resp = False
        i = 0
        while not valid_resp and i < 10:
            resp = get_response(lifted_pred_list, type_dict, img_fpath)
            # parse the summarized response
            response_lines = resp.strip().split("\n")
            true_grounded_preds = []
            try:
                for line in response_lines:
                    if not line.strip():
                        continue
                    grounded_pred = Predicate.from_string(line.strip())
                    grounded_pred.types = [p.types for p in lifted_pred_list if p.name == grounded_pred.name][0]  # assign types
                    logging.info(f'Evaluating predicate {grounded_pred} to be True in {img_fpath}')
                    true_grounded_preds.append(grounded_pred)
                    predicate_state.set_pred_value(grounded_pred, True)
            except:
                breakpoint()
                i += 1
                continue
            valid_resp = True

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

def eval_pred(img: str, grounded_pred: Predicate, model: GPT4, env: str, input_modality: str = "image", prompt_fpath='prompts/evaluate_pred.yaml', log=False) -> bool:
    '''
    evaluate truth value of a predicate using a dictionary of parameters
    init step and later steps use different prompts. hardcoded.

    Args:
        grounded_pred::dict:: grounded predicate with parameter type, e.g., {'name':"At", 'params':["location"]}
    '''

    def construct_prompt(prompt, grounded_pred):
        "replace placeholders in the prompt"
        # Predicate might have parameters don't belong to the skill
        place_holders = ['[GROUNDED_PRED]','[LIFTED_PRED]', '[SEMANTIC]']
        while any([p in prompt for p in place_holders]):
            prompt = prompt.replace('[GROUNDED_PRED]', str(grounded_pred))
            prompt = prompt.replace('[LIFTED_PRED]', str(grounded_pred.lifted()))
            prompt = prompt.replace('[SEMANTIC]', grounded_pred.semantic)
        return prompt
    
    if input_modality == "text":
        raise NotImplementedError("Text input modality is not implemented yet.")
    
    prompt = load_from_file(prompt_fpath)[env]
    prompt = construct_prompt(prompt, grounded_pred)

    model.engine = "gpt-5-nano"
    resp = model.generate_multimodal(prompt, [img])[0]
    result = True if "True" in resp.split('\n')[-1] else False

    if log:
        logging.info(f'Prompt:\n{prompt}')
        logging.info(f'Model response: {resp}')

    logging.info(f'{grounded_pred} evaluated to `{result}` in {img}')
    return result

def generate_pred(image_pair: list[str], grounded_skills: list[Skill], successes: list[bool], lifted_pred_list: list[Predicate], pred_type: str, model: GPT4, env, type_dict, skill2tried_pred={}, prompt_fpath='prompts/predicate_invention.yaml') -> Predicate:
    '''
    propose new predicates based on the contrastive pair.
    '''
    def construct_prompt(prompt: str, grounded_skills, successes, lifted_pred_list: list[Predicate], tried_pred: list[Predicate]):
        """
        replace placeholders in the prompt
        pred_list :: list of lifted predicates
        """
        placeholders = ["[LIFTED_SKILL]", "[PARAMETERS]", "[GROUNDED_SKILL_1]", "[GROUNDED_SKILL_2]", "[SUCCESS_1]", "[SUCCESS_2]", "[PRED_LIST]"]
        while any([p in prompt for p in placeholders]):
            prompt = prompt.replace("[LIFTED_SKILL]",  str(grounded_skills[0].lifted()))
            prompt = prompt.replace("[PARAMETERS]",  str(grounded_skills[0].types))
            prompt = prompt.replace("[GROUNDED_SKILL_1]",  str(grounded_skills[0]))
            prompt = prompt.replace("[GROUNDED_SKILL_2]",  str(grounded_skills[1]))
            prompt = prompt.replace("[SUCCESS_1]",  "succeeded" if bool(successes[0]) else "failed")
            prompt = prompt.replace("[SUCCESS_2]",  "succeeded" if bool(successes[1]) else "failed")
            # construct predicate list from pred_dict
            if lifted_pred_list:
                pred_list_str = "Avoid duplicates or near-duplicates of existing predicates. Existing predicates:\n"
                pred_list_str += '\n'.join([f'{str(pred)}: {pred.semantic}' for pred in lifted_pred_list])
                pred_list_str += "\n"
            else:
                pred_list_str = ""
            prompt = prompt.replace("[PRED_LIST]", pred_list_str)
            if tried_pred:
                tried_pred_str = "Avoid the following predicates that are previously proposed but rejected:\n"
                tried_pred_str += '\n'.join(", ".join([str(pred) for pred in tried_pred]))
            else:
                tried_pred_str = ""
            prompt = prompt.replace("[TRIED_PRED]", tried_pred_str)
        return prompt

    all_params = set(sum([ types for name, types in type_dict.items()], []))
    tried_pred = skill2tried_pred[grounded_skills[0].lifted()] if skill2tried_pred else []
    prompt = load_from_file(prompt_fpath)[env][pred_type]
    prompt = construct_prompt(prompt, grounded_skills, successes, lifted_pred_list, tried_pred)
    assert len(image_pair)==4 if pred_type=="eff" else len(image_pair)==2, "precondition need 2 images while effect need 4"
    logging.info('Generating predicate')
    # resp = model.generate(prompt)[0]
    model.engine = "gpt-5"
    resp = model.generate_multimodal(prompt, image_pair)[0]
    pred, sem = resp.split('\n')[-1].split(': ', 1)[0].strip('`'), resp.split(': ', 1)[1].strip()
    # parse the parameters from the output string into predicate parameters
    # e.g., "At(obj, loc)"" -> Predicate(name="At", types=["obj", "loc"])
    new_pred = Predicate(pred.split("(")[0], pred.split("(")[1].strip(")").split(", ")) # lifted
    new_pred.semantic = sem
    # breakpoint()
    return new_pred

# Adding to precondition or effect are different prompts
def update_empty_predicates(model, tasks: dict, lifted_pred_list: list[Predicate], type_dict, grounded_predicate_truth_value_log, env, skill: Skill = None):
    '''
    Find the grounded predicates with missing values and evaluate them.
    The grounded predicates are evaluated from the beginning to the end, and then lifted to the lifted predicates.

    Args:
        lifted_pred_list::list(Predicate):: List of all lifted predicates
        grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
        tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
        type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    Returns:
        grounded_predicate_truth_value_log
    '''
    # NOTE: step is a integer ranging from 0-?, where 0 is the init step and success==None. 1- are states after executions

    # look for predicates that haven't been evaluated
    # The truth values could be missing if:
    #    1. the predicate is newly added (assuming all possible grounded predicates are added, including the init step)
    #    2. a task is newly executed
    # NOTE: the dictionary could be partially complete because some truth values will be directly reused from the scoring function
    # generate all possible grounded predicates that match object types
    grounded_pred_list = possible_grounded_preds(lifted_pred_list, type_dict)
    # update if there are new tasks
    for task_id, steps in tasks.items():
        new_task = False
        if task_id not in grounded_predicate_truth_value_log:
            new_task = True
            grounded_predicate_truth_value_log[task_id] = {}
            for step in steps:
                grounded_predicate_truth_value_log[task_id][step] = PredicateState(grounded_pred_list)
        
        if not grounded_pred_list: # empty predicate list
            continue
        
        if new_task:
            for step, state in steps.items():

                skill_curr = tasks[task_id][step]["skill"]
                new_pred_added = grounded_predicate_truth_value_log[task_id][step].get_unevaluated_preds() # all are empty

                assert new_pred_added == grounded_pred_list, "All predicates should be new in a new task"

                if step == 0: # must evaluate first step
                    pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, state["image"], env, batched=True)
                    grounded_predicate_truth_value_log[task_id][step] = pred_state
                elif not state["success"]: # failed exec, copy from last step
                    grounded_predicate_truth_value_log[task_id][step] = deepcopy(grounded_predicate_truth_value_log[task_id][step-1])
                    logging.info(f'Skill {str(skill_curr)} failed at step {step}, copy the predicate state from last step')
                elif state["success"]:
                    # if only the skill arguments intersect with any predicates arguments, or there's a nullary predicate, then eval
                    new_params = set(sum([list(pred.params) for pred in new_pred_added], []))
                    skill_params = set(skill_curr.params)

                    nullary_pred_list = [pred for pred in new_pred_added if not pred.types]

                    if (new_params & skill_params) or nullary_pred_list: # only eval if there's intersection or nullary preds
                        pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, state["image"], env, batched=True)
                        grounded_predicate_truth_value_log[task_id][step] = pred_state
                    else: # copy from last step
                        grounded_predicate_truth_value_log[task_id][step] = deepcopy(grounded_predicate_truth_value_log[task_id][step-1])
                        logging.info(f'Skill {str(skill_curr)} has no overlapping parameters with new predicates and there\'s no nullary predicates, copy the predicate state from last step')

        elif not new_task: # updating existing task

            if not skill: # update for all skills

                for step, state in steps.items():
                    
                    skill_curr = tasks[task_id][step]["skill"]
                    new_pred_added: list[Predicate] = grounded_predicate_truth_value_log[task_id][step].add_pred_list(grounded_pred_list)

                    if new_pred_added: # new predicates added

                        if step == 0: # must evaluate first step
                            pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, state["image"], env, batched=True)
                            grounded_predicate_truth_value_log[task_id][step] = pred_state

                        elif not state["success"]: # failed exec, copy from last step
                            grounded_predicate_truth_value_log[task_id][step] = deepcopy(grounded_predicate_truth_value_log[task_id][step-1])
                            logging.info(f'Skill {str(skill_curr)} failed, copy the predicate state from last step')
                        
                        elif state["success"]: # 1+ step, sucess exec, eval
                            # if only the skill arguments intersect with any predicates arguments, or there's a nullary predicate, then eval
                            new_params = set(sum([list(pred.params) for pred in new_pred_added], []))
                            skill_params = set(skill_curr.params)

                            nullary_pred_list = [pred for pred in new_pred_added if not pred.types]
                            if (new_params & skill_params) or nullary_pred_list: # only eval if there's intersection or nullary preds
                                pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, state["image"], env, batched=True)
                                grounded_predicate_truth_value_log[task_id][step] = pred_state
                            else:
                                grounded_predicate_truth_value_log[task_id][step] = deepcopy(grounded_predicate_truth_value_log[task_id][step-1])
                                logging.info(f'Skill {str(skill_curr)} has no overlapping parameters with new predicates, copy the predicate state from last step')

                    elif not new_pred_added: # no new predicate added, could only happen when copying from a hypothetical grounded pred value dict with missing values
                        
                        # only need to update missing predicates
                        for step, state in steps.items():

                            pred_to_update = grounded_predicate_truth_value_log[task_id][step].get_unevaluated_preds()
                            skill_curr = tasks[task_id][step]["skill"]

                            if not pred_to_update:
                                continue
                            elif step == 0: # must eval init step
                                pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, state["image"], env, batched=True)
                                grounded_predicate_truth_value_log[task_id][step] = pred_state
                            elif not state["success"]: # failed exec, copy from last step
                                grounded_predicate_truth_value_log[task_id][step] = deepcopy(grounded_predicate_truth_value_log[task_id][step-1])
                                logging.info(f'Skill failed, copy the predicate state from last step')
                            elif state["success"]: # 1+ step, sucess exec, eval
                                new_params = set(sum([list(pred.params) for pred in pred_to_update], []))
                                skill_params = set(skill_curr.params)

                                nullary_pred_list = [pred for pred in pred_to_update if not pred.types]
                                if (new_params & skill_params) or nullary_pred_list: # only eval if there's intersection or nullary preds
                                    pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, state["image"], env, batched=True)
                                    grounded_predicate_truth_value_log[task_id][step] = pred_state
                                else:
                                    grounded_predicate_truth_value_log[task_id][step] = deepcopy(grounded_predicate_truth_value_log[task_id][step-1])
                                    logging.info(f'Skill {str(skill_curr)} has no overlapping parameters with new predicates, copy the predicate state from last step')

            elif skill: # for a specific lifted skill, only update state before and after the skill execution
                    assert not skill.params, "only lifted skills are allowed"
                    for step, state in steps.items():

                        new_pred_added: list[Predicate] = grounded_predicate_truth_value_log[task_id][step].add_pred_list(grounded_pred_list)

                        if new_pred_added: # new predicates added

                            # don't need to eval the first step

                            # if step == 0: # must evaluate first step
                            #     pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, state["image"], env, batched=True)
                            #     grounded_predicate_truth_value_log[task_id][step] = pred_state
                            if state["skill"]:
                                if state["skill"].lifted() == skill: # eval before and after the skill execution
                                    last_state = steps[step-1]
                                    last_pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, last_state["image"], env, batched=True)
                                    grounded_predicate_truth_value_log[task_id][step-1] = last_pred_state

                                    pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, state["image"], env, batched=True)
                                    grounded_predicate_truth_value_log[task_id][step] = pred_state
                            
                        else:
                            raise Exception("This shouldn't happen.")

        # # for each step, iterate through all steps and find empty predicates and update them
        # # calculate predicates to update based on the last action every step after init

        # # update predicates for all states 
        # for step, state in steps.items():
        #     new_pred_added: list[Predicate] = grounded_predicate_truth_value_log[task_id][step].add_pred_list(grounded_pred_list)

        #     # if no new predicates, skip
        #     if not new_pred_added:
        #         continue
            
        #     # union set of types that the newly added predicates have
        #     new_types = set(sum([pred.types for pred in new_pred_added], []))
        #     nullary_pred_list = [pred for pred in new_pred_added if not pred.types]
        #     # if the skill doesn't have any of the new types, skip
        #     if not step == 0:
        #         if skill and not (new_types & set(skill.types)):
        #             # grounded_predicate_truth_value_log[task_id][step] = deepcopy(grounded_predicate_truth_value_log[task_id][step-1])
        #             continue

        #     if not step == 0 and state["success"] == False:
        #         # copy and paste from last step if the skill failed
        #         grounded_predicate_truth_value_log[task_id][step] = deepcopy(grounded_predicate_truth_value_log[task_id][step-1])
        #         logging.info(f'Skill {str(state["skill"])} failed, copy the predicate state from last step')
        #         continue

        #     # 1. find states need to be eval or re-eval
        #     # At init step only evaluate empty ones
        #     # assuming the skill execution can only change the predicates with parameters overlapping with the skill
        #     pred_to_update = grounded_predicate_truth_value_log[task_id][step].get_unevaluated_preds() if step == 0 \
        #         else calculate_pred_to_update(grounded_pred_list, state["skill"])

        #     if pred_to_update:
        #         pred_state = eval_all_predicates(model, lifted_pred_list, type_dict, state["image"], env, batched=True)
        #         grounded_predicate_truth_value_log[task_id][step] = pred_state
            
        #     # # 2. re-eval grounded predicates
        #     # for grounded_pred in pred_to_update:
        #     #     # only update empty predicates
        #     #     if grounded_predicate_truth_value_log[task_id][step].get_pred_value(grounded_pred) == None:

        #     #         truth_value = eval_pred(state["image"], grounded_pred, model, env)
        #     #         grounded_predicate_truth_value_log[task_id][step].set_pred_value(grounded_pred, truth_value)
        #     # # 3.copy all empty predicates from previous state
        #     #     elif not step == 0: # if is a non-init state, update the predicates
        #     #         if (new_task) or (not new_task and grounded_predicate_truth_value_log[task_id][step].get_pred_value(grounded_pred) == None):
        #     #             truth_value = eval_pred(state["image"], grounded_pred, model, env)
        #     #             grounded_predicate_truth_value_log[task_id][step].set_pred_value(grounded_pred, truth_value)

        #     # unevaluated_pred: list[Predicate] = grounded_predicate_truth_value_log[task_id][step].get_unevaluated_preds()
        #     # # if not skill:
        #     # #     assert (unevaluated_pred==[]) if (step==0) else True, "Step 0 shouldn't have any predicate unevaluated"
        #     # for grounded_pred in unevaluated_pred:
        #     #     # fetch truth value from last state
        #     #     truth_value = grounded_predicate_truth_value_log[task_id][step-1].get_pred_value(grounded_pred)
        #     #     grounded_predicate_truth_value_log[task_id][step].set_pred_value(grounded_pred, truth_value)

    return grounded_predicate_truth_value_log

def grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks, success_only: bool=False):
    '''
    helper function to convert grounded predicate log into skill2task2state for predicate invention.

    Args:
        grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
        tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
        pred_type :str: {"precond", "eff"}
    Returns:
        skill2task2state::{grounded_skill: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}}
            task_step_tuple :: (task_name : str, step : int)
    '''
    skill2task2state: dict[Skill, dict[tuple, dict[str, Union[list[PredicateState, PredicateState], bool]]]] = defaultdict(dict)
    for task_name, steps in grounded_predicate_truth_value_log.items():
            for step, state in steps.items(): # state :: PredicateState class
                if not step == 0: # init state has no skill, and thus won't be included in (task_name, step)
                    grounded_skill = tasks[task_name][step]["skill"]
                    task_step_tuple: tuple[str, int] = (task_name, step)
                    if success_only and tasks[task_name][step]['success']:
                        skill2task2state[grounded_skill][task_step_tuple] = {'states':[last_state, state], 'success': tasks[task_name][step]['success']}
                    elif not success_only:
                        skill2task2state[grounded_skill][task_step_tuple] = {'states':[last_state, state], 'success': tasks[task_name][step]['success']}

                last_state = deepcopy(state)
    return skill2task2state

def in_alpha(possible_groundings, transition: list[PredicateState, PredicateState], grounded_skill, operator, type_dict, pred_type: str) -> bool:
    """
    Util function for detect_mismatch and score_by_partition:
    There exist a grounding such that the grounded state agree with the operator's precondition/effect
    """
    if not possible_groundings: # no valud groundings
        return False
    
    # also index all other objects if any
    for i, possible_grounding in enumerate(possible_groundings):
        obj_set = set()
        for state in transition:
            for pred in state.iter_predicates():
                obj_set.update(pred.params)
        remaining_obj_set = obj_set - set(possible_grounding.values())
        if remaining_obj_set:
            starting_idx = max(possible_grounding.keys()) + 1
            additional_possible_grounding = {i: obj for i, obj in zip(range(starting_idx, starting_idx + len(remaining_obj_set)), remaining_obj_set)}
            possible_groundings[i] = possible_grounding | additional_possible_grounding

    for grounding in possible_groundings:
        bridge = RCR_bridge()

        unified_transition, _ = unify_transition(transition, grounded_skill, type_dict)

        # map objects to lifted parameters
        pddl_state_list = [bridge.predicatestate_to_pddlstate(state, grounding) for state in unified_transition]
        param_name2param_object = {str(param): Parameter(param.pid, param.type, grounding[int(str(param).split("_p")[-1])]) for param in operator.parameters if not str(param).startswith("_")}
        for param_name, param in param_name2param_object.items(): param_name2param_object[param_name].pid = str(param).split("_p")[-1]
        param_name2param_object |= {'_p-1': Parameter(None, "", None)}
        grounded_operator = operator.get_grounded_action(param_name2param_object, 0) # don't know if action_id matters
        if pred_type == "precond":
            applicable = grounded_operator.check_applicability(pddl_state_list[0])
            if applicable:
                return True
        elif pred_type == "eff":
            if not (grounded_operator.effect.add_set or grounded_operator.effect.delete_set): # operator empty effect
                return True
            
            eff_add = pddl_state_list[1].true_set - pddl_state_list[0].true_set
            eff_del = pddl_state_list[1].false_set - pddl_state_list[0].false_set
            # assume skills cause state changes to the env
            # exploring the space of s' require stochastic failure mode
            if  eff_add == grounded_operator.effect.add_set and eff_del == grounded_operator.effect.delete_set \
                    and (eff_add or eff_del): # effect is not empty
                return True
    return False

def detect_mismatch(lifted_skill: Skill, skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, pred_type: str) -> list[list[tuple, tuple]]:
    """
    Find mismatch state pairs where they both belong to Union Precondition or Effect.

    Args:
        skill2operator :: {lifted_skill: [(LiftedPDDLAction, {pid: int: type: str}, {obj:str, type:str})]}
        grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
        tasks :: dict(task_name: (step: dict("skill": grounded_skill, 'image':img_path, 'success': bool))) ::
        pred_type::{'precond', 'eff'}
    Returns:
        mismatch_pairs :: [[task_step_tuple, task_step_tuple]...]
    """
    skill2task2state = grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks)
    # All grounded skills
    task2in_alpha: dict[str, bool] = {} # alpha is the union of grounding of precondition or effect of operators corresponding to one skill
    task2success: dict[str, bool] = {}
    task_step_tuple_list = []
    for grounded_skill, task2state in skill2task2state.items():
        # evaluate across all grounded skills of the same name and type
        if grounded_skill.lifted() == lifted_skill:

            # task2state :: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}
            for task_step_tuple, transition_meta in task2state.items():
                task_step_tuple_list.append(task_step_tuple)

                task2success[task_step_tuple] = transition_meta['success']
                state_in_alpha = False
                # first iteration when no operators set to true so we invent
                if skill2operator[lifted_skill] is None:
                    state_in_alpha = True

                else:
                    for operator, skill_param2pid in skill2operator[lifted_skill]:
                        possible_groundings = generate_possible_groundings(operator, grounded_skill, skill_param2pid, type_dict)
                        if in_alpha(possible_groundings, transition_meta["states"], grounded_skill, operator, type_dict, pred_type):
                            state_in_alpha = True
                            break

                task2in_alpha[task_step_tuple] = state_in_alpha

    assert len(task2in_alpha) == len(task2success), "length of both dictionaries state2in_alpha and state2success must equal"
    # looking for pairs of state where truth value of s1 and s2 agree in state2in_alpha but conflict in state2success
    mismatched_pairs = []
    for i in range(len(task2state)):
        for j in range(i + 1, len(task_step_tuple_list)):
            task_step_tuple_1, task_step_tuple_2 = task_step_tuple_list[i], task_step_tuple_list[j]
            if task2in_alpha[task_step_tuple_1] == task2in_alpha[task_step_tuple_2] and task2success[task_step_tuple_1] != task2success[task_step_tuple_2]:
                mismatched_pairs.append([task_step_tuple_1, task_step_tuple_2])
    return mismatched_pairs   

def invent_predicate_one(mismatch_pair: list[tuple, tuple], model: GPT4, lifted_skill: Skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type, env, skill2triedpred=defaultdict(list), threshold={"precond":0.6, "eff":0.6}) -> Predicate:
    """
    One iteration of predicate invention.

    Args:
        mismatch_pair :: two task step tuples that triggered predicate invention
    """
    # task_step_tuple :: tuple[str, int]
    task_0, index_0 = mismatch_pair[0]
    task_1, index_1 = mismatch_pair[1]
    state_0 = tasks[task_0][index_0]
    state_1 = tasks[task_1][index_1]
    image_0 = tasks[task_0][index_0 - 1]["image"]
    image_1 = tasks[task_1][index_1 - 1]["image"]
    if pred_type == "precond":
        image_pair = [image_0, image_1]
    elif pred_type == "eff":
        image_0_1 = state_0["image"]
        image_1_1 = state_1["image"]
        image_pair = [image_0, image_0_1, image_1, image_1_1]
    assert state_0['skill'] is not None and state_1['skill'] is not None, "Never use the first steps. They are empty!"

    logging.info("Inventing predicates for two transitions:\n")
    logging.info(f"1. task:{task_0}, step: {index_0}, skill: {str(state_0['skill'])}, success: {state_0['success']}\nstate before:\n{str(grounded_predicate_truth_value_log[task_0][index_0-1])}\nstate after:\n{str(grounded_predicate_truth_value_log[task_0][index_0])}")
    logging.info(f"2. task:{task_1}, step: {index_1}, skill: {str(state_1['skill'])}, success: {state_1['success']}\nstate before:\n{str(grounded_predicate_truth_value_log[task_1][index_1-1])}\nstate after:\n{str(grounded_predicate_truth_value_log[task_1][index_1])}")
    t = 0
    while True and t < 3:
        t += 1
        new_pred = generate_pred(image_pair,
                                [state_0["skill"], state_1["skill"]],
                                [state_0["success"], state_1["success"]],
                                lifted_pred_list, pred_type, model, env, type_dict, skill2triedpred)
        breakpoint()
        if new_pred: # valid predicate generated
            break
    try:
        logging.info(f"Generated new predicate {new_pred}: {new_pred.semantic}")
    except:
        logging.info(f"Generated new predicate {new_pred} but failed to parse")
        return lifted_pred_list, skill2triedpred, False, grounded_predicate_truth_value_log
    
    if len(new_pred.types) > 2:
        logging.info(f"Predicate {new_pred} is NOT added to predicate set because contain more then 2 parameters")
        skill2triedpred[lifted_skill].append(new_pred)
        return lifted_pred_list, skill2triedpred, False, grounded_predicate_truth_value_log
    # elif new_pred in lifted_pred_list or new_pred in skill2triedpred[lifted_skill]:
    #     logging.info(f"Predicate {new_pred} is already in the predicate set or tried before.")
    #     return lifted_pred_list, skill2triedpred, False, grounded_predicate_truth_value_log
    
    new_pred_accepted = False
    # evaluate the new predicate on all states
    # suppose we add the new predicate to the current predicate set
    hypothetical_pred_list = deepcopy(lifted_pred_list)
    hypothetical_pred_list.append(new_pred)
    hypothetical_grounded_predicate_truth_value_log = deepcopy(grounded_predicate_truth_value_log)
    # task unchanged, only add candidate predicate
    hypothetical_grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, hypothetical_pred_list, type_dict, hypothetical_grounded_predicate_truth_value_log, env, skill=lifted_skill)
    # breakpoint()
    add_new_pred = score_by_partition(lifted_skill, hypothetical_grounded_predicate_truth_value_log, tasks, pred_type, type_dict, threshold)
    if add_new_pred:
        logging.info(f"Predicate {new_pred} added to predicate set by {pred_type} check")
        lifted_pred_list.append(new_pred)
        grounded_predicate_truth_value_log = hypothetical_grounded_predicate_truth_value_log
        grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, env) # udpate for all skills
        new_pred_accepted = True
    else:
        logging.info(f"Predicate {new_pred} is NOT added to predicate set by {pred_type} check")
        skill2triedpred[lifted_skill].append(new_pred)
    
    return lifted_pred_list, skill2triedpred, new_pred_accepted, grounded_predicate_truth_value_log

def invent_predicates(model: GPT4, lifted_skill: Skill, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, env, skill2triedpred=defaultdict(list), max_t=3,):
    '''
    Main loop of generating predicates.
    Invent one pred for precondition and one for effect.
    '''
    # check precondition first
    t = 0
    pred_type = "precond"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, env)
    mismatch_pairs = detect_mismatch(lifted_skill, skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, pred_type=pred_type)
    logging.info(f"About to enter precondition check of skill {lifted_skill}")
    new_pred_accepted = False
    while mismatch_pairs and t < max_t:
        # Always solve the first mismatch pair
        lifted_pred_list, skill2triedpred, new_pred_accepted, grounded_predicate_truth_value_log = invent_predicate_one(random.choice(mismatch_pairs), model, lifted_skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type, env, skill2triedpred=skill2triedpred)
        if new_pred_accepted: break
        t += 1
    
    if mismatch_pairs and new_pred_accepted:
        skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict)

    # check effect
    t = 0
    pred_type = "eff"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, env)
    mismatch_pairs = detect_mismatch(lifted_skill, skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, pred_type=pred_type)
    logging.info(f"About to enter effect check of skill {lifted_skill}")
    new_pred_accepted = False
    while mismatch_pairs and t < max_t:
        lifted_pred_list, skill2triedpred, new_pred_accepted, grounded_predicate_truth_value_log = invent_predicate_one(random.choice(mismatch_pairs), model, lifted_skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type, env, skill2triedpred=skill2triedpred)
        if new_pred_accepted: break
        t += 1
    
    if mismatch_pairs and new_pred_accepted:
        skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict)

    logging.info(f"Done inventing predicates for skill {str(lifted_skill)}")
    # Not flushing tried predicate cache right now
    return skill2operator, lifted_pred_list, skill2triedpred, grounded_predicate_truth_value_log

def score_by_partition(lifted_skill: Skill, hypothetical_grounded_predicate_truth_value_log, tasks, pred_type: str, type_dict, threshold:  dict[str, float]={"precond":0.5, "eff":0.5}) -> bool:
    '''
    New scoring function that use the predicate invention condition
    Calculate hypothetical operators and then check

    This function is largely taken from detect mismatch
    '''
    # calculate hypotehtical operators
    hypothetical_skill2task2state = grounded_pred_log_to_skill2task2state(hypothetical_grounded_predicate_truth_value_log, tasks, success_only=False)
    hypothetical_skill2task2state_success = grounded_pred_log_to_skill2task2state(hypothetical_grounded_predicate_truth_value_log, tasks, success_only=True)
    skill2partition = partition_by_lifted_effect(hypothetical_skill2task2state_success, type_dict, skill=lifted_skill)

    hypothetical_operators = create_operators_from_partitions(lifted_skill, hypothetical_skill2task2state_success, hypothetical_skill2task2state, skill2partition, type_dict)

    # if the new operators can make sure fail execution outside alpha and successful execution inside alpha

    task_num = 0
    score = 0

    # for both success and failed tasks
    for grounded_skill, task2state in hypothetical_skill2task2state.items():
        # evaluate across all grounded skills of the same name and type
        if grounded_skill.lifted() == lifted_skill:
            # task2state :: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}
            for task_step_tuple, transition_meta in task2state.items():
                state_in_alpha = False
                # first iteration when no operators set to true so we invent
                assert hypothetical_operators, "There must be at least one operator learned"
                for operator, skill_param2pid in hypothetical_operators:
                    possible_groundings = generate_possible_groundings(operator, grounded_skill, skill_param2pid, type_dict)
                    # if not possible_groundings: breakpoint()
                    if in_alpha(possible_groundings, transition_meta["states"], grounded_skill, operator, type_dict, pred_type):
                        state_in_alpha = True
                        break
                # print(task_step_tuple, state_in_alpha, transition_meta['success'])
                # print(transition_meta["states"][0],'\n')
                # print(transition_meta["states"][1],'\n')
                # breakpoint()
                score += 1 if state_in_alpha == transition_meta['success'] else 0
                task_num += 1

    result = True if score/task_num > threshold[pred_type] else False
    logging.info(f"Predicate is {'' if result else 'not '}added for {pred_type}. Score = {score/task_num}")
    return result

# def score_by_partition_final(new_pred_lifted: Predicate, lifted_skill: Skill, skill2task2state, pred_type: str, threshold: dict[str, float]) -> bool:
#     '''
#     Partition by termination and effect and then score the predicates across each partition

#     Args:
#         skill :: grouded skill {"name":"PickUp", "types":["obj"], "params":["Apple"]}
#         threshold={"precond":float, "eff":float}
#     '''
#     # 1. find all states after executing the same grounded skill
#     _, _, skill2partition = partition_by_termination_n_eff(skill2task2state)

#     # 2. evaluate the score across all grounded skill of the lifted skill, return true if only one partition makes the score higher than threshold
#     for grounded_skill_outer, partitions in skill2partition.items():
#         for task_step_tuple_list in partitions:
#             for grounded_skill_inner, task2state in skill2task2state.items():
#                 if grounded_skill_outer == grounded_skill_inner and grounded_skill_outer.lifted() == lifted_skill:
#                     partitioned_task2state = {task_step_tuple: task2state[task_step_tuple] for task_step_tuple in task_step_tuple_list}
#                     new_pred_grounded = Predicate.ground_with(new_pred_lifted, grounded_skill_inner.params)
#                     t_score_t, f_score_t, t_score_f, f_score_f = score(new_pred_grounded, partitioned_task2state, pred_type)
#                     if pred_type == "precond":
#                         if (t_score_t > threshold[pred_type] or f_score_t > threshold[pred_type]) \
#                             or (t_score_f > threshold[pred_type] or f_score_f > threshold[pred_type]):
#                             return True
                        
#                     elif pred_type == "eff":
#                         if (t_score_t > threshold[pred_type] or f_score_t > threshold[pred_type]) \
#                             or (t_score_f > threshold[pred_type] or f_score_f > threshold[pred_type]):
#                             return True
#     return False

def calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, filtered_lifted_pred_list:list[Predicate]=None,):
    # partitioning
    # 1. partition by different termination and effect, success task only
    skill2task2state = grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks, success_only=True)

    skill2task2state_all = grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks, success_only=False)

    if filtered_lifted_pred_list: # when we perform final filtering after all iterations
        for grounded_skill, task2state in skill2task2state.items():
            for task_step_tuple, transition_meta in task2state.items():
                transition = transition_meta['states']
                transition[0].keep_pred_list(filtered_lifted_pred_list)
                transition[1].keep_pred_list(filtered_lifted_pred_list)

        for grounded_skill, task2state in skill2task2state_all.items():
            for task_step_tuple, transition_meta in task2state.items():
                transition = transition_meta['states']
                transition[0].keep_pred_list(filtered_lifted_pred_list)
                transition[1].keep_pred_list(filtered_lifted_pred_list)

    # _, _, skill2partition = partition_by_termination_n_eff(skill2task2state)
    skill2partition = partition_by_lifted_effect(skill2task2state, type_dict)
    # 2. create one operator for each partition
    for lifted_skill in skill2operator:
        # if "Pick" in str(lifted_skill):
        #     breakpoint()
        skill2operator[lifted_skill] = create_operators_from_partitions(lifted_skill, skill2task2state, skill2task2state_all, skill2partition, type_dict)

    return skill2operator

def filter_predicates(skill2operator, lifted_pred_list: list[Predicate], grounded_predicate_truth_value_log, tasks, type_dict, threshold={"precond":0.5, "eff":0.5}) -> list[Predicate]:
    """
    After running all iterations in main function, score all predicates again
    This function will only be called in main.
    """
    filtered_lifted_pred_list = []
    for lifted_pred in lifted_pred_list:
        for lifted_skill in skill2operator:
            if not skill2operator[lifted_skill]: continue
            for pred_type in ['precond', 'eff']:
                add_new_pred = score_by_partition(lifted_skill, grounded_predicate_truth_value_log, tasks, pred_type, type_dict, threshold)
                if add_new_pred and lifted_pred not in filtered_lifted_pred_list:
                    filtered_lifted_pred_list.append(lifted_pred)
    
    return filtered_lifted_pred_list

def partition_by_lifted_effect(skill2task2state, type_dict, skill=None) -> dict:
    '''
    Partition the a set of transitions using  their lifted effect set. Will be used again in scoring and final operators learning.
    Only successful execution will be used for partitioning.

    Args:
        skill2task2state :: {grounded_skill: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}}
        type_dict :: {str: str} :: {object: type}
    Returns:
        skill2partition :: {lifted_skill: [[task_step_tuple, task_step_tuple]]}
    '''
    def all_mappings(a , b, dedup: bool = True) -> list[dict[any, any]]:
        """
        Return all possible bijective mappings from elements in `a` to elements in `b`.
        Assumes len(a) == len(b). If `dedup` is True, duplicate maps from
        values in `b` are removed.
        """
        a = list(a)
        b = list(b)
        if len(a) != len(b):
            raise ValueError("Lists must have the same length for bijective mappings.")

        out = []
        if dedup:
            seen = set()
            for perm in permutations(b, len(b)):
                if perm in seen:
                    continue
                seen.add(perm)
                out.append(dict(zip(a, perm)))
        else:
            for perm in permutations(b, len(b)):
                out.append(dict(zip(a, perm)))
        return out

    def beam_grounding(abstract_state_list: list[list[tuple[Predicate, int]]], param, grounding_tuple: tuple[Predicate, int, int], obj_type: str) -> list[list[Predicate]]:
        """
        Try grounding one parameter of each abstract state, without violating the type constraint and truth value.
        If there are multiple possible groundings, return all of them, to account for multiple grounded predicates with same lifted form.

        Args:
            abstract_state_list :: list[list[tuple[Predicate, int]]] :: a list of abstract states in list of predicates, predicates can be partially grounded, truth value can be None
            grounding_tuple :: tuple[Predicate, int, bool] :: (predicate to ground, index of the parameter to ground, target truth value)
            type_list :: list[str] :: list of types for the parameter to ground
        """
        assert abstract_state_list, "abstract_state_list cannot be empty"

        new_abstract_state_list = []
        pred_to_ground, param_index, truth_change = grounding_tuple

        for abstract_state in abstract_state_list:
            new_state = []
            # if there are multiple predicates with the same lifted form, we create copies and try all of them
            pred_with_same_lifted_form = []
            for pred, change in abstract_state:
                if pred.name == pred_to_ground.name:
                    pred_with_same_lifted_form.append((pred, change))
                else:
                    new_state.append((pred, change))

            for i, (pred, change) in enumerate(pred_with_same_lifted_form):
                # make a copy of the state without the predicate to ground
                abstract_state_to_ground = deepcopy(new_state)
                pred_copy = deepcopy(pred)
                # try grounding the predicate
                # check type
                required_type = pred_copy.types[param_index]
                if not required_type == obj_type:
                    continue

                # check if pred's param is already grounded
                if not pred_copy.params:
                    pred_copy.params = [None] * len(pred_copy.types)

                if pred_copy.params[param_index] is not None and pred_copy.params[param_index] != param:
                    continue

                # check truth value
                if change is not None and change != truth_change:
                    continue

                # check is passed, ground this specific predicate, leave all other predicates unchanged

                pred_copy.params[param_index] = param
                abstract_state_to_ground.append((pred_copy, truth_change))

                # add back all other predicates in pred_with_same_lifted_form
                for j, (other_pred, other_change) in enumerate(pred_with_same_lifted_form):
                    if j != i:
                        abstract_state_to_ground.append((other_pred, other_change))
                new_abstract_state_list.append(abstract_state_to_ground)
        return new_abstract_state_list

    def equal_lifted_effect(value_tuple_1, value_tuple_2, grounded_skill_1, grounded_skill_2) -> bool:
        """
        Check if two value tuples share the same lifted effect.
        This function considers multi-type objects, multiple same lifted predicates with different parameters.
        
        Args:
            value_tuple_1 :: tuple[Predicate, int]
            value_tuple_2 :: tuple[Predicate, int]
        """
        # print("skill 1:", grounded_skill)
        # print("eff 1:")
        # for t in value_tuple:
        #     print(f"{t[0]} : {t[1]}")
        # print("\n")
        # print("skill 2:", existing_eff[1])
        # print("eff 2:")
        # for (pred, change) in existing_eff[0]:
        #     print(f"{pred} : {change}")
        # print("----")

        # check if their effect has same set of lifted predicates
        if len(list(value_tuple_1)) != len(list(value_tuple_2)):
            return False
        
        lifted_pred_list_1 = [pred.lifted() for pred, change in value_tuple_1]
        lifted_pred_list_2 = [pred.lifted() for pred, change in value_tuple_2]
        if set(lifted_pred_list_1) != set(lifted_pred_list_2):
            return False
        
        # build object mapping using groudned skills
        objects_1 = set(sum([list(pred.params) for pred, change in value_tuple_1], []))
        objects_2 = set(sum([list(pred.params) for pred, change in value_tuple_2], []))

        if len(objects_1) != len(objects_2): # must have equal number of objects
            return False

        assert grounded_skill_1.lifted() == grounded_skill_2.lifted(), "Only compare two grounded skills of the same lifted skill"

        # skill parameters always come first
        obj2idx_1 = {}
        for i, param in enumerate(grounded_skill_1.params):
            if len(obj2idx_1) == 0:
                obj2idx_1[param] = 0
            elif param not in obj2idx_1:
                obj2idx_1[param] = len(obj2idx_1)

        for obj in objects_1:
            if obj not in obj2idx_1:
                obj2idx_1[obj] = len(obj2idx_1)

        idx2pred_id_truth_list: dict[int, list[tuple]] = defaultdict(list)
        for pred, change in value_tuple_1:
            for i, param in enumerate(pred.params):
                idx2pred_id_truth_list[obj2idx_1[param]].append((pred.lifted(), i, change))

        
        # fix obj mapping for the first predicate set, search for match over possible obj mapping for the second predicate set
        obj2idx_2 = {}
        for i, param in enumerate(grounded_skill_2.params):
            if len(obj2idx_2) == 0:
                obj2idx_2[param] = 0
            elif param not in obj2idx_2:
                obj2idx_2[param] = len(obj2idx_2)

        for obj in objects_2:
            if obj not in obj2idx_2:
                obj2idx_2[obj] = len(obj2idx_2)

        if not len(obj2idx_1) == len(obj2idx_2): # Two value_dict must have the same number of parameters
            return False
        idx2obj_2 = {v:k for k,v in obj2idx_2.items()} # reverse mapping

        skill_param_mapping = {i: i for i in range(len(grounded_skill_1.params))}

        # type of each object in value_dict_2
        skill_param2type = {obj: t for obj, t in zip(grounded_skill_2.params, grounded_skill.types)}

        param2lowest_type = {}

        for pred, change in value_tuple_2:
            for obj, t in zip(pred.params, pred.types):
                if obj in skill_param2type:
                    param2lowest_type[obj] = skill_param2type[obj]
                elif obj in param2lowest_type:
                    if type_dict[obj].index(t) > type_dict[obj].index(param2lowest_type[obj]):
                        param2lowest_type[obj] = t
                else:
                    param2lowest_type[obj] = t


        # all possible mappings for the rest of the objects
        rest_index_1 = list(range(len(grounded_skill_1.params), len(obj2idx_1)))
        rest_index_2 = list(range(len(grounded_skill_2.params), len(obj2idx_2)))

        possible_mappings = all_mappings(rest_index_1, rest_index_2, dedup=True)
        # loop through all possible mappings, check if there exist one satisfy matches predicate pattern (idx2pred_id_list) and typing of predicates
        for mapping in possible_mappings:
            full_mapping = skill_param_mapping | mapping

            # try creating the same predicate set as value_tuple_1
            abstract_state_list = [[(pred.lifted(), None) for pred, change in value_tuple_1]] # initially all lifted
            for param_id, pred_id_truth_list in idx2pred_id_truth_list.items():
                param_id_2 = full_mapping[param_id]
                # type_list = type_dict[idx2obj_2[param_id_2]]
                try:
                    obj_type = param2lowest_type[idx2obj_2[param_id_2]]
                except:
                    breakpoint()
                for grounding_tuple in pred_id_truth_list:
                    predicate_to_ground, param_index, target_truth = grounding_tuple
                    abstract_state_list = beam_grounding(abstract_state_list, param_id, (predicate_to_ground, param_index, target_truth), obj_type)
                    if not abstract_state_list: # no possible grounding
                        break
                if not abstract_state_list: # no possible grounding
                    break
            if abstract_state_list:
                return True
        return False

    # dict[lifted skill, dict[ tuple[value_tuple, grounded_skill], list[task_step_tuple]]]
    skill2eff2task_step: dict[Skill, dict[ tuple, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    
    for grounded_skill, task2state in skill2task2state.items():
        if skill and not grounded_skill.lifted() == skill:
            skill2eff2task_step[grounded_skill.lifted()] = {}
            continue
        for task_step_tuple, transition_meta in task2state.items():

            transition, _ = unify_transition(transition_meta["states"], grounded_skill, type_dict)
            state_0, state_1 = transition
            value_tuple: tuple[Predicate, int] = tuple([(pred, state_1.get_pred_value(pred) - state_0.get_pred_value(pred)) \
                                                        for pred in state_0.iter_predicates() if state_1.get_pred_value(pred) - state_0.get_pred_value(pred) != 0])
            eff = (value_tuple, grounded_skill)
            # if grounded_skill.name == "Pick":
            #     print({str(p): change for p, change in value_tuple})
            if not skill2eff2task_step[grounded_skill.lifted()]:
                skill2eff2task_step[grounded_skill.lifted()][eff].append(task_step_tuple)
            else:
                found_match = False
                for existing_eff, task_step_tuple_list in skill2eff2task_step[grounded_skill.lifted()].items():
                    if equal_lifted_effect(value_tuple, existing_eff[0], grounded_skill, existing_eff[1]):
                        skill2eff2task_step[grounded_skill.lifted()][existing_eff].append(task_step_tuple)
                        found_match = True

                        break
                
                if not found_match:
                    skill2eff2task_step[grounded_skill.lifted()][eff].append(task_step_tuple)

    # merge partitions with same lifted effect across same lifted skill
    skill2partition: dict[Skill, list[list[tuple]]] = defaultdict(list)

    for lifted_skill, eff2task_step_list in skill2eff2task_step.items():
        if not eff2task_step_list:
            skill2partition[lifted_skill] = []
        skill2partition[lifted_skill] = [task_step_tuple_list for eff, task_step_tuple_list in eff2task_step_list.items()]
        # filter out partitions with few elements
        num_parts = len(sum(skill2partition[lifted_skill], []))
        skill2partition[lifted_skill] = [part for part in skill2partition[lifted_skill] if len(part) > max(1, num_parts * 0.15)] # NOTE: each partition must contain at least some amount of tasks

    return skill2partition

# def create_one_operator_from_one_partition(grounded_skill: Skill, task2state, task_step_tuple_list: list[tuple], type_dict: dict) -> LiftedPDDLAction:
#     """
#     Build operator from one partition using RCR code.

#     Args:
#         task2state :: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}
#         task_tuple_list: list of tuple of task_name and step number.
#     """
#     # no failure cases in the task2state partition
#     assert all([task2state[task_step_tuple]["success"] for task_step_tuple in task_step_tuple_list])

#     bridge = RCR_bridge()
#     transitions = [task2state[task_step_tuple]["states"] for task_step_tuple in task_step_tuple_list]
#     obj2type, _ = bridge.unify_obj_type(transitions, grounded_skill, type_dict)
#     unified_transitions = []
#     for t in transitions:
#         unified_transition = []
#         for state in t:
#             predicate_state = PredicateState([])
#             for grounded_pred, truth_value in state.pred_dict.items():
#                 types_list = []
#                 for idx, obj in enumerate(grounded_pred.params):
#                     if obj in obj2type:
#                         types_list.append(obj2type[obj])
#                     else:
#                         types_list.append(grounded_pred.types[idx])
#                 new_grounded_pred = Predicate(grounded_pred.name, types_list, grounded_pred.params)
#                 predicate_state.pred_dict[new_grounded_pred] = truth_value
#             unified_transition.append(predicate_state)
#         unified_transitions.append(unified_transition)
#     return bridge.operator_from_transitions(unified_transitions, grounded_skill, type_dict, obj2type, flush=True), bridge.get_pid_to_type(), obj2type

def create_one_operator_from_one_partition(task2state_skill, task_step_tuple_list: list[tuple], tautology_preds: list[Predicate]=False) -> LiftedPDDLAction:
    """
    Build operator from one partition using RCR code.

    Args:
        task2state_skill :: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool, "skill": grounded_skill}}
        task_step_tuple_list: list of tuple of task_name and step number.
        tautology_preds: list of predicates to remove that are always true or always false
    """

    bridge = RCR_bridge()
    transitions = []
    for task_step_tuple in task_step_tuple_list:
        transition = deepcopy(task2state_skill[task_step_tuple]["states"])
        state_0, state_1 = transition
        # remove tautology preds
        if tautology_preds:
            state_0.remove_pred_list(tautology_preds)
            state_1.remove_pred_list(tautology_preds)

        # remove predicates that doesn't have skill arguments or effect arguments
        skill_params = set(task2state_skill[task_step_tuple]["grounded_skill"].params)
        effect = tuple([(pred, state_1.get_pred_value(pred) - state_0.get_pred_value(pred)) \
                                                        for pred in state_0.iter_predicates() if state_1.get_pred_value(pred) - state_0.get_pred_value(pred) != 0])
        effect_params = set(sum([list(pred.params) for pred, change in effect], []))
        relevant_params = set(skill_params).union(effect_params)

        irrelevant_preds = []
        for pred in state_0.pred_dict.keys():
            if not set(pred.params).intersection(relevant_params):
                irrelevant_preds.append(pred)
        state_0.remove_pred_list(irrelevant_preds)
        state_1.remove_pred_list(irrelevant_preds)

        transitions.append([state_0, state_1])
        # breakpoint()

    return bridge.operator_from_transitions(transitions), bridge.obj2pid

def create_operators_from_partitions(lifted_skill: Skill, skill2task2state, skill2task2state_all, skill2partition, type_dict):
    """
    Calculate operators for one skill using the partitions by termination set.

    Args:
        lifted_skill :: Skill
        skill2task2state: only successful tasks
        skill2task2state_all: both successful and failed tasks
        skill2partition :: {lifted_skill: [[task_step_tuple, task_step_tuple]]}

    Returns:
        operators :: [(LiftedPDDLAction, {pid: int: type: str})]
    """

    operators = []
    # # create operators for each grounded skill
    # for grounded_skill, task2state in skill2task2state.items():
    #     if grounded_skill.lifted() == lifted_skill:
    #         for partition in skill2partition[grounded_skill]:
    #             operator, pid2type, obj2type = create_one_operator_from_one_partition(grounded_skill, task2state, partition, type_dict)
    #             if not operator in seen_operators:
    #                 seen_operators.add(operator)
    #                 operators.append((operator, pid2type, obj2type))

    # merge the first level of the dict by lifting the grounded skills, add grounded skills to the lowest level
    lifted_skill2task2state_skill = {}
    for grounded_skill, task2state in skill2task2state.items():
        if grounded_skill.lifted() == lifted_skill:
            task2state_skill = {}
            for task_step_tuple, transition_meta in task2state.items():
                task2state_skill[task_step_tuple] = {'states': transition_meta['states'], 'success': transition_meta['success'], 'grounded_skill': grounded_skill}
            
            if grounded_skill.lifted() in lifted_skill2task2state_skill:
                lifted_skill2task2state_skill[grounded_skill.lifted()].update(task2state_skill)
            else:
                lifted_skill2task2state_skill[grounded_skill.lifted()] = task2state_skill
    

    # calculate unnecessary predicates and remove them
    # Unnecessary predicates: 
    # 1. tautologies: always true or always false no matter success or failure
    # 2. ones that care about even lower level typing of the objects NOTE: not implemented yet

    all_transitions = []
    has_success = False
    has_failure = False
    for skill, task2state in skill2task2state_all.items():
        if skill.lifted() == lifted_skill:
            all_transitions += [transition_meta['states'] for transition_meta in task2state.values()]
            if any([transition_meta['success'] for transition_meta in task2state.values()]):
                has_success = True
            if any([not transition_meta['success'] for transition_meta in task2state.values()]):
                has_failure = True 

    if all_transitions:
        tautology_preds_set = set([(pred, value) for pred, value in all_transitions[0][0].pred_dict.items()]).intersection( \
                                set([(pred, value) for pred, value in all_transitions[0][1].pred_dict.items()]))

        for transition_i in all_transitions:
            state_preds_set_i = set([(pred, value) for pred, value in transition_i[0].pred_dict.items()]).intersection( \
                                set([(pred, value) for pred, value in transition_i[1].pred_dict.items()]))
            tautology_preds_set = tautology_preds_set.intersection(state_preds_set_i)

        tautology_preds = [pred for pred, value in tautology_preds_set]
    else:
        tautology_preds = []

    if not (has_success and has_failure):
        tautology_preds = [] # if all tasks are successful or failed, do not remove any predicates  


    # print("Tautology preds:", [str(p) for p in tautology_preds])

    # create operators for each partition
    partitions = skill2partition[lifted_skill]
    # print(partitions)
    for partition in partitions:
        operator, obj2pid = create_one_operator_from_one_partition(lifted_skill2task2state_skill[lifted_skill], partition, tautology_preds)
        operator.action_id = lifted_skill.name + "_" + str(operator.action_id)
        # remove higher level type parameters from the operator if exists
        params = {}
        # post process operators to remove duplicate parameters with higher level types
        for i, p in enumerate(operator.parameters):
            if str(p) not in params:
                params[str(p)] = (p, i)
            else:
                # see which one has the lowest type
                for types in type_dict.values():
                    if p.type in types and params[str(p)][0].type in types:
                        # Compare positions
                        remove_idx = params[str(p)][1] if types.index(str(p.type)) > types.index(params[str(p)][0].type) else i
                        operator.parameters.pop(remove_idx)
        # find param mapping and add skill argument if it is not in the operator parameters
        first_grounded_skill = lifted_skill2task2state_skill[lifted_skill][partition[0]]['grounded_skill']
        skill_param2pid = {}
        remaining_params = {}
        for i, param in enumerate(first_grounded_skill.params):
            if param in obj2pid:
                skill_param2pid[i] = obj2pid[param]
            else:
                remaining_params[i] = first_grounded_skill.types[i]

        # largest pid in the operator parameters
        largest_pid = max(obj2pid.values(), default=0)
        for remaining_idx, remaining_type in remaining_params.items():
            largest_pid += 1
            # create a new parameter
            new_param = Parameter(name=remaining_type, type=remaining_type, pid=f"{remaining_type}_p{largest_pid}")
            operator.parameters.append(new_param)
            skill_param2pid[remaining_idx] = largest_pid
            
        operators.append((operator, skill_param2pid))
    return operators

def score(pred, task2state, pred_type) -> tuple[float, float, float, float]:
    """
    score of a predicate as one skill's precondition or effect
    tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
    task2state :: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}
    type : {precond, eff}
    """
    # skill2task2state :: {skill_name: {task_step_tuple: [PredicateState, PredicateState]}}
    # task_step_tuple=(task_name, step)

    # step 0 will be skipped
    # In t_score, ratio of either p = True or p = False has to > threshold
    # but t_score and f_score need to agree with each other. i.e., if t_score has p=True f_score has to have p=False
    
    # PRECONDITION (s)
    # t_score_t: if P = True is a precond, P must equal to True if the task is successful
    # t_score_t = (Success & P=True)/Success = a / b
    # f_score_t: if P = True is a precond, the task must fail if P is False
    # f_score_t = (Fail & p=False)/p=False = c / d
    # t_score_f: if P = False is a precond, P must equal to False if the task is successful 
    # t_score_f = (Success & P=False)/Success e / b
    # f_score_f: if P = False is a precond, the task must fail if P is True
    # f_score_f = (Fail & p=True)/p=True f / g

    # EFFECT (s')
    # t_score_t: if P is a eff+, P must equal to True if the task is successful
    # t_score_t = (Success & P=True)/Success = a / b
    # f_score_t: if P is in eff+, the task must fail if P is False
    # f_score_t = (Fail & p=False)/p=False = c / d
    # t_score_f: if P is in eff-, P must equal to False if the task is successful 
    # t_score_f = (Success & P=False)/Success e / b
    # f_score_f: if P is in eff-, the task must fail if P is True
    # f_score_f = (Fail & p=True)/p=True f / g

    def sw_divide(a, b):
        "return 0 if devide by 0ß"
        return b and a / b
    
    a, b, c, d, e, f, g = 0, 0, 0, 0, 0, 0, 0
    state_idx = 0 if pred_type=="precond" else 1
    for task_step_id in task2state:
        # task_step_id is just for indexing purpose
        task_name, step = task_step_id
        # Using init state (s) for precondition and next state (s') for effect
        state = task2state[(task_name, step)]['states'][state_idx]
        success = task2state[(task_name, step)]["success"]
        pred_is_true = state.get_pred_value(pred)
        if step == 0:
            continue
        if success:
            b += 1
            if pred_is_true:
                a += 1
                g += 1
            elif not pred_is_true:
                d += 1
                e += 1
        elif not success:
            if pred_is_true:
                f += 1
            elif not pred_is_true:
                c += 1
    t_score_t, f_score_t, t_score_f, f_score_f = sw_divide(a, b), sw_divide(c, d), sw_divide(e, b), sw_divide(f, g)
    print(t_score_t, f_score_t, t_score_f, f_score_f)
    return t_score_t, f_score_t, t_score_f, f_score_f

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    else:
        return d

if __name__ == '__main__':
    model = GPT4(engine='gpt-4o-2024-11-20')
    
    # type_dict = {'PeanutButter': ['openable', 'pickupable'], 'Knife': ['pickupable', 'utensil'], 'Bread': ['food'], 'Cup': ['receptacle'], 'Table': ['location'], 'Shelf': ['location'], 'Robot': ['robot']}
    # lifted_skill = Skill("PickLeft", ["pickupable"])
    # threshold={"precond":0.5, "eff":0.5}
    # grounded_predicate_truth_value_log = load_from_file("hypo_test.yaml")
    # from utils import load_tasks
    # task_config = load_from_file("task_config/dorfl.yaml")
    # tasks = load_tasks("test_tasks/dorfl/", task_config)
    # # value = score_by_partition(lifted_skill, grounded_predicate_truth_value_log, tasks, pred_type, type_dict, threshold=threshold)
    # # breakpoint()
    # bridge = RCR_bridge(obj2pid={'PeanutButter': 0, 'Knife': 1, 'Robot': 2, None: -1})
    # test_pred_1 = Predicate('EnclosedByGripper', ['pickupable'], ['PeanutButter'])
    # test_pred_2 = Predicate('EnclosedByGripper', ['pickupable'], ['Knife'])
    # test_ps = PredicateState([test_pred_1, test_pred_2])
    # test_ps.set_pred_value(test_pred_1, False)
    # test_ps.set_pred_value(test_pred_2, False)
    # pddl_state = bridge.predicatestate_to_pddlstate(test_ps)
    def defaultdict_to_dict(d):
        if isinstance(d, defaultdict):
            return {k: defaultdict_to_dict(v) for k, v in d.items()}
        else:
            return d
    type_dict = {'Robot': ['robot'], 'Lettuce': ['pickupable', 'cuttable'], 'TopBun': ['pickupable'], 'BottomBun': ['pickupable'], 'Patty': ['pickupable', 'cookable'], 'Stove': ['station', 'cooker'], 'CuttingBoard': ['station', 'cuttingboard']}
    from src.utils import load_from_file
    skill2task2state = load_from_file("test_skill2task2state.yaml")
    tasks = load_from_file("results/oracle_predicates/burger/runs/0/0_partial/transitions/tasks.yaml")
    _,_,skill2partition = partition_by_lifted_effect(skill2task2state, type_dict)
    for skill, partition in skill2partition:
        print(skill)
        for task_steps in partition:
            print("Partition:")
            for task_step in task_steps:
                task, step = task_step
                print(tasks[task][str(step)], str(tasks[task][str(step)]['skill']))
        print('\n')
    breakpoint()