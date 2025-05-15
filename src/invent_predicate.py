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
import logging
import random
from typing import Union

from utils import GPT4, load_from_file
from data_structure import Skill, Predicate, PredicateState
from RCR_bridge import PDDLState, LiftedPDDLAction, Parameter, RCR_bridge, generate_possible_groundings

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

def eval_pred(img: str, grounded_pred: Predicate, model: GPT4, prompt_fpath='prompts/evaluate_pred.yaml', args=None) -> bool:
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
            # prompt = prompt.replace('[GROUNDED_SKILL]', str(grounded_skill))
            prompt = prompt.replace('[GROUNDED_PRED]', str(grounded_pred))
            prompt = prompt.replace('[LIFTED_PRED]', str(grounded_pred.lifted()))
            prompt = prompt.replace('[SEMANTIC]', grounded_pred.semantic)
        return prompt
    
    prompt = load_from_file(prompt_fpath)[args.env]
    prompt = construct_prompt(prompt, grounded_pred)
    # logging.info(f'Evaluating predicate {grounded_pred}')

    resp = model.generate_multimodal(prompt, [img])[0]
    result = True if "True" in resp.split('\n')[-1] else False
    logging.info(f'{grounded_pred} evaluated to `{result}` in {img}')
    # breakpoint()
    return result

def generate_pred(image_pair: list[str], grounded_skills: list[Skill], successes: list[bool], lifted_pred_list: list[Predicate], pred_type: str, model: GPT4, skill2tried_pred={}, prompt_fpath='prompts/predicate_invention.yaml', args=None) -> Predicate:
    '''
    propose new predicates based on the contrastive pair.
    '''
    def construct_prompt(prompt: str, grounded_skills, successes, lifted_pred_list: list[Predicate], tried_pred: list[Predicate]):
        """
        replace placeholders in the prompt
        pred_list :: list of lifted predicates
        """
        placeholders = ["[LIFTED_SKILL]", "[GROUNDED_SKILL_1]", "[GROUNDED_SKILL_2]", "[SUCCESS_1]", "[SUCCESS_2]", "[PRED_LIST]"]
        while any([p in prompt for p in placeholders]):
            prompt = prompt.replace("[LIFTED_SKILL]",  str(grounded_skills[0].lifted()))
            prompt = prompt.replace("[GROUNDED_SKILL_1]",  str(grounded_skills[0]))
            prompt = prompt.replace("[GROUNDED_SKILL_2]",  str(grounded_skills[1]))
            prompt = prompt.replace("[SUCCESS_1]",  "succeeded" if bool(successes[0]) else "failed")
            prompt = prompt.replace("[SUCCESS_2]",  "succeeded" if bool(successes[1]) else "failed")
            # construct predicate list from pred_dict
            pred_list_str = '\n'.join([f'{str(pred)}: {pred.semantic}' for pred in lifted_pred_list])
            prompt = prompt.replace("[PRED_LIST]", pred_list_str)
            prompt = prompt.replace("[TRIED_PRED]", ", ".join([str(pred) for pred in tried_pred]))
        return prompt

    tried_pred = skill2tried_pred[grounded_skills[0].lifted()] if skill2tried_pred else []
    prompt = load_from_file(prompt_fpath)[args.env][pred_type]
    prompt = construct_prompt(prompt, grounded_skills, successes, lifted_pred_list, tried_pred)
    assert len(image_pair)==4 if pred_type=="eff" else len(image_pair)==2, "precondition need 2 images while effect need 4"
    logging.info('Generating predicate')
    # resp = model.generate(prompt)[0]
    resp = model.generate_multimodal(prompt, image_pair)[0]
    pred, sem = resp.split('\n')[-1].split(': ', 1)[0].strip('`'), resp.split(': ', 1)[1].strip()
    # parse the parameters from the output string into predicate parameters
    # e.g., "At(obj, loc)"" -> Predicate(name="At", types=["obj", "loc"])
    new_pred = Predicate(pred.split("(")[0], pred.split("(")[1].strip(")").split(", ")) # lifted
    new_pred.semantic = sem
    # breakpoint()
    return new_pred

# Adding to precondition or effect are different prompts
def update_empty_predicates(model, tasks: dict, lifted_pred_list: list[Predicate], type_dict, grounded_predicate_truth_value_log, skill: Skill = None, args=None):
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
    # NOTE: step is a integer ranging from 0-8, where 0 is the init step and success==None. 1-8 are states after executions

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

        # for each step, iterate through all steps and find empty predicates and update them
        # calculate predicates to update based on the last action every step after init

        # update predicates for all states 
        for step, state in steps.items():
            grounded_predicate_truth_value_log[task_id][step].add_pred_list(grounded_pred_list)

            # 1. find states need to be eval or re-eval
            # At init step only evaluate empty ones
            # assuming the skill execution can only change the predicates with parameters overlapping with the skill
            pred_to_update = grounded_predicate_truth_value_log[task_id][step].get_unevaluated_preds() if step == 0 \
                else calculate_pred_to_update(grounded_pred_list, state["skill"])
            # print(f'step {step}, skill {str(state["skill"])}') # TODO: remove these loggings
            # print(f"task success: {state['success']}")
            # [print(p, grounded_predicate_truth_value_log[task_id][step].get_pred_value(p)) for p in pred_to_update]
            # print('\n')
            # 2. re-eval grounded predicates
            for grounded_pred in pred_to_update:
                # only update empty predicates
                if grounded_predicate_truth_value_log[task_id][step].get_pred_value(grounded_pred) == None:
                    # if skill and step != 0:
                    #     # evaluate the predicate before and after the skill execution
                    #     truth_value = eval_pred(state["image"], grounded_pred, model, args=args) if state["skill"].lifted() == skill or steps[step-1]["skill"].lifted() == skill\
                    #                     else None
                    # else:
                    #     truth_value = eval_pred(state["image"], grounded_pred, model, args=args)
                    
                    # if truth_value is not None:
                    #     grounded_predicate_truth_value_log[task_id][step].set_pred_value(grounded_pred, truth_value)

                    truth_value = eval_pred(state["image"], grounded_pred, model, args=args)
                    grounded_predicate_truth_value_log[task_id][step].set_pred_value(grounded_pred, truth_value)
            
            # 3.copy all empty predicates from previous state
                elif not step == 0: # if is a non-init state, update the predicates
                    if (new_task) or (not new_task and grounded_predicate_truth_value_log[task_id][step].get_pred_value(grounded_pred) == None):
                        truth_value = eval_pred(state["image"], grounded_pred, model, args=args)
                        grounded_predicate_truth_value_log[task_id][step].set_pred_value(grounded_pred, truth_value)

            unevaluated_pred: list[Predicate] = grounded_predicate_truth_value_log[task_id][step].get_unevaluated_preds()
            if not skill:
                assert (unevaluated_pred==[]) if (step==0) else True, "Step 0 shouldn't have any predicate unevaluated"
            for grounded_pred in unevaluated_pred:
                # fetch truth value from last state
                truth_value = grounded_predicate_truth_value_log[task_id][step-1].get_pred_value(grounded_pred)
                grounded_predicate_truth_value_log[task_id][step].set_pred_value(grounded_pred, truth_value)

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

def in_alpha(possible_groundings, transition: list[PredicateState, PredicateState], operator, pred_type: str, obj2type: dict[str, str]) -> bool:
    """
    Util function for detect_mismatch and score_by_partition
    There exist a grounding such that the grounded state agree with the operator's precondition/effect
    """
    for grounding in possible_groundings:
        bridge = RCR_bridge()

        # also construct these pddl state with obj2type
        unified_transition = []
        for state in transition:
            predicate_state = PredicateState([])
            for grounded_pred, truth_value in state.pred_dict.items():
                types_list = []
                for idx, obj in enumerate(grounded_pred.params):
                    if obj in obj2type:
                        types_list.append(obj2type[obj])
                    else:
                        types_list.append(grounded_pred.types[idx])
                new_grounded_pred = Predicate(grounded_pred.name, types_list, grounded_pred.params)
                predicate_state.pred_dict[new_grounded_pred] = truth_value
            unified_transition.append(predicate_state)

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
                    for operator, pid2type, obj2type in skill2operator[lifted_skill]:
                        possible_groundings = generate_possible_groundings(pid2type, type_dict, fixed_grounding=grounded_skill.params)
                        if in_alpha(possible_groundings, transition_meta["states"], operator, pred_type, obj2type):
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

def invent_predicate_one(mismatch_pair: list[tuple, tuple], model: GPT4, lifted_skill: Skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type, skill2triedpred=defaultdict(list), threshold={"precond":0.5, "eff":0.5}, args=None) -> Predicate:
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
    new_pred = generate_pred(image_pair,
                             [state_0["skill"], state_1["skill"]],
                             [state_0["success"], state_1["success"]],
                            lifted_pred_list, pred_type, model, skill2triedpred, args=args)
    logging.info(f"Generated new predicate {new_pred}")
    if len(new_pred.types) > 2:
        logging.info(f"Predicate {new_pred} is NOT added to predicate set because contain more then 2 parameters")
        skill2triedpred[lifted_skill].append(new_pred)
        return lifted_pred_list, skill2triedpred, False, grounded_predicate_truth_value_log
    elif new_pred in lifted_pred_list or new_pred in skill2triedpred[lifted_skill]:
        logging.info(f"Predicate {new_pred} is already in the predicate set or tried before.")
        return lifted_pred_list, skill2triedpred, False, grounded_predicate_truth_value_log
    
    new_pred_accepted = False
    # evaluate the new predicate on all states
    # suppose we add the new predicate to the current predicate set
    hypothetical_pred_list = deepcopy(lifted_pred_list)
    hypothetical_pred_list.append(new_pred)
    hypothetical_grounded_predicate_truth_value_log = deepcopy(grounded_predicate_truth_value_log)
    # task unchanged, only add candidate predicate
    hypothetical_grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, hypothetical_pred_list, type_dict, hypothetical_grounded_predicate_truth_value_log, skill=lifted_skill, args=args)
    add_new_pred = score_by_partition(lifted_skill, hypothetical_grounded_predicate_truth_value_log, tasks, pred_type, type_dict, threshold)
    if add_new_pred:
        logging.info(f"Predicate {new_pred} added to predicate set by {pred_type} check")
        lifted_pred_list.append(new_pred)
        grounded_predicate_truth_value_log = hypothetical_grounded_predicate_truth_value_log
        grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, args=args) # udpate for all skills
        new_pred_accepted = True
    else:
        logging.info(f"Predicate {new_pred} is NOT added to predicate set by {pred_type} check")
        skill2triedpred[lifted_skill].append(new_pred)
    
    return lifted_pred_list, skill2triedpred, new_pred_accepted, grounded_predicate_truth_value_log

def invent_predicates(model: GPT4, lifted_skill: Skill, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, skill2triedpred=defaultdict(list), max_t=3, args=None):
    '''
    Main loop of generating predicates.
    Invent one pred for precondition and one for effect.
    '''
    # check precondition first
    t = 0
    pred_type = "precond"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, args=args)
    mismatch_pairs = detect_mismatch(lifted_skill, skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, pred_type=pred_type)
    logging.info(f"About to enter precondition check of skill {lifted_skill}")
    new_pred_accepted = False
    while mismatch_pairs and t < max_t:
        # Always solve the first mismatch pair
        lifted_pred_list, skill2triedpred, new_pred_accepted, grounded_predicate_truth_value_log = invent_predicate_one(random.choice(mismatch_pairs), model, lifted_skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type, skill2triedpred=skill2triedpred, args=args)
        if new_pred_accepted: break
        t += 1
    
    if mismatch_pairs and new_pred_accepted:
        skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict)

    # check effect
    t = 0
    pred_type = "eff"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log, args=args)
    mismatch_pairs = detect_mismatch(lifted_skill, skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, pred_type=pred_type)
    logging.info(f"About to enter effect check of skill {lifted_skill}")
    new_pred_accepted = False
    while mismatch_pairs and t < max_t:
        lifted_pred_list, skill2triedpred, new_pred_accepted, grounded_predicate_truth_value_log = invent_predicate_one(random.choice(mismatch_pairs), model, lifted_skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type, skill2triedpred=skill2triedpred, args=args)
        if new_pred_accepted: break
        t += 1
    
    if mismatch_pairs and new_pred_accepted:
        skill2operator = calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict)

    logging.info(f"Done inventing predicates for skill {str(lifted_skill)}")
    # Not flushing tried predicate cache right now
    return skill2operator, lifted_pred_list, skill2triedpred, grounded_predicate_truth_value_log

def score_by_partition(lifted_skill: Skill, hypothetical_grounded_predicate_truth_value_log, tasks, pred_type: str, type_dict, threshold:  dict[str, float]) -> bool:
    '''
    New scoring function that use the predicate invention condition
    Calculate hypothetical operators and then check

    This function is largely taken from detect mismatch
    '''
    # calculate hypotehtical operators
    hypothetical_skill2task2state_success = grounded_pred_log_to_skill2task2state(hypothetical_grounded_predicate_truth_value_log, tasks, success_only=True)
    _, _, skill2partition = partition_by_termination_n_eff(hypothetical_skill2task2state_success)
    hypothetical_operators = create_operators_from_partitions(lifted_skill, hypothetical_skill2task2state_success, skill2partition, type_dict)

    # if the new operators can make sure fail execution outside alpha and successful execution inside alpha
    hypothetical_skill2task2state = grounded_pred_log_to_skill2task2state(hypothetical_grounded_predicate_truth_value_log, tasks, success_only=False)
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
                for operator, pid2type, obj2type in hypothetical_operators:
                    possible_groundings = generate_possible_groundings(pid2type, type_dict, fixed_grounding=grounded_skill.params)
                    if in_alpha(possible_groundings, transition_meta["states"], operator, pred_type, obj2type):
                        state_in_alpha = True
                        break
                print(task_step_tuple, state_in_alpha, transition_meta['success'])
                print(transition_meta["states"][0],'\n')
                print(transition_meta["states"][1],'\n')
                # breakpoint()
                score += 1 if state_in_alpha == transition_meta['success'] else 0
                task_num += 1

    result = True if score/task_num > threshold[pred_type] else False
    logging.info(f"Predicate is {'' if result else 'not'} added. Score = {score/task_num}")
    return result

def score_by_partition_final(new_pred_lifted: Predicate, lifted_skill: Skill, skill2task2state, pred_type: str, threshold: dict[str, float]) -> bool:
    '''
    Partition by termination and effect and then score the predicates across each partition

    Args:
        skill :: grouded skill {"name":"PickUp", "types":["obj"], "params":["Apple"]}
        threshold={"precond":float, "eff":float}
    '''
    # 1. find all states after executing the same grounded skill
    _, _, skill2partition = partition_by_termination_n_eff(skill2task2state)

    # 2. evaluate the score across all grounded skill of the lifted skill, return true if only one partition makes the score higher than threshold
    for grounded_skill_outer, partitions in skill2partition.items():
        for task_step_tuple_list in partitions:
            for grounded_skill_inner, task2state in skill2task2state.items():
                if grounded_skill_outer == grounded_skill_inner and grounded_skill_outer.lifted() == lifted_skill:
                    partitioned_task2state = {task_step_tuple: task2state[task_step_tuple] for task_step_tuple in task_step_tuple_list}
                    new_pred_grounded = Predicate.ground_with(new_pred_lifted, grounded_skill_inner.params)
                    t_score_t, f_score_t, t_score_f, f_score_f = score(new_pred_grounded, partitioned_task2state, pred_type)
                    if pred_type == "precond":
                        if (t_score_t > threshold[pred_type] or f_score_t > threshold[pred_type]) \
                            or (t_score_f > threshold[pred_type] or f_score_f > threshold[pred_type]):
                            return True
                        
                    elif pred_type == "eff":
                        if (t_score_t > threshold[pred_type] or f_score_t > threshold[pred_type]) \
                            or (t_score_f > threshold[pred_type] or f_score_f > threshold[pred_type]):
                            return True
    return False

def calculate_operators_for_all_skill(skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, filtered_lifted_pred_list:list[Predicate]=None,):
    # partitioning
    # 1. partition by different termination and effect, success task only
    skill2task2state = grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks, success_only=True)
    
    if filtered_lifted_pred_list: # when we perform final filtering after all iterations
        for grounded_skill, task2state in skill2task2state.items():
            for task_step_tuple, transition_meta in task2state.items():
                transition = transition_meta['states']
                new_transition = [
                    transition[0].filter_pred_list(filtered_lifted_pred_list),
                    transition[1].filter_pred_list(filtered_lifted_pred_list)
                ]
                transition_meta['states'] = new_transition

    _, _, skill2partition = partition_by_termination_n_eff(skill2task2state)
    # 2. create one operator for each partition
    for lifted_skill in skill2operator:
        skill2operator[lifted_skill] = create_operators_from_partitions(lifted_skill, skill2task2state, skill2partition, type_dict)

    return skill2operator

def filter_predicates(skill2operator, lifted_pred_list: list[Predicate], grounded_predicate_truth_value_log, tasks, threshold={"precond":0.5, "eff":0.5}) -> list[Predicate]:
    """
    After running all iterations in main function, score all predicates again
    This function will only be called in main.
    """
    filtered_lifted_pred_list = []
    skill2task2state = grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks, success_only=False)
    for lifted_pred in lifted_pred_list:
        for lifted_skill in skill2operator:
            for pred_type in ['precond', 'eff']:
                add_new_pred = score_by_partition_final(lifted_pred, lifted_skill, skill2task2state, pred_type, threshold=threshold)
                if add_new_pred:
                    filtered_lifted_pred_list.append(lifted_pred)
    
    return filtered_lifted_pred_list

def partition_by_termination_n_eff(skill2task2state) -> Union[dict, dict]:
    '''
    Partition the a set of transitions using termination set. Will be used again in scoring and final operators learning.
    Only successful execution will be used for partitioning.

    Args:
        skill2task2state :: {Skill: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}}
    Returns:
        {grounded_skill: [{PredicateState: [task_step_tuple]} , ...]}
    '''
    def apply_both_partition(partition_1, partition_2):
        "Find the intersection of applying termination and eff partition"
        # Map each item to its group index in both groupings
        partition_1_map = {}
        for i, g in enumerate(partition_1):
            for item in g:
                partition_1_map[item] = i
        
        partition_2_map = {}
        for i, g in enumerate(partition_2):
            for item in g:
                partition_2_map[item] = i
        
        # Use a dict to collect items that share the same (group_1_index, group_2_index)
        combined_partitions = {}
        for item in set(partition_1_map) & set(partition_2_map):  # In case not all items are covered
            key = (partition_1_map[item], partition_2_map[item])
            combined_partitions.setdefault(key, []).append(item)
        
        return list(combined_partitions.values())

    skill2state2partition: dict[Skill, dict[PredicateState, list[tuple]]] = {}
    skill2eff2partition: dict[Skill, dict[dict, list[tuple]]] = {}

    for grounded_skill, task2state in skill2task2state.items():
        termination_partition = defaultdict(list) # {task_step}
        eff_partition = defaultdict(list)
        for task_step_tuple, transition_meta in task2state.items():

            state_0, state_1 = transition_meta["states"]
            value_tuple = ((pred, state_1.get_pred_value(pred) - state_0.get_pred_value(pred)) for pred in state_0.iter_predicates() \
                                if state_1.get_pred_value(pred) - state_0.get_pred_value(pred) != 0)
            
            termination_partition[state_1].append(task_step_tuple)
            # value_dict is not hashable so 
            eff_partition[value_tuple].append(task_step_tuple)

        skill2state2partition[grounded_skill] = termination_partition
        skill2eff2partition[grounded_skill] = eff_partition
    # take intersection of both
    skill2partition: dict[Skill, list[list[tuple]]] = {grounded_skill: apply_both_partition(list(skill2state2partition[grounded_skill].values()), \
                                                            list(skill2eff2partition[grounded_skill].values()) ) \
                                                            for grounded_skill in skill2eff2partition}

    return skill2state2partition, skill2eff2partition, skill2partition

def create_one_operator_from_one_partition(grounded_skill: Skill, task2state, task_step_tuple_list: list[tuple], type_dict) -> LiftedPDDLAction:
    """
    Build operator from one partition using RCR code.

    Args:
        task2state :: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}
        task_tuple_list: list of tuple of task_name and step number.
    """
    # no failure cases in the task2state partition
    assert all([task2state[task_step_tuple]["success"] for task_step_tuple in task_step_tuple_list])

    bridge = RCR_bridge()
    transitions = [task2state[task_step_tuple]["states"] for task_step_tuple in task_step_tuple_list]
    obj2type, _ = bridge.unify_obj_type(transitions, grounded_skill, type_dict)
    unified_transitions = []
    for t in transitions:
        unified_transition = []
        for state in t:
            predicate_state = PredicateState([])
            for grounded_pred, truth_value in state.pred_dict.items():
                types_list = []
                for idx, obj in enumerate(grounded_pred.params):
                    if obj in obj2type:
                        types_list.append(obj2type[obj])
                    else:
                        types_list.append(grounded_pred.types[idx])
                new_grounded_pred = Predicate(grounded_pred.name, types_list, grounded_pred.params)
                predicate_state.pred_dict[new_grounded_pred] = truth_value
            unified_transition.append(predicate_state)
        unified_transitions.append(unified_transition)
    return bridge.operator_from_transitions(unified_transitions, grounded_skill, type_dict, obj2type, flush=True), bridge.get_pid_to_type(), obj2type

def create_operators_from_partitions(lifted_skill: Skill, skill2task2state, skill2partition, type_dict):
    """
    Calculate operators for one skill using the partitions by termination set.

    Returns:
        operators :: [(LiftedPDDLAction, {pid: int: type: str})]
    """
    seen_operators = set()
    operators = []
    # create operators for each grounded skill
    for grounded_skill, task2state in skill2task2state.items():
        if grounded_skill.lifted() == lifted_skill:
            for partition in skill2partition[grounded_skill]:
                operator ,pid2type, obj2type = create_one_operator_from_one_partition(grounded_skill, task2state, partition, type_dict)
                if not operator in seen_operators:
                    seen_operators.add(operator)
                    operators.append((operator, pid2type, obj2type))
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

if __name__ == '__main__':
    model = GPT4(engine='gpt-4o-2024-11-20')
    # # mock symbolic state
    # type_dict = {"Robot": ["robot"], "Apple": ['object'], "Banana": ['object'], "Table": ['location'], "Couch": ['location']}
    # type_dict = {
    #     "Robot": ["robot"], 
    #     "Apple": ['object'], 
    #     "Banana": ['object'], 
    #     "Table": ['location'], 
    #     "Couch": ['location']
    #     }

    # pred = Predicate("At", ["object", "location"])
    # grounded_pred = pred.ground_with(["Apple", "Table"], type_dict)
    # lifted_pred = grounded_pred.lifted()
    # skill = Skill("PlaceAt", ["object", "location"])
    # grounded_skill = skill.ground_with(["Apple", "Table"], type_dict)
    # lifted_skill = grounded_skill.lifted()

    # lifted_pred_list = [
    #     Predicate("At", ["object", "location"]),
    #     Predicate("CloseTo", ["robot", "location"]),
    #     Predicate("HandOccupied", []),
    #     Predicate("IsHolding", ["object"]),
    #     Predicate("EnoughBattery", []),
    #     Predicate('handEmpty', [])
    # ]
    # pred_state = PredicateState(lifted_pred_list)

    # # predicate invention tuning
    # from attrdict import AttrDict
    # args = AttrDict()
    # args.env = "dorfl"
    # # pickleft()
    # image_pair = [
    #     'test_tasks/task_imgs/1/1.jpg',
    #     'test_tasks/task_imgs/1/5.jpg'
    # ]
    # grounded_skills = [
    #     Skill('PickLeft', ['pickupable'], ['Knife']),
    #     Skill('PickLeft', ['pickupable'], ['PeanutButter'])
    # ]
    # successes = [
    #     False,
    #     True
    # ]
    # lifted_pred_list = []
    # pred_type = 'precond'

    # # open()
    # image_pair = [
    #     'test_tasks/dorfl_images/1/3.jpg',
    #     'test_tasks/dorfl_images/1/6.jpg'
    # ]
    # grounded_skills = [
    #     Skill('Open', ['openable'], ['PeanutButter']),
    #     Skill('Open', ['openable'], ['PeanutButter'])
    # ]
    # successes = [
    #     False,
    #     True
    # ]
    # lifted_pred_list = []
    # pred_type = 'precond'

    # # scoop
    # image_pair = [
    #     'test_tasks/dorfl_images/1/5.jpg',
    #     'test_tasks/dorfl_images/1/10.jpg'
    # ]
    # grounded_skills = [
    #     Skill('Scoop', ['utensil', 'openable'], ['Knife', 'PeanutButter']),
    #     Skill('Scoop', ['utensil', 'openable'], ['Knife', 'PeanutButter'])
    # ]
    # successes = [
    #     False,
    #     True
    # ]
    # lifted_pred_list = []
    # pred_type = 'precond'

    pred_type = "eff"
    # image_pair = [
    #     'test_tasks/dorfl_images/1/4.jpg',
    #     'test_tasks/dorfl_images/1/5.jpg',
    #     'test_tasks/dorfl_images/1/10.jpg',
    #     'test_tasks/dorfl_images/1/11.jpg'
    # ]

    # image_pair = [
    #     'test_tasks/dorfl_images/1/7.jpg',
    #     'test_tasks/dorfl_images/1/8.jpg',
    #     'test_tasks/dorfl_images/1/11.jpg',
    #     'test_tasks/dorfl_images/1/12.jpg'
    # ]
    # grounded_skills = [
    #     Skill('Spread', ['utensil', 'food'], ['Knife', 'Bread']),
    #     Skill('Spread', ['utensil', 'food'], ['Knife', 'Bread'])
    # ]



    # new_pred = generate_pred(image_pair, grounded_skills, successes, lifted_pred_list, pred_type, model, args=args)
    # new_pred = Predicate("IsOpen", ["openable"], semantic='the object (e.g., a jar) is visually open, with its lid removed or absent.')
    # grounded_pred = new_pred.ground_with(["PeanutButter"])
    # truth_value = eval_pred(image_pair[1], grounded_pred, model, args=args)

    type_dict = {'PeanutButter': ['openable', 'pickupable'], 'Knife': ['pickupable', 'utensil'], 'Bread': ['food'], 'Cup': ['receptacle'], 'Table': ['location'], 'Shelf': ['location'], 'Robot': ['robot']}
    lifted_skill = Skill("PickLeft", ["pickupable"])
    threshold={"precond":0.5, "eff":0.5}
    grounded_predicate_truth_value_log = load_from_file("hypo_test.yaml")
    from utils import load_tasks
    task_config = load_from_file("task_config/dorfl.yaml")
    tasks = load_tasks("test_tasks/dorfl/", task_config)
    # value = score_by_partition(lifted_skill, grounded_predicate_truth_value_log, tasks, pred_type, type_dict, threshold=threshold)
    # breakpoint()
    bridge = RCR_bridge(obj2pid={'PeanutButter': 0, 'Knife': 1, 'Robot': 2, None: -1})
    test_pred_1 = Predicate('EnclosedByGripper', ['pickupable'], ['PeanutButter'])
    test_pred_2 = Predicate('EnclosedByGripper', ['pickupable'], ['Knife'])
    test_ps = PredicateState([test_pred_1, test_pred_2])
    test_ps.set_pred_value(test_pred_1, False)
    test_ps.set_pred_value(test_pred_2, False)
    pddl_state = bridge.predicatestate_to_pddlstate(test_ps)

    breakpoint()