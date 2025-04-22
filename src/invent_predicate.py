'''
Get symbolic representation from skill semantic info and observation.
Data structures for logging:
    - read from data:
        - tasks :: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ::
            NOTE: step is int starting from 0. init state of the skill is at (step-1), next state is at step. step 0 has no skill
    - produced by skillwrapper
        - grounded_predicate_truth_value_log :: dict :: {task:{step:PredicateState}}
        - skill2operator :: {lifted_skill: [(LiftedPDDLAction, {pid: int: type: str})]}
    '''
from collections import defaultdict
from copy import deepcopy
import itertools
import logging
from typing import Union

from utils import GPT4, load_from_file
from data_structure import Skill, Predicate, PredicateState
from RCR_bridge import PDDLState, LiftedPDDLAction, RCR_bridge, generate_possible_groundings

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

def eval_pred(model, img: str, grounded_skill: Skill, grounded_pred: Predicate, prompt_fpath=['prompts/evaluate_pred_ai2thor.txt','prompts/evaluate_pred_ai2thor_init.txt'], init=False) -> bool:
    '''
    evaluate truth value of a predicate using a dictionary of parameters
    init step and later steps use different prompts. hardcoded.

    Args:
        grounded_skill::dict:: grounded skill with parameter type, e.g., {'name':'GoTo', "params":['location', 'location']}
        grounded_pred::dict:: grounded predicate with parameter type, e.g., {'name':"At", 'params':["location"]}
    '''
    def construct_prompt(prompt, grounded_skill, grounded_pred):
        "replace placeholders in the prompt"
        # Predicate might have parameters don't belong to the skill
        place_holders = ['[GROUNDED_SKILL]', '[GROUNDED_PRED]','[LIFTED_PRED]', '[SEMANTIC]']

        while any([p in prompt for p in place_holders]):
            prompt = prompt.replace('[GROUNDED_SKILL]', str(grounded_skill))
            prompt = prompt.replace('[GROUNDED_PRED]', str(grounded_pred))
            prompt = prompt.replace('[LIFTED_PRED]', str(grounded_pred.lifted()))
            prompt = prompt.replace('[SEMANTIC]', grounded_pred.semantic)
        return prompt
    
    prompt = load_from_file(prompt_fpath[0]) if not init else load_from_file(prompt_fpath[1])
    prompt = construct_prompt(prompt, grounded_skill, grounded_pred)
    logging.info(f'Evaluating predicate {grounded_pred} on skill {grounded_skill}')

    resp = model.generate_multimodal(prompt, img)[0]
    result = True if "True" in resp.split('\n')[-1] else False
    logging.info(f'{grounded_pred} evaluated to {result}')
    return result

def generate_pred(image_pair: list[str], grounded_skills: list[Skill], successes: list[bool], model: GPT4, lifted_pred_list: list[Predicate], pred_type: str, tried_pred=[], prompt_fpath='prompts/predicate_refining') -> Predicate:
    '''
    propose new predicates based on the contrastive pair.
    '''
    # TODO: polish the prompt
    def construct_prompt(prompt: str, grounded_skills, successes, lifted_pred_list: list[Predicate]):
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

    prompt_fpath += f"_{pred_type}.txt"
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, grounded_skills, successes, lifted_pred_list)

    logging.info('Generating predicate')
    # resp = model.generate(prompt)[0]
    resp = model.generate_multimodal(prompt, image_pair)[0]
    pred, sem = resp.split(': ', 1)[0].strip('`'), resp.split(': ', 1)[1].strip()
    # parse the parameters from the output string into predicate parameters
    # e.g., "At(obj, loc)"" -> {"name":"At", "types": ["obj", "loc"]}
    # new_pred = {'name': pred.split("(")[0], 'types': pred.split("(")[1].strip(")").split(", ")}
    new_pred = Predicate(pred.split("(")[0], pred.split("(")[1].strip(")").split(", ")) # lifted
    new_pred.semantic = sem
    return new_pred

# Adding to precondition or effect are different prompts
def update_empty_predicates(model, tasks: dict, lifted_pred_list: list[Predicate], type_dict, grounded_predicate_truth_value_log, skill: Skill = None):
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
    logging.info('looking for empty grounded predicates')
    # update if there are new tasks
    for task_id, steps in tasks.items():
        if task_id not in grounded_predicate_truth_value_log:
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
            pred_to_update = grounded_predicate_truth_value_log[task_id][step].get_unevaluated_preds() if step == 0 \
                else calculate_pred_to_update(grounded_pred_list, state["skill"])
            print(f'step {step}, skill {str(state["skill"])}')
            [print(p) for p in pred_to_update]
            print('\n')
            # 2. re-eval grounded predicates
            for pred in pred_to_update:
                # only update empty predicates
                if grounded_predicate_truth_value_log[task_id][step].get_pred_value(pred) == None:
                    # TODO: swicth it back after unit test
                    # truth_value = eval_pred(model, state["image"], state["skill"], pred, init=True) if state["skill"].lifted() == skill \
                    #                 else None
                    import random
                    if skill and step != 0:
                        truth_value = random.choice([True, False]) if state["skill"].lifted() == skill\
                                        else None
                    else:
                        truth_value = random.choice([True, False])
                    grounded_predicate_truth_value_log[task_id][step].set_pred_value(pred, truth_value)
            
            # 3.copy all empty predicates from previous state
                elif not step == 0: # if is a non-init state, update the predicates
                    truth_value = random.choice([True, False])
                    grounded_predicate_truth_value_log[task_id][step].set_pred_value(pred, truth_value)

            unevaluated_pred: list[Predicate] = grounded_predicate_truth_value_log[task_id][step].get_unevaluated_preds()
            if not skill:
                assert (unevaluated_pred==[]) == (step==0), "Step 0 shouldn't have any predicate unevaluated"
            for pred in unevaluated_pred:
                # fetch truth value from last state
                truth_value = grounded_predicate_truth_value_log[task_id][step-1].get_pred_value(pred)
                grounded_predicate_truth_value_log[task_id][step].set_pred_value(pred, truth_value)

    logging.info('Done updating predicate truth values')
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
                            skill2task2state[grounded_skill][task_step_tuple] = {'state':[last_state, state], 'success': tasks[task_name][step]['success']}
                    elif not success_only:
                        skill2task2state[grounded_skill][task_step_tuple] = {'state':[last_state, state], 'success': tasks[task_name][step]['success']}

                last_state = deepcopy(state)
    return skill2task2state

# TODO: Heavy unit test
def detect_mismatch(lifted_skill: Skill, skill2operator, grounded_predicate_truth_value_log, tasks, type_dict, pred_type: str) -> list[list[tuple, tuple]]:
    """
    Find mismatch state pairs where they both belong to Union Precondition or Effect.

    Args:
        skill2operator :: {lifted_skill: [(LiftedPDDLAction, {pid: int: type: str})]}
        grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
        tasks :: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': bool))) ::
        pred_type::{'precond', 'eff'}
    Returns:
        mismatch_pairs :: [[task_step_tuple, task_step_tuple]...]
    """
    def in_alpha(possible_groundings, pddl_state_list: list[PDDLState, PDDLState], operator, pred_type: str) -> bool:
        """
        There exist a grounding such that the grounded state agree with the operator's precondition/effect
        """
        for grounding in possible_groundings:
            grounded_operator = operator.get_grounded_action(grounding, 0) # don't know if action_id matters
            if pred_type == "precond":
                applicable = grounded_operator.check_applicability(pddl_state_list[0])
                if applicable:
                    return True
            elif pred_type == "eff":
                next_state = grounded_operator.apply(pddl_state_list[0])
                if next_state == pddl_state_list[1]:
                    return True
        return False
   
    skill2task2state = grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks)
    bridge = RCR_bridge()
    # All grounded skills
    # TODO: detect mismatch across grounded skill or lifted skill? Now across grounded
    task2in_alpha: dict[str, bool] = {} # alpha is the union of grounding of precondition or effect of operators corresponding to one skill
    task2success: dict[str, bool] = {}
    task_step_tuple_list = []
    for grounded_skill, task2state in skill2task2state.items():
        # evaluate across all grounded skill of the same name and type
        if grounded_skill.lifted() == lifted_skill:

            # task2state :: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}
            for task_step_tuple, transition_meta in task2state.items():
                task_step_tuple_list.append(task_step_tuple)

                task2success[task_step_tuple] = transition_meta['success']
                state_in_alpha = False
                for operator, pid2type in skill2operator[lifted_skill]:
                    possible_groundings = generate_possible_groundings(pid2type, type_dict, fixed_grounding=grounded_skill.params)
                    pddl_state_list = [bridge.predicatestate_to_pddlstate(state) for state in transition_meta["states"]]
                    if in_alpha(possible_groundings, pddl_state_list, operator, pred_type):
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

def invent_predicate_one(mismatch_pair: list[tuple, tuple], model: GPT4, lifted_skill: Skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type,  skill2triedpred={}, threshold={"precond":0.5, "eff":0.5}) -> Predicate:
    """
    One iteration of predicate invention.

    Args:
        mismatch_pair :: two task setp tuples that triggered predicate invention
    """
    # TODO: try different predicate invention: e.g., images, comparisons.
    # task_step_tuple = tuple[str, int]
    task_0, index_0 = mismatch_pair[0]
    task_1, index_1 = mismatch_pair[1]
    state_0 = tasks[task_0][index_0] if pred_type == "eff" else tasks[task_0][index_0-1]
    state_1 = tasks[task_1][index_1] if pred_type == "eff" else tasks[task_1][index_1-1]

    new_pred = generate_pred([state_0["image"], state_1["image"]],
                             [state_0["skill"], state_1["skill"]],
                             [state_0["success"], state_1["success"]],
                              model, lifted_pred_list, pred_type, tried_pred=skill2triedpred[lifted_skill])
    logging.info(f"new predicate {new_pred}")
    new_pred_accepted = False
    # evaluate the new predicate on all states
    # suppose we add the new predicate to the current predicate set
    hypothetical_pred_list = deepcopy(lifted_pred_list)
    hypothetical_pred_list.append(new_pred)
    hypothetical_grounded_predicate_truth_value_log = deepcopy(grounded_predicate_truth_value_log)
    # task unchanged, only add candidate predicate
    hypothetical_grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, hypothetical_pred_list, type_dict, hypothetical_grounded_predicate_truth_value_log, skill=lifted_skill)
    hypothetical_skill2task2state = grounded_pred_log_to_skill2task2state(hypothetical_grounded_predicate_truth_value_log, tasks, pred_type)
    # NOTE: socring now only done to a grounded skill instead of across all grounded of the lifted skill
    # TODO: scoring might need change entirely
    add_new_pred = score_by_partition(new_pred, lifted_skill, hypothetical_skill2task2state, pred_type, threshold)

    if add_new_pred:
        logging.info(f"Predicate {new_pred} added to predicate set by {pred_type} check")
        lifted_pred_list.append(new_pred)
        new_pred_accepted = True
    else:
        logging.info(f"Predicate {new_pred} is NOT added to predicate set by {pred_type} check")
        skill2triedpred[lifted_skill].append(new_pred)
    
    return lifted_pred_list, skill2triedpred, new_pred_accepted


def invent_predicates(model: GPT4, lifted_skill: Skill, skill2operator, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, skill2triedpred={}, max_t=3):
    '''
    Main loop of generating predicates.
    Invent one pred for precondition and one for effect.
    '''
    
    # check precondition first
    t = 0
    pred_type = "precond"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log)
    mismatch_pairs = detect_mismatch(lifted_skill, skill2operator, grounded_predicate_truth_value_log, tasks, pred_type=pred_type)
    # TODO: use mismatch tasks for predicate invention
    logging.info("About to enter precondition check")
    while mismatch_pairs and t < max_t:
        # Always solve the first mismatch pair
        lifted_pred_list, skill2triedpred, new_pred_accepted = invent_predicate_one(mismatch_pairs[0], model, lifted_skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type, skill2triedpred=skill2triedpred)
        if new_pred_accepted: break
        t += 1
    
    # check effect
    t = 0
    pred_type = "eff"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log)
    mismatch_pairs = detect_mismatch(lifted_skill, skill2operator, grounded_predicate_truth_value_log, tasks, pred_type=pred_type)
    logging.info("About to enter effect check")
    while mismatch_pairs and t < max_t:
        lifted_pred_list, skill2triedpred, new_pred_accepted = invent_predicate_one(mismatch_pairs[0], model, lifted_skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type, skill2triedpred=skill2triedpred)
        if new_pred_accepted: break
        t += 1
    
    # partitioning
    # 1. partition by different termination and effect, success task only
    skill2task2state = grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, success_only=True)
    _, _, skill2partition = partition_by_termination_n_eff(skill2task2state)
    # 2. create one operator for each partition
    skill2operator = create_operators_from_partitions(skill2partition)

    return skill2operator, lifted_pred_list, skill2triedpred

def score_by_partition(new_pred: Predicate, grounded_skill: Skill, skill2task2state, pred_type: str, threshold: dict[str, float]) -> bool:
    '''
    Partition by termination and effect and then score the predicates across each partition

    Args:
        skill :: grouded skill {"name":"PickUp", "types":["obj"], "params":["Apple"]}
        threshold={"precond":float, "eff":float}
    '''
    # 1. find all states after executing the same grounded skill
    _, _, skill2partition = partition_by_termination_n_eff(skill2task2state)

    # 2. evaluate the score for each task2state dictionary, pick the best one
    for grounded_skill, partitions in skill2partition.items():
        for task_step_tuple_list in partitions.values():
            for grounded_skill, task2state in skill2task2state:
                partitioned_task2state = {task_step_tuple: task2state[task_step_tuple] for task_step_tuple in task_step_tuple_list}

                new_pred_grounded = Predicate.ground_with_params(new_pred, grounded_skill.params)
                t_score_t, f_score_t, t_score_f, f_score_f = score(new_pred_grounded, partitioned_task2state, pred_type)
                # TODO: all 'or' for scoring the new partitioning method?
                if pred_type == "precond":
                    if (t_score_t > threshold[pred_type] or f_score_t > threshold[pred_type]) \
                        or (t_score_f > threshold[pred_type] or f_score_f > threshold[pred_type]):
                        return True
                    
                elif pred_type == "eff":
                    if (t_score_t > threshold[pred_type] or f_score_t > threshold[pred_type]) \
                        or (t_score_f > threshold[pred_type] or f_score_f > threshold[pred_type]):
                        return True
    return False

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
            value_dict = {pred: state_1.get_pred_value(pred) - state_0.get_pred_value(pred) for pred in state_0 \
                                if state_1.get_pred_value(pred) - state_0.get_pred_value(pred) != 0}
            
            termination_partition[state_1].append(task_step_tuple)
            eff_partition[value_dict].append(task_step_tuple)

        skill2state2partition[grounded_skill] = termination_partition
        skill2eff2partition[grounded_skill] = eff_partition
    # take intersection of both
    skill2partition: dict[Skill, list[list[tuple]]] = {grounded_skill: apply_both_partition( list(skill2state2partition[grounded_skill].values()), \
                                                            list(skill2eff2partition[grounded_skill].values()) ) \
                                                            for grounded_skill in skill2eff2partition}

    return skill2state2partition, skill2eff2partition, skill2partition

def create_one_operator_from_one_partition(grounded_skill: Skill, task2state, task_step_tuple_list: list[tuple]) -> LiftedPDDLAction:
    """
    Build operator from one partition using RCR code.

    Args:
        task2state :: {task_step_tuple: {"states": [PredicateState, PredicateState], "success": bool}}
        task_tuple_list: list of tuple of task_name and step number.
    """
    # no failure cases in the task2state partition
    assert all([task2state[task_step_tuple]["success"] for task_step_tuple in task_step_tuple_list])

    bridge = RCR_bridge()
    transitions = [task2state[task_step_tuple] for task_step_tuple in task_step_tuple_list]

    return bridge.operator_from_transitions(transitions, grounded_skill), bridge.get_pid_to_type()

def create_operators_from_partitions(skill2task2state, skill2partition):
    """
    Calculate operators using the partitions by termination set.

    Returns:
        skill2operators :: {lifted_skill: [(LiftedPDDLAction, {pid: int: type: str})]}
    """
    seen_operators = set()
    skill2operator = defaultdict(list)
    # create operators for each grounded skill
    for grounded_skill, task2state in skill2task2state.items():
        operator ,pid2type = create_one_operator_from_one_partition(grounded_skill, task2state, skill2partition[grounded_skill])
        if not operator in seen_operators:
            seen_operators.add(operator)
            skill2operator[grounded_skill.lifted()].append((operator, pid2type))
    return skill2operator

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
    
    a, b, c, d, e, f, g = 0, 0, 0, 0, 0, 0, 0
    state_idx = 0 if pred_type=="precond" else 1
    for task_step_id in task2state:
        # task_step_id is just for indexing purpose
        task_name, step = task_step_id
        # Using init state (s) for precondition and next state (s') for effect
        state = task2state[(task_name, step)]['state'][state_idx]
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
    t_score_t, f_score_t, t_score_f, f_score_f = a/b, c/d, e/b, f/g

    return t_score_t, f_score_t, t_score_f, f_score_f

if __name__ == '__main__':
    model = GPT4(engine='gpt-4o-2024-11-20')
    # mock symbolic state
    type_dict = {"Robot": ["robot"], "Apple": ['object'], "Banana": ['object'], "Table": ['location'], "Couch": ['location']}
    skill_1 = {'name': 'PickUp', 'types':['object'], 'params':[]}
    skill_2 = {"name": "GoTo", "types": ['location'], 'params':[]}
    skill_3 = {"name": "PlaceAt", "types": ['object', 'location'], 'params':[]}
    skill_list = [skill_1, skill_2, skill_3]
    type_dict = {
        "Robot": ["robot"], 
        "Apple": ['object'], 
        "Banana": ['object'], 
        "Table": ['location'], 
        "Couch": ['location']
        }

    pred = Predicate("At", ["object", "location"])
    grounded_pred = pred.ground_with(["Apple", "Table"], type_dict)
    lifted_pred = grounded_pred.lifted()
    skill = Skill("PlaceAt", ["object", "location"])
    grounded_skill = skill.ground_with(["Apple", "Table"], type_dict)
    lifted_skill = grounded_skill.lifted()

    lifted_pred_list = [
        Predicate("At", ["object", "location"]),
        Predicate("CloseTo", ["robot", "location"]),
        Predicate("HandOccupied", []),
        Predicate("IsHolding", ["object"]),
        Predicate("EnoughBattery", []),
        Predicate('handEmpty', [])
    ]

    pred_state = PredicateState(lifted_pred_list)
    breakpoint()
    
    example_lifted_predicates = [
        {'name':"At", 'types':["object", "location"], 'params':[], 'semantic': "At sem"},
        {'name':"CloseTo", 'types':["robot", "location"], 'params':[], 'semantic': "CloseTo sem"},
        {'name':"HandOccupied", 'types':[], 'params':[], 'semantic': "HandOccupied sem"},
    ]
    add_predicates = [
        {'name': "IsHolding", "types":["object"], 'params':[], 'semantic': "IsHolding sem"},
        {'name': "EnoughBattery", "types": [], 'params':[], 'semantic': "EnoughBattery sem"},
        {'name':'handEmpty', 'types':[],'params':[], 'semantic': "The robot's hand is empty"}
    ]
    test_pred_state = PredicateState(example_lifted_predicates)

    test_pred = {'name':"At", 'types':["object", "location"], 'params':[]}
    test_pred_keyified = PredicateState._keyify(test_pred)
    test_pred_restored = PredicateState.restore_pred_from_key(test_pred_keyified)

    value = test_pred_state.get_pred_value(test_pred)

    test_pred_state.add_pred_list(add_predicates)
    # caculate possible predicates based on lifed predicates and type dict
    possible_preds = possible_grounded_preds(test_pred_state.get_pred_list(lifted=True), type_dict)
    # add grounded predicates to PredicateState
    test_pred_state.add_pred_list(possible_preds)
    
    grounded_skill = {"name": "PlaceAt", "types": ['object', 'location'], 'params':['Banana', 'Couch']}
    pred_to_update = calculate_pred_to_update(possible_preds, grounded_skill)

    # test partitioning and scoring
    # construct multiple PredicateState instances
    s_0 = deepcopy(test_pred_state)
    # breakpoint()
    s_0.set_pred_value(
        {'name':"At", 'types':["object", "location"], 'params':['Banana', 'Couch']}, True
    )
    s_1 = deepcopy(s_0)
    s_1.set_pred_value(
        {'name':"IsHolding", 'types':["object"], 'params':['Banana']}, True
    )
    # test PredicateState hashable function
    hash_state_dict = {s_0: 0, s_1:1}
    breakpoint()
    # manually construct grounded_predicate_truth_value_log ::dict:: {task:{step:PredicateState}}
    grounded_predicate_truth_value_log = {
        '1': {

        }
    }