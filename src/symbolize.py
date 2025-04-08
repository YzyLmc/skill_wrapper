'''
Get symbolic representation from skill semantic info and observation.
Data structures:
    type_dict :: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    skill :: dict :: {'name':str, 'types':list,'params':list}
    pred :: dict :: {'name':str, 'types':list, 'params':list, 'semantic':str}
    lifted_pred_list :: [{'name':str, 'types':list, 'params':list, 'semantic':str}] ::
    tasks :: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ::
        step is int starting from 0. init state of the skill is at (step-1), next state is at step. step 0 has no skill
    grounded_predicate_truth_value_log :: dict :: {task:{step:[{'name':str, 'params':list, 'truth_value':bool}]}}
'''
from utils import GPT4, load_from_file
from collections import defaultdict
from copy import deepcopy
import random
import itertools
import logging

class PredicateState:
    def __init__(self, predicates):
        """
        Initializes the predicate state.
        Lifted predicates have empty 'params'
        predicates::list:: A list of predicate dictionaries {'name': str, 'params': tuple/list, "types": tuple/list, "semantic":str}
        """
        self.pred_dict = {self._keyify(pred): None for pred in predicates}

    @classmethod
    def _keyify(self, pred):
        """Generates a unique key from predicate name and parameters."""
        return (pred["name"], tuple(pred["types"]),tuple(pred["params"]))
    
    def get_pred_list(self, lifted=False):
        """
        Restores and returns a list of unique predicate dictionaries in their original form.
        """
        pred_list = []
        seen = set()
        for name, types, params in self.pred_dict.keys():
            key = (name, types, params)
            if key not in seen:
                pred_list.append({
                    "name": name,
                    "types": list(types),
                    "params": [] if lifted else list(params),
                    # Optional: include "semantic" as None since it's not stored in pred_dict
                    "semantic": None  
                })
                seen.add(key)
        return pred_list

    def set_pred_value(self, grounded_pred, value):
        """
        Sets the predicate value efficiently using dictionary lookup.
        """
        pred_key = self._keyify(grounded_pred)
        if pred_key in self.pred_dict:
            self.pred_dict[pred_key] = value

    def get_pred_value(self, grounded_pred):
        """
        Retrieves the predicate value.
        """
        pred_key = self._keyify(grounded_pred)
        return self.pred_dict.get(pred_key, {})
    
    def add_pred_list(self, new_pred_list):
        """
        Find and add new predicate dictionaries to the predicate state.
        new_pred_list::dict:: A list of dictionary representing the new predicate [{'name': str, 'params': tuple/list, ...}]
        """
        for new_pred in new_pred_list:
            pred_key = self._keyify(new_pred)
            if pred_key not in self.pred_dict:
                # ensure not to add repeated predicate because predicate has redundant parameters
                # hardcoded
                assert all(key in ["name", "types", "params", "semantic"] for key in new_pred.keys())
                self.pred_dict[pred_key] = None
            # # remove this text after tested
            # else:
            #     print(f"Predicate {dict_to_string(new_pred, lifted=True)} already exists.")

    def get_unevaluated_predicates(self):
        """
        Returns a list of predicates that do not have truth value evaluated.
        """
        return [pred for pred,truth_value in self.pred_dict.items() if not truth_value]

def dict_to_string(dict, lifted=False):
    """
    Assembly a dictionary of grounded/lifted representation to string
    dict::dict:: {'name':str, 'params':list}
    """
    variable_string = ', '.join(dict['types']) if lifted else ', '.join(dict['params'])
    return f"{dict['name']}({variable_string})"

def ground_with_params(lifted, params, type_dict):
    """
    Grounded a skill or a predicate with parameters and their types.
    Basically add another key, value to it
    lifted::dict:: lifted skill/predicate with parameter type, e.g., {'name':"GoTo", 'types':["[ROBOT]", "[LOC]", "[LOC]"]}/{'name':"At", 'types':["[LOC]"]}
    params::list:: list of parameters, e.g., ["Apple", "Table"]
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    """
    # tuple is applicable to the lifted representation
    # assert len(lifted['types']) == len(params), f"number of types ({len(lifted['types'])}) not agree with number of params ({len(params)})"
    for i, p in enumerate(params):
        assert p in type_dict
        assert lifted['types'][i] in type_dict[p]

    # grounded skill/predicate
    # return {'name': lifted['name'], 'params': params}
    grounded_pred = deepcopy(lifted)
    grounded_pred["params"] = params
    return grounded_pred

def possible_grounded_predicates(pred_list, type_dict):
    """
    Generate all possible grounded predicate using the combination of predicates and objects
    pred_list:: [{'name':str, 'types':list, 'params':list, 'semantic':str}]
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    return:: list:: list of possible grounded predicates
    """
    # build inverse type_dict
    type_dict_inv = defaultdict(list)
    for param, type_ls in type_dict.items():
        for type  in type_ls:
            type_dict_inv[type].append(param)
    
    # generate all possible grounded predicates
    grounded_predicates = []
    for pred in pred_list:
        for params in itertools.product(*[type_dict_inv[p] for p in pred['types']]):
            grounded_predicates.append(ground_with_params(pred, list(params), type_dict))
    return grounded_predicates

def calculate_pred_to_update(grounded_predicates, grounded_skill):
    '''
    Given a skill and its parameters, find the set of predicates that need updates
    grounded_skill::dict:: {'name':str, 'types':list,'params':list}
    grounded_predicates::list:: list of grounded predicates, e.g., [{'name': 'At', 'params': ['Apple', 'Table']}]
    '''
    # TODO: always include predicates that has 0 arity after every step
    return [gp for gp in grounded_predicates if any([p in gp['params'] for p in grounded_skill['params']])]

def lift_grounded_pred(grounded_pred, type_dict):
    """
    Lift a grounded predicate, e.g., {'name': 'At', 'types': ["object", "location"], 'params': ['Apple', 'Table']} to a {'name':"At", 'types':["object", "location"], 'params':[]}
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    """
    assert all([type in type_dict[param] for type, param in zip(grounded_pred['types'], grounded_pred['params'])])
    return {'name':grounded_pred['name'], 'types':grounded_pred['types'], 'params':grounded_pred['params'], 'semantic':grounded_pred['semantic']}
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

def eval_pred(model, img, grounded_skill, grounded_pred, type_dict, prompt_fpath=['prompts/evaluate_pred_ai2thor.txt','prompts/evaluate_pred_ai2thor_init.txt'], init=False):
    '''
    evaluate truth value of a predicate using a dictionary of parameters
    init step and later steps use different prompts. harded coded.
    grounded_skill::dict:: grounded skill with parameter type, e.g., {'name':'GoTo', "params":['location', 'location']}
    grounded_pred::dict:: grounded predicate with parameter type, e.g., {'name':"At", 'params':["location"]}
    '''
    def construct_prompt(prompt, grounded_skill, grounded_pred):
        "replace placeholders in the prompt"
        # return none if none of the parameters of the grounded predicate match any of the parameters of the grounded skill
        if not any([p in grounded_pred['params'] for p in grounded_skill['params']]):
            return None
        place_holders = ['[GROUNDED_SKILL]', '[GROUNDED_PRED]','[LIFTED_PRED]', '[SEMANTIC]']
        lifted_pred = lift_grounded_pred(grounded_pred, type_dict)
        while any([p in prompt for p in place_holders]):
            prompt = prompt.replace('[GROUNDED_SKILL]', grounded_skill)
            prompt = prompt.replace('[GROUNDED_PRED]', grounded_pred)
            prompt = prompt.replace('[LIFTED_PRED]', dict_to_string(lifted_pred, lifted=True))
            prompt = prompt.replace('[SEMANTIC]', lifted_pred['semantic'])
        return prompt
    
    prompt = load_from_file(prompt_fpath[0]) if not init else load_from_file(prompt_fpath[1])
    prompt = construct_prompt(prompt, grounded_skill, grounded_pred)
    logging.info(f'Evaluating predicate {grounded_pred} on skill {grounded_skill}')
    if prompt:
        logging.info('Calling GPT4')
        resp = model.generate_multimodal(prompt, img)[0]
        result = True if "True" in resp.split('\n')[-1] else False
        logging.info(f'{grounded_pred} evaluated to {result}')
        return result
    else:
        logging.info(f"mismatch skill and predicate: return False\n{grounded_skill} / {grounded_pred}")
        return False

def generate_pred(model, skill, pred_list, pred_type, tried_pred=[], prompt_fpath='prompts/predicate_refining'):
    '''
    propose new predicates based on the contrastive pair
    '''
    # TODO: add tried_pred back to the prompt
    def construct_prompt(prompt, grounded_skill, pred_list):
        "replace placeholders in the prompt"
        while "[SKILL]" in prompt or "[PRED_DICT]" in prompt or "[TRIED_PRED]" in prompt:
            prompt = prompt.replace("[SKILL]",  dict_to_string(grounded_skill))
            # construct predicate list from pred_dict
            pred_list_str = '\n'.join([f'{dict_to_string(pred, lifted=True)}: {pred["semantic"]}' for pred in pred_list])
            prompt = prompt.replace("[PRED_LIST]", pred_list_str)
            prompt = prompt.replace("[TRIED_PRED]", ", ".join([dict_to_string(pred, lifted=True) for pred in tried_pred]))

    prompt_fpath += f"_{pred_type}.txt"
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred_list)
    logging.info('Generating predicate')
    resp = model.generate(prompt)[0]
    pred, sem = resp.split(': ', 1)[0].strip('`'), resp.split(': ', 1)[1].strip()
    # separate the parameters from the predicate into dictionary
    # e.g., "At(obj, loc)"" -> {"name":"At", "types": ["obj", "loc"]}
    new_pred = {'name': pred.split("(")[0], 'types': pred.split("(")[1].strip(")").split(", ")}
    new_pred['semantic'] = sem
    return new_pred

# Adding to precondition or effect are different prompts
def update_empty_predicates(model, tasks, lifted_pred_list, type_dict,  grounded_predicate_truth_value_log):
    '''
    Find the grounded predicates with missing values and evaluate them.
    The grounded predicates are evaluated from the beginning to the end, and then lifted to the lifted predicates.
    lifted_pred_list::list(dict):: List of all lifted predicates
    grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
    tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    '''
    # NOTE: step is a integer ranging from 0-8, where 0 is the init step and success==None. 1-8 are states after executions

    # look for predicates that haven't been evaluated
    # The truth values could be missing if:
    #    1. the predicate is newly added (assuming all possible grounded predicates are added, including the init step)
    #    2. a task is newly executed
    # NOTE: the dictionary could be partially complete because some truth values will be directly reused from the scoring function
    
    # generate all possible grounded predicates that match object types
    grounded_pred_list = possible_grounded_predicates(lifted_pred_list, type_dict)
    logging.info('looking for empty grounded predicates')
    for task_id, steps in tasks.items():
        # NOTE: might not be necessary. tasks always get updated after every execution
        if task_id not in grounded_predicate_truth_value_log:
            grounded_predicate_truth_value_log[task_id] = {}
            for step in steps:
                grounded_predicate_truth_value_log[task_id][step] = PredicateState(grounded_pred_list)

        # for each step, iterate through all steps and find empty predicates and update them
        # calculate predicates to update based on the last action every step after init
        # 
        # only empty predicates will be updated
        # init step is updated separately

        # update predicates for all states 
        for step, state in steps.items():
            grounded_predicate_truth_value_log[task_id][step].add_pred_list(lifted_pred_list)

            # 1. find states need to be re-eval
            pred_to_update = grounded_pred_list if step == 0 else calculate_pred_to_update(grounded_pred_list, state["skill"])

            # 2. re-eval grounded predicates
            for pred in pred_to_update:
                # only update empty predicates
                if not grounded_predicate_truth_value_log[task_id][step].get_truth_value(pred):
                    truth_value = eval_pred(model, state["image"], state["skill"], pred, type_dict, init=step==0)
                    grounded_predicate_truth_value_log[task_id][step].set_pred_value(pred, truth_value)
            
            # 3.copy all empty predicates from previous state
                elif not step==0: # non-init state, 
                    truth_value =  grounded_predicate_truth_value_log[task_id][step].get_pred_value(pred)
                    grounded_predicate_truth_value_log[task_id][step].set_pred_value(pred, truth_value)
    logging.info('Done evaluation from mismatch_symbolic_state')
    return grounded_predicate_truth_value_log


def grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks):
    '''
    helper function to convert grounded predicate log into skill2task2state for predicate invention
    grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
    tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
    pred_type :str: {"precond", "eff"}
    returns:
    skill2task2state::{skill_keyified: {task_step_name: [PredicateState, PredicateState]}}
        skill_keyified :: (skill_name | str, skill_types | set(str), skill_params | set(str))
        task_step_name :: (task_name | str, step | int)
    '''
    skill2task2state = defaultdict(dict)
    for task_name, steps in grounded_predicate_truth_value_log.items():
            for step, state in steps.items(): # state :: PredicateState class
                if not step == 0: # init state has no skill, and thus won't be included in (task_name, step)
                    skill = tasks[task_name][step]["skill"]
                    task_name_stepped = (task_name, step)
                    # skill in a dictionary so need keyifying
                    skill_keyified = PredicateState._keyify(skill)
                    skill2task2state[skill_keyified][task_name_stepped] = [last_state, state]
                last_state = deepcopy(state)
    return skill2task2state

# TODO: change the criteria to the latest one
def detect_mismatch(grounded_predicate_truth_value_log, tasks, pred_type):
    '''
    find mismatch *grounded* states.
    grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
    pred_type::{'precond', 'eff'}
    '''
    def tasks_with_same_symbolic_states(task2state: dict):
        '''
        return:: {key:[duplicated_keys]}
        '''
        element_count = defaultdict(list)
        for key, value in task2state.items():
            element_tuple = tuple(sorted(value.items()))
            element_count[element_tuple].append(key)
        duplicates = {keys[0]: keys[1:] for keys in element_count.values() if len(keys) > 1}
        return duplicates
    
    # represent all tasks using (task,step) tuple as a key and list of list.{(task,step): PredicateState}
    dup_tasks = {}
    # precondition cares only about s
    skill2task2state = grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks)
    state_idx = 0 if pred_type==0 else 1
    for skill_keyified, task_name_stepped2state in skill2task2state.items():
        task2state = {task_name_stepped: state_tuple[state_idx] for task_name_stepped, state_tuple in task_name_stepped2state.items()}
        dup_tasks[skill_keyified] = tasks_with_same_symbolic_states(task2state)
    
    # skill2task2state :: {skill_keyified: {task_name_stepped: [PredicateState, PredicateState]}}
    # dup_tasks:: {skill_keyified: [{task_name_stepped:[task_name_stepped]}],...]}

    # TODO: fix this part
    # calculating mismatch
    mismatch_pairs = {}
    for skill_keyified, dups in dup_tasks.items():
        if dups:
            # (we do?) for effect we have to make sure successful execution have state changes
            tasks = {p: skill2tasks[skill][p] for p in list(dups.values())[0]} # task to success/fail
            tasks[list(dups.keys())[0]] = skill2tasks[skill_keyified][list(dups.keys())[0]] # ?
            task2success = {}
            success_tasks = {id: t for id, t in tasks.items() if t['success']}
            failed_tasks = {id: t for id, t in tasks.items() if not t['success']}
            if len(success_tasks) > 0 and len(failed_tasks) > 0:
                mismatch_pairs[skill_keyified] = [random.choice(list(success_tasks.keys())), random.choice(list(failed_tasks.keys()))]
    return mismatch_pairs

def invent_predicate_one(model, skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type,  skill2triedpred={}, threshold={"precond":0.5, "eff":0.5}):
    """
    One iteration of predicate invention.

    """
    # TODO: try different predicate invention: e.g., images, comparisons. Now it's just blindly generating predicates
    new_pred = generate_pred(model, skill, lifted_pred_list, pred_type, tried_pred=skill2triedpred[skill])
    logging.info(f"new predicate {new_pred}")
    new_pred_accepted = False
    # evaluate the new predicate on all states
    # suppose we add the new predicate to the current predicate set
    hypothetical_pred_list = deepcopy(lifted_pred_list)
    hypothetical_pred_list.append(new_pred)
    hypothetical_grounded_predicate_truth_value_log = deepcopy(grounded_predicate_truth_value_log)
    # TODO: Add skill option to update_empty_predicates() only update empty predicates for one skill to speed up and reduce cost
    # task unchanged, only add candidate predicate
    hypothetical_grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, hypothetical_pred_list, type_dict, hypothetical_grounded_predicate_truth_value_log)
    hypothetical_skill2task2state = grounded_pred_log_to_skill2task2state(hypothetical_grounded_predicate_truth_value_log, tasks, pred_type)
    # TODO: modify the score function
    t_score, f_score = score_by_partition(new_pred, skill, hypothetical_skill2task2state, hypothetical_pred_list, [], pred_type)
    logging.info(f"Precondition T score of predicate {dict_to_string(new_pred)}: {t_score}, F score: {f_score}")
    if t_score >= threshold[pred_type] and  f_score >= threshold[pred_type]:
        logging.info(f"Predicate {new_pred} added to predicate set by precondition check")
        lifted_pred_list.append(new_pred)
        new_pred_accepted = True
    else:
        skill2triedpred[skill].append(new_pred)
    
    return lifted_pred_list, skill2triedpred, new_pred, new_pred_accepted


def invent_predicates(model, skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, skill2triedpred={}, max_t=3):
    '''
    Main loop of generating predicates. It also evaluates empty predicates that introduced by new tasks or new predicates
    '''
    # TODO: number of if statements to create empty dicts or lists for on init
    
    # check precondition first
    t = 0
    pred_type = "precond"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log)
    mismatch_tasks = detect_mismatch(grounded_predicate_truth_value_log, tasks, pred_type=pred_type)
    logging.info("About to enter precondition check")
    logging.info(f"mismatch state for precondition generation {mismatch_tasks[skill][0]}, {mismatch_tasks[skill][1]}")
    while skill in mismatch_tasks and t < max_t:
        lifted_pred_list, skill2triedpred, new_pred, new_pred_accepted = invent_predicate_one(model, skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type,  skill2triedpred=skill2triedpred)
        logging.info(f"Iteration {t} of predicate invention. {new_pred} is accepted: {new_pred_accepted}")
        t += 1
    
    # check effect
    t = 0
    pred_type = "eff"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log)
    mismatch_tasks = detect_mismatch(grounded_predicate_truth_value_log, tasks, pred_type=pred_type)
    logging.info("About to enter effect check")
    logging.info(f"mismatch state for precondition generation {mismatch_tasks[skill][0]}, {mismatch_tasks[skill][1]}")
    while skill in mismatch_tasks and t < max_t:
        lifted_pred_list, skill2triedpred, new_pred, new_pred_accepted = invent_predicates(model, skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type,  skill2triedpred=skill2triedpred)
        logging.info(f"Iteration {t} of predicate invention. {new_pred} is accepted: {new_pred_accepted}")
        t += 1
    
    # partitioning
    # TODO: update partition_by_effect function
    partitioned_output = partition_by_effect(grounded_predicate_truth_value_log)
    
    return pred_list, skill2triedpred

def score_by_partition(new_pred, skill, skill2task2state, pred_list, pred_type):
    '''
    Partition by effect and then score the predicates across each partition
    skill :: grouded skill {"name":"PickUp", "types":["obj"], "params":["Apple"]}
    '''
    task2state = skill2task2state[skill]
    # 1. find all states after executing the same grounded skill
    state2partition = partition_by_termination(task2state)
    # 2. evaluate the score for each task2state dictionary, pick the best one
    for state, partition in state2partition.itesm():
        score(new_pred, skill, pred_type)
    pass

def partition_by_termination(task2state):
    '''
    Partition the a set of trajectory using termination set. Will be used again in scoring and final operators learning.
    Return a list of task2state?
    task2state:: {(task,step):[PredicateState, PredicateState]}
    grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
    return:: [task2state]
    '''
    def states_are_equal(state_1, state_2):
        return state_1.pred_dict == state_2.pred_dict
    
    state2partition: {PredicateState | list} = defaultdict(list)
    for task_step_id, state in task2state.items():
        find_partition = False
        for registered_state in state2partition:
            if states_are_equal(state.pred_dict, registered_state):
                state2partition[state.pred_dict].append(task_step_id)
                find_partition = True
        if not find_partition:
            state2partition[state.pred_dict].append(task_step_id)
            find_partition = False
    return state2partition

def partition_by_effect(pred_dict):
    'calculate partitioned skill from pred_dict'
    def convert_to_task2pred(pred_dict):
        'Returns {task:{pred:[s_0, s_1]}}'
        task2pred = {}
        for pred, content in pred_dict.items():
            tasks = content['task']
            for t, values in tasks.items():
                if not t in task2pred:
                    task2pred[t] = {}
                task2pred[t][pred] = values
        return task2pred
    def group_nested_dict(nested_dict):
        'calculate precondition and effect based on task2pred'
        'Effect is calculated as change of truth values'
        'Precondition is the intersection of all abstract states before execution'
        result = defaultdict(lambda: {'task': [], 'effect': {}, 'precondition': {}})

        # effect
        for outer_key, inner_dict in nested_dict.items():
            value_dict = {k: v[1] - v[0] for k, v in inner_dict.items() if v[1] - v[0] != 0}
            group_id = None
            for gid, group in result.items():
                if group['effect'] == value_dict:
                    group_id = gid
                    break
            if group_id is None:
                group_id = len(result)
                result[group_id]['effect'] = value_dict
            result[group_id]['task'].append(outer_key)

        result = dict(result)
        # precondition
        for k, v in result.items():
            cluster = v['task']
            temp_dict = defaultdict(list)
            for task in cluster:
                for pred in nested_dict[task]:
                    temp_dict[pred].append(nested_dict[task][pred][0])
            for pred, v_list in temp_dict.items():
                if len(set(v_list)) == 1:
                    result[k]['precondition'][pred] = v_list[0]

        return result
    
    task2pred = convert_to_task2pred(pred_dict)
    suc_task2pred = {}
    # only success tasks will be use to calculate precondition
    for task, value in task2pred.items():
        if not "False" in task:
            suc_task2pred[task] = value
    partitioned_tasks = group_nested_dict(suc_task2pred)

    return partitioned_tasks

def score(pred, tasks, task2state, pred_type):
    """
    score of a predicate as one skill's precondition or effect
    tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
    grounded_predicate_truth_value_log::dict::{task:{step:[{'name':str, 'params':list, 'truth_value':bool}]}}
    type : {precond, eff}
    """
    # skill2task2state :: {skill_name: {task_step_name: [PredicateState, PredicateState]}}
    # task_step_name=(task_name, step)

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
        state = task2state[(task_name, step)][state_idx]
        success = tasks[task_name][step]["success"]
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
    model = GPT4(engine='gpt-4o-2024-08-06')
    # mock symbolic state
    pred_list = [{'name':'handEmpty', 'types':[],'params':[], 'semantic': "The robot's hand is empty"}]
    type_dict = {"Robot": ["robot"], "Apple": ['object'], "Banana": ['object'], "Table": ['location'], "Couch": ['location']}
    skill_1 = {'name': 'PickUp', 'types':['object'], 'params':[]}
    skill_2 = {"name": "GoTo", "types": ['location'], 'params':[]}
    skill_3 = {"name": "PlaceAt", "types": ['object', 'location'], 'params':[]}
    skill_list = [skill_1, skill_2, skill_3]
    # pred_type = 'eff'
    pred_type = 'precond'
    # response = generate_pred(model, skill, pred_list, pred_type)
    # print(response)
    example_lifted_predicates = [
        {'name':"At", 'types':["object", "location"], 'params':[]},
        {'name':"CloseTo", 'types':["robot", "location"], 'params':[]},
        {'name':"HandOccupied", 'types':[], 'params':[]},
    ]
    add_predicates = [
        {'name': "IsHolding", "types":["object"], 'params':[]},
        {'name': "EnoughBattery", "types": [], 'params':[]},
        {'name':'handEmpty', 'types':[],'params':[], 'semantic': "The robot's hand is empty"}
    ]
    test_pred_ls = PredicateState(example_lifted_predicates)
    breakpoint()
    test_pred = {'name':"At", 'types':["object", "location"], 'params':[]}
    value = test_pred_ls.get_pred_value(test_pred)

    test_pred_ls.add_pred_list(add_predicates)

    possible_preds = possible_grounded_predicates(test_pred_ls.get_pred_list(lifted=True), type_dict)
    
    grounded_skill = {"name": "PlaceAt", "types": ['object', 'location'], 'params':['Banana', 'Couch']}
    pred_to_update = calculate_pred_to_update(possible_preds, grounded_skill)
    breakpoint()

    # test partitioning and scoring
    # construct multiple PredicateState instances
    s_0 = deepcopy(test_pred_ls)
    s_0.set_pred_value(

    )
    # manually construct grounded_predicate_truth_value_log ::dict:: {task:{step:PredicateState}}
    grounded_predicate_truth_value_log = {
        ('1',0): None
    }