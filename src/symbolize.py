'''
Get symbolic representation from skill semantic info and observation
Data structures:
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    lifted_pred_list:: [{'name':str, 'types':list, 'semantic':str}]
    tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
    grounded_predicate_truth_value_log::dict::{task:{step:[{'name':str, 'params':list, 'truth_value':bool}]}}
    # lifted_predicate_truth_value_log::dict::{pred_name:{task: {id: [Bool, Bool]}, sem: str}}
'''
from utils import GPT4, load_from_file
from collections import defaultdict
import inspect
from copy import deepcopy
import random
import itertools
import logging

class PredicateState:
    def __init__(self, predicates):
        """
        Initializes the predicate state.
        predicates::list:: A list of predicate dictionaries {'name': str, 'params': tuple/list, ...}
        """
        self.pred_dict = {self._keyify(pred): None for pred in predicates}

    def _keyify(self, pred):
        """Generates a unique key from predicate name and parameters."""
        return (pred["name"], tuple(pred["params"]))

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
            # remove this text after tested
            else:
                print(f"Predicate {dict_to_string(new_pred, lifted=True)} already exists.")

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
    assert len(lifted['types']) == len(params)
    for i, p in enumerate(params):
        assert p in type_dict
        assert lifted['types'][i] in type_dict[p]

    # grounded skill/predicate
    # return {'name': lifted['name'], 'params': params}
    grounded_pred = deepcopy(lifted)
    grounded_pred["params"] = params
    return grounded_pred

# TODO: revise data structure under tasks, record grounding parameters
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
        for params in itertools.product(*[type_dict_inv[p] for p in pred['params']]):
            grounded_predicates.append(ground_with_params(pred, params, type_dict))
    
    return grounded_predicates

def calculate_pred_to_update(grounded_predicates, skill):
    '''
    Given a skill and its parameters, find the set of predicates that need updates
    skill::dict:: {'name':str, 'params':list}
    grounded_predicates::list:: list of grounded predicates, e.g., [{'name': 'At', 'params': ['Apple', 'Table']}]
    '''
    return [gp for gp in grounded_predicates if any([p in gp['params'] for p in skill['params']])]

def lift_grounded_predicate(grounded_pred, type_dict, lifted_pred_list):
    """
    Lift a grounded predicate, e.g., {'name': 'At', 'params': ['Apple', 'Table']} to a {'name':"At", 'params':["object", "location"]}
    """
    assert all([p in type_dict for p in grounded_pred['params']])
    output_list = []
    # iterate through the pred_list to find the lifted representation
    for pred in lifted_pred_list:
        if pred['name'] == grounded_pred['name']:
            if all([p in type_dict for p in pred['params']]):
                output_list.append(pred)
    # cannot have more than one matched lifted predicate
    # e.g., existing two predicates At(a, b) and At(a, c) while the grounded version At(apple, banana) has type(banana) = [b, c]
    assert len(output_list) == 1
    return output_list[0]

# Not used
# evaluate an execution using foundation model. Expected acc to be ~ 70%
def eval_execution(model, skill, consecutive_pair, prompt_fpath='prompts/evalutate_task.txt'):
    'Get successfulness of the execution given images and the skill name'
    def construct_prompt(prompt, skill):
        while "[SKILL]" in prompt:
                prompt = prompt.replace("[SKILL]", skill)
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill)
    return model.generate_multimodal(prompt, consecutive_pair)[0]

def eval_pred(model, img, grounded_skill, grounded_pred, type_dict, lifted_pred_list, prompt_fpath=['prompts/evaluate_pred_ai2thor.txt','prompts/evaluate_pred_ai2thor_init.txt'], init=False):
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
        lifted_pred = lift_grounded_predicate(grounded_pred, type_dict, lifted_pred_list)
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
            prompt = prompt.replace("[TRIED_PRED]", [dict_to_string(pred, lifted=True) for pred in tried_pred])

    prompt_fpath += f"_{pred_type}.txt"
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred_list)
    logging.info('Generating predicate')
    resp = model.generate(prompt)[0]
    pred, sem = resp.split(': ', 1)[0].strip('`'), resp.split(': ', 1)[1].strip()
    # separate the parameters from the predicate into dictionary
    # e.g., "At(obj, loc)"" -> {"name":"At", "params": ["obj", "loc"]}
    new_pred = {'name': pred.split("(")[0], 'params': pred.split("(")[1].strip(")").split(", ")}
    new_pred['semantic'] = sem
    return new_pred

# Adding to precondition or effect are different prompts
def update_empty_predicates(model, tasks, grounded_pred_list, lifted_pred_list, type_dict,  grounded_predicate_truth_value_log):
    '''
    Find the grounded predicates with missing values and evaluate them.
    The grounded predicates are evaluated from the beginning to the end, and then lifted to the lifted predicates.
    lifted_pred_list::list(dict):: List of all lifted predicates
    grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
    # lifted_predicate_truth_value_log::dict:: {pred_name:{task: {id: [Bool, Bool]}, sem: str}}
    # skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    '''
    # NOTE: step is a integer ranging from 0-8, where 0 is the init step and success==None. 1-8 are states after executions

    # look for predicates that haven't been evaluated
    # The truth values could be missing if:
    #    1. the predicate is newly added (assuming all possible grounded predicates are added, including the init step)
    #    2. a task is newly executed
    # NOTE: the dictionary could be partially complete because some truth values will be directly reused from the scoring function
    
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
                    truth_value = eval_pred_new(model, state["image"], state["skill"], pred, type_dict, lifted_pred_list, init=step==0)
                    grounded_predicate_truth_value_log[task_id][step].set_pred_value(pred, truth_value)
            
            # 3.copy all empty predicates from previous state
                elif not step==0: # non-init state, 
                    truth_value =  grounded_predicate_truth_value_log[task_id][step].get_pred_value(pred)
                    grounded_predicate_truth_value_log[task_id][step].set_pred_value(pred, truth_value)
    logging.info('Done evaluation from mismatch_symbolic_state')
    return grounded_predicate_truth_value_log

def detect_mismatch(grounded_predicate_truth_value_log, tasks, pred_type):
    '''
    find mismatch *grounded* states.
    grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
    pred_type::{'precond', 'eff'}
    '''
    def tasks_with_same_symbolic_states(task2state):
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
    if pred_type == "precond":
        skill2task2state = defaultdict(dict)
        for task_name, steps in grounded_predicate_truth_value_log.items():
            for step, state in steps.items(): # state :: PredicateState class
                if not step == 0: # init state has no skill
                    skill = tasks[task_name][step]["skill"]
                    task_name_stepped = (task_name, step) 
                    skill2task2state[skill][task_name_stepped] = state.pred_dict # all predicates should have truth value
        for skill, task_name_stepped2pred in skill2task2state.items():
            dup_tasks[skill] = tasks_with_same_symbolic_states(task_name_stepped2pred)
    elif pred_type == "eff":
        skill2task2state = defaultdict(dict)
        for task_name, steps in grounded_predicate_truth_value_log.items():
            for step, state in steps.items():
                if not step == 0:
                    skill = tasks[task_name][step]["skill"]
                    task_name_stepped = (task_name, step)
                    skill2task2state[skill][task_name_stepped] = {pred: int(truth_value) - int(last_state.pred_dict[pred]['task']) for pred, truth_value in state.pred_dict.items()}
                last_state = state
        for skill, task_name_stepped2pred in skill2task2state.items():
            dup_tasks[skill] = tasks_with_same_symbolic_states(task_name_stepped2pred)
    
    # skill2task2state :: {skill_name: {task_step_name: {pred: bool}}}
    # dup_tasks:: {skill_name: [[task+step_name,...], []]}

    # calculating mismatch
    mismatch_pairs = {}
    for skill, pairs in dup_tasks.items():
        if pairs:
            # for effect we have to make sure successful execution will have state changes
            tasks = {p: skill2tasks[skill][p] for p in list(pairs.values())[0]}
            tasks[list(pairs.keys())[0]] = skill2tasks[skill][list(pairs.keys())[0]]
            success_tasks = {id: t for id, t in tasks.items() if t['success']}
            failed_tasks = {id: t for id, t in tasks.items() if not t['success']}
            if len(success_tasks) > 0 and len(failed_tasks) > 0:
                mismatch_pairs[skill] = [random.choice(list(success_tasks.keys())), random.choice(list(failed_tasks.keys()))]
    return mismatch_pairs

######################## NEW SYSTEM WITH PARTITIONING ##############################
def refine_pred_new(model, skill, skill2operators, skill2tasks, pred_dict, skill2triedpred={}, max_t=3):
    """
    New predicate refining function, will only add new predicates when there's a pair of mismatch states
    Only one union predicates set shared by all skills
    no skill2operators, only pred_dict is used
    """
    if not pred_dict:
        pred_dict = {}

    if not skill2operators:
        skill2operators = {}
    if not skill in skill2operators:
        skill2operators[skill] = [] # now a single list to hold all related predicates

    if not skill2triedpred:
        skill2triedpred = {}
    if not skill in skill2triedpred:
        skill2triedpred[skill] = [] # a list for tried predicates too
    
    # check precondition first
    t = 0
    pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'precond')
    new_p_added = False
    # breakpoint()
    logging.info("About to enter precondition refinement")
    while skill in mismatch_tasks and t < max_t:
        new_p, sem = generate_pred(model, skill, list(pred_dict.keys()),  'precond', tried_pred=skill2triedpred[skill])
        logging.info(f"mismatch state for precondition generation {mismatch_tasks[skill][0]}, {mismatch_tasks[skill][1]}")
        logging.info(f"new predicate {new_p}, {sem}")
        # evaluate the new predicate on all states
        new_p_tv = {idx: [eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s0']), eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s1'])] for idx in skill2tasks[skill]}
        pred_dict_tmp = deepcopy(pred_dict)
        pred_dict_tmp[new_p] = {}
        pred_dict_tmp[new_p]['task'] = new_p_tv
        pred_dict_tmp[new_p]['semantic'] = sem
        tscore_precond, fscore_precond = score(new_p, skill, skill2tasks, pred_dict_tmp, [], 'precond')
        logging.info(f"Precond tscore: {tscore_precond}, fscore: {fscore_precond}")
        # if new_p_mismatch[mismatch_tasks[skill][0]][0] != new_p_mismatch[mismatch_tasks[skill][1]][0]:
            # skill2operators[skill]['precond'][new_p] = True if skill2tasks[skill][mismatch_tasks[skill][0]]['success'] == new_p_mismatch[mismatch_tasks[skill][0]][0] else False

        # use tscore and fscore to evaluate new_p
        if tscore_precond >= 0.5 and  fscore_precond >= 0.5:
            logging.info(f"Predicate {new_p} added to predicate set by precondition check")
            new_p_added = True
            if new_p_added and new_p not in pred_dict:
                pred_dict[new_p] = {'task': {}}
                if ('[LOC]' in new_p) and ('[OBJ]' not in new_p):
                    new_p_goto_1 = new_p.replace('[LOC]', '[LOC_1]')
                    new_p_goto_2 = new_p.replace('[LOC]', '[LOC_2]')
                    if new_p_goto_1 not in pred_dict:
                        pred_dict[new_p_goto_1] = {'task': {}}
                        pred_dict[new_p_goto_1]['semantic'] = sem
                    if new_p_goto_2 not in pred_dict:
                        pred_dict[new_p_goto_2] = {'task': {}}
                        pred_dict[new_p_goto_2]['semantic'] = sem
                    # breakpoint()
                elif ('[LOC_1]' in new_p) != ('[LOC_2]' in new_p): # only one loc in new predicate
                    new_p_not_goto = new_p.replace('[LOC_1]', '[LOC]').replace('[LOC_2]', '[LOC]')
                    if new_p_not_goto not in pred_dict:
                        pred_dict[new_p_not_goto] = {'task': {}}
                        pred_dict[new_p_not_goto]['semantic'] = sem
                    new_p_the_other = new_p.replace("[LOC_1]", '[LOC_2]') if "[LOC_1]" in new_p else new_p.replace("[LOC_2]", '[LOC_1]')
                    if not new_p_the_other in pred_dict:
                        pred_dict[new_p_the_other] = {'task': {}}
                        pred_dict[new_p_the_other]['semantic'] = sem
                    # breakpoint()
                
                logging.info('Evaluating for precond')
                pred_dict[new_p]['task'] = new_p_tv
                pred_dict[new_p]['semantic'] = sem

            logging.info(f'Done evaluating predicate {new_p} for all tasks')
            new_p_added = False
            pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'precond')
            logging.info(f'Done evaluating predicate {new_p} for all tasks')
            skill2triedpred[skill] = []
        else:
            skill2triedpred[skill].append(new_p)
        t += 1
    
    # check effect
    t = 0
    pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'eff')
    logging.info("About to enter effect refinement")
    # breakpoint()
    while skill in mismatch_tasks and t < max_t:
        new_p, sem = generate_pred(model, skill, list(pred_dict.keys()), 'eff', tried_pred=[skill])
        logging.info(f"mismatch state for effect generation {mismatch_tasks[skill][0]}, {mismatch_tasks[skill][1]}")
        logging.info(f'new predicate {new_p}, {sem}')
        new_p_tv = {idx: [eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s0']), eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s1'])] for idx in skill2tasks[skill]}
        pred_dict_tmp = deepcopy(pred_dict)
        pred_dict_tmp[new_p] = {}
        pred_dict_tmp[new_p]['task'] = new_p_tv
        pred_dict_tmp[new_p]['semantic'] = sem
        tscore_eff, fscore_eff = score(new_p, skill, skill2tasks, pred_dict_tmp, [], 'eff')
        logging.info(f"Effect tscore: {tscore_eff}, fscore: {fscore_eff}")
        # eff representation might be wrong (s1-s2). The result value could be {-1, 0, 1}, 0 cases could be wrong?
        if abs(tscore_eff) >= 0.2 and fscore_eff >= 0.2:
            logging.info(f"Predicate {new_p} added to the predicate set by effect check")
            new_p_added = True
            if new_p_added and new_p not in pred_dict:
                pred_dict[new_p] = {}
                if ('[LOC]' in new_p) and ('[OBJ]' not in new_p):
                    new_p_goto_1 = new_p.replace('[LOC]', '[LOC_1]')
                    new_p_goto_2 = new_p.replace('[LOC]', '[LOC_2]')
                    if new_p_goto_1 not in pred_dict:
                        pred_dict[new_p_goto_1] = {'task': {}}
                        pred_dict[new_p_goto_1]['semantic'] = sem
                    if new_p_goto_2 not in pred_dict:
                        pred_dict[new_p_goto_2] = {'task': {}}
                        pred_dict[new_p_goto_2]['semantic'] = sem
                    # breakpoint()
                elif ('[LOC_1]' in new_p) != ('[LOC_2]' in new_p): # only one loc in new predicate
                    new_p_not_goto = new_p.replace('[LOC_1]', '[LOC]').replace('[LOC_2]', '[LOC]')
                    if new_p_not_goto not in pred_dict:
                        pred_dict[new_p_not_goto] = {'task': {}}
                        pred_dict[new_p_not_goto]['semantic'] = sem
                    new_p_the_other = new_p.replace("[LOC_1]", '[LOC_2]') if "[LOC_1]" in new_p else new_p.replace("[LOC_2]", '[LOC_1]')
                    if not new_p_the_other in pred_dict:
                        pred_dict[new_p_the_other] = {'task': {}}
                        pred_dict[new_p_the_other]['semantic'] = sem
                    # breakpoint()   
                pred_dict[new_p]['task'] = new_p_tv            
                pred_dict[new_p]['semantic'] = sem
            logging.info(f'Done evaluating predicate {new_p} for all tasks')
            new_p_added = False
            pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'eff')
            skill2triedpred[skill] = []
        else:
            skill2triedpred[skill].append(new_p)
        t += 1
    # breakpoint()
    
    # partitioning
    partitioned_output = partition_by_effect(pred_dict)
    grounded_skill_dictionary = defaultdict(dict)
    for idx, operator in partitioned_output.items():
        base_action_name = operator['task'][0].split('_')[0]
        action_counter = 1

        # hardcode skill name, will remove after workshop
        if "PickUp" in base_action_name or "DropAt" in base_action_name:
            action_name = f"{base_action_name}_{action_counter}(obj, loc)"
        else:
            action_name = f"{base_action_name}_{action_counter}(init, goal)"
        while action_name in grounded_skill_dictionary:
            action_counter += 1
            if "PickUp" in base_action_name or "DropAt" in base_action_name:
                action_name = f"{base_action_name}_{action_counter}(obj, loc)"
            else:
                action_name = f"{base_action_name}_{action_counter}(init, goal)"
        grounded_skill_dictionary[action_name]['task'] = operator['task']
        grounded_skill_dictionary[action_name]['precondition'] = {p.replace('([OBJ]', '(obj').replace('[OBJ])', 'obj)').replace('([LOC_1]', '(init').replace('[LOC_2])', 'goal)').replace('([LOC]', '(loc').replace('[LOC])', 'loc)'):value for p, value in operator['precondition'].items()}
        grounded_skill_dictionary[action_name]['effect'] = {p.replace('([OBJ]', '(obj').replace('[OBJ])', 'obj)').replace('([LOC_1]', '(init').replace('[LOC_2])', 'goal)').replace('([LOC]', '(loc').replace('[LOC])', 'loc)'):value for p, value in operator['effect'].items()}

    return skill2operators, pred_dict, skill2triedpred, grounded_skill_dictionary

############################# ORIGINAL REASSIGNMENT CODE ####################################
def merge_predicates(model, skill2operator, pred_dict, prompt_fpath='prompts/predicate_unify.txt'):
    '''
    merge predicates based on their semantic meaning
    skill2operator: {skill_name:{precond:{str:Bool}, eff:{str:int}}}
    pred_dict: {pred_name:{task: [Bool, Bool]}, semantic:str}
    '''
    def construct_prompt(prompt, skill2operator):
        for skill_name, operator in skill2operator.items():
            pred_ls = list(operator['precond'].keys()) + list(operator['eff'].keys())
            prompt += "\nSkill: " + skill_name
            prompt += "\nPredicates:\n" + "\n".join([f"- {pred}:{pred_dict[pred]['semantic']}" for pred in pred_ls])
            prompt += "\n"
        while "[OBJ]" in prompt or "[LOC]" in prompt or "[LOC_1]" in prompt or "[LOC_2]" in prompt:
            prompt = prompt.replace("[OBJ]", "obj")
            prompt = prompt.replace("[LOC]", "loc")
            prompt = prompt.replace("[LOC_1]", "init")
            prompt = prompt.replace("[LOC_2]", "goal")
        prompt += "\nEquivalent Predicates:"
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill2operator)

    response = model.generate(prompt)[0].split("Equivalent Predicates:")[0]
    equal_preds = response.split("\n\n")
    equal_preds = [pred.replace("-","").replace(" ","").split('|') for pred in equal_preds]

    unified_skill2operator = {}
    for skill, operator in skill2operator.items():
        unified_skill2operator[skill] = {}
        for pred_type, preds in operator.items():
            if not pred_type in unified_skill2operator[skill]:
                unified_skill2operator[skill][pred_type] = {}
            for pred in preds:
                dup = False
                for equal_pred in equal_preds:
                    equal_pred = [ p.replace('(obj', '([OBJ]').replace('obj)', '[OBJ])').replace('(init', '([LOC_1]').replace('goal)', '[LOC_2])').replace('(loc', '([LOC]').replace('loc)', '[LOC])') for p in equal_pred]
                    # _equal_pred = [p[:-10] for p in equal_pred] # remove [positive] and [negative] flag
                    pred_flagged = [p for p in equal_pred if pred in p]
                    # print(pred, equal_pred, pred_flagged)
                    if pred_flagged:
                        # check if the replaced predicate has same or opposite semantic
                        # print(pred_flagged, pred)
                        if pred_type == 'precond':
                            unified_skill2operator[skill][pred_type][equal_pred[0][:-10]] = True if ('[positive]' in pred_flagged[0] and '[positive]' in equal_pred[0]) or ('[negative]' in pred_flagged[0] and '[negative]' in equal_pred[0]) else False
                        elif pred_type == 'eff':
                            unified_skill2operator[skill][pred_type][equal_pred[0][:-10]] = skill2operator[skill][pred_type][pred] if ('[positive]' in pred_flagged[0] and '[positive]' in equal_pred[0]) or ('[negative]' in pred_flagged[0] and '[negative]' in equal_pred[0]) else -skill2operator[skill][pred_type][pred]
                        dup = True
                if not dup:
                    unified_skill2operator[skill][pred_type][pred] = skill2operator[skill][pred_type][pred]
    return unified_skill2operator, equal_preds

def refine_pred_new(model, skill, tasks, grounded_predicate_truth_value_log, type_dict, pred_list, skill2triedpred={}, max_t=3):
    '''
    Main loop of generating predicates. It also evaluates empty predicates that introduced by new tasks or new predicates
    '''
    # TODO: number of if statements to create empty dicts or lists for on init
    
    # precondition first
    # update empty predicates
    t = 0
    pred_type = "precond"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, pred_list, type_dict, grounded_predicate_truth_value_log)
    mismatch_tasks = detect_mismatch(grounded_predicate_truth_value_log, tasks, pred_type=pred_type)
    new_p_added = False
    logging.info("About to enter precondition refinement")
    while skill in mismatch_tasks and t < max_t:
        new_pred = generate_pred(model, skill, pred_list, pred_type, tried_pred=skill2triedpred[skill])
        logging.info(f"mismatch state for precondition generation {mismatch_tasks[skill][0]}, {mismatch_tasks[skill][1]}")
        logging.info(f"new predicate {new_pred}")
def score(pred, skill, skill2tasks, pred_dict, equal_preds, type):
        "score of a predicate as one skill's precondition or effect"
        "type : {precond, eff}"
        tasks = skill2tasks[skill]
        success_tasks = [id for id, t in tasks.items() if t['success']]
        fail_tasks = [id for id, t in tasks.items() if not t['success']]
        repeated = False
        # TODO: precondition could be false
        if type == 'precond':
            t_p, t_d, f_n, f_d = 0, 0, 0, 0
            for ps in equal_preds:
                if pred in ps:
                    repeated = True
                    for p in ps:
                        for t_suc in success_tasks:
                            t_p += 1 if pred_dict[p]['task'][t_suc][0] == True else 0
                            t_d += 1
                        for t_fail in tasks:
                            f_n += 1 if pred_dict[p]['task'][t_fail][0] == False and tasks[t_fail]['success'] == False else 0
                            f_d += 1 if pred_dict[p]['task'][t_fail][0] == False else 0
            if repeated == False:
                    for t_suc in success_tasks:
                        t_p += 1 if pred_dict[pred]['task'][t_suc][0] == True else 0
                        t_d += 1
                    for t_fail in tasks:
                        f_n += 1 if pred_dict[pred]['task'][t_fail][0] == False and tasks[t_fail]['success'] == False else 0
                        f_d += 1 if pred_dict[pred]['task'][t_fail][0] == False else 0
            if not tasks:
                return 0, 0
            elif t_d == 0 or not success_tasks:
                return 0, f_n/f_d
            elif f_d == 0 or not fail_tasks:
                return t_p/t_d, 0
            else:
                tscore_t = t_p/t_d
                fscore_t = f_n/f_d
            # calculate false score
            t_p, t_d, f_n, f_d = 0, 0, 0, 0
            for ps in equal_preds:
                if pred in ps:
                    repeated = True
                    for p in ps:
                        for t_suc in success_tasks:
                            t_p += 1 if pred_dict[p]['task'][t_suc][0] == False else 0
                            t_d += 1
                        for t_fail in tasks:
                            f_n += 1 if pred_dict[p]['task'][t_fail][0] == True and tasks[t_fail]['success'] == False else 0
                            f_d += 1 if pred_dict[p]['task'][t_fail][0] == True else 0
            if repeated == False:
                    for t_suc in success_tasks:
                        t_p += 1 if pred_dict[pred]['task'][t_suc][0] == False else 0
                        t_d += 1
                    for t_fail in tasks:
                        f_n += 1 if pred_dict[pred]['task'][t_fail][0] == True and tasks[t_fail]['success'] == False else 0
                        f_d += 1 if pred_dict[pred]['task'][t_fail][0] == True else 0
            if not tasks:
                return 0, 0
            elif t_d == 0 or not success_tasks:
                return 0, f_n/f_d
            elif f_d == 0 or not fail_tasks:
                return t_p/t_d, 0
            else:
                tscore_f = t_p/t_d
                fscore_f = f_n/f_d
            if tscore_t * fscore_t > tscore_f * fscore_f:
                tscore = tscore_t
                fscore = fscore_t
            else:
                tscore = tscore_f
                fscore = fscore_f
        if type == 'eff':
            # truth value change could be 1, 0, -1
            # tscore and fscore should both be high and agree with each other
            t, f, f_d = 0, 0, 0
            for ps in equal_preds:
                if pred in ps:
                    repeated = True
                    for p in ps:
                        t += sum([int(pred_dict[p]['task'][t_suc][1]==True) - int(pred_dict[p]['task'][t_suc][0]==True) for t_suc in success_tasks])
                    break
            
            if repeated == False:
                t = sum([int(pred_dict[pred]['task'][t_suc][1]==True) - int(pred_dict[pred]['task'][t_suc][0]==True) for t_suc in success_tasks])
                # breakpoint()
                if not success_tasks:
                    return 0, 0
                tscore = t/len(success_tasks)
            else:
                if not success_tasks:
                    return 0, 0
                tscore = t/(len(ps)*len(success_tasks))
            if t == 0:
                return 0, 0
            sign = 1 if tscore > 0 else -1
            # if has effect eff=1 on this predicate, eff={-1,0} must fail
            if repeated == False:
                for t in tasks:
                    f += 1 if int(pred_dict[pred]['task'][t][1]==True) - int(pred_dict[pred]['task'][t][0]==True) in [0, -sign] and tasks[t]['success'] == False else 0
                    f_d += 1 if int(pred_dict[pred]['task'][t][1]==True) - int(pred_dict[pred]['task'][t][0]==True) in [0, -sign] else 0
                if f_d == 0:
                    return tscore, 0
                fscore = f/f_d
            else:
                for p in ps:
                    for t in tasks:
                        f += 1 if int(pred_dict[p]['task'][t][1]==True) - int(pred_dict[p]['task'][t][0]==True) in [0, -sign] and tasks[t]['success'] == False else 0
                        f_d += 1 if int(pred_dict[p]['task'][t][1]==True) - int(pred_dict[p]['task'][t][0]==True) in [0, -sign] else 0
                    if f_d == 0:
                        return tscore, 0
                    fscore = f/f_d
        return tscore, fscore

def cross_assignment(skill2operator, skill2tasks, pred_dict, equal_preds=None, threshold=0.4):
    '''
    Assign precondtions of all skills to effect of each skill
    
    success execution -> precond == True, eff == 1
    precond == False , eff != 1 -> failed execution : ?for eff really? There's no gurantee on independent factor

    There should be a threshold for cross assignment, either if higher than it or lower than -1*threshold will be added
    skill2operator:: {skill_name:{precond:{str:Bool}, eff:{str:int}}}
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    pred_dict:: {pred_name:{task: [Bool, Bool]}, semantic:str}
    '''
    def equal_pred_dict_list(equal_preds):
        "input equal_preds has [postive] and [negative]"
        "equal_preds:: list(list(str))"
        "return: list(dict(str: bool))"
        dict_list = []
        for equal_pred in equal_preds:
            new_dict = {}
            for p in equal_pred:
                if 'postive' in p:
                    new_dict[p[:-10]] = True
                else:
                    new_dict[p[:-10]] = False
            dict_list.append(new_dict)
        return dict_list
        

    # all_pred = list(itertools.chain([list(skill2operator[skill]['precond'].keys()) for skill in skill2operator])) + list(itertools.chain([list(skill2operator[skill]['eff'].keys()) for skill in skill2operator]))
    # all_pred = set(list(itertools.chain(*all_pred)))
    all_pred = list(pred_dict.keys())

    reassigned_skill2operator = deepcopy(skill2operator)
    for skill in skill2operator:
        reassigned_skill2operator[skill] = {'precond':{}, 'eff':{}}
        for pred in all_pred:
            t_precond, f_precond = score(pred, skill, skill2tasks, pred_dict, equal_preds, 'precond')
            
            if t_precond > 0.5 and f_precond > 0.5:
                reassigned_skill2operator[skill]['precond'][pred] = True
            t_eff, f_eff = score(pred, skill, skill2tasks, pred_dict, equal_preds, 'eff')
            if abs(t_eff) > 0.5 and f_eff > 0.5:
                reassigned_skill2operator[skill]['eff'][pred] = 1 if t_eff > 0 else -1
            logging.info(f"skill name:{skill}, predicate:{pred}, tscore(precond):{t_precond}, fscore(precond):{f_precond}, tscore(eff):{t_eff}, fscore(eff):{f_eff}")
            print(f"skill name:{skill}, predicate:{pred}, tscore(precond):{t_precond}, fscore(precond):{f_precond}, tscore(eff):{t_eff}, fscore(eff):{f_eff}")
    return reassigned_skill2operator

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
        'Preconddtion is the intersection of all abstract states before execution'
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

# ### task proposing part below

# def scoring_chain(task, operators):
#     'returns the maximum steps that can be executed from the begining of a task'
#     pass

# def top_k_combo(coverage_table, k):
#     'calculate top k skill combination with highest information gain'
#     pass

# def update_coverage(coverage_table, task):
#     'update the coverage table with the task just been executed'
#     pass

# def task_proposing(skill, tasks):
#     '''
#     tasks::list(str):: previous tasks
#     '''
#     pass

if __name__ == '__main__':
    model = GPT4(engine='gpt-4o-2024-08-06')
    # # test predicate evaluation function
    # # pred = 'handEmpty()'
    # # pred = 'IsObjectReachable([OBJ], [LOC])'
    # # sem = "return true if the object is within the robot's reach at the given location, and false if it is not."
    # # sem = "Indicates whether the object is currently being held by the robot after attempting to pick it up."
    # sem = "return true if the agent is near the location else false."
    # # obj = 'KeyChain'
    # obj = 'Book'
    # loc = 'DiningTable'
    # loc_1 = 'Sofa'
    # # loc_2 = 'DiningTable'
    # loc_2 = 'Book'
    # img = ['Before_PickUp_2.jpg']
    # # img = ['test_new.jpg']
    # response = eval_pred(model, skill, pred, sem, obj, loc, loc_1, loc_2, img)
    # print(response)
    # breakpoint()

    # test predicate proposing for refining

    # mock symbolic state
    # # pred_dict = {'handEmpty()': True}
    # pred_dict = {'is_held([OBJ])': True}
    # # pred_dict = {}
    # # skill = 'PickUp([OBJ], [LOC])'
    # skill = 'GoTo([LOC_1], [LOC_2])'
    # pred_type = 'eff'
    # # pred_type = 'precond'
    # response = generate_pred(model, skill, pred_dict, pred_type)
    # print(response)
    # breakpoint()

    # # reassign
    # skill2operators = cross_assignment(skill2operators, skill2tasks, pred_dict, equal_preds=equal_preds, threshold=0.45)
    # print(skill2operators)
    # skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {'is_within_reach([OBJ], [LOC])': True}, 'eff': {'is_at_location([OBJ], [LOC])': -1}}, 
    #                    'DropAt([OBJ], [LOC])': {'precond': {'is_holding([OBJ])': True}, 'eff': {'is_at([OBJ], [LOC])': 1, 'is_dropped_at([OBJ], [LOC])': -1}}, 
    #                    'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {}}}
    skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {'is_within_reach([OBJ], [LOC])': True}, 'eff': {'is_at_location([OBJ], [LOC])': -1, 'is_within_reach([OBJ], [LOC])': -1, 'is_at([OBJ], [LOC])': -1}}, 
                       'DropAt([OBJ], [LOC])': {'precond': {'is_holding([OBJ])': True}, 'eff': {'is_at([OBJ], [LOC])': 1}}, 
                       'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {'is_at([OBJ], [LOC])': 1}}}
    # breakpoint()

    # # # merge after
    # unified_skill2operator, equal_preds = merge_predicates(model, skill2operators, pred_dict)
    # print(unified_skill2operator, '\n\n', equal_preds)

    # unified_skill2operator = {'PickUp([OBJ], [LOC])': {'precond': {'is_within_reach([OBJ], [LOC])': True}, 'eff': {'is_at_location([OBJ], [LOC])': -1, 'is_within_reach([OBJ], [LOC])': -1, 'is_at([OBJ], [LOC])': -1}}, 
    #                           'DropAt([OBJ], [LOC])': {'precond': {'is_holding([OBJ])': True}, 'eff': {'is_at([OBJ], [LOC])': 1}}, 
    #                           'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {}}} 

    # skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {}, 'eff': {}}, 'DropAt([OBJ], [LOC])': {'precond': {'isHolding([OBJ])': True}, 'eff': {}}, 'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {}}}
    # skill2tasks = {'DropAt([OBJ], [LOC])': {'DropAt_Vase_DiningTable_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Vase_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Vase_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Vase', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Bowl_CoffeeTable_True_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Bowl_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Bowl_CoffeeTable_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'CoffeeTable', 'loc_1': '', 'loc_2': ''}}, 'GoTo([LOC_1], [LOC_2])': {'GoTo_DiningTable_CoffeeTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_CoffeeTable_True_1.jpg'], 'success': True, 'loc_1': 'DiningTable', 'loc_2': 'CoffeeTable', 'obj': '', 'loc': ''}, 'GoTo_Sofa_DiningTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_1.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 'loc': ''}, 'GoTo_CoffeeTable_Sofa_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_CoffeeTable_Sofa_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_CoffeeTable_Sofa_True_1.jpg'], 'success': True, 'loc_1': 'CoffeeTable', 'loc_2': 'Sofa', 'obj': '', 'loc': ''}}, 'PickUp([OBJ], [LOC])': {'PickUp_TissueBox_Sofa_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_Sofa_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_Sofa_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'PickUp_Vase_CoffeeTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Vase_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Vase_CoffeeTable_True_1.jpg'], 'success': True, 'obj': 'Vase', 'loc': 'CoffeeTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_Bowl_DiningTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}}}
    # pred_dict = {'isHolding([OBJ])': {'task': {'GoTo_DiningTable_CoffeeTable_True_1': [False, False], 'GoTo_Sofa_DiningTable_True_1': [False, False], 'GoTo_CoffeeTable_Sofa_True_1': [False, False], 'PickUp_TissueBox_Sofa_True_1': [False, False], 'PickUp_Vase_CoffeeTable_True_1': [True, False], 'PickUp_Bowl_DiningTable_True_1': [False, True], 'DropAt_Bowl_CoffeeTable_True_1': [True, False], 'DropAt_Vase_DiningTable_False_1': [False, False]}, 'semantic': 'The robot is currently holding the object.'}}
    log_data = load_from_file('tasks/log/ai2thor_5_log_30.json')
    last_run_num = '5'
    skill2tasks, skill2operators, pred_dict, grounded_skill_dictionary, replay_buffer = log_data[last_run_num]["skill2tasks"], log_data[last_run_num]["skill2operators"], log_data[last_run_num]["pred_dict"], log_data[last_run_num]["grounded_skill_dictionary"], log_data[last_run_num]["replay_buffer"]
    merged_skill2operators, equal_preds = merge_predicates(model, skill2operators, pred_dict)
    print(equal_preds)
    assigned_skill2operators = cross_assignment(merged_skill2operators, skill2tasks, pred_dict, equal_preds=equal_preds)
    print(assigned_skill2operators)
    breakpoint()