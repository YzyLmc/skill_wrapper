'''
Get symbolic representation from skill semantic info and observation
Data structures:
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    pred_list:: [{'name':str, 'params':list, 'semantic':str}]
    grounded_predicate_truth_value_log::dict::{task:{step:[{'name':str, 'params':list, 'truth_value':bool}]}}
    lifted_predicate_truth_value_log::dict::{pred_name:{task: {id: [Bool, Bool]}, sem: str}}
'''
from utils import GPT4, load_from_file
from collections import defaultdict
import inspect
import random
import itertools
import logging

from manipula_skills import *

class PredicateState:
    def __init__(self, predicates):
        """
        Initializes the predicate state.
        predicates::list:: A list of predicate dictionaries {'name': str, 'params': tuple/list, ...}
        """
        self.pred_dict = {self._keyify(pred): pred for pred in predicates}

    def _keyify(self, pred):
        """Generates a unique key from predicate name and parameters."""
        return (pred["name"], tuple(pred["params"]))

    def set_pred_value(self, grounded_pred, key, value):
        """
        Sets the predicate value efficiently using dictionary lookup.
        """
        pred_key = self._keyify(grounded_pred)
        if pred_key in self.pred_dict:
            self.pred_dict[pred_key][key] = value

    def get_pred_value(self, grounded_pred, key):
        """
        Retrieves the predicate value.
        """
        pred_key = self._keyify(grounded_pred)
        return self.pred_dict.get(pred_key, {}).get(key, None)
    
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
                assert all(key in ["name", "params", "semantic"] for key in new_pred.keys())
                self.pred_dict[pred_key] = new_pred
            else:
                print(f"Predicate {dict_to_string(new_pred)} already exists.")

    def get_unevaluated_predicates(self):
        """
        Returns a list of predicates that do not have truth value evaluated.
        """
        return [pred for pred in self.pred_dict.values() if 'success' not in pred]

# def set_pred_value(pred_list, grounded_pred, key, value):
#     """
#     Set predicate value by looking up pred_list and find the matching predicate.
#     grounded_pred::dict:: {'name':str, 'params':list}
#     """
#     for pred in pred_list:
#         pred_short = {'name':pred['name'], 'params':pred['params']}
#         if pred_short == grounded_pred:
#             pred[key] = value
#             break

def dict_to_string(dict):
    """
    Assembly a dictionary of grounded/lifted representation to string
    dict::dict:: {'name':str, 'params':list}
    """
    return f"{dict['name']}({', '.join(dict['params'])})"

def ground_with_params(lifted, params, type_dict):
    """
    Grounded a skill or a predicate with parameters and their types.
    lifted::dict:: lifted skill/predicate with parameter type, e.g., {'name':"GoTo", 'params':["[ROBOT]", "[LOC]", "[LOC]"]}/{'name':"At", 'params':["[LOC]"]}
    params::list:: list of parameters, e.g., ["Apple", "Table"]
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    """
    # tuple is applicable to the lifted representation
    assert len(lifted['params']) == len(params)
    for i, p in enumerate(params):
        assert p in type_dict
        assert lifted['params'][i] in type_dict[p]

    # grounded skill/predicate
    return {'name': lifted['name'], 'params': params}

# TODO: revise data structure under tasks, record grounding parameters
def possible_grounded_predicates(pred_list, type_dict):
    """
    Generate all possible grounded predicate using the combination of predicates and objects
    pred_dict::dict:: {pred_name:{semantic: str, params:list(str)}}
    pred_list:: [{'name':str, 'params':list, 'semantic':str}]
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
            grounded_predicates.append({'name': pred['name'], 'params': params})
    
    return grounded_predicates

def calculate_pred_to_update(grounded_predicates, skill):
    '''
    Given a skill and its parameters, find the set of predicates that need updates
    skill::dict:: {'name':str, 'params':list}
    grounded_predicates::list:: list of grounded predicates, e.g., [{'name': 'At', 'params': ['Apple', 'Table']}]
    '''
    return [gp for gp in grounded_predicates if any([p in gp['params'] for p in skill['params']])]

def lift_grounded_predicate(grounded_pred, type_dict, pred_list):
    """
    Lift a grounded predicate, e.g., {'name': 'At', 'params': ['Apple', 'Table']} to a {'name':"At", 'params':["[OBJ]", "[LOC]"]}
    """
    assert all([p in type_dict for p in grounded_pred['params']])
    output_list = []
    # iterate through the pred_list to find the lifted representation
    for pred in pred_list:
        if pred['name'] == grounded_pred['name']:
            if all([p in type_dict for p in pred['params']]):
                output_list.append(pred)
    # cannot have more than one matched lifted predicate
    # e.g., existing two predicates At(a, b) and At(a, c) while the grounded version At(apple, banana) has type(banana) = [b, c]
    assert len(output_list) == 1
    return output_list[0]

# Not used: evaluate an execution using foundation model. Expected acc to be ~ 70%
def eval_execution(model, skill, consecutive_pair, prompt_fpath='prompts/evalutate_task.txt'):
    'Get successfulness of the execution given images and the skill name'
    def construct_prompt(prompt, skill):
        while "[SKILL]" in prompt:
                prompt = prompt.replace("[SKILL]", skill)
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill)
    return model.generate_multimodal(prompt, consecutive_pair)[0]

def eval_pred_new(model, img, grounded_skill, grounded_pred, type_dict, prompt_fpath='prompts/evaluate_pred_ai2thor.txt'):
    '''
    evaluate truth value of a predicate using a dictionary of parameters
    grounded_skill::dict:: grounded skill with parameter type, e.g., {'name':'GoTo', "params":['location', 'location']}
    grounded_pred::dict:: grounded predicate with parameter type, e.g., {'name':"At", 'params':["location"]}
    '''
    def construct_prompt(prompt, grounded_skill, grounded_pred):
        # return none if none of the parameters of the grounded predicate match any of the parameters of the grounded skill
        if not any([p in grounded_pred['params'] for p in grounded_skill['params']]):
            return None
        place_holders = ['[GROUNDED_SKILL]', '[GROUNDED_PRED]','[LIFTED_PRED]', '[SEMANTIC]']
        lifted_pred = lift_grounded_predicate(grounded_pred, type_dict, pred_list)
        while any([p in prompt for p in place_holders]):
            prompt = prompt.replace('[GROUNDED_SKILL]', grounded_skill)
            prompt = prompt.replace('[GROUNDED_PRED]', grounded_pred)
            prompt = prompt.replace('[LIFTED_PRED]', dict_to_string(lifted_pred))
            prompt = prompt.replace('[SEMANTIC]', lifted_pred['semantic'])
        return prompt
    
    prompt = load_from_file(prompt_fpath)
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
    
def eval_pred_init(model, img, grounded_skill, grounded_pred, type_dict, prompt_fpath='prompts/evaluate_pred_ai2thor_init.txt'):
    '''
    Evaluate the predicates in the initial state with multiple images, using a different prompt.
    '''
    # TODO: test and tune prompt for evaluate initial state
    return eval_pred_new(model, img, grounded_skill, grounded_pred, type_dict, prompt_fpath=prompt_fpath)
            
# hard-coded eval function, will be replaced by the one above
def eval_pred(model, skill, pred, sem, obj, loc, loc_1, loc_2, img, prompt_fpath='prompts/evaluate_pred_ai2thor.txt', obj_ls = ['Book', 'Vase', 'TissueBox', 'Bowl'], loc_ls = ['DiningTable', 'Sofa']):
    '''
    Evaluate one predicate given one image.
    If skill or predicate takes variables they should contain [OBJ] [LOC] [LOC_1]or [LOC_2] in the input.
    For all mismatch arguments, output false.

    Obsoleted: --- GoTo(loc_1, loc_2) will be evaluated differently since the arguments are different. 
    Empty string '' means no argument
    '''
    def construct_prompt(prompt, skill, pred, obj, loc, loc_1, loc_2):
        arg_vec = [int('[OBJ]' in  pred) + int('[OBJ]' in  skill), 
                   int('[LOC]' in  pred) + int('[LOC]' in  skill), 
                   int('[LOC_1]' in  pred) + int('[LOC_1]' in  skill), 
                   int('[LOC_2]' in  pred) + int('[LOC_2]' in  skill)]
        # breakpoint()
        if sum(arg_vec[:2]) > 0 and sum(arg_vec[2:]) > 0:
            return None
        
        elif sum(arg_vec[:2]) > 0:
            while "[OBJ]" in pred or "[LOC]" in pred:
                pred = pred.replace("[OBJ]", obj).replace("[LOC]", loc)
            while "[OBJ]" in skill or "[LOC]" in skill:
                skill = skill.replace("[OBJ]", obj).replace("[LOC]", loc)

        elif sum(arg_vec[2:]) > 0:
            while "[LOC_1]" in pred or "[LOC_2]" in pred:
                pred = pred.replace("[LOC_1]", loc_1).replace("[LOC_2]", loc_2)
            while "[LOC_1]" in skill or "[LOC_2]" in skill:
                skill = skill.replace("[LOC_1]", loc_1).replace("[LOC_2]", loc_2)

        while "[PRED]" in prompt or "[SKILL]" in prompt or "[SEMANTIC]" in prompt:
            prompt = prompt.replace("[PRED]", pred)
            prompt = prompt.replace("[SKILL]", skill)
            prompt = prompt.replace("[SEMANTIC]", sem)

        return prompt

    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred, obj, loc, loc_1, loc_2)
    logging.info(f'Evaluating predicate {pred} on skill {skill} with arguments {obj} {loc} {loc_1} {loc_2}')
    if prompt:
        logging.info('Calling GPT4')
        resp = model.generate_multimodal(prompt, img)[0]
        result = True if "True" in resp.split('\n')[-1] else False
        print(result)
        return result
    else:
        logging.info(f"mismatch skill and predicate: return False\n{skill} / {pred}")
        return False

# def eval_pred_set(model, skill, pred2sem, obj, loc,loc_1, loc_2, img):
#     '''
#     Evaluate set of predicates
#     pred_set::list(str)
#     returns::Dict(str:bool)
#     '''
#     return {pred: eval_pred(model, skill, pred, sem, obj, loc, loc_1, loc_2, img) for pred, sem in pred2sem.items()}

def generate_pred_new(model, skill, pred_list, pred_type, tried_pred=[], prompt_fpath='prompts/predicate_refining'):
    '''
    propose new predicates based on the contrastive pair
    '''
    # TODO: add tried_pred back to the prompt
    def construct_prompt(prompt, grounded_skill, pred_list):
        while "[SKILL]" in prompt or "[PRED_DICT]" in prompt or "[TRIED_PRED]" in prompt:
            prompt = prompt.replace("[SKILL]",  dict_to_string(grounded_skill))
            # construct predicate list from pred_dict
            pred_list_str = '\n'.join([f'{dict_to_string(pred)}: {pred["semantic"]}' for pred in pred_list])
            prompt = prompt.replace("[PRED_LIST]", pred_list_str)
            prompt = prompt.replace("[TRIED_PRED]", [dict_to_string(pred) for pred in tried_pred])

    # model = GPT4(engine='o1-preview')
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
def generate_pred(model, skill, pred_dict, pred_type, tried_pred=[], prompt_fpath='prompts/predicate_refining'):
    '''
    generate_predicates based on existing predicates dictionary describing the same symbolic state
    pred_dict:: {pred_name:{task: {id: [Bool, Bool]}, sem: str}}
    type:: str:: 'precond' or 'eff' 
    '''
    def construct_prompt(prompt, skill, pred_dict):
        while "[SKILL]" in prompt or "[PRED_DICT]" in prompt or "[TRIED_PRED]" in prompt:
            prompt = prompt.replace("[SKILL]", skill)
            prompt = prompt.replace("[PRED_DICT]", str(pred_dict))
            prompt = prompt.replace("[TRIED_PRED]", str(tried_pred))
        # convert the placeholders back to 'obj' and 'loc'
        while "[OBJ]" in prompt or "[LOC]" in prompt or "[LOC_1]" in prompt or "[LOC_2]" in prompt:
            prompt = prompt.replace("[OBJ]", "obj")
            prompt = prompt.replace("[LOC]", "loc")
            prompt = prompt.replace("[LOC_1]", "init")
            prompt = prompt.replace("[LOC_2]", "goal")
        return prompt
    model = GPT4(engine='o1-preview')
    prompt_fpath += f"_{pred_type}.txt"
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred_dict)
    # breakpoint()
    # response consists of predicate and its semantic
    logging.info('Generating predicate')
    resp = model.generate(prompt)[0]
    pred, sem = resp.split(': ', 1)[0].strip('`'), resp.split(': ', 1)[1].strip()
    pred = pred.replace('(obj', '([OBJ]').replace('obj)', '[OBJ])').replace('(init', '([LOC_1]').replace('goal)', '[LOC_2])').replace('(loc', '([LOC]').replace('loc)', '[LOC])').replace(' ','')
    return pred, sem

def update_missing_predicates(model, pred_list, tasks, grounded_predicate_truth_value_log, lifted_predicate_truth_value_log, type_dict):
    '''
    Find the grounded predicates with missing values and evaluate them.
    The grounded predicates are evaluated from the beginning to the end, and then lifted to the lifted predicates.
    pred_list::list(dict):: List of all possible grounded predicates
    grounded_predicate_truth_value_log::dict:: {task:{step:[{'name':str, 'params':list, 'truth_value':bool}]}}
    grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
    lifted_predicate_truth_value_log::dict:: (same as pred_dict before) {pred_name:{task: {id: [Bool, Bool]}, sem: str}}
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    '''
    # NOTE: step is a integer ranging from 0-8, where 0 is the init step and success==None. 1-8 are states after executions

    # look for predicates that haven't been evaluated
    # The truth values could be missing if:
    #    1. the predicate is newly added (assuming all possible grounded predicates are added, including the init step)
    #    2. a task is newly executed
    # NOTE: the dictionary could be partially complete because some truth values will be directly reused from the scoring function

    logging.info('looking for empty grounded predicates')
    for task_id, steps in tasks.items():
        # NOTE: might not be necessary. tasks always get updated after every execution
        if task_id not in grounded_predicate_truth_value_log:
            grounded_predicate_truth_value_log[task_id] = {}
            for step in steps:
                grounded_predicate_truth_value_log[task_id][step] = PredicateState(pred_list)

        # for each step, iterate through all steps and find empty predicates and update them
        # calculate predicates to update based on the last action every step after init
        # init step is updated separately

        # 1. update predicates for all states 
        for step, state in steps.items():
            grounded_predicate_truth_value_log[task_id][step].add_pred_list(pred_list)

            # 2. find states need to be re-eval
            pred_to_update = pred_list if step == 0 else calculate_pred_to_update(pred_list, skill)

            # 3. re-eval
            for pred in pred_to_update:
                truth_value = eval_pred_new(model, img, grounded_skill, ground_pred, type_dict)

        for step_id, pred_list in grounded_predicate_truth_value_log[task_id].items():
            for pred in pred_list:
                # pred is grounded
                items_per_step = steps[step_id]
                # case 1: missing truth value in init state
                if step_id == 'init' and pred['truth_value'] == None:
                    # evaluate the predicate
                    truth_value = eval_pred_init(model, items_per_step['image'], items_per_step['skill'], pred, type_dict, pred_list)
                    set_pred_value(grounded_predicate_truth_value_log[task_id][step_id], pred, "truth_value", truth_value)                   
                    # if no need to update, copy from previous state
                    
                    # NOTE: cannot lift from grounded log due to conflicting truth values, will directly log when scoring
                    
                elif not step_id == 'init':
                    skill = items_per_step['skill']

                
    logging.info('Done evaluation from mismatch_symbolic_state')
    

def detect_mismatch():
    pass

# TODO: Separate predicate update from detecting mismatch states
def mismatch_symbolic_state(model, pred_dict, skill2tasks, pred_type):
    '''
    look for same symbolic states (same s1 or s2 - s1) in task dictionary
    pred_dict::{pred_name:{task: {id: [Bool, Bool]}, sem: str}}
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    pred_type::{'precond', 'eff'}
    returns::dict(skill, list(id: dict('s0':img, 's1':img, 'success': Bool)))
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
    
    # evaluate new task first
    logging.info('evaluating from mismatch_symbolic_state')
    for pred in pred_dict.keys():
        for skill, tasks in skill2tasks.items():
            for id, task in tasks.items():
                if id not in pred_dict[pred]['task']:
                    pred_dict[pred]['task'][id] = [
                        eval_pred(model, skill, pred, pred_dict[pred]['semantic'], task['obj'], task['loc'], task['loc_1'], task['loc_2'], task['s0']), \
                        eval_pred(model, skill, pred, pred_dict[pred]['semantic'], task['obj'], task['loc'], task['loc_1'], task['loc_2'], task['s1'])
                    ]    
    logging.info('Done evaluation from mismatch_symbolic_state')
    # look for duplicated tasks
    dup_tasks = {}
    if pred_type == "precond":
        # return two same symbolic states BEFORE EXECUTION
        for skill, tasks in skill2tasks.items():
            task2state = {}
            for id, task in tasks.items():
                task2state[id] = {pred: pred_dict[pred]['task'][id][0] for pred in pred_dict}
            dup_tasks[skill] = tasks_with_same_symbolic_states(task2state)
    elif pred_type == "eff":
        # return two same symbolic states TRANSITION (symbolic)
        for skill, tasks in skill2tasks.items():
            task2state = {}
            for id, task in tasks.items():
                task2state[id] = {pred: int(pred_dict[pred]['task'][id][1]==True) - int(pred_dict[pred]['task'][id][0]==True) for pred in pred_dict}
            dup_tasks[skill] = tasks_with_same_symbolic_states(task2state)

    # breakpoint()
    # find the ones that jas different successfuness
    # try:
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
    return pred_dict, mismatch_pairs
    # except:
    #     return {}
    # if dup_tasks:
    #     t1 = random.choice(list(dup_tasks.keys()))
    #     t2 = random.choice(dup_tasks[t1])
    #     return [task[t] for t in [t1, t2]]
    # else:
    #     return []
        
# TODO: code for seperate tasks in terms of skill names:: dict(skill: list(dict('id':num, 's0':img, 's1':img, 'success': Bool)))
def refine_pred(model, skill, skill2operators, skill2tasks, pred_dict, skill2triedpred={}, max_t=3):
    '''
    Refine predicates of precondition and effect of one skill for precondition and effect
    pred_dict::{pred_name:{task{id: [Bool, Bool]}, sem:str}}
    skill2operators:: {skill_name: {{'precond':{str:Bool},'eff':{str:int}}}}
    skill2triedpred:: {skill_name: {"precond":[str], "eff": [str]}}
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    '''

    # keep tracking truth value of predicates before and after a skill execution
    # {pred_name:{task: [Bool, Bool]}} Does it need skill name as keys?
    if not pred_dict:
        pred_dict = {}

    if not skill2operators:
        skill2operators = {}
    if not skill in skill2operators:
        skill2operators[skill] = {}
        skill2operators[skill]['precond'] = {}
        skill2operators[skill]['eff'] = {}

    if not skill2triedpred:
        skill2triedpred = {}
    if not skill in skill2triedpred:
        skill2triedpred[skill] = {}
        skill2triedpred[skill]['precond'] = []
        skill2triedpred[skill]['eff'] = []
    
    # check precondition first
    t = 0
    pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'precond')
    new_p_added = False
    # breakpoint()
    logging.info("About to enter precondition refinement")
    while skill in mismatch_tasks and t < max_t:
        new_p, sem = generate_pred(model, skill, list(skill2operators[skill]['precond'].keys()),  'precond', tried_pred=skill2triedpred[skill]['precond'])
        logging.info(f"mismatch state for precondition generation {mismatch_tasks[skill][0]}, {mismatch_tasks[skill][1]}")
        # logging.info('new predicate', new_p, sem)
        logging.info(f"new predicate {new_p}, {sem}")
        new_p_mismatch = {idx: [eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s0']), eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s1'])] for idx in mismatch_tasks[skill]}
        # logging.info('new predicate truth value', new_p_mismatch)
        logging.info(f'new predicate truth value{new_p_mismatch}')
        if new_p_mismatch[mismatch_tasks[skill][0]][0] != new_p_mismatch[mismatch_tasks[skill][1]][0]:
            # print('Entering #1 if')
            skill2operators[skill]['precond'][new_p] = True if skill2tasks[skill][mismatch_tasks[skill][0]]['success'] == new_p_mismatch[mismatch_tasks[skill][0]][0] else False
            logging.info(f"Predicate {new_p} added to precondition with truth value {skill2operators[skill]['precond'][new_p]}")
            # logging.info(f"Predicate {new_p} added to precondition with truth value {skill2operators[skill]['precond'][new_p]}")
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
                for s in skill2tasks:  
                    for idx, task in skill2tasks[s].items():
                        if idx not in new_p_mismatch:
                                logging.info('Evaluating for precond')
                                pred_dict[new_p]['task'][idx] = [eval_pred(model, s, new_p, sem, task['obj'], task['loc'], skill2tasks[s][idx]['loc_1'], skill2tasks[s][idx]['loc_2'], task['s0']), eval_pred(model, s, new_p, sem, task['obj'], task['loc'], skill2tasks[s][idx]['loc_1'], skill2tasks[s][idx]['loc_2'], task['s1'])]
                for new_p_id in new_p_mismatch:
                    pred_dict[new_p]['task'][new_p_id] = new_p_mismatch[new_p_id]
                pred_dict[new_p]['semantic'] = sem

            logging.info(f'Done evaluating predicate {new_p} for all tasks')
            new_p_added = False
            pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'precond')
            logging.info(f'Done evaluating predicate {new_p} for all tasks')
            skill2triedpred[skill]['precond'] = []
        else:
            skill2triedpred[skill]['precond'].append(new_p)
        t += 1
    

    # breakpoint()
    # check effect
    t = 0
    pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'eff')
    logging.info("About to enter effect refinement")
    while skill in mismatch_tasks and t < max_t:
        new_p, sem = generate_pred(model, skill, list(skill2operators[skill]['eff'].keys()), 'eff', tried_pred=skill2triedpred[skill]['eff'])
        logging.info(f"mismatch state for effect generation {mismatch_tasks[skill][0]}, {mismatch_tasks[skill][1]}")
        logging.info(f"current predicates in effect set of skill {skill}:\n{list(skill2operators[skill]['eff'].keys())}")
        # logging.info('new predicate', new_p, sem)
        logging.info(f'new predicate {new_p}, {sem}')
        new_p_mismatch = {idx: [eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s0']), eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s1'])] for idx in mismatch_tasks[skill]}
        # logging.info('new predicate truth value', new_p_mismatch)
        logging.info(f'new predicate truth value {new_p_mismatch}')
        # first index for task number, second index for before and after
        s_1_1 = new_p_mismatch[mismatch_tasks[skill][0]][0]
        s_1_2 = new_p_mismatch[mismatch_tasks[skill][0]][1]
        s_2_1 = new_p_mismatch[mismatch_tasks[skill][1]][0]
        s_2_2 = new_p_mismatch[mismatch_tasks[skill][1]][1]
        success_task = mismatch_tasks[skill][0] if skill2tasks[skill][mismatch_tasks[skill][0]]['success'] == True else mismatch_tasks[skill][1]
        state_change_success = int(new_p_mismatch[success_task][1]==True) - int(new_p_mismatch[success_task][0]==True)
        # print('Before Entering #2 if for eff')
        # eff representation might be wrong (s1-s2). The result value could be {-1, 0, 1}, 0 cases could be wrong?
        if (int(s_1_2==True) - int(s_1_1==True) != int(s_2_2==True) - int(s_2_1==True)) and state_change_success != 0:
            
            skill2operators[skill]['eff'][new_p] = state_change_success
            logging.info(f"Predicate {new_p} added to effect with truth value {skill2operators[skill]['eff'][new_p]}")
            # logging.info(f"Predicate {new_p} added to effect with truth value {skill2operators[skill]['eff'][new_p]}")
            new_p_added = True
            if new_p_added and new_p not in pred_dict:
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
                pred_dict[new_p] = {'task': {}}
                for s in skill2tasks:
                    for idx, task in skill2tasks[s].items():
                        if idx not in new_p_mismatch:
                            pred_dict[new_p]['task'][idx] = [eval_pred(model, s, new_p, sem, task['obj'], task['loc'], skill2tasks[s][idx]['loc_1'], skill2tasks[s][idx]['loc_2'], task['s0']), eval_pred(model, s, new_p, sem, task['obj'], task['loc'], skill2tasks[s][idx]['loc_1'], skill2tasks[s][idx]['loc_2'], task['s1'])]
                for new_p_id in new_p_mismatch:
                    pred_dict[new_p]['task'][new_p_id] = new_p_mismatch[new_p_id]
                pred_dict[new_p]['semantic'] = sem
            logging.info(f'Done evaluating predicate {new_p} for all tasks')
            new_p_added = False
            pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'eff')
            skill2triedpred[skill]['eff'] = []
        else:
            skill2triedpred[skill]['eff'].append(new_p)
        t += 1
    
            
    return skill2operators, pred_dict, skill2triedpred

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

    # for skill in skill2operator:
    #     tasks = skill2tasks[skill]
    #     # skill2operator[skill]['eff'] = {}
    #     # print(skill)
    #     for pred in all_pred:
    #         # try:
    #             # accuracy here means the portion of state change from true to false
    #             state_pair_all = [pred_dict[pred]['task'][id] for id, t in tasks.items() if t['success']]
    #             if not state_pair_all: # if no success case for the skill
    #                 continue
    #             acc_ls = [int(state_pair[1]==True) - int(state_pair[0]==True) for state_pair in state_pair_all]
    #             total_num = len(acc_ls)
    #             for ps in equal_preds:
    #                 if pred in ps:
    #                     for p in ps:
    #                         state_pair_all = [pred_dict[p]['task'][id] for id, t in tasks.items() if t['success']]
    #                         acc_ls += state_pair_all
    #                         total_num += len(state_pair_all)
    #             acc = sum(acc_ls)/total_num
    #             print(skill, pred, acc, state_pair_all)
    #             # breakpoint()
    #             if acc > threshold:
    #                 skill2operator[skill]['eff'][pred] = 1
    #             elif acc < - threshold:
    #                 skill2operator[skill]['eff'][pred] = -1
    #         # except:
    #         #     breakpoint()
    # return skill2operator


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
    # # pred = 'is_held([OBJ])'
    # # pred = 'isPathClear([LOC_1], [LOC_2])'
    # pred = 'At([LOC_1])'
    # # pred = 'at([LOC])'
    # # pred = 'at([OBJ])'
    # skill = 'PickUp([OBJ], [LOC])'
    # # skill = 'GoTo([LOC_1], [LOC_2])'
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

    # test mismatch state
    # pred_dict::{pred_name:{task: [Bool, Bool]}}
    # skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    # return: dict(skill: list(id: dict('s0':img, 's1':img, 'success': Bool)))

    # mock_pred_dict = {
    #     "At(loc)": {'task': {
    #         "PickUp_0": [True, True],
    #         "PickUp_1": [False, False],
    #         "PickUp_2": [True, True],
    #         "PickUp_3": [True, True]
    #     },
    #     'semantic': 'test semantic'
    #     },
    #     "IsFreeHand()":{'task':{
    #         "PickUp_0": [True, False],
    #         "PickUp_1": [True, True],
    #         "PickUp_2": [True, True],
    #         "PickUp_3": [True, False]
    #     },
    #     'sem': 'test semantic'
    # }
    # }
    # mock_pred_dict = {}
    # mock_skill2tasks = {
    #     "PickUp": {
    #         "PickUp_0": {"s0": ["pickup_t0_s0_success.jpg"], "s1":["pickup_t0_s1_success.jpg"], "obj":"test", "loc":"test", "success": True},
    #         "PickUp_1": {"s0": ["pickup_t1_s0_fail.jpg"], "s1":["pickup_t1_s1_fail.jpg"], "obj":"test", "loc":"test", "success": False},
    #         "PickUp_2": {"s0": ["pickup_t2_s0_fail.jpg"], "s1":["pickup_t2_s1_fail.jpg"], "obj":"test", "loc":"test", "success": False},
    #         "PickUp_3": {"s0": ["pickup_t3_s0_fail.jpg"], "s1":["pickup_t3_s1_fail.jpg"], "obj":"test", "loc":"test", "success": False}
    #     }
    # }
    # pred_type = 'precond'
    # pred_dict, mismatch_states = mismatch_symbolic_state(model, mock_pred_dict, mock_skill2tasks, pred_type)
    # print(mismatch_states) # {'PickUp': {'PickUp_0': ['PickUp_2']}}

    # pred_type = 'eff'
    # pred_dict, mismatch_states = mismatch_symbolic_state(model, mock_pred_dict, mock_skill2tasks, pred_type)
    # print(mismatch_states) # {'PickUp': ['PickUp_0', 'PickUp_3']}

    # # test main refining function
    # skill = 'PickUp'
    # precond = {}
    # eff = {}
    # precond, eff, pred_dict = refine_pred(model, skill, mock_skill2tasks, mock_pred_dict, precond, eff)
    # print(precond, eff, pred_dict)

    # test with real images and tasks
    # pred_dict = {}

    # skill2tasks = {
    #     "PickUp([OBJ], [LOC])": {
    #         "PickUp_0": {"s0": ["Before_PickUp_2.jpg"], "s1":["After_PickUp_2.jpg"], "obj":"Book", "loc":"DiningTable", "loc_1":"", "loc_2":"", "success": True},
    #         "PickUp_1": {"s0": ["Before_PickUp_1.jpg"], "s1":["After_PickUp_1.jpg"], "obj":"KeyChain", "loc":"DiningTable", "loc_1":"", "loc_2":"", "success": False}
    #     },
    #     "GoTo([LOC_1], [LOC_2])":{

    #     }
    # }
    # skill2tasks = {
    #     'PickUp(obj, loc)': {'PickUp_TissueBox_Sofa_True_1': {'s0': 'Before_PickUp_TissueBox_Sofa_True_1.jpg', 's1': 'After_PickUp_TissueBox_Sofa_True_1.jpg', 'success': True}, 'PickUp_Book_DiningTable_False_1': {'s0': 'Before_PickUp_Book_DiningTable_False_1.jpg', 's1': 'After_PickUp_Book_DiningTable_False_1.jpg', 'success': False}}, 
    #     'DropAt(obj, loc)': {'DropAt_TissueBox_Sofa_False_1': {'s0': 'Before_DropAt_TissueBox_Sofa_False_1.jpg', 's1': 'After_DropAt_TissueBox_Sofa_False_1.jpg', 'success': False}, 'DropAt_Book_DiningTable_False_1': {'s0': 'Before_DropAt_Book_DiningTable_False_1.jpg', 's1': 'After_DropAt_Book_DiningTable_False_1.jpg', 'success': False}, 'DropAt_TissueBox_DiningTable_True_1': {'s0': 'Before_DropAt_TissueBox_DiningTable_True_1.jpg', 's1': 'After_DropAt_TissueBox_DiningTable_True_1.jpg', 'success': True}}, 
    #     'GoTo(init, goal)': {'GoTo_Sofa_Sofa_True_1': {'s0': 'Before_GoTo_Sofa_Sofa_True_1.jpg', 's1': 'After_GoTo_Sofa_Sofa_True_1.jpg', 'success': True}, 'GoTo_Sofa_DiningTable_True_1': {'s0': 'Before_GoTo_Sofa_DiningTable_True_1.jpg', 's1': 'After_GoTo_Sofa_DiningTable_True_1.jpg', 'success': True}}
    #                }

    # pred_type = 'precond'
    # pred_dict, mismatch_states = mismatch_symbolic_state(model, pred_dict, skill2tasks, pred_type)
    # print(mismatch_states) # {'PickUp': {'PickUp_0': ['PickUp_2']}}
    # breakpoint()
    # pred_type = 'eff'
    # pred_dict, mismatch_states = mismatch_symbolic_state(model, pred_dict, skill2tasks, pred_type)
    # print(mismatch_states) # {'PickUp': ['PickUp_0', 'PickUp_3']}
    # breakpoint()
    # skill = 'PickUp([OBJ], [LOC])'
    # precond = {}
    # eff = {}
    # skill2operators, pred_dict, skill2triedpred = refine_pred(model, skill, skill2tasks, pred_dict, precond, eff)
    # print(skill2operators, pred_dict)

    # test merge perdicate
    # pred_dict = {
    #             'At([LOC])': {'semantic': "return true if robot is at the location, else return false"}, 
    #             'IsObjectAtLocation([OBJ], [LOC])': {'semantic': "return true if the object is at the lcoation, otherwise return false"}, 
    #             "IsHandFree()": {'semantic': "return true if the robot hand is available now, and return false if it's not available."}, 
    #             'isHeld([OBJ])': {'semantic': "return true if the object is held by the robot, otherwise return false."}, 
    #             'IsHolding([OBJ])': {'semantic': "if the robot is holding the object, return true, otherwaise return false."}, 
    #             'IsAtLocation([LOC])': {'semantic': "if robot is at the location, return true, else return false."}, 
    #             'handIsOccupied()': {'semantic':"if the robot hand is occupied then return true, if it's free return false."}
    #             }

    # # Target operator              
    # skill2operator = {'PickUp([OBJ], [LOC])': {'precond':{'IsHandFree()':True, 'IsAtLocation([LOC])':True, 'IsObjectAtLocation([OBJ], [LOC])':True}, 'eff':{'isHeld([OBJ])':1, 'handIsOccupied()':1}},
    #                    'DropAt([OBJ], [LOC])': {'precond':{'IsHolding([OBJ])':True, 'IsAtLocation([LOC])':True}, 'eff':{'handIsOccupied()':-1}},
    #                    'GoTo([LOC_1], [LOC_2])': {'precond':{'At([LOC])':True}, 'eff':{'At([LOC])':-1, 'At([LOC])':1}}}
    
    # unified_skill2operator, equal_preds = merge_predicates(model, skill2operator, pred_dict)
    # print(unified_skill2operator)
    # print(equal_preds)

    # # test cross assignment
    # pred_dict = {
    #                 'At([LOC])': {'task':{"PickUp_0":[True, True],"PickUp_1":[True, True]}, 'semantic': "return true if robot is at the location, else return false"}, 
    #                 'IsObjectAtLocation([OBJ], [LOC])': {'task':{"PickUp_0":[False, True],"PickUp_1":[False, True]},'semantic': "return true if the object is at the lcoation, otherwise return false"}, 
    #                 "IsHandFree()": {'task':{"PickUp_0":[True, False], "PickUp_1":[True, False]},'semantic': "return true if the robot hand is available now, and return false if it's not available."}, 
    #                 'isHeld([OBJ])': {'task':{"PickUp_0":[False, True],"PickUp_1":[False, True]},'semantic': "return true if the object is held by the robot, otherwise return false."}, 
    #                 'IsHolding([OBJ])': {'task':{"PickUp_0":[False, True],"PickUp_1":[False, True]},'semantic': "if the robot is holding the object, return true, otherwaise return false."}, 
    #                 'IsAtLocation([LOC])': {'task':{"PickUp_0":[True, True],"PickUp_1":[True, True]},'semantic': "if robot is at the location, return true, else return false."}, 
    #                 'handIsOccupied()': {'task':{"PickUp_0":[False, True],"PickUp_1":[False, True]},'semantic':"if the robot hand is occupied then return true, if it's free return false."}
    #                 }
    
    # skill2operator = {'PickUp([OBJ], [LOC])': {'precond':{'IsHandFree()':True, 'IsAtLocation([LOC])':True, 'IsObjectAtLocation([OBJ], [LOC])':True}, 'eff':{'isHeld([OBJ])':1}},
    #                    'DropAt([OBJ], [LOC])': {'precond':{'IsHolding([OBJ])':True, 'IsAtLocation([LOC])':True}, 'eff':{'handIsOccupied()':-1}},
    #                    'GoTo([LOC_1], [LOC_2])': {'precond':{'At([LOC])':True}, 'eff':{'At([LOC])':-1, 'At([LOC])':1}}}
    
    # skill2tasks = {
    #     "PickUp([OBJ], [LOC])": {
    #         "PickUp_0": {"s0": ["Before_PickUp_2.jpg"], "s1":["After_PickUp_2.jpg"], "obj":"Book", "loc":"DiningTable", "loc_1":"", "loc_2":"", "success": True},
    #         "PickUp_1": {"s0": ["Before_PickUp_1.jpg"], "s1":["After_PickUp_1.jpg"], "obj":"KeyChain", "loc":"DiningTable", "loc_1":"", "loc_2":"", "success": False}
    #     },
    #     "GoTo([LOC_1], [LOC_2])":{
    #         "GoTo_0": {"s0": ["Before_PickUp_2.jpg"], "s1":["After_PickUp_2.jpg"], "obj":"Book", "loc":"DiningTable", "loc_1":"", "loc_2":"", "success": True}
    #     },
    #     "DropAt([OBJ], [LOC])":{
    #         "DropAt_0": {"s0": ["Before_PickUp_2.jpg"], "s1":["After_PickUp_2.jpg"], "obj":"Book", "loc":"DiningTable", "loc_1":"", "loc_2":"", "success": True}
    #     }
    # }

    # skill2operator = cross_assignment(skill2operator, skill2tasks, pred_dict)
    # print(skill2operator)

    # real test
    pred_dict = {}

    skill2tasks = {'PickUp([OBJ], [LOC])': {'PickUp_TissueBox_Sofa_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_Sofa_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_Sofa_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'PickUp_Book_DiningTable_False_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Book_DiningTable_False_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Book_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Book', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}}, 'DropAt([OBJ], [LOC])': {'DropAt_TissueBox_Sofa_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_TissueBox_Sofa_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_TissueBox_Sofa_False_1.jpg'], 'success': False, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'DropAt_Book_DiningTable_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Book_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Book_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Book', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_TissueBox_DiningTable_True_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_TissueBox_DiningTable_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_TissueBox_DiningTable_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}}, 'GoTo([LOC_1], [LOC_2])': {'GoTo_Sofa_Sofa_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_Sofa_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_Sofa_True_1.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'Sofa', 'obj': '', 'loc': ''}, 'GoTo_Sofa_DiningTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_1.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 'loc': ''}}}
    skill2operators = {
        'PickUp([OBJ], [LOC])':{},
        'DropAt([OBJ], [LOC])':{},
        'GoTo([LOC_1], [LOC_2])':{}
    }
    pred_type = 'precond'
    pred_dict, mismatch_states = mismatch_symbolic_state(model, pred_dict, skill2tasks, pred_type)
    # print(mismatch_states) {'PickUp([OBJ], [LOC])': ['PickUp_TissueBox_Sofa_True_1', 'PickUp_Book_DiningTable_False_1'], 'DropAt([OBJ], [LOC])': ['DropAt_TissueBox_DiningTable_True_1', 'DropAt_TissueBox_Sofa_False_1']}
    # breakpoint()
    pred_type = 'eff'
    pred_dict, mismatch_states = mismatch_symbolic_state(model, pred_dict, skill2tasks, pred_type)
    # print(mismatch_states) # {'PickUp([OBJ], [LOC])': ['PickUp_TissueBox_Sofa_True_1', 'PickUp_Book_DiningTable_False_1'], 'DropAt([OBJ], [LOC])': ['DropAt_TissueBox_DiningTable_True_1', 'DropAt_Book_DiningTable_False_1']}
    # breakpoint()
    skill = 'PickUp([OBJ], [LOC])'
    precond = {}
    eff = {}
    
    # R1
    # pickup
    # skill2operators, pred_dict, skill2triedpred = refine_pred(model, skill, skill2operators, skill2tasks, pred_dict)
    # print(skill2operators, '\n\n', pred_dict)

    skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {}, 'eff': {'is_at_location([OBJ], [LOC])': -1}}, 'DropAt([OBJ], [LOC])': {}, 'GoTo([LOC_1], [LOC_2])': {}}
    pred_dict = {'is_at_location([OBJ], [LOC])': {'task': {'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Book_DiningTable_False_1': [True, True]}, 'semantic': 'The object `obj` is currently located at the location `loc`.'}}

    # dropat
    skill = 'DropAt([OBJ], [LOC])'
    # skill2operators, pred_dict, skill2triedpred = refine_pred(model, skill, skill2operators, skill2tasks, pred_dict)
    # print(skill2operators, '\n\n', pred_dict)

    skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {}, 'eff': {'is_at_location([OBJ], [LOC])': -1}}, 'DropAt([OBJ], [LOC])': {'precond': {'is_holding([OBJ])': True}, 'eff': {'is_at([OBJ], [LOC])': 1}}, 'GoTo([LOC_1], [LOC_2])': {'precond':{},'eff':{}}}
    pred_dict =  {'is_at_location([OBJ], [LOC])': {'task': {'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [True, True], 'GoTo_Sofa_Sofa_True_1': [True, True], 'GoTo_Sofa_DiningTable_True_1': [False, True]}, 'semantic': 'The object `obj` is currently located at the location `loc`.'}, 'is_holding([OBJ])': {'task': {'DropAt_TissueBox_Sofa_False_1': [True, False], 'DropAt_Book_DiningTable_False_1': [False, False], 'DropAt_TissueBox_DiningTable_True_1': [True, False], 'PickUp_TissueBox_Sofa_True_1': [False, False], 'PickUp_Book_DiningTable_False_1': [False, False], 'GoTo_Sofa_Sofa_True_1': [False, False], 'GoTo_Sofa_DiningTable_True_1': [False, False]}, 'semantic': 'The robot is currently holding the object `obj` before attempting to drop it at the location `loc`.'}, 'is_at([OBJ], [LOC])': {'task': {'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [False, True]}, 'semantic': 'The object `obj` is located at the location `loc` after the execution of the skill.'}}

    # goto will have no change because all tasks succeeded
    # but pred_dict is updated
    skill = 'GoTo([LOC_1], [LOC_2])'
    # skill2operators, pred_dict, skill2triedpred = refine_pred(model, skill, skill2operators, skill2tasks, pred_dict)
    # print(skill2operators, '\n\n', pred_dict)

    skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {}, 'eff': {'is_at_location([OBJ], [LOC])': -1}}, 'DropAt([OBJ], [LOC])': {'precond': {'is_holding([OBJ])': True}, 'eff': {'is_at([OBJ], [LOC])': 1}}, 'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {}}} 
    pred_dict = {'is_at_location([OBJ], [LOC])': {'task': {'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [True, True], 'GoTo_Sofa_Sofa_True_1': [True, True], 'GoTo_Sofa_DiningTable_True_1': [False, True]}, 'semantic': 'The object `obj` is currently located at the location `loc`.'}, 
                 'is_holding([OBJ])': {'task': {'DropAt_TissueBox_Sofa_False_1': [True, False], 'DropAt_Book_DiningTable_False_1': [False, False], 'DropAt_TissueBox_DiningTable_True_1': [True, False], 'PickUp_TissueBox_Sofa_True_1': [False, False], 'PickUp_Book_DiningTable_False_1': [False, False], 'GoTo_Sofa_Sofa_True_1': [False, False], 'GoTo_Sofa_DiningTable_True_1': [False, False]}, 'semantic': 'The robot is currently holding the object `obj` before attempting to drop it at the location `loc`.'}, 
                 'is_at([OBJ], [LOC])': {'task': {'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [False, True], 'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Book_DiningTable_False_1': [True, True], 'GoTo_Sofa_Sofa_True_1': [True, True], 'GoTo_Sofa_DiningTable_True_1': [False, True]}, 'semantic': 'The object `obj` is located at the location `loc` after the execution of the skill.'}}
    

    # R2
    # new skill2tasks
    skill2tasks = {'PickUp([OBJ], [LOC])': {'PickUp_TissueBox_Sofa_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_Sofa_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_Sofa_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'PickUp_Bowl_DiningTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_Book_DiningTable_False_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Book_DiningTable_False_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Book_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Book', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_TissueBox_DiningTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_DiningTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_DiningTable_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}}, 'DropAt([OBJ], [LOC])': {'DropAt_TissueBox_DiningTable_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_TissueBox_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_TissueBox_DiningTable_False_1.jpg'], 'success': False, 'obj': 'TissueBox', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Bowl_Sofa_True_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Bowl_Sofa_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Bowl_Sofa_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'DropAt_TissueBox_Sofa_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_TissueBox_Sofa_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_TissueBox_Sofa_False_1.jpg'], 'success': False, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'DropAt_Book_DiningTable_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Book_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Book_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Book', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_TissueBox_DiningTable_True_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_TissueBox_DiningTable_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_TissueBox_DiningTable_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Book_Sofa_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Book_Sofa_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Book_Sofa_False_1.jpg'], 'success': False, 'obj': 'Book', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}}, 'GoTo([LOC_1], [LOC_2])': {'GoTo_DiningTable_Sofa_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_Sofa_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_Sofa_True_1.jpg'], 'success': True, 'loc_1': 'DiningTable', 'loc_2': 'Sofa', 'obj': '', 'loc': ''}, 'GoTo_Sofa_Sofa_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_Sofa_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_Sofa_True_1.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'Sofa', 'obj': '', 'loc': ''}, 'GoTo_Sofa_DiningTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_1.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 'loc': ''}}}

    # pickup
    skill = 'PickUp([OBJ], [LOC])'
    # skill2operators, pred_dict, skill2triedpred = refine_pred(model, skill, skill2operators, skill2tasks, pred_dict)
    # print(skill2operators, '\n\n', pred_dict)

    skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {'is_within_reach([OBJ], [LOC])': True}, 'eff': {'is_at_location([OBJ], [LOC])': -1}}, 'DropAt([OBJ], [LOC])': {'precond': {'is_holding([OBJ])': True}, 'eff': {'is_at([OBJ], [LOC])': 1}}, 'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {}}}
    pred_dict = {'is_at_location([OBJ], [LOC])': {'task': {'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [True, True], 'GoTo_Sofa_Sofa_True_1': [True, True], 'GoTo_Sofa_DiningTable_True_1': [False, True], 'PickUp_Bowl_DiningTable_True_1': [False, False], 'PickUp_TissueBox_DiningTable_True_1': [True, False], 'DropAt_TissueBox_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [False, False], 'DropAt_Book_Sofa_False_1': [True, True], 'GoTo_DiningTable_Sofa_True_1': [True, True]}, 'semantic': 'The object `obj` is currently located at the location `loc`.'}, 
                 'is_holding([OBJ])': {'task': {'DropAt_TissueBox_Sofa_False_1': [True, False], 'DropAt_Book_DiningTable_False_1': [False, False], 'DropAt_TissueBox_DiningTable_True_1': [True, False], 'PickUp_TissueBox_Sofa_True_1': [False, False], 'PickUp_Book_DiningTable_False_1': [False, False], 'GoTo_Sofa_Sofa_True_1': [False, False], 'GoTo_Sofa_DiningTable_True_1': [False, False], 'PickUp_Bowl_DiningTable_True_1': [False, False], 'PickUp_TissueBox_DiningTable_True_1': [False, False], 'DropAt_TissueBox_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [False, False], 'DropAt_Book_Sofa_False_1': [False, False], 'GoTo_DiningTable_Sofa_True_1': [False, False]}, 'semantic': 'The robot is currently holding the object `obj` before attempting to drop it at the location `loc`.'}, 
                 'is_at([OBJ], [LOC])': {'task': {'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [False, True], 'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Book_DiningTable_False_1': [True, True], 'GoTo_Sofa_Sofa_True_1': [True, True], 'GoTo_Sofa_DiningTable_True_1': [False, True], 'PickUp_Bowl_DiningTable_True_1': [False, False], 'PickUp_TissueBox_DiningTable_True_1': [True, False], 'DropAt_TissueBox_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [False, False], 'DropAt_Book_Sofa_False_1': [True, True], 'GoTo_DiningTable_Sofa_True_1': [False, True]}, 'semantic': 'The object `obj` is located at the location `loc` after the execution of the skill.'}, 
                 'is_within_reach([OBJ], [LOC])': {'task': {'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Bowl_DiningTable_True_1': [False, False], 'PickUp_Book_DiningTable_False_1': [False, False], 'PickUp_TissueBox_DiningTable_True_1': [True, False], 'DropAt_TissueBox_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [True, True], 'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [False, True], 'DropAt_Book_Sofa_False_1': [True, True], 'GoTo_DiningTable_Sofa_True_1': [True, True], 'GoTo_Sofa_Sofa_True_1': [True, True], 'GoTo_Sofa_DiningTable_True_1': [True, True]}, 'semantic': "The object is within the robot's reachable distance at the specified location."}}

    # dropat
    skill = 'DropAt([OBJ], [LOC])'
    # skill2operators, pred_dict, skill2triedpred = refine_pred(model, skill, skill2operators, skill2tasks, pred_dict)
    # print(skill2operators, '\n\n', pred_dict)

    skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {'is_within_reach([OBJ], [LOC])': True}, 'eff': {'is_at_location([OBJ], [LOC])': -1}}, 'DropAt([OBJ], [LOC])': {'precond': {'is_holding([OBJ])': True}, 'eff': {'is_at([OBJ], [LOC])': 1}}, 'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {}}} 

    pred_dict = {'is_at_location([OBJ], [LOC])': {'task': {'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [True, True], 'GoTo_Sofa_Sofa_True_1': [True, True], 'GoTo_Sofa_DiningTable_True_1': [False, True], 'PickUp_Bowl_DiningTable_True_1': [False, False], 'PickUp_TissueBox_DiningTable_True_1': [True, False], 'DropAt_TissueBox_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [False, False], 'DropAt_Book_Sofa_False_1': [True, True], 'GoTo_DiningTable_Sofa_True_1': [True, True]}, 'semantic': 'The object `obj` is currently located at the location `loc`.'}, 'is_holding([OBJ])': {'task': {'DropAt_TissueBox_Sofa_False_1': [True, False], 'DropAt_Book_DiningTable_False_1': [False, False], 'DropAt_TissueBox_DiningTable_True_1': [True, False], 'PickUp_TissueBox_Sofa_True_1': [False, False], 'PickUp_Book_DiningTable_False_1': [False, False], 'GoTo_Sofa_Sofa_True_1': [False, False], 'GoTo_Sofa_DiningTable_True_1': [False, False], 'PickUp_Bowl_DiningTable_True_1': [False, False], 'PickUp_TissueBox_DiningTable_True_1': [False, False], 'DropAt_TissueBox_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [False, False], 'DropAt_Book_Sofa_False_1': [False, False], 'GoTo_DiningTable_Sofa_True_1': [False, False]}, 'semantic': 'The robot is currently holding the object `obj` before attempting to drop it at the location `loc`.'}, 'is_at([OBJ], [LOC])': {'task': {'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [False, True], 'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Book_DiningTable_False_1': [True, True], 'GoTo_Sofa_Sofa_True_1': [True, True], 'GoTo_Sofa_DiningTable_True_1': [False, True], 'PickUp_Bowl_DiningTable_True_1': [False, False], 'PickUp_TissueBox_DiningTable_True_1': [True, False], 'DropAt_TissueBox_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [False, False], 'DropAt_Book_Sofa_False_1': [True, True], 'GoTo_DiningTable_Sofa_True_1': [False, True]}, 'semantic': 'The object `obj` is located at the location `loc` after the execution of the skill.'}, 'is_within_reach([OBJ], [LOC])': {'task': {'PickUp_TissueBox_Sofa_True_1': [True, False], 'PickUp_Bowl_DiningTable_True_1': [False, False], 'PickUp_Book_DiningTable_False_1': [False, False], 'PickUp_TissueBox_DiningTable_True_1': [True, False], 'DropAt_TissueBox_DiningTable_False_1': [False, False], 'DropAt_Bowl_Sofa_True_1': [True, True], 'DropAt_TissueBox_Sofa_False_1': [False, False], 'DropAt_Book_DiningTable_False_1': [True, True], 'DropAt_TissueBox_DiningTable_True_1': [False, True], 'DropAt_Book_Sofa_False_1': [True, True], 'GoTo_DiningTable_Sofa_True_1': [True, True], 'GoTo_Sofa_Sofa_True_1': [True, True], 'GoTo_Sofa_DiningTable_True_1': [True, True]}, 'semantic': "The object is within the robot's reachable distance at the specified location."}}

    # goto will have no change because all tasks succeeded
    # but pred_dict is updated
    skill = 'GoTo([LOC_1], [LOC_2])'
    # skill2operators, pred_dict, skill2triedpred = refine_pred(model, skill, skill2operators, skill2tasks, pred_dict)
    # print(skill2operators, '\n\n', pred_dict)

    # breakpoint()

    # merge before
    # unified_skill2operator, equal_preds = merge_predicates(model, skill2operators, pred_dict)
    # print(unified_skill2operator, '\n\n', equal_preds)

    unified_skill2operator = {'PickUp([OBJ], [LOC])': {'precond': {'is_within_reach([OBJ], [LOC])': True}, 'eff': {'is_at_location([OBJ], [LOC])': -1}}, 'DropAt([OBJ], [LOC])': {'precond': {'is_holding([OBJ])': True}, 'eff': {'is_at([OBJ], [LOC])': 1, 'is_dropped_at([OBJ], [LOC])': -1}}, 'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {}}}

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