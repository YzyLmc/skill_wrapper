'''
Get symbolic representation from skill semantic info and observation.
Data structures:
    type_dict :: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    skill :: dict :: {'name':str, 'types':list,'params':list}
    predicate :: dict :: {'name':str, 'types':list, 'params':list, 'semantic':str}
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
        # NOTE: this will re-save sematic for each new grounding of the predicates
        self.semantic_dict = {self._keyify(pred): pred["semantic"] for pred in predicates}
    
    def __eq__(self, other):
        """
        Checks if two PredicateState instances are equal.
        Two PredicateState instances are considered equal if they contain the same predicates
        with the same truth values.
        Requried for making the class hashable
        """
        if not isinstance(other, PredicateState):
            return False
        return self.pred_dict == other.pred_dict
    
    def __hash__(self):
        """
        Generates a hash based on the predicate dictionary.
        """
        # Sort keys for deterministic order, and convert to a tuple of key-value pairs
        items = tuple(sorted(self.pred_dict.items()))
        return hash(items)

    @staticmethod
    def _keyify(pred):
        """Generates a unique key from predicate name and parameters."""
        return (pred["name"], tuple(pred["types"]),tuple(pred["params"]))
    
    @staticmethod
    def restore_pred_from_key(key_tuple, semantic_dict=None):
        """
        Restores a predicate dictionary from a keyified tuple: (name, types, params).
        'semantic' is set to None since it's not stored in the key.
        """
        name, types, params = key_tuple
        semantic = semantic_dict.get(key_tuple, None) if semantic_dict else None
        return {
            "name": name,
            "types": list(types),
            "params": list(params),
            "semantic": semantic
        }
    
    def iter_predicates(self):
        """
        Generator that yields each grounded predicate as a dictionary in original format.
        """
        for name, types, params in self.pred_dict.keys():
            yield {
                "name": name,
                "types": list(types),
                "params": list(params),
                # Optional: include semantic as None
                "semantic": None
            }
    
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
                    "semantic": self.semantic_dict[key]
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
        else:
            raise Exception("Predicate not found!")

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
                assert all(key in ["name", "types", "params", "semantic"] for key in new_pred.keys())
                self.pred_dict[pred_key] = None
                self.semantic_dict[pred_key] = new_pred["semantic"]

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

def ground_with_params(lifted, params, type_dict=None):
    """
    Grounded a skill or a predicate with parameters and their types.
    Basically add another key, value to it
    lifted::dict:: lifted skill/predicate with parameter type, e.g., {'name':"GoTo", 'types':["[ROBOT]", "[LOC]", "[LOC]"]}/{'name':"At", 'types':["[LOC]"]}
    params::list:: list of parameters, e.g., ["Apple", "Table"]
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    """
    # tuple is applicable to the lifted representation
    if type_dict:
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

def lift_grounded_pred(grounded_pred, type_dict=None):
    """
    Lift a grounded predicate, e.g., {'name': 'At', 'types': ["object", "location"], 'params': ['Apple', 'Table']} to a {'name':"At", 'types':["object", "location"], 'params':[]}
    type_dict:: dict:: {param: type}, e.g., {"Apple": ['object'], "Table": ['location']}
    """
    if type_dict:
        assert all([type in type_dict[param] for type, param in zip(grounded_pred['types'], grounded_pred['params'])])
    return {'name':grounded_pred['name'], 'types':grounded_pred['types'], 'params':[], 'semantic':grounded_pred['semantic']}

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

def eval_pred(model, img, grounded_skill, grounded_pred, type_dict, prompt_fpath=['prompts/evaluate_pred_ai2thor.txt','prompts/evaluate_pred_ai2thor_init.txt'], init=False) -> bool:
    '''
    evaluate truth value of a predicate using a dictionary of parameters
    init step and later steps use different prompts. hardcoded.
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

def generate_pred(model, skill, pred_list, pred_type, tried_pred=[], prompt_fpath='prompts/predicate_refining') -> dict:
    '''
    propose new predicates based on the contrastive pair

    return:
    new_pred :: {'name':str, 'types':list, 'params':list, 'semantic':str}
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
    skill2task2state::{skill_keyified: {task_step_name: {"states": [PredicateState, PredicateState], "success": bool}}}
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
                    skill2task2state[skill_keyified][task_name_stepped] = {'state':[last_state, state], 'success': tasks[task_name][step]['success']}
                last_state = deepcopy(state)
    return skill2task2state

def detect_mismatch(skill, operators, grounded_predicate_truth_value_log, tasks, pred_type) -> list[list[str, str]]:
    """
    Find mismatch state pairs where they both belong to Union Precondition or Effect.
    operators :: [{"skill": skill_name | str, "precond": {lift_pred | tuple : bool}, "eff+":{...}, "eff-":{...}}]
    grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
    tasks :: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': bool))) ::
    pred_type::{'precond', 'eff'}

    Returns:
    mismatch_pairs :: [[task_name_stepped, task_name_stepped]...]
    """
    # TODO: compatibility with empty precond and eff, and empty operator
    def in_alpha(state_tuple, grounded_skill, operators, pred_type):
        """
        If a state is within the union set of precond/eff of learned operators.
        The predicates need to be grounded with the same params as the skill so that we can evaluate over grounded states
        state :: PredicateState
        """
        skill_params = grounded_skill['params']
        if pred_type == 'precond':
            state = state_tuple[0]
            for operator in operators:
                # if truth values of precond are all satisfied by state
                # massive line but basically saying truth value in PredicateState should agree with predicate grounded with skill params
                if any([state.get_pred_value(ground_with_params(PredicateState.restore_pred_from_key(lift_pred_tuple), skill_params))!=value for lift_pred_tuple, value in operator['precond'].items()]):
                    return False
            return True

        # effect is a bit tricky
        # if not specified by precond:
        #   if not specified by eff: no care
        #   if specified by eff:
        #       if specified by eff+: final value has to be true
        #       if specified by eff-: final value has to be false
        # if specified by precond:
        #   if not specified by eff: final value has to agree with precond
        #   if specified by precond and eff:
        #       if specified by eff+: final value has to be true
        #       if specified by eff-: final value has to be false

        elif pred_type == "eff":
            state = state_tuple[1]
            for operator in operators:
                # construct grounded predicate with skill params
                for lifted_pred_tuple, value in operator["eff+"].items():
                    grounded_pred = ground_with_params(PredicateState.restore_pred_from_key(lift_grounded_pred), skill_params)
                    if not state.get_pred_value(grounded_pred) == value:
                        return False
                for lifted_pred_tuple, value in operator["eff-"].items():
                    grounded_pred = ground_with_params(PredicateState.restore_pred_from_key(lift_grounded_pred), skill_params)
                    if not state.get_pred_value(grounded_pred) == value:
                        return False
                # predicates in precond but not in eff+/- have to remain the same
                for lifted_pred_tuple, value in operator["precond"].items():
                    if (lifted_pred_tuple not in operator["eff+"]) and (lifted_pred_tuple not in operator["eff-"]):
                        grounded_pred = ground_with_params(PredicateState.restore_pred_from_key(lift_grounded_pred), skill_params)
                        if not state.get_pred_value(grounded_pred) == value:
                            return False
            return True
        
    skill2task2state = grounded_pred_log_to_skill2task2state(grounded_predicate_truth_value_log, tasks)
    task2state = skill2task2state[PredicateState._keyify(skill)]
    task2in_alpha = {task_name_stepped: in_alpha(state_meta['state'], grounded_skill, operators, pred_type) for task_name_stepped, state_meta in task2state.items()} # {task_name_stepped: in_alpha | bool}
    task2success = {task_name_stepped: state_meta['success'] for task_name_stepped, state_meta in task2state.items()} # {task_name_stepped: success | bool}
    assert len(task2in_alpha) == len(task2success), "length of both dictionaries state2in_alpha and state2success must equal"
    task_name_stepped_list = list(task2in_alpha.keys())
    mismatched_pairs = []
    # looking for pairs of state where truth value of s1 and s2 agree in state2in_alpha but conflict in state2success
    for i in range(len(task_name_stepped_list)):
        for j in range(i + 1, len(task_name_stepped_list)):
            task_name_stepped_1, task_name_stepped_2 = task_name_stepped_list[i], task_name_stepped_list[j]
            if task2in_alpha[task_name_stepped_1] == task2in_alpha[task_name_stepped_2] and task2success[task_name_stepped_1] != task2success[task_name_stepped_2]:
                mismatched_pairs.append([task_name_stepped_1, task_name_stepped_2])
    
    return mismatched_pairs

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
    # NOTE: effect score will be different now due to the new partition
    add_new_pred = score_by_partition(new_pred, skill, hypothetical_skill2task2state, pred_type, threshold)
    # logging.info(f"Precondition T score of predicate {dict_to_string(new_pred)}: {t_score}, F score: {f_score}")
    if add_new_pred:
        logging.info(f"Predicate {new_pred} added to predicate set by {pred_type} check")
        lifted_pred_list.append(new_pred)
        new_pred_accepted = True
    else:
        logging.info(f"Predicate {new_pred} is NOT added to predicate set by {pred_type} check")
        skill2triedpred[skill].append(new_pred)
    
    return lifted_pred_list, skill2triedpred, new_pred, new_pred_accepted


def invent_predicates(model, skill, operators, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, skill2triedpred={}, max_t=3):
    '''
    Main loop of generating predicates. It also evaluates empty predicates that introduced by new tasks or new predicates
    '''
    # TODO: number of if statements to create empty dicts or lists for on init
    
    # check precondition first
    t = 0
    pred_type = "precond"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log)
    mismatch_tasks = detect_mismatch(skill, operators, grounded_predicate_truth_value_log, tasks, pred_type=pred_type)
    logging.info("About to enter precondition check")
    logging.info(f"mismatch state for precondition generation {mismatch_tasks[skill][0]}, {mismatch_tasks[skill][1]}")
    while mismatch_tasks and t < max_t:
        lifted_pred_list, skill2triedpred, new_pred, new_pred_accepted = invent_predicate_one(model, skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type,  skill2triedpred=skill2triedpred)
        logging.info(f"Iteration {t} of predicate invention. {new_pred} is accepted: {new_pred_accepted}")
        t += 1
    
    # check effect
    t = 0
    pred_type = "eff"
    grounded_predicate_truth_value_log = update_empty_predicates(model, tasks, lifted_pred_list, type_dict, grounded_predicate_truth_value_log)
    mismatch_tasks = detect_mismatch(skill, operators, grounded_predicate_truth_value_log, tasks, pred_type=pred_type)
    logging.info("About to enter effect check")
    logging.info(f"mismatch state for precondition generation {mismatch_tasks[skill][0]}, {mismatch_tasks[skill][1]}")
    while mismatch_tasks and t < max_t:
        lifted_pred_list, skill2triedpred, new_pred, new_pred_accepted = invent_predicates(model, skill, tasks, grounded_predicate_truth_value_log, type_dict, lifted_pred_list, pred_type,  skill2triedpred=skill2triedpred)
        logging.info(f"Iteration {t} of predicate invention. {new_pred} is accepted: {new_pred_accepted}")
        t += 1
    
    # partitioning
    # 1. partition by each different termination set
    partitioned_output = partition_by_termination(grounded_predicate_truth_value_log)
    # 2. create one operator for each partition
    operators = create_operators_from_partitions(partitioned_output)
    # 3. merge operators
    # NOTE: We use a mutable dictionary to represent operators, so they can by merged just using '='
    return operators, lifted_pred_list, skill2triedpred

def score_by_partition(new_pred, skill, skill2task2state, pred_type, threshold) -> bool:
    '''
    Partition by effect and then score the predicates across each partition
    skill :: grouded skill {"name":"PickUp", "types":["obj"], "params":["Apple"]}
    threshold={"precond":float, "eff":float}
    '''

    skill_keyified_list = skill2task2state.keys()
    skill_keyified = PredicateState._keyify(skill)
    task2state = skill2task2state[skill_keyified]
    # 1. find all states after executing the same grounded skill
    skill2state2partition = partition_by_termination(task2state)
    state2partition = skill2state2partition[skill_keyified]

    # 2. evaluate the score for each task2state dictionary, pick the best one
    for state, partition in state2partition.items():
        for task_name_stepped_list in partition.values():
            for skill_keyified in skill_keyified_list:
                partitioned_task2state = {task_name_stepped: skill2task2state[skill_keyified] for task_name_stepped in task_name_stepped_list}
                # TODO: check if this yields good result
                new_pred_grounded = ground_with_params(new_pred, skill_keyified[2])
                t_score_t, f_score_t, t_score_f, f_score_f = score(new_pred_grounded, partitioned_task2state, pred_type)
                if pred_type == "precond":
                    if (t_score_t > threshold[pred_type] and f_score_t > threshold[pred_type]) \
                        or (t_score_f > threshold[pred_type] and f_score_f > threshold[pred_type]):
                        return True
                    
                # NOTE: effect score will be different now due to the new partition method
                elif pred_type == "eff":
                    if (t_score_t > threshold[pred_type] or f_score_t > threshold[pred_type]) \
                        or (t_score_f > threshold[pred_type] or f_score_f > threshold[pred_type]):
                        return True
                
    return False

def partition_by_termination(skill2task2state) -> dict[tuple, list[dict[PredicateState, list[tuple]]]]:
    '''
    Partition the a set of trajectory using termination set. Will be used again in scoring and final operators learning.
    Only successful execution will be used for partitioning
    skill :: grouded skill {"name":"PickUp", "types":["obj"], "params":["Apple"]}
    skill2task2state::{skill_keyified: {task_name_stepped: {"states": [PredicateState, PredicateState], "success": bool}}}
    grounded_predicate_truth_value_log::dict:: {task:{step:PredicateState}}
    return:: {skill: [{PredicateState: [task_name_stepped]} , ...]}
    '''
    def states_are_equal(state_1, state_2):
        return state_1.pred_dict == state_2.pred_dict
    
    skill2partition = defaultdict(dict)
    for skill_keyified, task2state in skill2task2state.items():
        partition = defaultdict(list) # {task_step}
        for task_name_stepped, state_meta in task2state.items():
            find_partition = False
            state = state_meta['state'][1] # s1 for next state
            for registered_state in partition:
                if states_are_equal(state, registered_state):
                    partition[state].append(task_name_stepped)
                    find_partition = True
            if not find_partition:
                partition[state].append(task_name_stepped)
                find_partition = False
        skill2partition[skill_keyified] = partition
    return skill2partition

def create_operators_from_one_partition(task2state, task_name_stepped_list) -> list[tuple[dict[PredicateState, bool], dict[str, list[PredicateState]]]]:
    """
    Create operators from one partition.
    One partition should have only one operator since they have same termination set
    and from a same skill, but not guaranteed.
    task2state :: {task_name_stepped: {"states": [PredicateState, PredicateState], "success": bool}}
    task_name_stepped_list :: [str]
    Return:
    precond_n_effect :: [({PredicateState: bool}, {eff+: [PredicateState], eff-: [PredicateState]})]
    """
    def calculate_effect(s_1: PredicateState, s_2: PredicateState) -> dict[str, list[dict]]:
        """
        Compares two states before and after execution and calculate eff+ and eff-.
        Assumes both have the same predicate keys.
        Returns:
        effect_keyified :: (eff+, eff-)
            effect :: {'eff+': [Predicate: dict], 'eff-': [Predicate: dict]}
        """
        # TODO: consider invariants in effect?
        assert s_1.pred_dict.keys() == s_2.pred_dict.keys(), "PredicateStates must have identical keys."

        effect = defaultdict(list)
        for grounded_pred in s_1:
            val1 = s_1.get_pred_value(grounded_pred)
            val2 = s_2.get_pred_value(grounded_pred)
            if val1 != val2:
                eff_type = 'eff+' if val1 == True else 'eff-'
                effect[eff_type].append(grounded_pred)
        return effect
    
    def calculate_precondition(state_list) -> dict[tuple, bool]:
        """
        Calculate precondition of a set of effect by taking intersection of the init states
        state_list :: [PredicateState]
        """
        # Start with the full set of keys from the first PredicateState
        precond = state_list[0].pred_dict.copy()

        for ps in state_list[1:]:
            new_precond = {}
            for pred_keyified in precond:
                assert pred_keyified in ps.pred_dict, "all state should have same predicates"
                if ps.pred_dict[pred_keyified] == precond[pred_keyified]:
                    new_precond[pred_keyified] = precond[pred_keyified]
            precond = new_precond  # Narrow down to only shared + matching predicates

            if not precond:
                break  # Early exit if nothing is shared

        return precond
    
    # 1. calculate different effect in the partition
    effect2task_name_stepped_list = defaultdict(list)
    for task_name_stepped in task_name_stepped_list:
        state_tuple = task2state[task_name_stepped]['state']
        effect = calculate_effect(state_tuple[0], state_tuple[1])
        effect_keified = (set(sorted(effect['eff+'])), set(sorted(effect['eff-'])))
        effect2task_name_stepped_list[effect_keified].append(task_name_stepped)
    
    # 2. calculate precondition for each different effect
    precond_n_effect = []
    for effect, task_name_stepped_list in effect2task_name_stepped_list.items():
        state_list = [task2state[task_name_stepped]['state'][0] for task_name_stepped in task_name_stepped_list]
        precond_n_effect.append((calculate_precondition(state_list), effect))

    return precond_n_effect

def create_operators_from_partitions(skill2task2state, skill2partition):
    """
    Calculate operators using the partitions by termination set.

    Returns:
    operators :: [{"skill": lifted_skill | str, "precond": {lift_pred | tuple : bool}, "eff+":{...}, "eff-":{...}}]
        lifted_skill :: {'name':str, 'types':list,'params':list} # params is empty
    """
    def lift_operator(grounded_operator, params, type_dict):
        """
        Lift a grounded predicate by replacing its parameters with typed arguments
        """
        pass
    operators = []
    # create operators for each grounded skill
    for skill_keyified, task2state in skill2task2state.items():
        precond_n_effect = create_operators_from_one_partition(task2state, skill2partition[skill_keyified])
        # lift the variables in precondition and effect if agree with params of the skill

    pass

def score(pred, task2state, pred_type) -> tuple[float, float, float, float]:
    """
    score of a predicate as one skill's precondition or effect
    tasks:: dict(id: (step: dict("skill": grounded_skill, 'image':img_path, 'success': Bool))) ; step is int ranging from 0-8
    task2state :: {task_name_stepped: {"states": [PredicateState, PredicateState], "success": bool}}
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
    possible_preds = possible_grounded_predicates(test_pred_state.get_pred_list(lifted=True), type_dict)
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