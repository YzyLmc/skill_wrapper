'Get symbolic representation from skill semantic info and observation'
from utils import GPT4, load_from_file
from collections import defaultdict
import inspect
import random

from manipula_skills import *

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

def eval_pred(model, skill, pred, obj, loc, img, prompt_fpath='prompts/evaluate_pred_ai2thor.txt'):
    '''
    Evaluate one predicate given one image.
    If skill or predicate takes variables they should contain [OBJ] or [LOC] in the input.
    Empty string '' means no argument
    '''
    def construct_prompt(prompt, skill, pred, obj, loc):
        while "[PRED]" in prompt or "[SKILL]" in prompt:
            prompt = prompt.replace("[PRED]", pred)
            prompt = prompt.replace("[SKILL]", skill)
        while "[OBJ]" in prompt or "[LOC]" in prompt:
            prompt = prompt.replace("[OBJ]", obj)
            prompt = prompt.replace("[LOC]", loc)
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred, obj, loc)
    resp = model.generate_multimodal(prompt, img)[0]
    return True if "True" in resp.split('\n')[-1] else False

def eval_pred_set(model, skill, pred_set, obj, loc, img):
    '''
    Evaluate set of predicates
    pred_set::list(str)
    returns::Dict(str:bool)
    '''
    return {pred: eval_pred(model, skill, pred, obj, loc, img) for pred in pred_set}

# TODO: have to track predicates have been tried in the prompt?
# Adding to precondition or effect are different prompts
def generate_pred(model, skill, pred_dict, pred_type, prompt_fpath='prompts/predicate_refining'):
    '''
    generate_predicates based on existing predicates dictionary describing the same symbolic state
    pred_dict:: Dict(pred_name: Bool)
    type:: str:: 'precond' or 'eff' 
    '''
    def construct_prompt(prompt, skill, pred_dict):
        while "[SKILL]" in prompt or "[PRED_DICT]" in prompt:
            prompt = prompt.replace("[SKILL]", skill)
            prompt = prompt.replace("[PRED_DICT]", str(pred_dict))
        # convert the placeholders back to 'obj' and 'loc'
        while "[OBJ]" in prompt or "[LOC]" in prompt:
            prompt = prompt.replace("[OBJ]", "obj")
            prompt = prompt.replace("[LOC]", "loc")
        return prompt
    prompt_fpath += f"_{pred_type}.txt"
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred_dict)
    return model.generate(prompt)[0]

# TODO: code for seperate tasks in terms of skill names:: dict(skill: list(dict('id':num, 's0':img, 's1':img, 'success': Bool)))
def refine_pred(model, skill, tasks, pred_dict, precond, eff, max_t=3):
    '''
    Refine predicates of precondition and effect of one skill for precondition and effect
    tasks:: dict(id, dict('s0':img, 's1':img, 'success': Bool))
    '''
    def mismatch_symbolic_state(pred_dict, tasks, pred_type):
        '''
        look for same symbolic states (same s1 or s2) in task dictionary
        pred_dict::{pred_name:{task: [Bool, Bool]}}
        pred_type::{'precond', 'eff'}
        returns::list(id: dict('s0':img, 's1':img, 'success': Bool))
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

        task2state = {}
        if pred_type == "precond":
            # return two same symbolic states BEFORE EXECUTION with different successfulness
            for task in tasks:
                task2state[task] = {pred: pred_dict[task][0] for pred in pred_dict}
            dup_tasks = tasks_with_same_symbolic_states(task2state)
        elif pred_type == "eff":
            # return two same symbolic states TRANSITION (symbolic) with different successfulness
            for task in tasks:
                task2state[task] = {pred: int(pred_dict[task][1]==True) - int(pred_dict[task][0]==True) for pred in pred_dict}
            dup_tasks = tasks_with_same_symbolic_states(task2state)
        
        if dup_tasks:
            t1 = random.choice(list(dup_tasks.keys()))
            t2 = random.choice(dup_tasks[t1])
            return [task[t] for t in [t1, t2]]
        else:
            return []

    # keep tracking truth value of predicates before and after a skill execution
    # {pred_name:{task: [Bool, Bool]}}
    if not pred_dict:
        pred_dict = {}
    if not precond:
         precond = {}
    if not eff:
         eff = {}
    
    # check precondition first
    t = 0
    mismatch_tasks = mismatch_symbolic_state(pred_dict, tasks, 'pred')
    new_p_added = False
    while mismatch_tasks and t < max_t:
        new_p = generate_pred(model, skill, list(pred_dict.keys()))
        new_p_mismatch = {idx: [eval_pred(model, new_p, task['s0']), eval_pred(model, new_p, task['s0'])] for idx, task in mismatch_tasks.items()}
        if new_p_mismatch[list(mismatch_tasks.keys())[0]][0] != new_p_mismatch[list(mismatch_tasks.keys())[1]][0]:
            precond[new_p] = True if list(mismatch_tasks.keys())[0]['success'] == new_p_mismatch[list(mismatch_tasks.keys())[0]][0] else False
            new_p_added = True
        t += 1
    if new_p_added:
        pred_dict[new_p] = {idx: [eval_pred(model, new_p, task['s0']), eval_pred(model, new_p, task['s0'])] for idx, task in tasks.items()}
        new_p_added = False

    # check effect
    t = 0
    mismatch_tasks = mismatch_symbolic_state(pred_dict, tasks, 'eff')
    while mismatch_tasks and t < max_t:
        new_p = generate_pred(model, skill, list(pred_dict.keys()))
        new_p_mismatch = {idx: [eval_pred(model, new_p, task['s0']), eval_pred(model, new_p, task['s0'])] for idx, task in mismatch_tasks.items()}
        # first index for task number, second index for before and after
        s_1_1 = new_p_mismatch[list(mismatch_tasks.keys())[0]][0]
        s_1_2 = new_p_mismatch[list(mismatch_tasks.keys())[0]][1]
        s_2_1 = new_p_mismatch[list(mismatch_tasks.keys())[1]][0]
        s_2_2 = new_p_mismatch[list(mismatch_tasks.keys())[1]][1]
        if not (int(s_1_2==True) - int(s_1_1==True) != int(s_2_2==True) - int(s_2_1==True)):
            success_task = list(mismatch_tasks.values())[0] if list(mismatch_tasks.values())[0]['success'] == True else list(mismatch_tasks.values())[1]
            eff[new_p] = int(new_p_mismatch[success_task][1]==True) - int(new_p_mismatch[success_task][0]==True)
            new_p_added = True
        t += 1
    if new_p_added:
        pred_dict[new_p] = {idx: [eval_pred(model, new_p, task['s0']), eval_pred(model, new_p, task['s0'])] for idx, task in tasks.items()}
        new_p_added = False
            
    return precond, eff, pred_dict

### task proposing part below

def scoring_chain(task, operators):
    'returns the maximum steps that can be executed from the begining of a task'
    pass

def top_k_combo(coverage_table, k):
    'calculate top k skill combination with highest information gain'
    pass

def update_coverage(coverage_table, task):
    'update the coverage table with the task just been executed'
    pass

def task_proposing(skill, tasks):
    '''
    tasks::list(str):: previous tasks
    '''
    pass

if __name__ == '__main__':
    model = GPT4()
    # # test predicate evaluation function
    # pred = 'handEmpty()'
    # skill = 'PickUp([OBJ])'
    # obj = 'Book'
    # loc = ''
    # img = ['pickup_t2_s0_fail.jpg']
    # response = eval_pred(model, skill, pred, obj, loc, img)
    # print(response)

    # # test predicate proposing for refining

    # # mock symbolic state
    # pred_dict = {'handEmpty()': True}
    # skill = 'PickUp([OBJ])'
    # pred_type = 'precond'
    # response = generate_pred(model, skill, pred_dict, pred_type)
    # print(response)

    # test main refining function
