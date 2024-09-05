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

def eval_pred(model, skill, pred, sem, obj, loc, img, prompt_fpath='prompts/evaluate_pred_ai2thor.txt'):
    '''
    Evaluate one predicate given one image.
    If skill or predicate takes variables they should contain [OBJ] or [LOC] in the input.
    Empty string '' means no argument
    '''
    def construct_prompt(prompt, skill, pred, obj, loc):
        while "[PRED]" in prompt or "[SKILL]" in prompt or "[SEMANTIC]" in  prompt:
            prompt = prompt.replace("[PRED]", pred)
            prompt = prompt.replace("[SKILL]", skill)
            prompt = prompt.replace("[SEMANTIC]", sem)
        while "[OBJ]" in prompt or "[LOC]" in prompt:
            prompt = prompt.replace("[OBJ]", obj)
            prompt = prompt.replace("[LOC]", loc)
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred, obj, loc)
    resp = model.generate_multimodal(prompt, img)[0]
    # breakpoint()
    return True if "True" in resp.split('\n')[-1] else False

def eval_pred_set(model, skill, pred2sem, obj, loc, img):
    '''
    Evaluate set of predicates
    pred_set::list(str)
    returns::Dict(str:bool)
    '''
    return {pred: eval_pred(model, skill, pred, sem, obj, loc, img) for pred, sem in pred2sem.items()}

# TODO: have to track predicates have been tried in the prompt?
# Adding to precondition or effect are different prompts
def generate_pred(model, skill, pred_dict, pred_type, tried_pred=[], prompt_fpath='prompts/predicate_refining'):
    '''
    generate_predicates based on existing predicates dictionary describing the same symbolic state
    pred_dict:: Dict(pred_name: Bool)
    type:: str:: 'precond' or 'eff' 
    '''
    def construct_prompt(prompt, skill, pred_dict):
        while "[SKILL]" in prompt or "[PRED_DICT]" in prompt or "[TRIED_PRED]" in prompt:
            prompt = prompt.replace("[SKILL]", skill)
            prompt = prompt.replace("[PRED_DICT]", str(pred_dict))
            prompt = prompt.replace("[TRIED_PRED]", str(tried_pred))
        # convert the placeholders back to 'obj' and 'loc'
        while "[OBJ]" in prompt or "[LOC]" in prompt:
            prompt = prompt.replace("[OBJ]", "obj")
            prompt = prompt.replace("[LOC]", "loc")
        return prompt
    prompt_fpath += f"_{pred_type}.txt"
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred_dict)
    breakpoint()
    # response consists of predicate and its semantic
    resp = model.generate(prompt)[0]
    pred, sem = resp.split(': ', 1)[0].strip('`'), resp.split(': ', 1)[1].strip()
    pred = pred.replace('(obj', '([OBJ]').replace('obj)', '[OBJ])').replace('loc', '[LOC]')
    return pred, sem

def mismatch_symbolic_state(pred_dict, skill2tasks, pred_type):
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
    
    for pred in pred_dict.keys():
        for skill, tasks in skill2tasks.items():
            for id, task in tasks.items():
                if id not in pred_dict[pred]['task']:
                    pred_dict[pred]['task'][id] = [
                        eval_pred(model, skill, pred, pred_dict[pred]['semantic'], task['obj'], task['loc'], task['s0']), \
                        eval_pred(model, skill, pred, pred_dict[pred]['semantic'], task['obj'], task['loc'], task['s1'])
                    ]    
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
        # success_task = mismatch_tasks[skill][0] if skill2tasks[skill][mismatch_tasks[skill][0]]['success'] == True else mismatch_tasks[skill][1]
        # state_change_success = int(new_p_mismatch[success_task][1]==True) - int(new_p_mismatch[success_task][0]==True)
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
    return mismatch_pairs
    # except:
    #     return {}
    # if dup_tasks:
    #     t1 = random.choice(list(dup_tasks.keys()))
    #     t2 = random.choice(dup_tasks[t1])
    #     return [task[t] for t in [t1, t2]]
    # else:
    #     return []
        
# TODO: code for seperate tasks in terms of skill names:: dict(skill: list(dict('id':num, 's0':img, 's1':img, 'success': Bool)))
def refine_pred(model, skill, skill2tasks, pred_dict, precond, eff, max_t=3):
    '''
    Refine predicates of precondition and effect of one skill for precondition and effect
    pred_dict::{pred_name:{task{id: [Bool, Bool]}, sem:str}}
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    '''

    # keep tracking truth value of predicates before and after a skill execution
    # {pred_name:{task: [Bool, Bool]}} Does it need skill name as keys?
    if not pred_dict:
        pred_dict = {}
    if not precond:
         precond = {}
    if not eff:
         eff = {}
    
    # check precondition first
    t = 0
    mismatch_tasks = mismatch_symbolic_state(pred_dict, skill2tasks, 'precond')
    new_p_added = False
    breakpoint()
    tried_pred_precond = []
    while mismatch_tasks and t < max_t:
        new_p, sem = generate_pred(model, skill, list(pred_dict.keys()),  'precond', tried_pred=tried_pred_precond)
        print('new predicate', new_p, sem)
        new_p_mismatch = {idx: [eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['s0']), eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['s1'])] for idx in mismatch_tasks[skill]}
        print('new predicate truth value', new_p_mismatch)
        # breakpoint()
        if new_p_mismatch[mismatch_tasks[skill][0]][0] != new_p_mismatch[mismatch_tasks[skill][1]][0]:
            precond[new_p] = True if skill2tasks[skill][mismatch_tasks[skill][0]]['success'] == new_p_mismatch[mismatch_tasks[skill][0]][0] else False
            new_p_added = True
            break
        else:
            tried_pred_precond.append(new_p)
        t += 1
    if new_p_added and new_p not in pred_dict:
        pred_dict[new_p] = {}
        pred_dict[new_p]['task'] = {idx: [eval_pred(model, skill, new_p, sem, task['obj'], task['loc'], task['s0']), eval_pred(model, skill, new_p, sem, task['obj'], task['loc'], task['s1'])] if idx not in new_p_mismatch else new_p_mismatch[idx] for idx, task in skill2tasks[skill].items()}
        pred_dict[new_p]['semantic'] = sem
        new_p_added = False

    # check effect
    t = 0
    mismatch_tasks = mismatch_symbolic_state(pred_dict, skill2tasks, 'eff')
    tried_pred_eff = []
    breakpoint()
    while mismatch_tasks and t < max_t:
        new_p, sem = generate_pred(model, skill, list(pred_dict.keys()), 'eff', tried_pred=tried_pred_eff)
        print('new predicate', new_p, sem)
        new_p_mismatch = {idx: [eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['s0']), eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['s1'])] for idx in mismatch_tasks[skill]}
        print('new predicate truth value', new_p_mismatch)
        # first index for task number, second index for before and after
        s_1_1 = new_p_mismatch[mismatch_tasks[skill][0]][0]
        s_1_2 = new_p_mismatch[mismatch_tasks[skill][0]][1]
        s_2_1 = new_p_mismatch[mismatch_tasks[skill][1]][0]
        s_2_2 = new_p_mismatch[mismatch_tasks[skill][1]][1]
        success_task = mismatch_tasks[skill][0] if skill2tasks[skill][mismatch_tasks[skill][0]]['success'] == True else mismatch_tasks[skill][1]
        state_change_success = int(new_p_mismatch[success_task][1]==True) - int(new_p_mismatch[success_task][0]==True)
        # eff representation might be wrong (s1-s2). The result value could be {-1, 0, 1}, 0 cases could be wrong?
        if (int(s_1_2==True) - int(s_1_1==True) != int(s_2_2==True) - int(s_2_1==True)) and state_change_success != 0:
            eff[new_p] = state_change_success
            new_p_added = True
            break
        else:
            tried_pred_eff.append(new_p)
        t += 1
    if new_p_added and new_p not in pred_dict:
        pred_dict[new_p] = {}
        pred_dict[new_p]['task'] = {idx: [eval_pred(model, skill, new_p, sem, task['obj'], task['loc'], task['s0']), eval_pred(model, skill, new_p, sem, task['obj'], task['loc'], task['s1'])] if idx not in new_p_mismatch else new_p_mismatch[idx] for idx, task in skill2tasks[skill].items()}
        pred_dict[new_p]['semantic'] = sem
        new_p_added = False
            
    return precond, eff, pred_dict, tried_pred_precond, tried_pred_precond

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
    model = GPT4(engine='gpt-4o-2024-08-06')
    # test predicate evaluation function
    # # pred = 'handEmpty()'
    # pred = 'IsObjectReachable([OBJ], [LOC])'
    # # pred = 'is_held([OBJ])'
    # skill = 'PickUp([OBJ], [LOC])'
    # sem = "return true if the object is within the robot's reach at the given location, and false if it is not."
    # # sem = "Indicates whether the object is currently being held by the robot after attempting to pick it up."
    # obj = 'KeyChain'
    # loc = 'DiningTable'
    # img = ['Before_PickUp_2.jpg']
    # # img = ['test1.jpg']
    # response = eval_pred(model, skill, pred, sem, obj, loc, img)
    # print(response)
    # breakpoint()

    # test predicate proposing for refining

    # mock symbolic state
    # pred_dict = {'handEmpty()': True}
    pred_dict = {'is_held([OBJ])': True}
    # pred_dict = {}
    skill = 'PickUp([OBJ], [LOC])'
    pred_type = 'eff'
    # pred_type = 'precond'
    response = generate_pred(model, skill, pred_dict, pred_type)
    print(response)
    breakpoint()

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
    # mismatch_states = mismatch_symbolic_state(mock_pred_dict, mock_skill2tasks, pred_type)
    # print(mismatch_states) # {'PickUp': {'PickUp_0': ['PickUp_2']}}

    # pred_type = 'eff'
    # mismatch_states = mismatch_symbolic_state(mock_pred_dict, mock_skill2tasks, pred_type)
    # print(mismatch_states) # {'PickUp': ['PickUp_0', 'PickUp_3']}

    # # test main refining function
    # skill = 'PickUp'
    # precond = {}
    # eff = {}
    # precond, eff, pred_dict = refine_pred(model, skill, mock_skill2tasks, mock_pred_dict, precond, eff)
    # print(precond, eff, pred_dict)

    # test with real images and tasks
    pred_dict = {}

    skill2tasks = {
        "PickUp([OBJ], [LOC])": {
            "PickUp_0": {"s0": ["Before_PickUp_2.jpg"], "s1":["After_PickUp_2.jpg"], "obj":"Book", "loc":"DiningTable", "success": True},
            "PickUp_1": {"s0": ["Before_PickUp_1.jpg"], "s1":["After_PickUp_1.jpg"], "obj":"KeyChain", "loc":"DiningTable", "success": False}
        }
    }

    pred_type = 'precond'
    mismatch_states = mismatch_symbolic_state(pred_dict, skill2tasks, pred_type)
    print(mismatch_states) # {'PickUp': {'PickUp_0': ['PickUp_2']}}

    pred_type = 'eff'
    mismatch_states = mismatch_symbolic_state(pred_dict, skill2tasks, pred_type)
    print(mismatch_states) # {'PickUp': ['PickUp_0', 'PickUp_3']}

    skill = 'PickUp([OBJ], [LOC])'
    precond = {}
    eff = {}
    precond, eff, pred_dict, tried_pred_precond, tried_pred_eff = refine_pred(model, skill, skill2tasks, pred_dict, precond, eff)
    print(precond, eff, pred_dict)