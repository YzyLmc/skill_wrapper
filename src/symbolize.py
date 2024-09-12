'Get symbolic representation from skill semantic info and observation'
from utils import GPT4, load_from_file
from collections import defaultdict
import inspect
import random
import itertools

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

def eval_pred(model, skill, pred, sem, obj, loc, loc_1, loc_2, img, prompt_fpath='prompts/evaluate_pred_ai2thor.txt', obj_ls = ['Book', 'Vase', 'TissueBox', 'Bowl'], loc_ls = ['DiningTable', 'Sofa']):
    '''
    Evaluate one predicate given one image.
    If skill or predicate takes variables they should contain [OBJ] [LOC] [LOC_1]or [LOC_2] in the input.
    For all mismatch arguments, output false.

    Obsoleted: --- GoTo(loc_1, loc_2) will be evaluated differently since the arguments are different. 
    Empty string '' means no argument
    '''
    # def construct_prompt(prompt, skill, pred, obj, loc, loc_1, loc_2):
    #     if 'GoTo' not in skill: # pickup or dropat
    #         if '[LOC_1]' in pred or '[LOC_2]' in pred: # convert goto pred to pickup and dropat
    #             if not ('[LOC_1]' in pred and '[LOC_2]' in pred):
    #                 # e.g., At([LOC_1]),  At([LOC_2])
    #                 pred = pred.replace('[LOC_1]', loc)
    #                 pred = pred.replace('[LOC_2]', loc)
    #             else: # both [LOC_1] and [LOC_2] are in the prompt
    #                 # e.g. clearPath([LOC_1], [LOC_2]), N/A
    #                 # always return one value so if 50% success and 50% fail this pred will not be added to precond or eff
    #                 return None
                
    #         else: # pickup and dropat pred, or not argument
    #             while "[OBJ]" in pred or "[LOC]" in pred:
    #                 pred = pred.replace("[OBJ]", obj)
    #                 pred = pred.replace("[LOC]", loc)

    #         while "[LOC]" in skill or "[OBJ]" in skill:
    #             skill = skill.replace("[LOC]", loc)
    #             skill = skill.replace("[OBJ]", obj)
    #     else:
    #         assert 'GoTo' in skill
    #         if ('[LOC_1]' in pred or '[LOC_2]' in pred): # GoTo pred
    #             # e.g., At([LOC_1]),  At([LOC_2])
    #             # e.g. clearPath([LOC_1], [LOC_2])
    #             pred = pred.replace('[LOC_1]', loc_1)
    #             pred = pred.replace('[LOC_2]', loc_2)
    #         else: # pickup and drop at on goto
    #             # first determine the type of the arguments are loc or obj
    #             # goto might have 2 obj, 1 obj 1 loc, or 2 loc
    #             # pred might have 0 arg, 1 obj, 1 loc, 1 obj and 1 loc
    #             goto_obj_num = len([obj for obj in obj_ls if obj in [loc_1, loc_2]])
    #             goto_loc_num = len([obj for obj in loc_ls if obj in [loc_1, loc_2]])
    #             pred_obj_num = 1 if '[OBJ]' in pred else 0
    #             pred_loc_num = 1 if '[LOC]' in pred else 0
    #             if pred_obj_num + pred_loc_num == 0:
    #                 pass
    #             elif pred_obj_num == 1 and pred_loc_num == 0:
    #                 if (goto_obj_num == 2 and goto_loc_num == 0):
    #                     if loc_2 in obj_ls:
    #                         while "[OBJ]" in pred:
    #                             pred = pred.replace("[OBJ]", loc_2)
    #                     else:
    #                         return None
    #                 elif goto_obj_num == 0 and goto_loc_num== 2:
    #                     return None
    #                 elif goto_obj_num == 1 and goto_loc_num == 1:
    #                     obj = loc_1 if loc_1 in obj_ls else loc_2
    #                     while "[OBJ]" in pred:
    #                         pred = pred.replace("[OBJ]", obj)
    #             elif pred_obj_num == 0 and pred_loc_num == 1:
    #                 if goto_obj_num == 2 and goto_loc_num == 0:
    #                     return None
    #                 elif goto_obj_num == 0 and goto_loc_num== 2:
    #                     if loc_2 in loc_ls:
    #                         while "[LOC]" in pred:
    #                             pred = pred.replace("[LOC]", loc_2)
    #                     else:
    #                         return None
    #                 elif goto_obj_num == 1 and goto_loc_num == 1:
    #                     loc = loc_1 if loc_1 in loc_ls else loc_2
    #                     while "[LOC]" in pred:
    #                         pred = pred.replace("[LOC]", loc)
    #             elif pred_obj_num == 1 and pred_loc_num == 1:
    #                 if (goto_obj_num == 2 and goto_loc_num == 0) or (goto_obj_num == 0 and goto_loc_num== 0):
    #                     return None
    #                 elif goto_obj_num == 1 and goto_loc_num == 1:
    #                     obj = loc_1 if loc_1 in obj_ls else loc_2
    #                     loc = loc_1 if loc_1 in loc_ls else loc_2
    #                     while "[LOC]" in pred or '[OBJ]' in pred:
    #                         # breakpoint()
    #                         pred = pred.replace("[OBJ]", obj)
    #                         pred = pred.replace("[LOC]", loc)
    #         while "[LOC_1]" in skill or "[LOC_2]" in skill:
    #             skill = skill.replace("[LOC_1]", loc_1)
    #             skill = skill.replace("[LOC_2]", loc_2)
    #     while "[PRED]" in prompt or "[SKILL]" in prompt or "[SEMANTIC]" in prompt:
    #         prompt = prompt.replace("[PRED]", pred)
    #         prompt = prompt.replace("[SKILL]", skill)
    #         prompt = prompt.replace("[SEMANTIC]", sem)
    #     return prompt

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
    print(f'Evaluating predicate {pred} on skill {skill} with arguments {obj} {loc} {loc_1} {loc_2}')
    if prompt:
        print('Calling GPT4')
        resp = model.generate_multimodal(prompt, img)[0]
        return True if "True" in resp.split('\n')[-1] else False
    else:
        print(f"mismatch skill and predicate: return False\n{skill} / {pred}")
        return False

def eval_pred_set(model, skill, pred2sem, obj, loc,loc_1, loc_2, img):
    '''
    Evaluate set of predicates
    pred_set::list(str)
    returns::Dict(str:bool)
    '''
    return {pred: eval_pred(model, skill, pred, sem, obj, loc, loc_1, loc_2, img) for pred, sem in pred2sem.items()}

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
        while "[OBJ]" in prompt or "[LOC]" in prompt or "[LOC_1]" in prompt or "[LOC_2]" in prompt:
            prompt = prompt.replace("[OBJ]", "obj")
            prompt = prompt.replace("[LOC]", "loc")
            prompt = prompt.replace("[LOC_1]", "init")
            prompt = prompt.replace("[LOC_2]", "goal")
        return prompt
    prompt_fpath += f"_{pred_type}.txt"
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred_dict)
    # breakpoint()
    # response consists of predicate and its semantic
    print('Generating predicate')
    resp = model.generate(prompt)[0]
    pred, sem = resp.split(': ', 1)[0].strip('`'), resp.split(': ', 1)[1].strip()
    pred = pred.replace('(obj', '([OBJ]').replace('obj)', '[OBJ])').replace('(init', '([LOC_1]').replace('goal)', '[LOC_2])').replace('(loc', '([LOC]').replace('loc)', '[LOC])')
    return pred, sem

# def evaluate_all_tasks(pred_dict, skill2tasks):
#     'Evaluate all tasks that hasn't been evaluated '

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
    print('evaluating from mismatch_symbolic_state')
    for pred in pred_dict.keys():
        for skill, tasks in skill2tasks.items():
            for id, task in tasks.items():
                if id not in pred_dict[pred]['task']:
                    pred_dict[pred]['task'][id] = [
                        eval_pred(model, skill, pred, pred_dict[pred]['semantic'], task['obj'], task['loc'], task['loc_1'], task['loc_2'], task['s0']), \
                        eval_pred(model, skill, pred, pred_dict[pred]['semantic'], task['obj'], task['loc'], task['loc_1'], task['loc_2'], task['s1'])
                    ]    
    print('Done evaluation from mismatch_symbolic_state')
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
    print("About to enter precondition refinement")
    while skill in mismatch_tasks and t < max_t:
        new_p, sem = generate_pred(model, skill, list(skill2operators[skill]['precond'].keys()),  'precond', tried_pred=skill2triedpred[skill]['precond'])
        print('new predicate', new_p, sem)
        new_p_mismatch = {idx: [eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s0']), eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s1'])] for idx in mismatch_tasks[skill]}
        print('new predicate truth value', new_p_mismatch)
        if new_p_mismatch[mismatch_tasks[skill][0]][0] != new_p_mismatch[mismatch_tasks[skill][1]][0]:
            print('Entering #1 if')
            skill2operators[skill]['precond'][new_p] = True if skill2tasks[skill][mismatch_tasks[skill][0]]['success'] == new_p_mismatch[mismatch_tasks[skill][0]][0] else False
            print(f"Predicate {new_p} added to precondition with truth value {skill2operators[skill]['precond'][new_p]}")
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
                    breakpoint()
                elif ('[LOC_1]' in new_p) != ('[LOC_2]' in new_p): # only one loc in new predicate
                    new_p_not_goto = new_p.replace('[LOC_1]', '[LOC]').replace('[LOC_2]', '[LOC]')
                    if new_p_not_goto not in pred_dict:
                        pred_dict[new_p_not_goto] = {'task': {}}
                        pred_dict[new_p_not_goto]['semantic'] = sem
                    breakpoint()
                for s in skill2tasks:  
                    for idx, task in skill2tasks[s].items():
                        if idx not in new_p_mismatch:
                                print('Evaluating for precond')
                                pred_dict[new_p]['task'][idx] = [eval_pred(model, s, new_p, sem, task['obj'], task['loc'], skill2tasks[s][idx]['loc_1'], skill2tasks[s][idx]['loc_2'], task['s0']), eval_pred(model, s, new_p, sem, task['obj'], task['loc'], skill2tasks[s][idx]['loc_1'], skill2tasks[s][idx]['loc_2'], task['s1'])]
                for new_p_id in new_p_mismatch:
                    pred_dict[new_p]['task'][new_p_id] = new_p_mismatch[new_p_id]
                pred_dict[new_p]['semantic'] = sem

            print(f'Done evaluating predicate {new_p} for all tasks')
            new_p_added = False
            pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'precond')
            print(f'Done evaluating predicate {new_p} for all tasks')
            skill2triedpred[skill]['precond'] = []
        else:
            skill2triedpred[skill]['precond'].append(new_p)
        t += 1
    

    # breakpoint()
    # check effect
    t = 0
    pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'eff')
    print("About to enter effect refinement")
    while skill in mismatch_tasks and t < max_t:
        new_p, sem = generate_pred(model, skill, list(skill2operators[skill]['eff'].keys()), 'eff', tried_pred=skill2triedpred[skill]['eff'])
        print('new predicate', new_p, sem)
        new_p_mismatch = {idx: [eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s0']), eval_pred(model, skill, new_p, sem, skill2tasks[skill][idx]['obj'], skill2tasks[skill][idx]['loc'], skill2tasks[skill][idx]['loc_1'], skill2tasks[skill][idx]['loc_2'], skill2tasks[skill][idx]['s1'])] for idx in mismatch_tasks[skill]}
        print('new predicate truth value', new_p_mismatch)
        # first index for task number, second index for before and after
        s_1_1 = new_p_mismatch[mismatch_tasks[skill][0]][0]
        s_1_2 = new_p_mismatch[mismatch_tasks[skill][0]][1]
        s_2_1 = new_p_mismatch[mismatch_tasks[skill][1]][0]
        s_2_2 = new_p_mismatch[mismatch_tasks[skill][1]][1]
        success_task = mismatch_tasks[skill][0] if skill2tasks[skill][mismatch_tasks[skill][0]]['success'] == True else mismatch_tasks[skill][1]
        state_change_success = int(new_p_mismatch[success_task][1]==True) - int(new_p_mismatch[success_task][0]==True)
        print('Before Entering #2 if for eff')
        # eff representation might be wrong (s1-s2). The result value could be {-1, 0, 1}, 0 cases could be wrong?
        if (int(s_1_2==True) - int(s_1_1==True) != int(s_2_2==True) - int(s_2_1==True)) and state_change_success != 0:
            
            skill2operators[skill]['eff'][new_p] = state_change_success
            print(f"Predicate {new_p} added to effect with truth value {skill2operators[skill]['eff'][new_p]}")
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
                    breakpoint()
                elif ('[LOC_1]' in new_p) != ('[LOC_2]' in new_p): # only one loc in new predicate
                    new_p_not_goto = new_p.replace('[LOC_1]', '[LOC]').replace('[LOC_2]', '[LOC]')
                    if new_p_not_goto not in pred_dict:
                        pred_dict[new_p_not_goto] = {'task': {}}
                        pred_dict[new_p_not_goto]['semantic'] = sem
                    breakpoint()
                pred_dict[new_p] = {'task': {}}
                for s in skill2tasks:
                    for idx, task in skill2tasks[s].items():
                        if idx not in new_p_mismatch:
                            pred_dict[new_p]['task'][idx] = [eval_pred(model, s, new_p, sem, task['obj'], task['loc'], skill2tasks[s][idx]['loc_1'], skill2tasks[s][idx]['loc_2'], task['s0']), eval_pred(model, s, new_p, sem, task['obj'], task['loc'], skill2tasks[s][idx]['loc_1'], skill2tasks[s][idx]['loc_2'], task['s1'])]
                for new_p_id in new_p_mismatch:
                    pred_dict[new_p]['task'][new_p_id] = new_p_mismatch[new_p_id]
                pred_dict[new_p]['semantic'] = sem
            print(f'Done evaluating predicate {new_p} for all tasks')
            new_p_added = False
            pred_dict, mismatch_tasks = mismatch_symbolic_state(model, pred_dict, skill2tasks, 'eff')
            skill2triedpred[skill]['eff'] = []
        else:
            skill2triedpred[skill]['eff'].append(new_p)
        t += 1
    
            
    return skill2operators, pred_dict, skill2triedpred

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

def cross_assignment(skill2operator, skill2tasks, pred_dict, equal_preds=None, threshold=0.4):
    '''
    Assign precondtions of all skills to effect of each skill
    There should be a threshold for cross assignment, either if higher than it or lower than -1*threshold will be added
    skill2operator:: {skill_name:{precond:{str:Bool}, eff:{str:int}}}
    skill2tasks:: dict(skill:dict(id: dict('s0':img_path, 's1':img_path, 'obj':str, 'loc':str, 'success': Bool)))
    pred_dict:: {pred_name:{task: [Bool, Bool]}, semantic:str}
    '''
    def f1_score(pred, skill, skill2tasks, pred_dict):
        "f1 score of a predicate as one skill's precondition"
        tasks = skill2tasks[skill]
        success_tasks = [id for id, t in tasks.items() if t['success']]
        fail_tasks = [t for t in tasks if not t['success']]
        for t_suc in success_tasks:
            tp = [p for p in pred_dict if pred_dict[p]['task'][t][0] == True]

    all_precond = list(itertools.chain([list(skill2operator[skill]['precond'].keys()) for skill in skill2operator])) + list(itertools.chain([list(skill2operator[skill]['eff'].keys()) for skill in skill2operator]))
    all_precond = set(list(itertools.chain(*all_precond)))

    for skill in skill2operator:
        tasks = skill2tasks[skill]
        # skill2operator[skill]['eff'] = {}
        # print(skill)
        for pred in all_precond:
            # try:
                # accuracy here means the portion of state change from true to false
                state_pair_all = [pred_dict[pred]['task'][id] for id, t in tasks.items() if t['success']]
                if not state_pair_all: # if no success case for the skill
                    continue
                acc_ls = [int(state_pair[1]==True) - int(state_pair[0]==True) for state_pair in state_pair_all]
                total_num = len(acc_ls)
                for ps in equal_preds:
                    if pred in ps:
                        for p in ps:
                            state_pair_all = [pred_dict[p]['task'][id] for id, t in tasks.items() if t['success']]
                            acc_ls += state_pair_all
                            total_num += len(state_pair_all)
                acc = sum(acc_ls)/total_num
                print(skill, pred, acc, state_pair_all)
                # breakpoint()
                if acc > threshold:
                    skill2operator[skill]['eff'][pred] = 1
                elif acc < - threshold:
                    skill2operator[skill]['eff'][pred] = -1
            # except:
            #     breakpoint()
    return skill2operator


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

    skill2operators = {'PickUp([OBJ], [LOC])': {'precond': {}, 'eff': {}}, 'DropAt([OBJ], [LOC])': {'precond': {'isHolding([OBJ])': True}, 'eff': {}}, 'GoTo([LOC_1], [LOC_2])': {'precond': {}, 'eff': {}}}
    skill2tasks = {'DropAt([OBJ], [LOC])': {'DropAt_Vase_DiningTable_False_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Vase_DiningTable_False_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Vase_DiningTable_False_1.jpg'], 'success': False, 'obj': 'Vase', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}, 'DropAt_Bowl_CoffeeTable_True_1': {'s0': ['tasks/exps/DropAt/Before_DropAt_Bowl_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/DropAt/After_DropAt_Bowl_CoffeeTable_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'CoffeeTable', 'loc_1': '', 'loc_2': ''}}, 'GoTo([LOC_1], [LOC_2])': {'GoTo_DiningTable_CoffeeTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_DiningTable_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_DiningTable_CoffeeTable_True_1.jpg'], 'success': True, 'loc_1': 'DiningTable', 'loc_2': 'CoffeeTable', 'obj': '', 'loc': ''}, 'GoTo_Sofa_DiningTable_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_Sofa_DiningTable_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_Sofa_DiningTable_True_1.jpg'], 'success': True, 'loc_1': 'Sofa', 'loc_2': 'DiningTable', 'obj': '', 'loc': ''}, 'GoTo_CoffeeTable_Sofa_True_1': {'s0': ['tasks/exps/GoTo/Before_GoTo_CoffeeTable_Sofa_True_1.jpg'], 's1': ['tasks/exps/GoTo/After_GoTo_CoffeeTable_Sofa_True_1.jpg'], 'success': True, 'loc_1': 'CoffeeTable', 'loc_2': 'Sofa', 'obj': '', 'loc': ''}}, 'PickUp([OBJ], [LOC])': {'PickUp_TissueBox_Sofa_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_TissueBox_Sofa_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_TissueBox_Sofa_True_1.jpg'], 'success': True, 'obj': 'TissueBox', 'loc': 'Sofa', 'loc_1': '', 'loc_2': ''}, 'PickUp_Vase_CoffeeTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Vase_CoffeeTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Vase_CoffeeTable_True_1.jpg'], 'success': True, 'obj': 'Vase', 'loc': 'CoffeeTable', 'loc_1': '', 'loc_2': ''}, 'PickUp_Bowl_DiningTable_True_1': {'s0': ['tasks/exps/PickUp/Before_PickUp_Bowl_DiningTable_True_1.jpg'], 's1': ['tasks/exps/PickUp/After_PickUp_Bowl_DiningTable_True_1.jpg'], 'success': True, 'obj': 'Bowl', 'loc': 'DiningTable', 'loc_1': '', 'loc_2': ''}}}
    pred_dict = {'isHolding([OBJ])': {'task': {'GoTo_DiningTable_CoffeeTable_True_1': [False, False], 'GoTo_Sofa_DiningTable_True_1': [False, False], 'GoTo_CoffeeTable_Sofa_True_1': [False, False], 'PickUp_TissueBox_Sofa_True_1': [False, False], 'PickUp_Vase_CoffeeTable_True_1': [True, False], 'PickUp_Bowl_DiningTable_True_1': [False, True], 'DropAt_Bowl_CoffeeTable_True_1': [True, False], 'DropAt_Vase_DiningTable_False_1': [False, False]}, 'semantic': 'The robot is currently holding the object.'}}
    merged_skill2operators, equal_preds = merge_predicates(model, skill2operators, pred_dict)
    assigned_skill2operators = cross_assignment(merged_skill2operators, skill2tasks, pred_dict, equal_preds=equal_preds)
    print(assigned_skill2operators)
    breakpoint()