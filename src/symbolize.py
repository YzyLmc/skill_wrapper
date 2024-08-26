'Get symbolic representation from skill semantic info and observation'
from utils import GPT4, load_from_file
import inspect
from manipula_skills import *

def eval_execution(model, skill, consecutive_pair, prompt_fpath='prompts/evalutate_task.txt'):
    'Get successfulness of the execution given images and the skill name'
    def construct_prompt(prompt, skill):
        while "[SKILL]" in prompt:
                prompt = prompt.replace("[SKILL]", skill)
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill)
    return model.generate_multimodal(prompt, consecutive_pair)[0]

def eval_pred(model, pred, img, prompt_fpath='prompts/.txt'):
    'Evaluate one predicate given one image'
    def construct_prompt(prompt, pred):
        while "[PRED]" in prompt:
                prompt = prompt.replace("[PRED]", pred)
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, pred)
    return model.generate_multimodal(prompt, img)[0]

def eval_pred_set(model, pred_set, img):
    '''
    Evaluate set of predicates
    pred_set::list(str)
    returns::Dict(str:bool)
    '''
    return {pred: eval_pred(model, pred, img) for pred in pred_set}

# TODO: have to track predicates have been tried in the prompt
def generate_pred(model, skill, pred_list, prompt_fpath='prompts/.txt'):
    'generate_predicates based on existing predicates set'
    def construct_prompt(prompt, skill, pred_list):
        while "[SKILL]" in prompt or "[PREDICATE_LIST]" in prompt:
                prompt = prompt.replace("[SKILL]", skill)
                prompt = prompt.replace("[PREDICATE_LIST]", pred_list)
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill, pred_list)
    return model.generate(prompt)[0]

def refine_pred(model, skill, tasks, precond, eff, max_t=3):
    '''
    Refine predicates set for precondition and effect
    tasks::dict(id: dict('s0':img, 's1':img, 'success': Bool))
    '''
    def mismatch_symbolic_state(pred_dict, tasks, type):
        '''
        look for same symbolic states (same s1 or s2) in task dictionary
        type: {'precond', 'eff'}
        returns::list(dict('id':num, 's0':img, 's1':img, 'success': Bool))
        '''
        if type == "precond":
            # return two same states with different successfulness
        elif type == "eff":
            # return two same states transition (symbolic) with different successfulness
        else:
            raise NotImplementedError

    # keep tracking truth value of predicates before and after a skill execution
    # {pred_name:{task: [Bool, Bool]}
    if not pred_dict:
        pred_dict = {}
    if not precond:
         precond = {}
    if not eff:
         eff = {}
    
    mismatch_tasks = mismatch_symbolic_state(pred_dict, tasks, 'pred')

    # check precondition first
    t = 0
    while mismatch_tasks and t < max_t:
        new_p = generate_pred(model, skill, list(pred_dict.keys()))
        new_p_mismatch = {task['id']: [eval_pred(model, new_p, task['s0']), eval_pred(model, new_p, task['s0'])] for task in mismatch_tasks}
        # new_p_value = {task['id']: [eval_pred(model, new_p, task['s0']), eval_pred(model, new_p, task['s0'])] for task in tasks}
        if new_p_mismatch[mismatch_tasks[0]][0] != new_p_mismatch[mismatch_tasks[1]][0]:
            precond[new_p] = True if mismatch_tasks[0]['success'] == new_p_mismatch[mismatch_tasks[0]][0] else False
        t += 1

    pred_dict[new_p] = {task['id']: [eval_pred(model, new_p, task['s0']), eval_pred(model, new_p, task['s0'])] for task in tasks}

    # check effect
    t = 0
    mismatch_tasks = mismatch_symbolic_state(pred_dict, tasks, 'eff')
    while mismatch_tasks and t < max_t:
        new_p = generate_pred(model, skill, list(pred_dict.keys()))
        new_p_mismatch = {task['id']: [eval_pred(model, new_p, task['s0']), eval_pred(model, new_p, task['s0'])] for task in mismatch_tasks}
        # first index for task number, second index for before and after
        s_1_1 = new_p_mismatch[mismatch_tasks[0]][0]
        s_1_2 = new_p_mismatch[mismatch_tasks[0]][1]
        s_2_1 = new_p_mismatch[mismatch_tasks[1]][0]
        s_2_2 = new_p_mismatch[mismatch_tasks[1]][1]
        if int(s_1_2==True) - int(s_1_2==True) != int(s_2_2==True) - int(s_2_2==True):
            success_task = mismatch_tasks[0] if mismatch_tasks[0]['success'] == True else mismatch_tasks[1]
            eff[new_p] = new_p_mismatch[success_task][1] - new_p_mismatch[success_task][0]
        t += 1
    
    pred_dict[new_p] = {task['id']: [eval_pred(model, new_p, task['s0']), eval_pred(model, new_p, task['s0'])] for task in tasks}
            
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