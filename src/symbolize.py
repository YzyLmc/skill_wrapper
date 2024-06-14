'Get symbolic representation from skill semantic info and observation'
from utils import GPT4, load_from_file
from manipula_skills import *

def predicates_per_skill(model, skill, obs=None, prompt_fpath = "prompts/predicates_proposing.txt"):
    '''
    Propose set of predicates for one skill and observations (optionally)
    '''
    def construct_prompt(prompt, skill):
        'replace placeholders with skill name'
        skill_name = skill.__name__
        while "[SKILL]" in prompt:
            prompt = prompt.replace("[SKILL]", skill_name)
        breakpoint()
        return prompt

    if obs:
        # not used for now
        # propose predicates with visual observation
        return model.generate_multimodal(prompt,obs)
    else:
        prompt = load_from_file(prompt_fpath)
        prompt = construct_prompt(prompt, skill)
        return model.generate(prompt)

def unifiy_predicates(model, skill2pred, prompt_fpath='prompts/predicates_unify.txt'):
    '''
    Make sure predicates for different skills have consistent names
    skill2predicts :: dict(String: list(String))
    '''
    def construct_prompt(prompt, skill2pred):
        for skill_name, pred_names in skill2pred:
            prompt += skill_name + ": " + ", ".join(pred_names)
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill2pred)
    return model.generate(prompt)
    

def task_proposal(model, predicate, skill, action_list=[MoveForward, MoveBackward, MoveLeft, MoveRight, TurnLeft, TurnRight, GoTo, GripperUp, GripperDown, GripperLeft, GripperRight, GripperForward, GripperBackward, PickUp, Drop], prompt_fpath='prompts/task_proposing.txt'):
    '''
    Proposing task for each predicate of a skill.
    It should be able to prpoposed based on existing roll-outs.
    predicate: str
    action_list: list(func)
    '''
    def construct_prompt(prompt, predicate_name, action_list):
        actiom_list_strs = [a.__name__ for a in action_list]
        action_list_strs_joined = ", ".join(actiom_list_strs)
        while "[PRED]" in prompt or "[ACTION_LIST]" in prompt or '[SKILL]' in prompt:
            prompt = prompt.replace("[SKILL]", skill.__name__)
            prompt = prompt.replace("[PRED]", predicate_name)
            prompt = prompt.replace("[ACTION_LIST]", action_list_strs_joined)
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, predicate, action_list)
    return model.generate(prompt)

def pred2precond(model, skill2pred, contrastive_pair, prompt_fpath='prompts/pred2precond.txt'):
    '''
    Compose candidate predicates to precondition
    skills2pred: {str:[str]}
    contrastive_pair: {str:img} : one "success", one "fail"
    '''
    def construct_prompt(prompt, skill2pred):
        for skill, pred in skill2pred.items():
            while "[SKILL]" or "[PREDICATE_LIST]" in prompt:
                prompt = prompt.replace("[SKILL]", skill)
                prompt = prompt.replace("[PREDICATE_LIST]", pred)
    prompt = load_from_file(prompt_fpath)



    

def pred2effect(model, skill2pred, consecutive_pair, prompt_fpath=''):
    '''
    Compose predicates to effect
    skills2pred: {str:[str]}
    contrastive_pair: {str:img}: one "before", one "after"
    '''
    def construct_prompt(prompt, skill2pred):
        for skill, pred in skill2pred.items():
            while "[SKILL]" or "[PREDICATE_LIST]" in prompt:
                prompt = prompt.replace("[SKILL]", skill)
                prompt = prompt.replace("[PREDICATE_LIST]", pred)
    prompt = load_from_file(prompt_fpath)

def symbolize():
    '''
    Not implemented because execution of propsoed tasks are handled manually now
    '''
    pass

if __name__ == "__main__":
    imgs = ["test_imgs/test_0.png"]
    prompt_path = 'prompts/predicates_proposing.txt'
    prompt = load_from_file(prompt_path)
    model = GPT4()
    # predicates = predicates_per_skill(model, PickUp)
    # predicates = predicates_per_skill(model, Drop)
    # print(predicates)
    task = task_proposal(model, "withinDistance()", PickUp)
    print(task)
