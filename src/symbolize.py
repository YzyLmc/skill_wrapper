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
    

def task_proposal(model, prompt_fpath='prompts/task_proposing.txt'):
    '''
    Proposing task for each predicate of a skill.
    It should be able to prpoposed based on existing roll-outs.
    '''
    prompt = load_from_file(prompt_fpath)
    return model.generate(prompt)

def predicates2precond():
    '''
    Compose predicates to precondition
    '''
    pass

def predicates2effect():
    '''
    Compose predicates to effect
    '''
    pass

def symbolize():
    '''
    put together all components
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
    task = task_proposal(model)
    print(task)
