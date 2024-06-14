'Get symbolic representation from skill semantic info and observation'
from utils import GPT4, load_from_file
from manipula_skills import *

def predicates_per_skill(model, skill, basic_info=None, obs=None, prompt_fpath = "prompts/predicates_proposing.txt"):
    '''
    Propose set of predicates for one skill and observations (optionally)
    '''
    def construct_prompt(prompt, skill, basic_info):
        'replace placeholders with skill name'
        skill_name = skill.__name__
        while "[SKILL]" in prompt or "[BASIC_INFO]" in prompt:
            prompt = prompt.replace("[SKILL]", skill_name)
            prompt = prompt.replace("[BASIC_INFO]", basic_info)
        return prompt
    basic_info = 'This image appears to depict a residential indoor environment, characterized by typical household furnishings. The robot in the image features an arm-like mechanism, suggesting it is designed for manipulation tasks, such as picking up or moving objects around the house.'
    if obs:
        # not used for now
        # propose predicates with visual observation
        return model.generate_multimodal(prompt,obs)
    else:
        prompt = load_from_file(prompt_fpath)
        prompt = construct_prompt(prompt, skill, basic_info)
        return model.generate(prompt)[0].replace('-','').replace(' ','').split('\n')

def unifiy_predicates(model, skill2pred, prompt_fpath='prompts/predicates_unify.txt'):
    '''
    Make sure predicates for different skills have consistent names
    skill2predicts :: dict(String: list(String))
    '''
    def construct_prompt(prompt, skill2pred):
        for skill_name, pred_names in skill2pred.items():
            prompt += "\nSkill: " + skill_name
            prompt += "\nPredicates: " + " | ".join(pred_names)
            prompt += "\n"
        prompt += "\nEquivalent Predicates:"
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill2pred)
    # breakpoint()
    response = model.generate(prompt)[0].split("Equivalent Predicates:")[1]
    equal_preds = response.split("\n\n")[1:]
    equal_preds = [pred.replace("-","").replace(" ","").split('|') for pred in equal_preds]
    breakpoint()
    return equal_preds
    

def task_proposal(model, predicate, skill, action_list=[MoveForward, MoveBackward, MoveLeft, MoveRight, TurnLeft, TurnRight, GoTo, GripperUp, GripperDown, GripperLeft, GripperRight, GripperForward, GripperBackward, PickUp, DropAt], prompt_fpath='prompts/task_proposing.txt'):
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
    # predicates = predicates_per_skill(model, DropAt)
    # print(predicates)
    # task = task_proposal(model, "withinDistance()", PickUp)
    # print(task)
    skill2pred = {'PickUp': ['IsObjectReachable(object)', 'IsObjectGraspable(object)', 'IsObjectOnSurface(object,surface)', 'IsSurfaceStable(surface)', 'IsObjectClearOfObstructions(object)', 'IsObjectWithinArmRange(object)', 'IsObjectTypeSupported(object)', 'IsObjectWeightSupported(object)', 'IsObjectPositionKnown(object)', 'IsArmInPositionForGrasp(object)'], 'DropAt': ['IsHolding(object)', 'IsAtLocation(robot,location)', 'IsClear(location)', 'IsObjectType(object,type)', 'IsWithinReach(robot,location)', 'IsStableSurface(location)', 'IsAligned(robot,location)', 'IsDropHeightSafe(robot,location)', 'IsObjectIntact(object)', 'IsLocationAccessible(robot,location)']}
    unified_pred = unifiy_predicates(model, skill2pred)
    print(unified_pred)