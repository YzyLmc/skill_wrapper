'Get symbolic representation from skill semantic info and observation'
from utils import GPT4, load_from_file
import inspect
from manipula_skills import *

def predicates_per_skill(model, skill, basic_info=None, obs=None, prompt_fpath = "prompts/predicates_proposing.txt"):
    '''
    Propose set of predicates for one skill and observations (optionally)
    '''
    def construct_prompt(prompt, skill, basic_info):
        'replace placeholders with skill name'
        skill_name = f'{skill.__name__}({", ".join(inspect.getfullargspec(skill)[0][:-2])})'
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
    def convert_skill2preds(skill2preds, equal_preds):
        '''
        skill2preds :dict(str:list(str)):
        equal_preds :list(list(str)): list of equivalent predicates
        '''
        unified_skill2preds = {}
        for skill, preds in skill2preds.items():
            unified_preds = []
            for pred in preds:
                for equal_pred in equal_preds:
                    dup = False
                    if pred in equal_pred:
                        unified_preds.append(equal_pred[0])
                        dup = True
                if not dup:
                    unified_preds.append(pred)
            unified_skill2preds[skill] = unified_preds
        return unified_skill2preds
    
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
    unified_skill2preds = convert_skill2preds(skill2pred, equal_preds)
    breakpoint()
    return unified_skill2preds
    

def task_proposal(model, predicate, skill, basic_info, history=None, highlv_actions=[PickUp, DropAt, GoTo], lowlv_actions=[MoveForward, MoveBackward, MoveLeft, MoveRight, TurnLeft, TurnRight, MoveGripperUp, MoveGripperDown, MoveGripperLeft, MoveGripperRight, MoveGripperForward, MoveGripperBackward], prompt_fpath='prompts/task_proposing_init.txt'):
    '''
    Proposing task for one predicate of a skill.
    It should be able to prpoposed based on existing roll-outs.
    predicate: str
    skill: str
    action_list: list(func)
    '''
    def construct_prompt(prompt, predicate_name, highlv_actions, lowlv_actions, basic_info, history):
        highlv_actions_strs = [f'{a.__name__}({", ".join(inspect.getfullargspec(a)[0][:-2])})' for a in highlv_actions]
        lowlv_actions_strs = [f'{a.__name__}({", ".join(inspect.getfullargspec(a)[0][:-2])})' for a in lowlv_actions]
        # action_list_strs_joined = ", ".join(actiom_list_strs)
        while "[PRED]" in prompt or "[HIGHLV_ACTION_LIST]" in prompt or "[LOWLV_ACTION_LIST]" in prompt or '[SKILL]' in prompt or '[BASIC_INFO]' in prompt:
            prompt = prompt.replace("[SKILL]", f'{skill.__name__}({", ".join(inspect.getfullargspec(skill)[0][:-2])})')
            prompt = prompt.replace("[PRED]", predicate_name)
            prompt = prompt.replace("[HIGHLV_ACTION_LIST]", ", ".join(highlv_actions_strs))
            prompt = prompt.replace("[LOWLV_ACTION_LIST]", ", ".join(lowlv_actions_strs))
            prompt = prompt.replace("[BASIC_INFO]", basic_info)
        if history:
            prompt = prompt.replace("[TASK_HISTORY]", history)
        return prompt
    prompt_fpath = 'prompts/task_proposing_next.txt' if history else prompt_fpath
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, predicate, highlv_actions, lowlv_actions, basic_info, history)
    return model.generate(prompt)[0]

def pred2precond(model, skill2pred, contrastive_pair, prompt_fpath='prompts/pred2precond.txt'):
    '''
    Compose candidate predicates to precondition
    skills2pred: {str:[str]}
    contrastive_pair: {str:img} : first one "success", second one "fail"
    '''
    def construct_prompt(prompt, skill2pred):
        for skill, pred in skill2pred.items():
            while "[SKILL]" in prompt or "[PREDICATE_LIST]" in prompt:
                prompt = prompt.replace("[SKILL]", skill)
                prompt = prompt.replace("[PREDICATE_LIST]", ", ".join(pred))
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill2pred)
    print(prompt)
    return model.generate_multimodal(prompt, contrastive_pair)[0]

def pred2effect(model, skill2pred, consecutive_pair, prompt_fpath='prompts/pred2effect.txt'):
    '''
    Compose predicates to effect
    skills2pred: {str:[str]}
    contrastive_pair: {str:img}: one "before", one "after"
    '''
    def construct_prompt(prompt, skill2pred):
        for skill, pred in skill2pred.items():
            while "[SKILL]" in prompt or "[PREDICATE_LIST]" in prompt:
                prompt = prompt.replace("[SKILL]", skill)
                prompt = prompt.replace("[PREDICATE_LIST]", ", ".join(pred))
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill2pred)
    return model.generate_multimodal(prompt, consecutive_pair)[0]

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
    # predicates = predicates_per_skill(model, GoTo)
    # print(predicates)
    basic_info = 'The robot visible in the image appears to have a humanoid form with an articulated arm, which suggests it is designed for multipurpose tasks, such as handling objects, performing household chores, or interacting with the environment in a human-like manner. It seems to possess mobility, indicated by its positioning away from the walls and navigating the hallway, allowing it to perform tasks throughout the home. The embodiment of the robot, featuring multiple joints and a sleek design, points towards advanced robotics meant for a domestic setting, enabling it to maneuver and operate effectively within the confines of typical household spaces.'
    # task = task_proposal(model, 'IsFreeHand()', PickUp, basic_info)
    history = '1.\nGoTo(Book)\nPickUp(Book, Table)\n\nSuccess'
    # history = '1.\nPickUp(Book)\n\nFailed\n\n2.\nGoTo(KeyChain)\nMoveGripperForward()\nMoveGripperDown()\nPickUp(KeyChain)\n\nSuccess'
    # history = '1.\nPickUp(Book)\n\nFailed\n\n2.\nGoTo(KeyChain)\nMoveGripperForward()\nMoveGripperDown()\nPickUp(KeyChain)\n\nSuccess\n\n3.\nGoTo(CellPhone)\nMoveGripperForward()\nMoveGripperDown()\nPickUp(CellPhone)\n\nSuccess'
    # history = 'GoTo(Book)\nPickUp(Book)\n\nSuccess'
    # history = 'GoTo(Book)\nPickUp(Book)\n\nSuccess\n\nMoveForward()\nGoTo(KeyChain)\nPickUp(KeyChain)\n\nSuccess'
    # task = task_proposal(model, 'IsReachable(object)', PickUp, basic_info, history=history)
    task = task_proposal(model, 'IsFreeHand()', PickUp, basic_info, history=history)
    print(task)
    # skill2pred = {'PickUp': ['IsObjectReachable(object)', 'IsObjectGraspable(object)', 'IsObjectOnSurface(object,surface)', 'IsSurfaceStable(surface)', 'IsObjectClearOfObstructions(object)', 'IsObjectWithinArmRange(object)', 'IsObjectTypeSupported(object)', 'IsObjectWeightSupported(object)', 'IsObjectPositionKnown(object)', 'IsArmInPositionForGrasp(object)'], 'DropAt': ['IsHolding(object)', 'IsAtLocation(robot,location)', 'IsClear(location)', 'IsObjectType(object,type)', 'IsWithinReach(robot,location)', 'IsStableSurface(location)', 'IsAligned(robot,location)', 'IsDropHeightSafe(robot,location)', 'IsObjectIntact(object)', 'IsLocationAccessible(robot,location)']}
    # unified_pred = unifiy_predicates(model, skill2pred)
    # print(unified_pred)
    # contrastive_pair = ['test_imgs/2.jpg', 'test_imgs/1.jpg']
    # response = pred2precond(model, skill2pred, contrastive_pair)
    # print(response)
    # consecutive_pair = ['test_imgs/2.jpg', 'test_imgs/3.jpg']
    # response = pred2effect(model, skill2pred, consecutive_pair)
    # print(response)