'adapt original skill into current environment by combining it with lower level actions'

from copy import deepcopy
import inspect
from image_similarity_measures.evaluate import evaluation
from manipula_skills import *
from utils import load_from_file, GPT4

def get_contrastive_pair(skill, obs, database, metrics=['fsim']):
    '''
    get a contrastive pair from the database
    obs: image_path
    database: {"Skill": {"Success":[img_path], "Fail":[img_path]}}
    '''
    # find success and fail imgs seprately
    success_list = deepcopy(database[skill.__name__]["Success"])
    fail_list = deepcopy(database[skill.__name__]["Fail"])
    # breakpoint()
    contrastive_pair = [
        sorted(success_list, key=lambda img_path:evaluation(org_img_path=obs, pred_img_path=img_path, metrics=metrics)[metrics[0]])[-1],
        sorted(fail_list, key=lambda img_path:evaluation(org_img_path=obs, pred_img_path=img_path, metrics=metrics)[metrics[0]])[-1]
    ]
    return contrastive_pair

def predict_success(model, skill, skill2pred, obs, contrastive_pair, prompt_fpath="prompts/predict_success.txt"):
    '''
    predict if the execution will succeed.
    Multiple backends including GPT, cosine similarity.
    '''
    def construct_prompt(prompt, skill2pred):
        while "[SKILL]" in prompt or "[PREDICATE_LIST]" in prompt:
            prompt = prompt.replace("[SKILL]", skill.__name__)
            prompt = prompt.replace("[PREDICATE_LIST]", ", ".join(skill2pred[skill.__name__]))
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill2pred)
    contrastive_pair.append(obs)
    return model.generate_multimodal(prompt, contrastive_pair)[0]

def back_to_success_state(model, skill, skill2pred, obs, success_state, action_list= [MoveForward, MoveBackward, MoveLeft, MoveRight, TurnLeft, TurnRight, MoveGripperUp, MoveGripperDown, MoveGripperLeft, MoveGripperRight, MoveGripperForward, MoveGripperBackward] , prompt_fpath="prompts/adaptation.txt"):
    '''
    Drive back the agent to the successful state. Only triggered when predicted to fail
    Only low-level skills are used for adaptation.
    '''
    def construct_prompt(prompt, skill2pred, action_list):
        action_list_strs = [f'{a.__name__}({", ".join(inspect.getfullargspec(a)[0][:-2])})' for a in action_list]
        action_list_strs_joined = ", ".join(action_list_strs)
        skill_name = f'{skill.__name__}({", ".join(inspect.getfullargspec(skill)[0][:-2])})'
        while "[PREDICATE_LIST]" in prompt or "[ACTION_LIST]" in prompt or '[SKILL]' in prompt:
            prompt = prompt.replace("[SKILL]", skill_name)
            prompt = prompt.replace("[PREDICATE_LIST]", ' ,'.join((skill2pred[skill_name])))
            prompt = prompt.replace("[ACTION_LIST]", action_list_strs_joined)
        return prompt
    prompt = load_from_file(prompt_fpath)
    prompt = construct_prompt(prompt, skill2pred, action_list)
    contrastive_pair = [success_state, obs]
    return model.generate_multimodal(prompt, contrastive_pair)[0]

def adapt(observation, database, prompt):
    '''
    Main function not implemented yet.
    '''
    pass

if __name__ == "__main__":
    database = {"PickUp":{"Success":['test_imgs/1.jpg', 'test_imgs/2.jpg', 'test_imgs/3.jpg'],"Fail": ['test_imgs/4.jpg', 'test_imgs/5.jpg']}}
    obs = 'test_imgs/1.jpg' # 1 should succeed
    obs = 'test_imgs/5.jpg' # 5 should fail
    obs = 'test_imgs/0.jpg'
    
    # contrastive_pair = get_contrastive_pair(PickUp, obs, database)
    # contrastive_pair = ['test_imgs/2.jpg', 'test_imgs/0.jpg']
    model = GPT4()
    # skill2pred = {'PickUp': ['IsObjectReachable(object)', 'IsObjectGraspable(object)', 'IsObjectOnSurface(object,surface)', 'IsSurfaceStable(surface)', 'IsObjectClearOfObstructions(object)', 'IsObjectWithinArmRange(object)', 'IsObjectTypeSupported(object)', 'IsObjectWeightSupported(object)', 'IsObjectPositionKnown(object)', 'IsArmInPositionForGrasp(object)'], 'DropAt': ['IsHolding(object)', 'IsAtLocation(robot,location)', 'IsClear(location)', 'IsObjectType(object,type)', 'IsWithinReach(robot,location)', 'IsStableSurface(location)', 'IsAligned(robot,location)', 'IsDropHeightSafe(robot,location)', 'IsObjectIntact(object)', 'IsLocationAccessible(robot,location)']}
    skill2preds = {
    'PickUp(object, location)': ['AtLocation(object,location)', 'Holding(object)', 'At(location)', 'IsReachable(object)', 'IsFreeHand()'],
    'DropAt(object, location)': ['Clear(location)', 'AtLocation(object,location)', 'Holding(object)', 'At(location)', 'IsFreeHand()'],
    'GoTo(location)': ['At(location)', 'IsFreeHand()', 'Clear(location)', 'BatterySufficient()']
               }
    # # response = predict_success(model, PickUp, skill2pred, obs, contrastive_pair)
    # # print(response)
    success_state = 'test_imgs/1.jpg' # 1 should succeed
    obs = 'test_imgs/4.jpg' #  4 should fail too
    # skill_name = f'{PickUp.__name__}({", ".join(inspect.getfullargspec(PickUp)[0][:-2])})'
    response = back_to_success_state(model, PickUp, skill2preds, obs, success_state)
    print(response)
