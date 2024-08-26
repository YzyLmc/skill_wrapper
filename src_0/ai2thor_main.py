import numpy as np
import random
from PIL import Image

from ai2thor.controller import Controller

from utils import get_top_down_frame, GPT4, load_from_file
from symbolize import *
from manipula_skills import *

ENGINE = "gpt-4o" # Omni is better

# import predefined skills
from manipula_skills import *
## high-level actions are GoTo, PickUp, Drop,Open, Close
## low-level actions are {Gripper}*{Up, Down, Left, Right, Forward, Backward}, {Base}*{Left, Right, Forward, Backward}


# # first collect images of the environment
# controller = Controller(
#                 massThreshold = 1,
#                 agentMode="arm",
#                 scene = "FloorPlan212",
#                 snapToGrid=False,
#                 visibilityDistance=1.5,
#                 gridSize=0.25,
#                 renderDepthImage=True,
#                 renderInstanceSegmentation=True,
#                 renderObjectImage = True,
#                 width= 1280,
#                 height= 720,
#                 fieldOfView=60
#             )
# controller.reset(scene="FloorPlan212", fieldOfView=100)

# top_down_img = get_top_down_frame(controller)
# top_down_img.show()
# show the gripper more by looking down a bit
# event = controller.step('LookDown')
# im = Image.fromarray(event.frame)
# im.save('running_imgs/env.jpg')

# get basic information of the environment
model = GPT4(engine=ENGINE)
basic_prompt = load_from_file(f"prompts/basic_info.txt")

# # top_down_view doesn't work yet, use something manually taken for now
# env_img = ['running_imgs/env.jpg']
# basic_info = model.generate_multimodal(basic_prompt, env_img)[0]
# print("Basic_info:",basic_info)
# breakpoint()

basic_info = 'The robot visible in the image appears to have a humanoid form with an articulated arm, which suggests it is designed for multipurpose tasks, such as handling objects, performing household chores, or interacting with the environment in a human-like manner. It seems to possess mobility, indicated by its positioning away from the walls and navigating the hallway, allowing it to perform tasks throughout the home. The embodiment of the robot, featuring multiple joints and a sleek design, points towards advanced robotics meant for a domestic setting, enabling it to maneuver and operate effectively within the confines of typical household spaces.'

# # propose predicates
# ## text-only
# highlevel_skills = [PickUp, DropAt, GoTo]
# skill2preds = {}
# for skill in highlevel_skills:
#     skill_name = skill_name = f'{skill.__name__}({", ".join(inspect.getfullargspec(skill)[0][:-2])})'
#     skill2preds[skill_name] = predicates_per_skill(model, skill, basic_info)
# print("Skill:predicates", skill2preds)

# skill2preds saved
skill2preds = {
    'PickUp(object)':['Holding(object)', 'AtLocation(location)', 'AtLocation(object,location)', 'IsReachable(object)', 'IsFreeHand()'],
    'DropAt(location)':['Holding(object)', 'AtLocation(location)', 'Clear(location)', 'ObjectAt(object,location)', 'ArmEmpty()'],
    'GoTo(location)':['At(location)', 'PathClear(location)', 'BatterySufficient()', 'LocationAccessible(location)', 'NotCarryingObject()']
}
breakpoint()

# #unify predicates bewteen skills
# equal_preds = unifiy_predicates(model, skill2preds)
# equal_preds saved
equal_preds = [['IsFreeHand()', 'NotCarryingObject()', 'ArmEmpty()'], ['At(location)', 'AtLocation(location)'], ['AtLocation(object,location)', 'ObjectAt(object,location)'], ['Clear(location)', 'PathClear(location)', 'LocationAccessible(location)']]

unified_skill2preds = {}
for skill, preds in skill2preds.items():
    unified_preds = []
    for pred in preds:
        dup = False
        for equal_pred in equal_preds:
            if pred in equal_pred:
                unified_preds.append(equal_pred[0])
                dup = True
        if not dup:
            unified_preds.append(pred)
    unified_skill2preds[skill] = list(set(unified_preds))

print("Skill: Unified predicates", unified_skill2preds)

breakpoint()

# skill2preds = {'PickUp': ['GripperOpen(robot)', 'ObjectGrasped(robot,object)', 'ObjectReachable(robot,object)', 'ObjectLifted(robot,object)', 'ObjectDetected(object)'], 'DropAt': ['IsClear(location)', 'IsObjectAt(object,location)', 'IsHolding(object)', 'IsAtLocation(robot,location)', 'ObjectDetected(object)']}
skill2preds = {
    'PickUp(object, location)': ['AtLocation(object,location)', 'Holding(object)', 'At(location)', 'IsReachable(object)', 'IsFreeHand()'],
    'DropAt(object, location)': ['Clear(location)', 'AtLocation(object,location)', 'Holding(object)', 'At(location)', 'IsFreeHand()'],
    'GoTo(location)': ['At(location)', 'IsFreeHand()', 'Clear(location)', 'BatterySufficient()']
               }
for skill, preds in skill2preds.items():
    task  = task_proposal(model, 'ObjectReachable(robot,object)', skill, basic_info=basic_info)[0]
    print(task)
    breakpoint()
# Generated tasks are stored under /tasks/
# executing tasks in a different script. For now the tasks are manullay converted into python codes