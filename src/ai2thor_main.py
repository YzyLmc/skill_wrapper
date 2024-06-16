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


# first collect images of the environment
controller = Controller(
                massThreshold = 1,
                agentMode="arm",
                scene = "FloorPlan212",
                snapToGrid=False,
                visibilityDistance=1.5,
                gridSize=0.25,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                renderObjectImage = True,
                width= 1280,
                height= 720,
                fieldOfView=60
            )
controller.reset(scene="FloorPlan212", fieldOfView=100)

# top_down_img = get_top_down_frame(controller)
# top_down_img.show()
# show the gripper more by looking down a bit
event = controller.step('LookDown')
im = Image.fromarray(event.frame)
im.save('running_imgs/env.jpg')

# get basic information of the environment
model = GPT4(engine=ENGINE)
basic_prompt = load_from_file(f"prompts/basic_info.txt")
breakpoint()

# # top_down_view doesn't work yet, use something manually taken for now
# env_img = ['running_imgs/env.jpg']
# basic_info = model.generate_multimodal(basic_prompt, env_img)[0]
# print("Basic_info:",basic_info)
# breakpoint()

# # propose predicates
# ## text-only
# highlevel_skills = [PickUp, DropAt]
# skill2preds = {}
# for skill in highlevel_skills:
#     skill2preds[skill.__name__] = predicates_per_skill(model, skill, basic_info)
# print("Skill:predicates", skill2preds)
# breakpoint()

# #unify predicates bewteen skills
# equal_preds = unifiy_predicates(model, skill2preds)

# for skill, preds in skill2preds.items():
#     unified_preds = []
#     for pred in preds:
#         for equal_pred in equal_preds:
#             if pred in equal_pred:
#                 pred = equal_pred[0]
#             else:
#                 continue
#         unified_preds.append(pred)
#     unified_preds= list(set(unified_preds))
#     skill2preds[skill] = unified_preds

# print("Skill: Unified predicates", skill2preds)

# breakpoint()

skill2preds = {'PickUp': ['GripperOpen(robot)', 'ObjectGrasped(robot,object)', 'ObjectReachable(robot,object)', 'ObjectLifted(robot,object)', 'ObjectDetected(object)'], 'DropAt': ['IsClear(location)', 'IsObjectAt(object,location)', 'IsHolding(object)', 'IsAtLocation(robot,location)', 'ObjectDetected(object)']}

for skill, preds in skill2preds.items():
    task  = task_proposal(model, 'ObjectReachable(robot,object)', skill)[0]
    print(task)
    breakpoint()

# assume we have this task generated are these
task1 = '''
GoTo(Object)
PickUp(Object)
'''
task2 = '''
GoTo(Object)
MoveLeft()
MoveBackward()
PickUp(Object)
'''

task3 = '''
GoTo(Object)
PickUp(Object)
'''

# executing tasks in a different script. For now the tasks are manullay converted into python codes