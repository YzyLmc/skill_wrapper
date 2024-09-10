'Convert generated tasks into python script'
import numpy as np
import random
from PIL import Image

from ai2thor.controller import Controller

from manipula_skills import *


def convert_task_to_code(task):
    'task: str: sequence of commands separate by \n'
    template = '''
import numpy as np
import random
import os
from PIL import Image

from ai2thor.controller import Controller

from manipula_skills import *

def capture_obs(controller, file_prefix):
    counter = 1
    directory = f"tasks/exps/{file_prefix.split('_')[1]}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    while True:
        screenshot_path = directory + f"{file_prefix}_{counter}.jpg"
        screenshot_path_suc = directory + f"{file_prefix}_True_{counter}.jpg"
        screenshot_path_fail = directory + f"{file_prefix}_False_{counter}.jpg"
        if not os.path.exists(screenshot_path) or not os.path.exists(screenshot_path_suc) or not os.path.exists(screenshot_path_fail):
            break
        counter += 1
    event = controller.step('Pass')
    im = Image.fromarray(event.frame)
    im.save(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")
    return screenshot_path

# init ai2thor controller
controller = Controller(
    massThreshold = 1,
                agentMode="arm",
                scene = "FloorPlan203",
                snapToGrid=False,
                visibilityDistance=1.5,
                gridSize=0.1,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                renderObjectImage = True,
                width= 1280,
                height= 720,
                fieldOfView=90
            )
event = controller.reset(scene="FloorPlan203", fieldOfView=100)
controller.step(action="SetHandSphereRadius", radius=0.15)
sofa_pose = {'name': 'agent', 'position': {'x': -0.1749999225139618, 'y': 0.9070531129837036, 'z': 3.083493709564209}, 'rotation': {'x': -0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': 30.00001525878906, 'isStanding': True, 'inHighFrictionArea': False}
controller.step(
            action = 'Teleport',
            position = sofa_pose['position'],
            rotation = sofa_pose['rotation'],
            horizon = int(sofa_pose['cameraHorizon']),
            standing = sofa_pose['isStanding']
        )

for obj in [obj for obj in event.metadata["objects"] if 'Chair' in obj['objectId']]:
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

for obj in [obj for obj in event.metadata["objects"] if 'Pencil' in obj['objectId']]:
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

for obj in [obj for obj in event.metadata["objects"] if 'Plate' in obj['objectId']]:
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

for obj in [obj for obj in event.metadata["objects"] if 'CellPhone' in obj['objectId']]:
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

for obj in [obj for obj in event.metadata["objects"] if 'RemoteControl' in obj['objectId']]:
    remote = deepcopy(obj)
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "Book" not in obj['name']]
object = "Book"
obj = [obj for obj in event.metadata["objects"] if object in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x']-0.2, 'y': obj['position']['y'], 'z': obj['position']['z']-0.2}})
event = controller.step('SetObjectPoses',objectPoses = poses)

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "TissueBox" not in obj['name']]
replace_with = 'TissueBox'# replace remotecontrol
obj_replace_with = [obj for obj in event.metadata["objects"] if replace_with in obj['objectId']][0]
obj = remote
poses.append({'objectName':obj_replace_with['name'], "position":{'x': obj['position']['x'] + 0.2, 'y': obj['position']['y'], 'z': obj['position']['z']-0.4}})
event = controller.step('SetObjectPoses',objectPoses = poses)

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "Bowl" not in obj['name']]
object = "Bowl"
obj = [obj for obj in event.metadata["objects"] if object in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'] - 0.1, 'y': obj['position']['y'], 'z': obj['position']['z']}})
event = controller.step('SetObjectPoses',objectPoses = poses)

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "Vase" not in obj['name']]
object = "Vase"
obj = [obj for obj in event.metadata["objects"] if object in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'] - 0.1, 'y': obj['position']['y'], 'z': obj['position']['z']}})
event = controller.step('SetObjectPoses',objectPoses = poses)
'''
    # commands = task.splitlines()
    commands = task
    formatted_commands = []
    
    for command in commands:
        if command.startswith("GoTo"):
            args = command[5:-1].replace(' ','').split(",")  # Extract arguments
            formatted_commands.append(f'screenshot_path = capture_obs(controller, f"Before_{command.split("(")[0]}_{args[0]}_{args[1]}")')
            formatted_command = f'suc, event = GoTo("{args[0]}", "{args[1]}", controller, event)'
        elif command.startswith("PickUp"):
            args = command[7:-1].replace(' ','').split(",")  # Extract arguments
            formatted_commands.append(f'screenshot_path = capture_obs(controller, f"Before_{command.split("(")[0]}_{args[0]}_{args[1]}")')
            formatted_command = f'suc, event = PickUp("{args[0]}", "{args[1]}", controller, event)'
        elif command.startswith("DropAt"):
            args = command[7:-1].replace(' ','').split(",")  # Extract arguments
            formatted_commands.append(f'screenshot_path = capture_obs(controller, f"Before_{command.split("(")[0]}_{args[0]}_{args[1]}")')
            formatted_command = f'suc, event = DropAt("{args[0]}", "{args[1]}", controller, event)'
        else:
            continue  # Skip any unrecognized commands
        formatted_commands.append(formatted_command)

        formatted_commands.append(f'capture_obs(controller, f"After_{command.split("(")[0]}_{args[0]}_{args[1]}_{"{suc}"}")')
        formatted_commands.append(f'os.rename(screenshot_path, screenshot_path.replace(f"Before_{command.split("(")[0]}_{args[0]}_{args[1]}", f"Before_{command.split("(")[0]}_{args[0]}_{args[1]}_{"{suc}"}"))')

    formatted_code = "\n".join(formatted_commands)
    return template + formatted_code

if __name__ == "__main__":
    # task = "GoTo(Sofa, Book)\nPickUp(Book, DiningTable)\nGoTo(DiningTable, Sofa)\nDropAt(Book, Sofa)"
    # task = "GoTo(Sofa, DiningTable)\nPickUp(RemoteControl, DiningTable)"

    # task 1
    # task = ['GoTo(Sofa,Sofa)', 'PickUp(TissueBox,Sofa)', 'GoTo(Sofa,DiningTable)', 'DropAt(TissueBox,DiningTable)', 'PickUp(Book,DiningTable)', 'DropAt(Book,DiningTable)', 'PickUp(TissueBox,DiningTable)', 'DropAt(TissueBox,Sofa)']
    task = ['GoTo(Sofa,DiningTable)', 'PickUp(Bowl,DiningTable)', 'GoTo(DiningTable,Sofa)', 'DropAt(Bowl,Sofa)', 'PickUp(TissueBox,Sofa)', 'DropAt(TissueBox,DiningTable)', 'PickUp(Book,DiningTable)', 'DropAt(Book,Sofa)']
    generated_code = convert_task_to_code(task)
    print(generated_code)
    exec(generated_code)