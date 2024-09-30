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
    from PIL import Image
    import os
    counter = 1
    directory = f"tasks/exps/{file_prefix.split('_')[1]}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    f_list = os.listdir(directory)
    while True:
        screenshot_path = f"{file_prefix.replace('_True', '').replace('_False','')}_{counter}.jpg"
        # screenshot_path = f"{file_prefix}_{counter}.jpg"
        screenshot_path_suc = f"{file_prefix.replace('_True', '').replace('_False','')}_True_{counter}.jpg"
        screenshot_path_fail = f"{file_prefix.replace('_True', '').replace('_False','')}_False_{counter}.jpg"
        if not (screenshot_path in f_list or screenshot_path_suc in f_list or screenshot_path_fail in f_list):
            break
        counter += 1
    event = controller.step('Pass')
    im = Image.fromarray(event.frame)
    im.save(f"{directory}{file_prefix}_{counter}.jpg")
    print(f"Screenshot saved to {file_prefix}_{counter}.jpg")
    return f"{directory}{file_prefix}_{counter}.jpg"

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
controller.step(action="SetHandSphereRadius", radius=0.04)
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

for obj in [obj for obj in event.metadata["objects"] if 'Watch' in obj['objectId']]:
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

for obj in [obj for obj in event.metadata["objects"] if 'RemoteControl' in obj['objectId']]:
    remote = deepcopy(obj)
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

for obj in [obj for obj in event.metadata["objects"] if 'Laptop' in obj['objectId']]:
    laptop = deepcopy(obj)
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if not "Book" in obj['name']]
object = "Book"
obj = [obj for obj in event.metadata["objects"] if "Book" in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x']-0.2, 'y': obj['position']['y'], 'z': obj['position']['z']-0.2}})
event = controller.step('SetObjectPoses',objectPoses = poses)

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if not "TissueBox" in obj['name']]
replace_with = 'TissueBox'# replace remotecontrol
obj_replace_with = [obj for obj in event.metadata["objects"] if 'TissueBox' in obj['objectId']][0]
obj = remote
poses.append({'objectName':obj_replace_with['name'], "position":{'x': obj['position']['x'] + 0.2, 'y': obj['position']['y'], 'z': obj['position']['z']-0.4}})
event = controller.step('SetObjectPoses',objectPoses = poses)

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "Vase" not in obj['name']]
replace_with = 'Vase'# replace laptop
obj_replace_with = [obj for obj in event.metadata["objects"] if "Vase" in obj['objectId']][0]
obj = laptop
poses.append({'objectName':obj_replace_with['name'], "position":{'x': obj['position']['x'] + 0.2, 'y': obj['position']['y'], 'z': obj['position']['z']-0.4}})
event = controller.step('SetObjectPoses',objectPoses = poses)

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if not "Bowl" in obj['name']]
object = "Bowl"
obj = [obj for obj in event.metadata["objects"] if "Bowl" in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'] + 0.3, 'y': obj['position']['y'], 'z': obj['position']['z'] + 0.3}})
event = controller.step('SetObjectPoses',objectPoses = poses)

suc, event = GoTo('Sofa', 'DiningTable', controller, event)
suc, event = GoTo('DiningTable', 'Sofa', controller, event)
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
    # task = ['GoTo(Sofa,DiningTable)', 'PickUp(Bowl,DiningTable)', 'GoTo(DiningTable,Sofa)', 'DropAt(Bowl,Sofa)', 'PickUp(TissueBox,Sofa)', 'DropAt(TissueBox,DiningTable)', 'PickUp(Book,DiningTable)', 'DropAt(Book,Sofa)']
    # task = ['PickUp(Bowl,DiningTable)', 'GoTo(DiningTable,Sofa)', 'DropAt(Bowl,Sofa)', 'PickUp(TissueBox,Sofa)', 'GoTo(Sofa,DiningTable)', 'DropAt(TissueBox,DiningTable)', 'GoTo(DiningTable,Sofa)', 'PickUp(Book,DiningTable)', 'DropAt(Book,DiningTable)']
    # task = ['GoTo(DiningTable,DiningTable)', 
    #         'PickUp(Book,DiningTable)', 
    #         'PickUp(Book,DiningTable)', 
    #         'DropAt(Book,Sofa)', 
    #         'PickUp(Vase,DiningTable)', 
    #         'DropAt(Vase,Sofa)', 
    #         'DropAt(Vase,Sofa)', 
    #         'PickUp(Bowl,DiningTable)']
    task = ['GoTo(Sofa,DiningTable)', 'PickUp(Bowl,DiningTable)', 'GoTo(DiningTable,Sofa)', 'DropAt(Bowl,Sofa)', 'PickUp(TissueBox,Sofa)', 'DropAt(TissueBox,DiningTable)', 'GoTo(DiningTable,CoffeeTable)', 'PickUp(Vase,CoffeeTable)']
    generated_code = convert_task_to_code(task)
    print(generated_code)
    breakpoint()
    exec(generated_code)