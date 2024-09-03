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
    while True:
        screenshot_path = f"{file_prefix}_{counter}.png"
        if not os.path.exists(screenshot_path):
            break
        counter += 1
    event = controller.step('Pass')
    im = Image.fromarray(event.frame)
    im.save(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")

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
                fieldOfView=60
            )
event = controller.reset(scene="FloorPlan203", fieldOfView=100)

# remove chairs and place objects to easier places
for obj in [obj for obj in event.metadata["objects"] if 'Chair' in obj['objectId']]:
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "Book" not in obj['name']]

object = "Book"
obj = [obj for obj in event.metadata["objects"] if object in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'], 'y': obj['position']['y'], 'z': obj['position']['z']-0.2}})
event = controller.step('SetObjectPoses',objectPoses = poses)

pickupable_objs = [obj['objectId'] for obj in  event.metadata["objects"] if obj["pickupable"]]
receptacle_objs = [obj['objectId'] for obj in  event.metadata["objects"] if obj['receptacle']]
'''
    commands = task.splitlines()
    formatted_commands = []
    
    for command in commands:
        formatted_commands.append(f'capture_obs(controller, "Before_{command.split("(")[0]}")')
        if command.startswith("GoTo"):
            object_name = command[5:-1]  # Extract object name
            formatted_command = f'event = GoTo("{object_name}", controller, event.metadata)'
        elif command.startswith("PickUp"):
            args = command[7:-1].split(", ")  # Extract arguments
            formatted_command = f'event = PickUp("{args[0]}", "{args[1]}", controller, event.metadata)'
        elif command.startswith("DropAt"):
            args = command[7:-1].split(", ")  # Extract arguments
            formatted_command = f'event = DropAt("{args[0]}", "{args[1]}", controller, event.metadata)'
        else:
            continue  # Skip any unrecognized commands
        formatted_commands.append(formatted_command)

        formatted_commands.append(f'capture_obs(controller, "After_{command.split("(")[0]}")')

    formatted_code = "\n".join(formatted_commands)
    return template + formatted_code

if __name__ == "__main__":
    task = "GoTo(Book)\nPickUp(Book, Table)\nGoTo(Sofa)\nDropAt(Book, Sofa)"
    task = "GoTo(DiningTable)\nPickUp(RemoteControl, DiningTable)"
    generated_code = convert_task_to_code(task)
    print(generated_code)
    exec(generated_code)