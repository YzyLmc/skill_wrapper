'Execute python script to get initial observation of the env and save them under /tasks/exps/init_obs/'
import numpy as np
import random
from PIL import Image

from ai2thor.controller import Controller

from manipula_skills import *


def get_initial_obs():
    'task: str: sequence of commands separate by \n'
    template = '''
import numpy as np
import random
import os
from PIL import Image

from ai2thor.controller import Controller

from manipula_skills import *

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
                width= 720,
                height= 720,
                fieldOfView=90
            )
event = controller.reset(scene="FloorPlan203", fieldOfView=100)
controller.step(action="SetHandSphereRadius", radius=0.04)
sofa_pose = {'name': 'agent', 'position': {'x': -0.1749999225139618, 'y': 0.9070531129837036, 'z': 3.083493709564209}, 'rotation': {'x': -0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': 30.00001525878906, 'isStanding': True, 'inHighFrictionArea': False}
dining_table_pose = {'name': 'agent', 'position': {'x': -4.324999809265137, 'y': 0.9070531129837036, 'z': 0.5165063142776489}, 'rotation': {'x': -0.0, 'y': 180.0, 'z': 0.0}, 'cameraHorizon': 30.000003814697266, 'isStanding': True, 'inHighFrictionArea': False}
coffee_table_pose = {'name': 'agent', 'position': {'x': -0.3915063440799713, 'y': 0.9070531129837036, 'z': 2.458493709564209}, 'rotation': {'x': -0.0, 'y': 90.0, 'z': 0.0}, 'cameraHorizon': 30.000003814697266, 'isStanding': True, 'inHighFrictionArea': False}
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

event = controller.step(
    action="AddThirdPartyCamera",
    position=sofa_pose['position'],
    rotation=sofa_pose['rotation'],
    fieldOfView=90
)
event = controller.step(
    action="AddThirdPartyCamera",
    position=dining_table_pose['position'],
    rotation=dining_table_pose['rotation'],
    fieldOfView=90
)
event = controller.step(
    action="AddThirdPartyCamera",
    position=coffee_table_pose['position'],
    rotation=coffee_table_pose['rotation'],
    fieldOfView=90
)

directory = f"tasks/exps/init_obs/"
if not os.path.exists(directory):
        os.makedirs(directory)
counter = 0
for img in event.third_party_camera_frames:
    im = Image.fromarray(img)
    im.save(f"{directory}obs_{counter}.jpg")
    print(f"Screenshot saved to {directory}obs_{counter}.jpg")
    counter += 1

'''

if __name__ == "__main__":
    generated_code = get_initial_obs()
    print(generated_code)
    breakpoint()
    exec(generated_code)