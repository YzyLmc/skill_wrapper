import numpy as np
import random
from PIL import Image

from ai2thor.controller import Controller

from manipula_skills import *

# test open action
# def open_obj(obj_id):

def get_obj(obj_id, obj_list):
    return [obj for obj in  event.metadata["objects"] if obj["objectId"] == obj_id]

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

for obj in [obj for obj in event.metadata["objects"] if 'Chair' in obj['objectId']]:
    event = controller.step('RemoveFromScene', objectId=obj["objectId"])

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "Book" not in obj['name']]
object = "Book"
obj = [obj for obj in event.metadata["objects"] if object in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'], 'y': obj['position']['y'], 'z': obj['position']['z']-0.2}})
event = controller.step('SetObjectPoses',objectPoses = poses)

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "RemoteControl" not in obj['name']]
object = "RemoteControl"
obj = [obj for obj in event.metadata["objects"] if object in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'] + 0.2, 'y': obj['position']['y'], 'z': obj['position']['z']}})
event = controller.step('SetObjectPoses',objectPoses = poses)

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "Bowl" not in obj['name']]
object = "Bowl"
obj = [obj for obj in event.metadata["objects"] if object in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'] - 0.2, 'y': obj['position']['y'], 'z': obj['position']['z']}})
event = controller.step('SetObjectPoses',objectPoses = poses)

poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if "Vase" not in obj['name']]
object = "Vase"
obj = [obj for obj in event.metadata["objects"] if object in obj['objectId']][0]
poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'], 'y': obj['position']['y'], 'z': obj['position']['z'] + 0.2}})
event = controller.step('SetObjectPoses',objectPoses = poses)

# sofa position
{'name': 'agent', 'position': {'x': -0.4249999523162842, 'y': 0.9070531129837036, 'z': 3.083493709564209}, 'rotation': {'x': -0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': 30.00000762939453, 'isStanding': True, 'inHighFrictionArea': False}

pickupable_objs = [obj['objectId'] for obj in  event.metadata["objects"] if obj["pickupable"]]
# find openable object
openable_objs = [obj['objectId'] for obj in  event.metadata["objects"] if obj["openable"]]
receptacle_objs = [obj['objectId'] for obj in  event.metadata["objects"] if obj['receptacle']]

breakpoint()

# use drawer as target
# drawer_id = 'Drawer|+00.60|+00.23|+02.60'
drawer_id = [obj for obj in openable_objs if "Drawer" in obj][0]
# find a pose to interact
event = controller.step(
    action="GetInteractablePoses",
    objectId=drawer_id,
    horizons=[np.linspace(-30, 60, 30)],
    standings=[True, False]
)
poses = event.metadata["actionReturn"]

# teleport to interactable pose
pose = random.choice(poses)

event = controller.step("Teleport", **pose)
im = Image.fromarray(event.frame)
im.show()
# print state of the drawer
drawer = get_obj(drawer_id, event.metadata["objects"])
breakpoint() # drawer should be closed
im = Image.fromarray(event.frame)
im.show()
# open it
event = controller.step(
    action="OpenObject",
    objectId=drawer_id,
    openness=1,
    forceAction=False
)
# see if it's opened
drawer = get_obj(drawer_id, event.metadata["objects"])
breakpoint()

im = Image.fromarray(event.frame)
im.show()

breakpoint()
event = controller.step("Teleport", **pose)
event = controller.step("MoveBack")
im = Image.fromarray(event.frame)
im.show()
