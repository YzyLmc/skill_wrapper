'''
evaluate script for learned operators in ai2thor
We handpick the evaluation cases
'''
import numpy as np
import random
from PIL import Image

from ai2thor.controller import Controller

from manipula_skills import *

# same init state as 
def init_env(controller):
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

    poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if not "Bowl" in obj['name']]
    object = "Bowl"
    obj = [obj for obj in event.metadata["objects"] if "Bowl" in obj['objectId']][0]
    poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'] - 0.1, 'y': obj['position']['y'], 'z': obj['position']['z']}})
    event = controller.step('SetObjectPoses',objectPoses = poses)

    poses = [{'objectName':obj['name'], "position":obj['position'], "rotation": obj['rotation']} for obj in event.metadata['objects'] if not "Vase" in obj['name']]
    object = "Vase"
    obj = [obj for obj in event.metadata["objects"] if "Vase" in obj['objectId']][0]
    poses.append({'objectName':obj['name'], "position":{'x': obj['position']['x'] - 0.1, 'y': obj['position']['y'], 'z': obj['position']['z']}})
    event = controller.step('SetObjectPoses',objectPoses = poses)
    return event, controller

def eval_init_state(controller, skill_list):
    
    return event, controller