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

def eval_init_state(controller, command_list):
    'drive the agent to the init state for eval'
    'command_list example: '
    event, controller = init_env(controller)
    for command in command_list:
        executable_command = 
        exec
    return event, controller


    def sampler()
    from itertools import product

    # Define objects and locations
    objects = ['a', 'b', 'c']
    containers = ['A', 'B', 'C']

    # Generate all possible states (robot location, object locations, robot holding state)
    # States are tuples: (robot_location, object_a_location, object_b_location, object_c_location, holding)
    states = []
    for robot_loc, obj_a_loc, obj_b_loc, obj_c_loc in product(containers, containers + ['holding'], containers + ['holding'], containers + ['holding']):
        # Ensure that at most one object is 'holding'
        held_objects = [obj_a_loc == 'holding', obj_b_loc == 'holding', obj_c_loc == 'holding']
        if sum(held_objects) <= 1:  # Only one object can be held at a time
            if any(held_objects):  # If an object is being held, robot can't be 'hand_empty'
                holding = 'holding'
            else:
                holding = 'hand_empty'
            states.append((robot_loc, obj_a_loc, obj_b_loc, obj_c_loc, holding))

    # Function to filter states based on predicates
    def get_states_satisfying_predicates(at=None, at_obj=None, holding=None, hand_empty=None):
        satisfying_states = []
        
        for state in states:
            robot_loc, obj_a_loc, obj_b_loc, obj_c_loc, robot_holding = state

            # Check At(loc)
            if at and robot_loc != at:
                continue

            # Check AtObj(obj, loc)
            if at_obj:
                obj, loc = at_obj
                if obj == 'a' and obj_a_loc != loc:
                    continue
                if obj == 'b' and obj_b_loc != loc:
                    continue
                if obj == 'c' and obj_c_loc != loc:
                    continue

            # Check Holding(obj)
            if holding:
                if robot_holding != 'holding' or (holding == 'a' and obj_a_loc != 'holding') or (holding == 'b' and obj_b_loc != 'holding') or (holding == 'c' and obj_c_loc != 'holding'):
                    continue

            # Check HandEmpty()
            if hand_empty is not None and (hand_empty and robot_holding != 'hand_empty'):
                continue

            # If all checks pass, add the state
            satisfying_states.append(state)

        return satisfying_states

    # # Example usage:
    # # Find all states where the robot is at A, and is holding object 'a'
    # example_states = get_states_satisfying_predicates(at='A', holding='a')
    # print("Example states:", example_states)