'''wrapped skills for manipulaThor agent desgined for the adaptation pipeline, both high-level and low-level'''
import random
from copy import deepcopy
import numpy as np

# Base movement
def MoveForward(controller, interval=0.25):
    event = controller.step(
        action='MoveAgent',
                    ahead=0.25
    )
    return event

def MoveBackward(controller):
    event = controller.step(
        action='MoveAgent',
        ahead=-0.25
    )
    return event

def MoveLeft(controller):
    event = controller.step(
        action='MoveAgent',
        right=-0.25
    )
    return event

def MoveRight(controller):
    event = controller.step(
        action='MoveAgent',
        right=0.25
    )
    return event

def TurnLeft(controller):
    event = controller.step(
        action='RotateAgent',
        degrees=-30
    )

def TurnRight(controller):
    event = controller.step(
        action='RotateAgent',
        degrees=30
    )

def GoTo(object_or_location, controller, metadata):
    '''Teleport to a distance from the obj *The goto function has to be imperfect*'''
    def dist_pose(obj1, obj2):
        x1, y1, z1 = obj1["x"], obj1["y"], obj1["z"]
        x2, y2, z2 = obj2["x"], obj2["y"], obj2["z"]
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        return np.sqrt(np.sum((p1-p2)**2, axis=0))

    # obj_names = [obj['objectId'] for obj in metadata["objects"]]
    obj = [obj for obj in metadata["objects"] if object_or_location in obj['objectId']][0]
    avail_positions = controller.step(
        action="GetReachablePositions"
    ).metadata["actionReturn"]
    event = controller.step(
        action="GetInteractablePoses",
        objectId=obj['objectId'],
        horizons=np.linspace(-30, 0),
        standings=[True]
    )
    poses = event.metadata["actionReturn"]
    # breakpoint() 
    poses = [p for p in poses if {'x':p['x'], 'y':p['y'], 'z':p['z']} in avail_positions]
    poses = sorted(poses, key=lambda p:dist_pose(p, obj['position']))
    # poses = sorted(avail_positions, key=lambda p:dist_pose(p, obj['position']))
    # pose = random.choice(poses[:10])
    for i in range(len(poses)):
        pose = poses[i]
        # assert {'x':pose['x'], 'y':pose['y'], 'z':pose['z']} in avail_positions
        event = controller.step("TeleportFull", **pose)
        if event.metadata["lastActionSuccess"]:
            break
    # event = controller.step("Teleport", **pose)
    controller.step('LookDown')
    controller.step('Done')
    return event

def LookAt(controller, obj_name):
    '''rotate the base to center the obj. No idea how to look up or down for centering.'''
    pass

# Gripper movement
def MoveGripperUp(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=0.5, z=0),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def MoveGripperDown(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=-0.5, z=0),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def MoveGripperRight(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0.5, y=0, z=0),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def MoveGripperLeft(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=-0.5, y=0, z=0),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def MoveGripperForward(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=0, z=0.5),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def MoveGripperBackward(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=0, z=-0.5),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

# TODO： hover above the obj to avoid collision
def PickUp(object, location, controller, metadata):
    '''
    object: str : name of the object
    '''
    # event = controller.step('MoveArmBase', y = 0.5)
    # obj_names = [obj['objectId'] for obj in metadata["objects"]]
    controller.step(action="SetHandSphereRadius", radius=0.04)
    obj = [obj for obj in metadata["objects"] if object in obj['objectId']][0]
    position = deepcopy(obj["position"])
    if "Book" in object:
        position = {'x': position['x']+0.1, 'y': position['y'] + 0.05, 'z': position['z']-0.2}
    # controller.step(action="SetHandSphereRadius", radius=0.25)
    event = controller.step(
        "MoveArm",
        position=position,
        coordinateSpace="world",
        returnToStart=False
    )
    event = controller.step(
        action="PickupObject",
        objectIdCandidates=[obj['objectId']],
    )
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=0.5, z=0),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=True
    )
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=0, z=-0.25),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=True
    )
    # event = controller.step('MoveArmBase', y = 0.5)
    # controller.step(action="SetHandSphereRadius", radius=0.04)
    controller.step('Done')
    
    return event

def DropAt(object, location, controller, metadata):
    '''
    location: str : name of the object to drop. Named location for LLM to reason
    '''
    # obj_names = [obj['objectId'] for obj in metadata["objects"]]
    obj = [obj for obj in metadata["objects"] if object in obj['objectId']][0]
    obj_pos = obj['position']
    hover_pos = dict(x=obj_pos['x'], y=obj_pos['y']+0.5, z=obj_pos['z'])
    event = controller.step(
        "MoveArm",
        position=obj["position"],
        coordinateSpace="world",
        returnToStart=False
    )
    event = controller.step(action="ReleaseObject")
    # make sure the thing is dropped
    for i in range(10):
        controller.step(action="Done")
    return event

# not for the workshop. Too tricky to implement
def Open(controller):
    pass

def Close(controller):
    pass
