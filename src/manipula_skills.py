'''wrapped skills for manipulaThor agent desgined for the adaptation pipeline, both high-level and low-level'''
import random
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

    obj_names = [obj['objectId'] for obj in metadata["objects"]]
    obj = [obj for obj in obj_names if object_or_location in obj][0]
    poses = sorted(poses, key=lambda p:dist_pose(p, obj['position']))
    pose = random.choice(poses[:10])
    event = controller.step("TeleportFull", **pose)
    return event

def LookAt(controller, obj_name):
    '''rotate the base to center the obj. No idea how to look up or down for centering.'''
    pass

# Gripper movement
def GripperUp(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=0.5, z=0),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def GripperDown(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=-0.5, z=0),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def GripperRight(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0.5, y=0, z=0),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def GripperLeft(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=-0.5, y=0, z=0),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def GripperForward(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=0, z=0.5),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def GripperBackward(controller):
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=0, z=0.5),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    return event

def PickUp(object, controller, metadata):
    '''
    object: str : name of the object
    '''
    obj_names = [obj['objectId'] for obj in metadata["objects"]]
    obj = [obj for obj in obj_names if object in obj][0]
    event = controller.step(
        "MoveArm",
        position=obj["position"],
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
        returnToStart=False
    )

    return event

def DropAt(location, controller, metadata):
    '''
    location: str : name of the object to drop. Named location for LLM to reason
    '''
    obj_names = [obj['objectId'] for obj in metadata["objects"]]
    obj = [obj for obj in obj_names if location in obj][0]
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
