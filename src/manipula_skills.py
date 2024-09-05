'''wrapped skills for manipulaThor agent desgined for the adaptation pipeline, both high-level and low-level'''
import random
from copy import deepcopy
import numpy as np

def No_op(controller):
    event = controller.step('Pass')
    return event

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
    
    obj = [obj for obj in metadata["objects"] if object_or_location in obj['objectId']][0]
    try:
        receptacle = obj['parentReceptacles'][0]
    except:
        receptacle = ''

    if "Sofa" in object_or_location or "Sofa" in receptacle:
        # hardcode locations for GoTo
        # target format example for teleportfull: {'x': -3.5250000953674316, 'y': 0.9070531129837036, 'z': -1.616506576538086, 'rotation': 0.0, 'standing': True, 'horizon': -20.816326141357422}
        # input format from event.metadata['agent']: {'name': 'agent', 'position': {'x': -0.4249999523162842, 'y': 0.9070531129837036, 'z': 3.083493709564209}, 'rotation': {'x': -0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': 30.00000762939453, 'isStanding': True, 'inHighFrictionArea': False}
        sofa_pose = {'name': 'agent', 'position': {'x': -0.1749999225139618, 'y': 0.9070531129837036, 'z': 3.083493709564209}, 'rotation': {'x': -0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': 30.00001525878906, 'isStanding': True, 'inHighFrictionArea': False}
        controller.step(
            action = 'Teleport',
            position = sofa_pose['position'],
            rotation = sofa_pose['rotation'],
            horizon = int(sofa_pose['cameraHorizon']),
            standing = sofa_pose['isStanding']
        )
    else:
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
        
        for i in range(len(poses)):
            pose = poses[i]
            pose['horizon'] = 30
            # assert {'x':pose['x'], 'y':pose['y'], 'z':pose['z']} in avail_positions
            event = controller.step("TeleportFull", **pose)
            if event.metadata["lastActionSuccess"]:
                break
    
    event = controller.step('Pass')
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
    obj = [obj for obj in metadata["objects"] if object in obj['objectId']][0]
    try:
        receptacle = obj['parentReceptacles'][0]
    except:
        receptacle = ''
    print(receptacle)
    # sofa is low so set standing to false
    if 'Sofa' in receptacle:
        agent_pose = metadata['agent']
        # sofa_pose = {'name': 'agent', 'position': agent_pose['position'], 'rotation': agent_pose['rotation'], 'cameraHorizon': , 'isStanding': True, 'inHighFrictionArea': False}
        controller.step(
            action = 'Teleport',
            position = agent_pose['position'],
            rotation = agent_pose['rotation'],
            horizon = int(agent_pose['cameraHorizon']),
            standing = False
        )
        standing = False
        event = controller.step('MoveArmBase', y = 0.2)
    elif location == 'DiningTable':
        event = controller.step('MoveArmBase', y = 0.5)
        standing = True
    else:
        standing = True
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
        returnToStart=False
    )
    event = controller.step(
        action='MoveArm',
        position=dict(x=0, y=0, z=-0.25),
        coordinateSpace='wrist',
        speed=1,
        returnToStart=False
    )
    # event = controller.step('MoveArmBase', y = 0.5)
    # controller.step(action="SetHandSphereRadius", radius=0.04)
    if not standing:
        agent_pose = event.metadata['agent']
        # sofa_pose = {'name': 'agent', 'position': agent_pose['position'], 'rotation': agent_pose['rotation'], 'cameraHorizon': , 'isStanding': True, 'inHighFrictionArea': False}
        controller.step(
            action = 'Teleport',
            position = agent_pose['position'],
            rotation = agent_pose['rotation'],
            horizon = int(agent_pose['cameraHorizon']),
            standing = True
        )
    controller.step(action='MoveArmBase',y=0.5)
    controller.step('Done')

    
    return event

def DropAt(object, location, controller, metadata):
    '''
    location: str : name of the object to drop. Named location for LLM to reason
    '''
    # obj_names = [obj['objectId'] for obj in metadata["objects"]]
    obj = [obj for obj in metadata["objects"] if location in obj['objectId']][0]
    obj_pos = obj['position']
    hover_pos = dict(x=obj_pos['x'], y=obj_pos['y']+0.15, z=obj_pos['z'])
    event = controller.step(
        "MoveArm",
        position=hover_pos,
        coordinateSpace="world",
        returnToStart=False
    )
    event = controller.step(action="ReleaseObject")
    # make sure the thing is dropped
    MoveGripperBackward(controller)
    event = controller.step(action="SetHandSphereRadius", radius=0.15)
    return event

# not for the workshop. Too tricky to implement
def Open(controller):
    pass

def Close(controller):
    pass
