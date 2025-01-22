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

def GoTo(object_or_location_1, object_or_location_2, controller, event):
    '''
    Teleport to a fixed location associated with an object.
    It now only works with FloorPlan203

    '''
    def dist_pose(obj1, obj2):
        x1, y1, z1 = obj1["x"], obj1["y"], obj1["z"]
        x2, y2, z2 = obj2["x"], obj2["y"], obj2["z"]
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        return np.sqrt(np.sum((p1-p2)**2, axis=0))
    
    if object_or_location_1 == object_or_location_2: # cannot go to same place
        return False, event
    
    metadata = event.metadata
    # start = [obj for obj in metadata["objects"] if object_or_location_1 in obj['objectId']][0]
    event = controller.step('Pass')
    pose_dict = {'Sofa': {'x': -0.1749999225139618, 'y': 0.9070531129837036, 'z': 3.083493709564209},
                 'DiningTable': {'x': -4.324999809265137, 'y': 0.9070531129837036, 'z': 0.5165063142776489},
                 'CoffeeTable': {'x': -0.3915063440799713, 'y': 0.9070531129837036, 'z': 2.458493709564209}}
    for loc in pose_dict:
        if object_or_location_1 == loc and dist_pose(event.metadata['agent']['position'], pose_dict[loc]) > 0.1: # cannot start at a place far away
            # breakpoint()
            return False, event

    obj = [obj for obj in metadata["objects"] if object_or_location_2 in obj['objectId']][0]
    try:
        receptacle = obj['parentReceptacles'][0]
    except:
        receptacle = ''

    if "Sofa" in object_or_location_2 or "Sofa" in receptacle:
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
    elif "DiningTable" in object_or_location_2:
        dining_table_pose = {'name': 'agent', 'position': {'x': -4.324999809265137, 'y': 0.9070531129837036, 'z': 0.5165063142776489}, 'rotation': {'x': -0.0, 'y': 180.0, 'z': 0.0}, 'cameraHorizon': 30.000003814697266, 'isStanding': True, 'inHighFrictionArea': False}
        controller.step(
            action = 'Teleport',
            position = dining_table_pose['position'],
            rotation = dining_table_pose['rotation'],
            horizon = int(dining_table_pose['cameraHorizon']),
            standing = dining_table_pose['isStanding']
        )
    elif "CoffeeTable" in object_or_location_2:
        coffee_table_pose = {'name': 'agent', 'position': {'x': -0.3915063440799713, 'y': 0.9070531129837036, 'z': 2.458493709564209}, 'rotation': {'x': -0.0, 'y': 90.0, 'z': 0.0}, 'cameraHorizon': 30.000003814697266, 'isStanding': True, 'inHighFrictionArea': False}
        controller.step(
            action = 'Teleport',
            position = coffee_table_pose['position'],
            rotation = coffee_table_pose['rotation'],
            horizon = int(coffee_table_pose['cameraHorizon']),
            standing = coffee_table_pose['isStanding']
        )
    else:
        # obj_names = [obj['objectId'] for obj in metadata["objects"]]
        obj = [obj for obj in metadata["objects"] if object_or_location_2 in obj['objectId']][0]
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
    success = dist_pose(event.metadata['agent']['position'], obj['position']) < 2
    return success, event

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


def PickUp(object, location, controller, event):
    '''
    object: str : name of the object
    '''
    # add try so will do nothing for unchainable tasks
    metadata = event.metadata
    obj = [obj for obj in metadata["objects"] if object in obj['objectId']][0]
    try:
        receptacle = obj['parentReceptacles'][0]
        if not location in receptacle:
            return False, event
    except:
        receptacle = ''

    try:
        for _ in range(2):

            if metadata['arm']['heldObjects'] and _ == 0:
                return False, event
            print('receptacle', receptacle)
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
                event = controller.step('MoveArmBase', y = 0.2)

            elif 'CoffeeTable' in receptacle:
                agent_pose = metadata['agent']
                # sofa_pose = {'name': 'agent', 'position': agent_pose['position'], 'rotation': agent_pose['rotation'], 'cameraHorizon': , 'isStanding': True, 'inHighFrictionArea': False}
                controller.step(
                    action = 'Teleport',
                    position = agent_pose['position'],
                    rotation = agent_pose['rotation'],
                    horizon = int(agent_pose['cameraHorizon']),
                    standing = True
                )
                event = controller.step('MoveArmBase', y = 0.3)

            controller.step(action="SetHandSphereRadius", radius=0.06)
            obj = [obj for obj in metadata["objects"] if object in obj['objectId']][0]
            position = deepcopy(obj["position"])
            # if "Book" in object:
            #     position = {'x': position['x']+0.1, 'y': position['y'] + 0.05, 'z': position['z']-0.2}

            event = controller.step(
                "MoveArm",
                position=position,
                coordinateSpace="world",
                returnToStart=False
            )
            if "Book" in object:
                event = controller.step(
                    action='MoveArm',
                    position=dict(x=0.15, y=0, z=-0.2),
                    coordinateSpace='wrist',
                    speed=1,
                    returnToStart=False
                )

            elif "Vase" in object:
                event = controller.step(
                    action='MoveArm',
                    position=dict(x=0, y=0.15, z=-0.1),
                    coordinateSpace='wrist',
                    speed=1,
                    returnToStart=False
                )

            elif "TissueBox" in object:
                event = controller.step(
                    action='MoveArm',
                    position=dict(x=-0.05, y=0.02, z=-0.1),
                    coordinateSpace='wrist',
                    speed=1,
                    returnToStart=False
                )

            elif "Bowl" in object:
                event = controller.step(
                    action='MoveArm',
                    position=dict(x=0.05, y=0, z=-0.12),
                    coordinateSpace='wrist',
                    speed=1,
                    returnToStart=False
                )
            # if _ == 1:
            event = controller.step(
                action="PickupObject",
                objectIdCandidates=[obj['objectId']],
            )
            agent_pose = event.metadata['agent']
            while not agent_pose['isStanding']:
                event = controller.step(
                    action = 'Teleport',
                    position = agent_pose['position'],
                    rotation = agent_pose['rotation'],
                    horizon = int(agent_pose['cameraHorizon']),
                    standing = True
                )
                agent_pose = event.metadata['agent']
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
            controller.step(action='MoveArmBase',y=0.5)
            event = controller.step(
                action="MoveArm",
                position=dict(x=-0.25, y=0.45, z=0.1),
                coordinateSpace="armBase",
                restrictMovement=False,
                speed=1,        
                returnToStart=True,        
                # fixedDeltaTime=0.02    
                )
            event = controller.step('Done')

        success = True if obj["objectId"] in event.metadata['arm']['heldObjects'][0] else False
        return success, event
    except:
        return False, event

def DropAt(object, location, controller, event):
    '''
    location: str : name of the object to drop. Named location for LLM to reason
    '''
    def dist_pose(obj1, obj2):
        x1, y1, z1 = obj1["x"], obj1["y"], obj1["z"]
        x2, y2, z2 = obj2["x"], obj2["y"], obj2["z"]
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        return np.sqrt(np.sum((p1-p2)**2, axis=0))
    
    try:
        pose_dict = {'Sofa': {'x': -0.1749999225139618, 'y': 0.9070531129837036, 'z': 3.083493709564209},
                    'DiningTable': {'x': -4.324999809265137, 'y': 0.9070531129837036, 'z': 0.5165063142776489},
                    'CoffeeTable': {'x': -0.3915063440799713, 'y': 0.9070531129837036, 'z': 2.458493709564209}}
        for loc in pose_dict:
            if location == loc and dist_pose(event.metadata['agent']['position'], pose_dict[loc]) > 0.1: # cannot drop at a place far away
                # breakpoint()
                return False, event
            
        metadata = event.metadata
        if not metadata['arm']['heldObjects']:
            return False, event
        if object not in metadata['arm']['heldObjects'][0]:
            return False, event
        loc = [obj for obj in metadata["objects"] if location in obj['objectId']][0]
        loc_pos = loc['position']
        # hover_pos = dict(x=loc_pos['x'], y=loc_pos['y']+2, z=loc_pos['z'])
        # event = controller.step(
        #     "MoveArm",
        #     position=hover_pos,
        #     coordinateSpace="world",
        #     returnToStart=False
        # )
        # event = controller.step(
        #     "MoveArm",
        #     position=dict(x=0,y=-0.5,z=0),
        #     coordinateSpace="wrist",
        #     returnToStart=False
        # )
        if "DiningTable" in location:
            event = controller.step(
                "MoveArm",
                position=dict(x=-0.2,y=-0.2,z=0.45),
                coordinateSpace="armBase",
                returnToStart=False
            )
        elif "CoffeeTable" in location:
            event = controller.step(
                "MoveArm",
                position=dict(x=-0.3,y=-0.25,z=0.5),
                coordinateSpace="armBase",
                returnToStart=False
            )
        elif "Sofa" in location:
            event = controller.step(
                "MoveArm",
                position=dict(x=-0.1,y=-0.5,z=0.45),
                coordinateSpace="armBase",
                returnToStart=False
            )
        event = controller.step(action="ReleaseObject")
        # make sure the thing is dropped
        controller.step(
            action="MoveArm",
            position=dict(x=0, y=0, z=0.4),
            coordinateSpace="armBase",
            restrictMovement=False,
            speed=1,
            returnToStart=True,
            fixedDeltaTime=0.02
        )
        # MoveGripperBackward(controller)
        event = controller.step(action="SetHandSphereRadius", radius=0.04)
        for i in range(10):
            controller.step(    
                action="AdvancePhysicsStep",    
                timeStep=0.05)
        event = No_op(controller)
        obj = [obj for obj in event.metadata["objects"] if object in obj['objectId']][0]
        # print('obj')
        receptacle = obj['parentReceptacles'][0]
        # print(receptacle)
        success = True if location in receptacle else False
        return success, event
    except:
        controller.step(
            action="MoveArm",
            position=dict(x=0, y=0, z=0.4),
            coordinateSpace="armBase",
            restrictMovement=False,
            speed=1,
            fixedDeltaTime=0.02
        )
        event = controller.step(action="SetHandSphereRadius", radius=0.04)
        event = No_op(controller)
        return False, event

# not for the workshop. Too tricky to implement
def Open(controller):
    pass

def Close(controller):
    pass
