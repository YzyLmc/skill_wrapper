import numpy as np
import random
from PIL import Image

from ai2thor.controller import Controller

# test open action
# def open_obj(obj_id):

def get_obj(obj_id, obj_list):
    return [obj for obj in  event.metadata["objects"] if obj["objectId"] == obj_id]

def dist_pose(obj1, obj2):
    x1, y1, z1 = obj1["x"], obj1["y"], obj1["z"]
    x2, y2, z2 = obj2["x"], obj2["y"], obj2["z"]
    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])

    return np.sqrt(np.sum((p1-p2)**2, axis=0))

def LookAt(metadata, object):
    pass

# init ai2thor controller
controller = Controller(
    agentMode="arm",
    massThreshold=None,
    scene="FloorPlan203",
    visibilityDistance=1.5,
    gridSize=0.25,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=1000,
    height=1000,
    fieldOfView=100
)
controller.reset(scene="FloorPlan212", fieldOfView=100)

controller.step(
    action="SetHandSphereRadius",
    radius=0.5
)

# controller = Controller()
event = controller.step("MoveAhead")
# get event, for metadata
# event = controller.step(
#     action="MoveAgent",
#     ahead=0.25,
#     right=0.25,
#     returnToStart=True,
#     speed=1,
#     fixedDeltaTime=0.02
# )

# find openable object
openable_objs = [obj['objectId'] for obj in  event.metadata["objects"] if obj["openable"]]
pickupable_objs = [obj['objectId'] for obj in  event.metadata["objects"] if obj["pickupable"]]
breakpoint()

# use drawer as target
# drawer_id = 'Drawer|+00.60|+00.23|+02.60'
drawer_id = [obj for obj in openable_objs if "Drawer" in obj][0]
remote_id = [obj for obj in pickupable_objs if "RemoteControl" in obj][0]
laptop_id = [obj for obj in pickupable_objs if "Laptop" in obj][0]
creditcard_id = [obj for obj in pickupable_objs if "CreditCard" in obj][0]
# find a pose to interact
event = controller.step(
    action="GetInteractablePoses",
    # objectId=drawer_id,
    objectId=laptop_id,
    horizons=np.linspace(-30, 60, 30),
    standings=[True, False]
)
poses = event.metadata["actionReturn"]

# teleport to interactable pose
# pose = random.choice(poses)

drawer = get_obj(drawer_id, event.metadata["objects"])[0]
laptop = get_obj(laptop_id, event.metadata["objects"])[0]
creditcard = get_obj(creditcard_id, event.metadata["objects"])[0]

pose = min(poses, key = lambda p:dist_pose(p, drawer['position']))
pose = min(poses, key = lambda p:dist_pose(p, laptop['position']))
pose = min(poses, key = lambda p:dist_pose(p, creditcard['position']))
pose = min(poses, key = lambda p:dist_pose(p, drawer['position']))

event = controller.step("Teleport", **pose)
# controller.step("Teleport", **pose)
im = Image.fromarray(event.frame)
im.show()
# print state of the drawer

breakpoint() # drawer should be closed
im = Image.fromarray(event.frame)
im.show()
# move to the drawer
event = controller.step("MoveArm",
                # position=laptop_pos,
                position=drawer['position'],
                coordinateSpace="world",
                returnToStart=False)
breakpoint()
# open it
event = controller.step(
    action="OpenObject",
    objectId=drawer_id,
    openness=1,
    forceAction=False
)

breakpoint()

drawer = get_obj(drawer_id, event.metadata["objects"])[0]
remote = get_obj(remote_id, event.metadata["objects"])[0]
laptop = get_obj(laptop_id, event.metadata["objects"])[0]
creditcard = get_obj(creditcard_id, event.metadata["objects"])[0]

remote_pos = remote["position"]
laptop_pos = laptop['position']
creditcard_pos = creditcard["position"]

event = controller.step("MoveArm",
                # position=laptop_pos,
                position=creditcard_pos,
                coordinateSpace="world",
                returnToStart=False)

im = Image.fromarray(event.frame)
im.show()         
breakpoint()    
# Pick the remote up
event = controller.step(
    action="PickupObject",
    # objectId=laptop_id,
    objectIdCandidates=[creditcard_id],
)
# see if it's opened
drawer = get_obj(drawer_id, event.metadata["objects"])
remote = get_obj(remote_id, event.metadata["objects"])
laptop = get_obj(laptop_id, event.metadata["objects"])


im = Image.fromarray(event.frame)
im.show()

breakpoint()
event = controller.step("Teleport", **pose)
# event = controller.step()
im = Image.fromarray(event.frame)
im.show()
