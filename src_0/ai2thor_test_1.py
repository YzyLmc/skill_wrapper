import numpy as np
import random
from PIL import Image

from ai2thor.controller import Controller

from manipula_skills import *

# test open action
# def open_obj(obj_id):

def get_obj(obj_id, obj_list):
    return [obj for obj in  event.metadata["objects"] if obj["objectId"] == obj_id]

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
controller.reset(scene="FloorPlan203", fieldOfView=100)

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

pickupable_objs = [obj['objectId'] for obj in  event.metadata["objects"] if obj["pickupable"]]
# find openable object
openable_objs = [obj['objectId'] for obj in  event.metadata["objects"] if obj["openable"]]
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
