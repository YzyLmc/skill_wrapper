import numpy as np
import random
from PIL import Image

from ai2thor.controller import Controller

from utils import get_top_down_frame, GPT4

# first collect images of the environment
controller = Controller(
                massThreshold = 1,
                agentMode="arm",
                scene = "FloorPlan212",
                snapToGrid=False,
                visibilityDistance=1.5,
                gridSize=0.25,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                renderObjectImage = True,
                width= 1280,
                height= 720,
                fieldOfView=60
            )
controller.reset(scene="FloorPlan212", fieldOfView=100)

top_down_img = get_top_down_frame(controller)
top_down_img.show()

# get basic information of the environment

