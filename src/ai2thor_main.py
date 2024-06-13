import numpy as np
import random
from PIL import Image

from ai2thor.controller import Controller

from utils import get_top_down_frame, GPT4, load_from_file

ENGINE = "gpt-4o" # Omni is better
PROMPT_DIR = "prompts/"

# import predefined skills
from manipula_skills import *
## high-level actions are GoTo, PickUp, Drop,Open, Close
## low-level actions are {Gripper}*{Up, Down, Left, Right, Forward, Backward}, {Base}*{Left, Right, Forward, Backward}


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
model = GPT4(engine=ENGINE)
basic_prompt = load_from_file(f"{PROMPT_DIR}basic_info.txt")

# top_down_view doesn't work yet, use something manually taken for now
top_down_img = ["test_imgs/pickup.png"]
basic_info = model.generate_multimodal(basic_prompt, top_down_img)

# propose predicates

## text-only



