"""
Test script by Ziyi. Setup a simple scene with a chair and a table and a red block on top. Use IK solver to drive teh arm to the block for grasping.
Mostly reuse ik example and grasping example from the original simulator.
"""

import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options

## Copy and paste from grasping demo
GRASPING_MODES = dict(
    sticky="Sticky Mitten - Objects are magnetized when they touch the fingers and a CLOSE command is given",
    assisted="Assisted Grasping - Objects are magnetized when they touch the fingers, are within the hand, and a CLOSE command is given",
    physical="Physical Grasping - No additional grasping assistance applied",
)

# Don't use GPU dynamics and Use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot grasping mode demo with selection
    Queries the user to select a type of grasping mode
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose type of grasping
    grasping_mode = choose_from_options(options=GRASPING_MODES, name="grasping mode", random_selection=random_selection)

    # Create environment configuration to use
    scene_cfg = dict(type="Scene")
    robot0_cfg = dict(
        type="Fetch",
        obs_modalities=["rgb"],  # we're just doing a grasping demo so we don't need all observation modalities
        action_type="continuous",
        action_normalize=True,
        grasping_mode="sticky", # sticky | assisted | physical
    )

    # Define objects to load
    table_cfg = dict(
        type="DatasetObject",
        name="table",
        category="breakfast_table",
        model="lcsizg",
        bounding_box=[0.5, 0.5, 0.8],
        fixed_base=True,
        position=[0.7, -0.1, 0.6],
        orientation=[0, 0, 0.707, 0.707],
    )

    chair_cfg = dict(
        type="DatasetObject",
        name="chair",
        category="straight_chair",
        model="amgwaw",
        bounding_box=None,
        fixed_base=False,
        position=[0.45, 0.65, 0.425],
        orientation=[0, 0, -0.9990215, -0.0442276],
    )

    box_cfg = dict(
        type="PrimitiveObject",
        name="box",
        primitive_type="Cube",
        rgba=[1.0, 0, 0, 1.0],
        size=0.05,
        position=[0.53, -0.1, 0.97],
    )

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg], objects=[table_cfg, chair_cfg, box_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)

    # Reset the robot
    robot = env.robots[0]
    robot.set_position_orientation(position=[0, 0, 0])
    robot.reset()
    robot.keep_still() # what is the use of this line?

    # Make the robot's camera(s) high-res
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.image_height = 1440
            sensor.image_width = 1440

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-2.39951, 2.26469, 2.66227]),
        orientation=th.tensor([-0.23898481, 0.48475231, 0.75464013, -0.37204802]),
    )

    #TODO: Call IK to pick up the block

    og.clear()

