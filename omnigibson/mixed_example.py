import os

import torch as th
import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson_scripts.omnigibson_primitive_actions.semantic_action_primitives import (
    SemanticActionPrimitives,
    SemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omnigibson import object_states

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def main():
    """
    Demonstrates how to use the action primitives to solve a simple BEHAVIOR-1K task.

    It loads Benevolence_1_int with a robot, and the robot attempts to solve the
    picking_up_trash task using a hardcoded sequence of primitives.
    """
    # Load the config
    config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to run a grocery shopping task
    # config["scene"]["scene_model"] = "Benevolence_1_int"
    # config["scene"]["load_task_relevant_only"] = True
    # config["scene"]["not_load_object_categories"] = ["ceilings"]
    # config["task"] = {
    #     "type": "BehaviorTask",
    #     "activity_name": "picking_up_trash",
    #     "activity_definition_id": 0,
    #     "activity_instance_id": 0,
    #     "predefined_problem": None,
    #     "online_object_sampling": False,
    # }
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        },
    }
    # config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table", "breakfast_table"]
    config["scene"]["load_object_categories"] = [
        'oven', 'toilet', 'sofa', 'trash_can', 'electric_switch', 'standing_tv', 
        'ceilings', 'sink', 'walls', 'top_cabinet', 'microwave', 'shelf', 'laptop',
        'table_lamp', 'countertop', 'coffee_table', 'door', 'mirror', 
        'picture', 'dishwasher', 'bottom_cabinet', 'floors', 'fridge', 'floor_lamp', 
        'towel_rack', 'shower_stall', 'bed', 
        'window', 'loudspeaker', 'breakfast_table']
    config["objects"] = [
        dict(
        type="DatasetObject",
        name="toy_dice",
        category="toy_dice",
        model="akguod",
        rgba=[1.0, 0, 0, 1.0],
        size=0.05,
        position=[-0.3, -1.0, 1.0],
    ),
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "scale": [0.3, 0.3, 0.3],
            "position": [-0.7, 0.5, 0.2],
            "orientation": [0, 0, 0, 1],
        },
        # {
        #     "type": "DatasetObject",
        #     "name": "cologne",
        #     "category": "bottle_of_cologne",
        #     "model": "lyipur",
        #     "position": [-0.15, -0.4, 0.],
        #     "orientation": [0, 0, 0, 1],
        # }
    ]
    # config['robots'][0]['grasping_mode'] = 'assisted'
    config['robots'][0]['grasping_mode'] = 'sticky'

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]
    # Make the robot's camera(s) high-res
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.image_height = 1440
            sensor.image_width = 1440

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()
    
    # motion_controller = SemanticActionPrimitives(env, enable_head_tracking=False)
    controller = SemanticActionPrimitives(env)

    # Navigate tp can of soda
    # grasp_obj = env.task.object_scope["can__of__soda.n.01_2"]
    # trash = env.task.object_scope["ashcan.n.01_1"]
    grasp_obj = scene.object_registry("name", "toy_dice")
    target_table = scene.object_registry("name", "breakfast_table")
    cabinet = scene.object_registry("name", "bottom_cabinet")
    breakpoint()
    # grasp_obj.set_position_orientation(orientation=th.tensor([0.0093, 0.0149, 0.3568, -0.9340]))
    breakpoint()
    # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.NAVIGATE_TO, (grasp_obj, grasp_obj.states[object_states.Pose].get_value()), attempts=5), env)

    # Grasp can of soda
    print("Heading to pick-up position")
    execute_controller(controller.apply_ref(SemanticActionPrimitiveSet.NAVIGATE_TO_OBJ_TELEPORT, grasp_obj, attempts=1), env)
    # execute_controller(controller.apply_ref(SemanticActionPrimitiveSet.NAVIGATE_TO_POSE_TELEPORT, (-9.0344e-02, -3.3580e+00, 0.5)), env)

    # test pose to teleport or navigate to
    # test_pose = (th.tensor([-9.0344e-02, -3.3580e+00,  7.5462e-04]), th.tensor([ 0.0093,  0.0149,  0.3568, -0.9340]))
    # controller.robot.set_position_orientation(*test_pose)
    # execute_controller(controller.apply_ref(SemanticActionPrimitiveSet.NO_OP), env)
    breakpoint()
    print("Executing controller")
    # execute_controller(controller.apply_ref(SemanticActionPrimitiveSet.GRASP, grasp_obj, attempts=1), env)
    execute_controller(controller.apply_ref(SemanticActionPrimitiveSet.GRASP_TELEPORT, grasp_obj, attempts=3), env)
    print("Finished executing grasp")

    # Heading to trash can
    print("Heading to place position")
    execute_controller(controller.apply_ref(SemanticActionPrimitiveSet.NAVIGATE_TO_OBJ_TELEPORT, target_table, attempts=1), env)
    # Place can in trash can
    breakpoint()
    # print("Executing controller")
    execute_controller(controller.apply_ref(SemanticActionPrimitiveSet.PLACE_ON_TOP, target_table, attempts=3), env)
    # execute_controller(controller.apply_ref(SemanticActionPrimitiveSet.PLACE_ON_TOP_TELEPORT, target_table, attempts=3), env)
    # print("Finished executing place")


if __name__ == "__main__":
    main()
