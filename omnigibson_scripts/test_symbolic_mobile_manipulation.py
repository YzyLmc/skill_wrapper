import os

import torch as th
import yaml

import omnigibson as og
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from omnigibson import object_states

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True


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
    config["scene"]["scene_model"] = "Benevolence_1_int"
    config["scene"]["load_task_relevant_only"] = True
    config["scene"]["not_load_object_categories"] = ["ceilings"]
    config["task"] = {
        "type": "BehaviorTask",
        "activity_name": "picking_up_trash",
        "activity_definition_id": 0,
        "activity_instance_id": 0,
        "predefined_problem": None,
        "online_object_sampling": False,
    }

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = SymbolicSemanticActionPrimitives(env)

    # Navigate tp can of soda
    grasp_obj = env.task.object_scope["can__of__soda.n.01_2"]
    trash = env.task.object_scope["ashcan.n.01_1"]
    # breakpoint()
    # execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO, grasp_obj), env)

    # Grasp can of soda
    
    print("Executing controller")
    execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, grasp_obj), env)
    breakpoint()
    print("Finished executing grasp")

    # Place can in trash can
    print("Executing controller")
    trash = env.task.object_scope["ashcan.n.01_1"]
    # execute_controller(controller.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_INSIDE, trash), env)
    execute_controller(controller._place_with_predicate(trash, object_states.OnTop), env)
    # breakpoint()
    print("Finished executing place")
    for i in range(100):
        execute_controller(controller._settle_robot(), env)



if __name__ == "__main__":
    main()
