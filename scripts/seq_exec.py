from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import cv2
import yaml
import rclpy
from rclpy.node import Node
import time
from everest_real_client import Executor
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import re


###############################################################################
# User‑adjustable parameters                                                   #
###############################################################################
YAML_PATH = Path("skill_sequences.yaml")
IMAGE_ROOT = Path("skill_images")
IMAGE_ROOT.mkdir(exist_ok=True)

Skill = Tuple[str]
Sequence = List[Skill]

SEQUENCES: List[Sequence] = [
    # seq 1
    [
        'Pick(Robot, Teapot)', 
        'Pour(Robot, Teapot, Mug)', 
        'PlaceBack(Robot, Teapot)', 
        'Pick(Robot, Sponge)', 
        'Wipe(Robot, Sponge, Plate)', 
        'PlaceBack(Robot, Sponge)', 
        'Pick(Robot, Bowl)', 
        'Stack(Robot, Bowl, Plate)', 
        'Stack(Robot, Bowl, Plate)', 
        'Pick(Robot, Sponge)', 
        'Wipe(Robot, Sponge, Plate)'
    ],
    # seq 2
    [
        'Pick(Robot, Teapot)', 
        'Pour(Robot, Teapot, Mug)', 
        'Pick(Robot, Sponge)', 
        'Wipe(Robot, Sponge, Plate)', 
        'PlaceBack(Robot, Bowl)', 
        'Pick(Robot, Bowl)', 
        'Stack(Robot, Bowl, Plate)', 
        'Pour(Robot, Teapot, Mug)', 
        'PlaceBack(Robot, Teapot)', 
        'Wipe(Robot, Sponge, Plate)', 
        'Pick(Robot, Bowl)', 
        'Stack(Robot, Bowl, Plate)'
    ]
]


def extract_skill_and_args(skill_string):
    match = re.match(r"(\w+)\(Robot, ?([^)]+)\)", skill_string)
    if match:
        skill = match.group(1)
        args_str = match.group(2)
        args_list = [arg.strip() for arg in args_str.split(',')]
        arg_vars = {}
        for i, arg in enumerate(args_list):
            arg_vars[f"arg{i+1}"] = arg
        return skill, arg_vars

###############################################################################
# Image helpers
###############################################################################

class ImageClient(Node):
    def __init__(self):
        super().__init__('image_client')
        self.bridge = CvBridge()
        

    def capture_image(self, topic: str, timeout_sec: float = 5.0) -> np.ndarray:
        """
        Subscribe once to a camera topic and return the OpenCV image.
        Raises TimeoutError if no image arrives in time.
        """
        img = None

        def _cb(msg):
            nonlocal img
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        sub = self.create_subscription(Image, topic, _cb, 10)
        start = time.time()
        while rclpy.ok() and img is None and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.destroy_subscription(sub)

        if img is None:
            raise TimeoutError(f"No image on '{topic}' after {timeout_sec}s")
        return img


###############################################################################
# YAML persistence                                                             #
###############################################################################

def load_yaml() -> Dict:
    if YAML_PATH.exists():
        with open(YAML_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml(data: Dict):
    with open(YAML_PATH, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=True)


###############################################################################
# Progress helpers                                                             #
###############################################################################

def find_resume_point(data: Dict) -> Tuple[int, int]:
    """Return (seq_idx, step_idx) to resume from (both 0‑based)."""
    for s_idx, seq in enumerate(SEQUENCES):
        task_key = f"seq_{s_idx+1}"
        if task_key not in data:
            return s_idx, 0
        steps = data[task_key]
        # If step not yet logged, resume there
        for st_idx in range(len(seq) + 1):  # +1 for step 0
            if str(st_idx) not in steps:
                return s_idx, st_idx
    return len(SEQUENCES), 0  # done

###############################################################################
# Main routine                                                                 #
###############################################################################

def main():
    time.sleep(3)
    rclpy.init()
    node = ImageClient()

    data = load_yaml()
    seq_i, step_i = find_resume_point(data)
    if seq_i >= len(SEQUENCES):
        print("All sequences complete! Nothing to do.")
        return

    for s_idx in range(seq_i, len(SEQUENCES)):
        exec = Executor()

        while not (exec.latest_rgb and exec.latest_depth and exec.latest_info):
            rclpy.spin_once(exec, timeout_sec=0.1)

        task_key = f"seq_{s_idx+1}"
        seq = SEQUENCES[s_idx]
        data.setdefault(task_key, {})
        print(f"\n=== Starting {task_key} ===")
        # ─── make a folder for this sequence ───
        sequence_dir = IMAGE_ROOT / task_key
        sequence_dir.mkdir(exist_ok=True)

        for st_idx in range((step_i if s_idx == seq_i else 0), len(seq) + 1):
            step_dict = {}

            # step 0 has no skill string / success flag from list
            if st_idx == 0:
                skill_str = None
                success: Optional[bool] = None
                img_path = None
            else:
                skill_str = seq[st_idx - 1]
                skill_str = skill_str
                success = input(f"Enter next skill success ({skill_str}) (True/False): ").strip().lower() == "true"
            
            
            print(f"\n[{task_key} | step {st_idx}] Skill: {skill_str if skill_str else 'INITIAL'} | success label: {success}")

            if st_idx != 0:
                # ─── run the skill ─────────────────────────────────────────────
                if success:
                    skill, skill_args = extract_skill_and_args(skill_str)
                                 
                    if skill == 'Pick':
                        if skill_args['arg1'] == "Teapot":
                            skill_args['arg1'] = 'purple_teapot'
                            if skill_args['arg1'].lower() not in exec.obj2pose:
                                exec.get_pose(skill_args['arg1'].lower())
                            exec.pick(skill_args['arg1'].lower())
                        elif skill_args['arg1'] == "Sponge":
                            exec.replay_demo(exec.PICK_SPONGE_DEMO_PATH, skip_threshold=0.02)
                        else:
                            if skill_args['arg1'].lower() not in exec.obj2pose:
                                exec.get_pose(skill_args['arg1'].lower())
                            exec.pick(skill_args['arg1'].lower())
                    elif skill == 'PlaceBack':
                        if skill_args['arg1'] == "Teapot":
                            skill_args['arg1'] = 'purple_teapot'
                            exec.place(skill_args['arg1'].lower())
                        elif skill_args['arg1'] == "Sponge":
                            exec.replay_demo(exec.PLACE_SPONGE_DEMO_PATH)
                        else:
                            exec.place(skill_args['arg1'].lower())
                    elif skill == 'Pour':
                        if skill_args['arg2'] == "Mug":
                            skill_args['arg2'] = 'white_mug'
                        exec.pour(skill_args['arg2'].lower())
                    elif skill == 'Wipe':
                        exec.replay_demo(exec.WIPE_DEMO_PATH)
                    elif skill == 'Stack':
                        exec.stack()
                    else:
                        raise NotImplementedError
                    exec.go_home()

            # ─── capture image ─────────────────────────────────────────────
            img_name = f"{task_key}_step{st_idx}_img.jpg"
            img_path = sequence_dir / img_name
            img = node.capture_image("/dave/dave_camera/color/image_raw")
            cv2.imwrite(str(img_path), img)
            print(f"Captured ⇒ {str(img_path)}")

            # record to dict
            step_dict['image'] = str(img_path)
            step_dict["skill"] = skill_str
            step_dict["success"] = success

            data[task_key][str(st_idx)] = step_dict
            save_yaml(data)
            print("Step saved. YAML checkpoint updated.")

        step_i = 0  # reset for next sequence
        print(f"=== Finished {task_key} ===\n")

    print("\nAll sequences finished! Data stored in", YAML_PATH)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted – progress saved to YAML. Bye!")
