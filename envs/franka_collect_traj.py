import yaml
import json
import os
import cv2
import numpy as np
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datetime import datetime
from franka_env import FrankaEnv
from src.data_structure import Skill


class FrankaTrajectoryCollector:
    def __init__(self):
        self.env = FrankaEnv()
        self.yaml_domain_path = "../../task_config/franka.yaml"
        self.task_config = self.load_yaml(self.yaml_domain_path)
        
        # Action mapping based on franka environment actions
        self.action_to_skill = {
            0: ["Pick", "Robot", "Teapot"],              # Pick teapot
            1: ["Pick", "Robot", "Bowl"],                # Pick bowl  
            2: ["Pick", "Robot", "Sponge"],              # Pick sponge
            3: ["Place", "Robot", "Teapot"],             # Place teapot
            4: ["Place", "Robot", "Bowl"],               # Place bowl
            5: ["Place", "Robot", "Sponge"],             # Place sponge
            6: ["Stack", "Robot", "Teapot", "Plate"],    # Stack teapot on plate
            7: ["Stack", "Robot", "Bowl", "Plate"],      # Stack bowl on plate
            8: ["Stack", "Robot", "Sponge", "Plate"],    # Stack sponge on plate
            9: ["Pour", "Robot", "Teapot", "Mug"],       # Pour teapot into mug
            10: ["Wipe", "Robot", "Sponge", "Plate"]     # Wipe plate with sponge
        }
        
        # Skill string to action mapping for trajs_to_use.yaml
        self.skill_to_action = {
            "Pick(Robot, Teapot)": 0,
            "Pick(Robot, Bowl)": 1,
            "Pick(Robot, Sponge)": 2,
            "Place(Robot, Teapot)": 3,
            "Place(Robot, Bowl)": 4,
            "Place(Robot, Sponge)": 5,
            "Stack(Robot, Teapot, Plate)": 6,
            "Stack(Robot, Bowl, Plate)": 7,
            "Stack(Robot, Sponge, Plate)": 8,
            "Pour(Robot, Teapot, Mug)": 9,
            "Wipe(Robot, Sponge, Plate)": 10
        }
    
    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    
    def save_image(self, image, output_dir, filename):
        """Save image to the specified directory"""
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, filename)
        cv2.imwrite(image_path, image)
        return filename
    
    def load_trajectories_from_yaml(self, yaml_path="trajs_to_use.yaml"):
        """Load trajectories from trajs_to_use.yaml file"""
        return self.load_yaml(yaml_path)
    
    def convert_skill_seq_to_actions(self, skill_sequence):
        """Convert skill strings to action integers"""
        actions = []
        for skill in skill_sequence:
            skill_str = str(skill)
            if skill_str in self.skill_to_action:
                actions.append(self.skill_to_action[skill_str])
            else:
                raise ValueError(f"Unknown skill: {skill_str}")
        return actions
    
    def collect_trajectory(self, action_sequence, traj_name=None):
        """
        Collect trajectory by executing action sequence in the environment
        
        Args:
            action_sequence: List of action integers (0-10)
            traj_name: Name for the trajectory (for subfolder)
            
        Returns:
            Dict containing trajectory data in the format expected by convert_franka_data.py
        """
        time_now = datetime.now()
        timestamp = str(time_now.year) + "-" + str(time_now.month) + "-" + str(time_now.day) + "-" + str(time_now.hour) + "-" + str(time_now.minute) + "-" + str(time_now.second)
        data_folder = "transitions"
        output_dir = f"{data_folder}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Reset environment and get initial state
        obs, info = self.env.reset()
        
        trajectory_data = {
            "time_stamp": timestamp,
            "seq": []
        }
        
        # Save initial state image
        pre_img_name = f"0.png"
        self.save_image(obs, output_dir, pre_img_name)
        
        # Execute each action in the sequence
        for i, action in enumerate(action_sequence):
            if action not in self.action_to_skill:
                raise ValueError(f"Invalid action: {action}. Must be 0-10.")
            
            # Get pre-action state
            pre_state = self.env.state.copy()
            pre_image = obs.copy()
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Get post-action state  
            post_state = self.env.state.copy()
            success = info['success']
            
            # Save post-action image
            post_img_name = f"{i + 1}.png"
            self.save_image(obs, output_dir, post_img_name)
            
            # Create trajectory entry
            traj_entry = {
                "action": self.get_action_name(action),
                "pre_img_name": pre_img_name if i == 0 else f"post_{(i-1):05d}.png",
                "post_img_name": post_img_name,
                "pre_state": pre_state.tolist(),
                "post_state": post_state.tolist(),
                "success": success,
                "reward": reward
            }
            
            trajectory_data["seq"].append(traj_entry)
            
            if terminated:
                print(f"Episode terminated at step {i+1}")
                break
        
        # Generate skill wrapper format YAML
        skill_wrapper_data = self.convert_to_skill_wrapper_format(trajectory_data, output_dir)
        # Save skill wrapper YAML
        yaml_path = os.path.join(data_folder, "tasks.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                existing_data = yaml.load(f, Loader=yaml.FullLoader) or {}
            existing_data.update(skill_wrapper_data)
            skill_wrapper_data = existing_data
        with open(yaml_path, 'w') as f:
            yaml.dump(skill_wrapper_data, f, default_flow_style=False)
        
        print(f"Trajectory collected and saved to: {output_dir}")
        print(f"Total steps: {len(trajectory_data['seq'])}")
        
        return trajectory_data, skill_wrapper_data
    
    def get_action_name(self, action):
        """Convert action integer to action name string"""
        action_names = {
            0: "pick_teapot",
            1: "pick_bowl", 
            2: "pick_sponge",
            3: "place_teapot",
            4: "place_bowl",
            5: "place_sponge",
            6: "stack_teapot_on_plate",
            7: "stack_bowl_on_plate",
            8: "stack_sponge_on_plate",
            9: "pour_teapot_into_mug",
            10: "wipe_plate_with_sponge"
        }
        return action_names[action]
    
    def convert_to_skill_wrapper_format(self, trajectory_data, output_dir):
        """
        Convert trajectory data to skill wrapper format matching convert_franka_data.py output
        """
        timestamp = trajectory_data["time_stamp"]
        seq = trajectory_data["seq"]
        
        skill_wrapper_result = {}
        dir_prefix = f"results/{args.baseline}/franka/runs/{args.run_idx}/{args.iter_idx}_partial"
        
        # Initial state (step 0)
        skill_wrapper_result[str(0)] = {
            "image": f"{dir_prefix}/transitions/{timestamp}/0.png",
            "skill": None,
            "success": None
        }
        
        # Process each trajectory step
        for i, traj in enumerate(seq):
            action_int = self.get_action_int_from_name(traj["action"])
            skill_info = self.action_to_skill[action_int]
            skill_name = skill_info[0]
            parameters = skill_info[1:]
            types = self.task_config['skills'][skill_name].types
            
            skill = Skill(name=skill_name, params=parameters, types=types)
            
            skill_wrapper_result[str(i + 1)] = {
                "skill": skill,
                "image": f"{dir_prefix}/transitions/{timestamp}/{i+1}.png", 
                "success": traj["success"]
            }
        
        return {timestamp: skill_wrapper_result}
    
    def get_action_int_from_name(self, action_name):
        """Convert action name back to integer"""
        name_to_int = {
            "pick_teapot": 0,
            "pick_bowl": 1,
            "pick_sponge": 2, 
            "place_teapot": 3,
            "place_bowl": 4,
            "place_sponge": 5,
            "stack_teapot_on_plate": 6,
            "stack_bowl_on_plate": 7,
            "stack_sponge_on_plate": 8,
            "pour_teapot_into_mug": 9,
            "wipe_plate_with_sponge": 10
        }
        return name_to_int[action_name]

    def collect_all_trajectories(self):
        """
        Load trajectories from trajs_to_use.yaml and collect data for each one
        """
        trajectories = [self.load_trajectories_from_yaml(args.traj_yaml)]
        all_results = {}

        for skill_sequence in trajectories:
            time_now = datetime.now()
            timestamp = str(time_now.year) + "-" + str(time_now.month) + "-" + str(time_now.day) + "-" + str(time_now.hour) + "-" + str(time_now.minute) + "-" + str(time_now.second)
            traj_name = timestamp

            print(f"\nProcessing trajectory: {traj_name}")
            print(f"Skills: {skill_sequence}")
            
            # Convert skill strings to action integers
            try:
                action_sequence = self.convert_skill_seq_to_actions(skill_sequence)
                print(f"Actions: {action_sequence}")
                
                # Collect trajectory
                trajectory_data, skill_wrapper_data = self.collect_trajectory(action_sequence, traj_name)
                all_results[traj_name] = {
                    'trajectory_data': trajectory_data,
                    'skill_wrapper_data': skill_wrapper_data
                }
                
                print(f"Completed {traj_name}: {len(trajectory_data['seq'])} steps")
                
            except ValueError as e:
                print(f"Error processing {traj_name}: {e}")
                continue
        
        return all_results


def main():
    collector = FrankaTrajectoryCollector()
    
    # Collect all trajectories from trajs_to_use.yaml
    print("Loading trajectories from trajs_to_use.yaml...")
    results = collector.collect_all_trajectories()
    
    print(f"\nTrajectory collection completed!")
    print(f"Processed {len(results)} trajectories")
    
    for traj_name, result in results.items():
        traj_data = result['trajectory_data']
        print(f"\n{traj_name}:")
        print(f"  Timestamp: {traj_data['time_stamp']}")
        print(f"  Steps: {len(traj_data['seq'])}")
        for i, step in enumerate(traj_data['seq']):
            print(f"    Step {i+1}: {step['action']} -> Success: {step['success']}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--traj_yaml", type=str, default="trajs_to_use.yaml")
    argparser.add_argument("--output_dir", type=str, default="output")
    argparser.add_argument("--run_idx", type=int, default=0, help="index of the run that produce the best operators.")
    argparser.add_argument("--iter_idx", type=int, help="index of iter run the full refinement and proposal loop.")
    argparser.add_argument("--baseline", type=str, choices=["skillwrapper", "oracle_predicates"], default="skillwrapper", help="which baseline to run")
    args = argparser.parse_args()
    main()