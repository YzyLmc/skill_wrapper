import yaml
import json
import os
import cv2
import numpy as np
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datetime import datetime
from dorfl_env import DorflEnv
from src.data_structure import Skill



class DorflTrajectoryCollector:
    def __init__(self):
        self.env = DorflEnv()
        self.yaml_domain_path = "../task_config/dorfl.yaml"
        self.task_config = self.load_yaml(self.yaml_domain_path)
        
        # Action mapping from convert_dorfl_data.py
        self.action_to_skill = {
            0: ["LeftArmPick", "Robot", "PeanutButterJar"],    # grasp_pb_jar
            1: ["Open", "Robot", "PeanutButterJar"],           # open_pb_jar  
            2: ["RightArmPick", "Robot", "Knife"],             # grasp_knife1
            3: ["Scoop", "Robot", "Knife", "PeanutButterJar"], # scoop
            4: ["Spread", "Robot", "Knife", "Bread"],          # spread
            5: ["Drop", "Robot", "Knife"]                      # drop_knife1
        }
        
        # Skill string to action mapping for trajs_to_use.yaml
        self.skill_to_action = {
            "LeftArmPick(Robot, PeanutButterJar)": 0,
            "Open(Robot, PeanutButterJar)": 1,
            "RightArmPick(Robot, Knife)": 2,
            "Scoop(Robot, Knife, PeanutButterJar)": 3,
            "Spread(Robot, Knife, Bread)": 4,
            "Drop(Robot, Knife)": 5
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
        for skill_str in skill_sequence:
            if skill_str in self.skill_to_action:
                actions.append(self.skill_to_action[skill_str])
            else:
                raise ValueError(f"Unknown skill: {skill_str}")
        return actions
    
    def collect_trajectory(self, action_sequence, traj_name=None):
        """
        Collect trajectory by executing action sequence in the environment
        
        Args:
            action_sequence: List of action integers (0-5)
            traj_name: Name for the trajectory (for subfolder)
            
        Returns:
            Dict containing trajectory data in the format expected by convert_dorfl_data.py
        """
        time_now = datetime.now()
        timestamp = str(time_now.year) + "-" + str(time_now.month) + "-" + str(time_now.day) + "-" + str(time_now.hour) + "-" + str(time_now.minute) + "-" + str(time_now.second)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_folder = "output_trajs"
        # if traj_name:
        #     output_dir = f"{data_folder}/{traj_name}"
        # else:
        #     output_dir = f"{data_folder}/{timestamp}"
        output_dir = f"{data_folder}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Reset environment and get initial state
        obs, info = self.env.reset()
        
        trajectory_data = {
            "time_stamp": timestamp,
            "seq": []
        }
        
        # Save initial state image
        pre_img_name = f"0.jpg"
        self.save_image(obs, output_dir, pre_img_name)
        
        # Execute each action in the sequence
        for i, action in enumerate(action_sequence):
            if action not in self.action_to_skill:
                raise ValueError(f"Invalid action: {action}. Must be 0-5.")
            
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
        
        # # Save trajectory data as JSON
        # traj_json_path = os.path.join(output_dir, "traj.json")
        # with open(traj_json_path, 'w') as f:
        #     json.dump(trajectory_data, f, indent=2)
        
        # Generate skill wrapper format YAML
        skill_wrapper_data = self.convert_to_skill_wrapper_format(trajectory_data, output_dir)
        breakpoint()
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
            0: "grasp_pb_jar",
            1: "open_jar", 
            2: "grasp_knife1",
            3: "scoop",
            4: "spread_second_time",
            5: "drop_knife1"
        }
        return action_names[action]
    
    def convert_to_skill_wrapper_format(self, trajectory_data, output_dir):
        """
        Convert trajectory data to skill wrapper format matching convert_dorfl_data.py output
        """
        timestamp = trajectory_data["time_stamp"]
        seq = trajectory_data["seq"]
        
        skill_wrapper_result = {}
        
        # Initial state (step 0)
        skill_wrapper_result[0] = {
            "image": f"transitions/{timestamp}/0.png",
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
            
            skill_wrapper_result[i + 1] = {
                "skill": skill,
                "image": f"transitions/{timestamp}/{i+1}.png", 
                "success": traj["success"]
            }
        
        return {timestamp: skill_wrapper_result}
    
    def get_action_int_from_name(self, action_name):
        """Convert action name back to integer"""
        name_to_int = {
            "grasp_pb_jar": 0,
            "open_jar": 1,
            "grasp_knife1": 2, 
            "scoop": 3,
            "spread_second_time": 4,
            "drop_knife1": 5
        }
        return name_to_int[action_name]


    def collect_all_trajectories(self):
        """
        Load trajectories from trajs_to_use.yaml and collect data for each one
        """
        trajectories = self.load_trajectories_from_yaml(args.traj_yaml)
        breakpoint()
        all_results = {}
        
        for traj_name, traj_data in trajectories.items():
            print(f"\nProcessing trajectory: {traj_name}")
            skill_sequence = traj_data['seq']
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
    collector = DorflTrajectoryCollector()
    
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
    args = argparser.parse_args()
    main()
