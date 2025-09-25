"""
Reproduction of vila baseline: given image of current state and final state and the action history, return a next skill
Franka, Dorfl, and Spot need manual input of the next image after executing the proposed skill
Burger is automated.

example command:
    python baselines/vila.py ++env=burger +dataset=test ++max_steps=20
"""

import os
import sys
import argparse
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
sys.path.append(f".") # if you run this script from the root directory
sys.path.append("robotouille")
import robotouille
from robotouille.run_skill_sequence import exec_and_record
from src.utils import GPT4, load_from_file, save_to_file, setup_logging, get_save_fpath
from src.data_structure import Skill

@hydra.main(version_base=None, config_path="../hydra_conf", config_name="vila_config")
def main(cfg: DictConfig):
    model = GPT4(engine=cfg.model)
    prompt = load_from_file("prompts/vila_prompt.yaml")[cfg.env]

    task_config = load_from_file(f"task_config/{cfg.env}.yaml")
    log_dir = f"results/vila/{task_config['env']}/log"
    setup_logging(log_dir, task_config["env"])

    if cfg.env == "dorfl":
        prompt = prompt.replace("<robot_description>", "a robot with two arms")
    elif cfg.env == "spot":
        prompt = prompt.replace("<robot_description>", "a quadruped robot with a single arm")
    elif cfg.env == "franka":
        prompt = prompt.replace("<robot_description>", "a single-armed robot mounted on a table")
    elif cfg.env == "burger":
        kwcfg = OmegaConf.to_container(cfg.game, resolve=True)
        _ = kwcfg.pop('environment_name')
        prompt = prompt.replace("<robot_description>", "a kitchen robot with a single arm and a torso")

    # -- let's formulate the prompt to include the skills and objects for the robot:
    skills = task_config["skills"]
    skills_str = [str(skills[P]) for P in skills]
    skills_str = [f"{sk+1}. {skills_str[sk]}" for sk in range(len(skills_str))]
    prompt = prompt.replace("<skills>", "\n".join(skills_str))
    objects = task_config["objects"]
    objects_str = [f"- {O}: {objects[O]['types']}" for O in objects]
    prompt = prompt.replace("<objects>", "\n".join(objects_str))

    problem_dir = f"eval/data/{cfg.env}/{cfg.dataset}/problems/"
    result_dir = f"results/vila/{cfg.env}/plans/{cfg.dataset}/"

    logging.info(f"Loading problems from {problem_dir}")
    if not os.path.exists(problem_dir):
        raise ValueError(f"Problem directory {problem_dir} does not exist.")
    
    # Run for all problems
    for root, dirs, files in os.walk(problem_dir):
        for d in dirs: # d is the problem name

            save_path = get_save_path(result_dir, d)
            # if the plan file already exists, skip
            if os.path.exists(os.path.join(save_path, "plan.yaml")):
                logging.info(f"Problem {d} already has a plan at {save_path}, skipping...")
                continue
            else:
                logging.info(f"Solving problem {d} in {cfg.dataset}...")

            current_img = os.path.join(root, d, "init_state.jpg")
            goal_img = os.path.join(root, d, "goal_state.jpg")
            root_components = root.split(os.sep)[-3:]
            root_path = os.sep.join(root_components)
            environment_name = os.path.join(root_path, d, "problem")

            plan = []
            step = 0
            while step < cfg.max_steps:
                step += 1
                new_prompt = str(prompt)

                if len(plan):
                    new_prompt += f" \nHere is the sequence of actions you have tried to execute:\n"
                    for y in range(len(plan)):
                        new_prompt += f"{y+1}. {plan[y]}\n"
                    new_prompt += f"If you see repeatitive patterns, or you just tried the same action as the first action of your plan about to be generated, it indicates that you are stuck. You should try to propose a different skill.\n"
                    new_prompt += "Generate the step-by-step reasoning in one paragraph and the plan from current state to the goal state:"
                resp = model.generate_multimodal(new_prompt, imgs=[current_img, goal_img])
                logging.info(resp[0])

                if "impossible" in resp[0].lower():
                    logging.info("impossible")
                    plan = ["impossible"]
                    break
                skill_string = resp[0].strip().split('\n\n')[1].split('\n')[0].strip()
                if "done" in skill_string.lower():
                    logging.info("done")
                    break
                try:
                    proposed_skill = Skill.from_string(skill_string)
                    logging.info(f"Proposed skill: {str(proposed_skill)}")
                except:
                    logging.info(f"Failed to parse skill from {skill_string}, try again.")
                    step -= 1
                    continue
                # If skill arguments match the types
                primitive_skill = [s for sname, s in skills.items() if s.name == proposed_skill.name][0]
                # object types of the proposed skill should match the types of the primitive skill
                type_matched = True

                for i, obj in enumerate(proposed_skill.params):
                    if obj not in objects:
                        type_matched = False
                        break
                    if i >= len(primitive_skill.types):
                        type_matched = False
                        break
                    if not primitive_skill.types[i] in objects[obj]["types"]:
                        type_matched = False
                        break
                if not type_matched:
                    logging.info("Type mismatch. Try again.")
                    continue
                
                # try executing
                new_plan = plan + [proposed_skill]

                if cfg.env == "burger":
                    last_img_path, suc = run_burger(environment_name, new_plan, cfg, **kwcfg)
                    # find the last image in the tmp_dir
                    next_img = last_img_path
                    if suc:
                        plan = new_plan

                else:
                    next_img = input("Enter path to the current image (or type 'done' to finish): ").strip()

                    if next_img == 'done':
                        break

                    if not os.path.exists(next_img):
                        logging.info("Image path does not exist. Try again.")
                        continue
                    logging.info(f"Next image: {next_img}")

                current_img = next_img
                logging.info(f"Current plan:\n{[str(s) for s in plan]}")
                # breakpoint()
            os.makedirs(save_path, exist_ok=True)
            results = {
                "env": cfg.env,
                "plan_method": "vila",
                0: {"all_parsed_plans": [plan]} 
                }
            save_to_file(results, os.path.join(save_path, "plan.yaml"))
    # delete the cached images in tmp_dir
    if cfg.env == "burger":
        os.system(f"rm -r {cfg.tmp_dir}*")

def run_burger(environment_name, plan, cfg, **kwcfg):
    "Take in skill sequence and execute them, save files to tmp_dir"
    img_save_path = robotouille.run_skill_sequence.exec_and_record(environment_name, plan, cfg.tmp_dir, eval=True, **kwcfg)
    # split each level of the path
    img_save_path_components = img_save_path.split(os.sep)
    task_name = img_save_path_components[1]
    tasks = load_from_file(os.path.join(cfg.tmp_dir, "tasks.yaml"))
    task = tasks[task_name]
    # find the max step number in the task
    max_step_num = max([int(i) for i in task.keys()])
    suc = task[str(max_step_num)]["success"]
    return img_save_path, suc

def get_save_path(save_fpath, problem_name):
    save_path = os.path.join(save_fpath, problem_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path

# def save_results(plan, save_fpath, problem_name):
#     os.makedirs(save_fpath, exist_ok=True)
#     save_path = os.path.join(save_fpath, problem_name)
#     os.makedirs(save_path, exist_ok=True)
#     save_path = get_save_fpath(save_path, "plan", "yaml")
#     save_to_file({"plan": plan}, save_path)
#     logging.info(f"Plan saved to {save_path}")


if __name__ == "__main__":
    """
    Arguments & Default values:
    dataset: ["test", "seen", "unseen"]
    env: dorfl
    model: gpt-4o
    max_steps: 10
    tmp_dir: tmp/
    """
    main()