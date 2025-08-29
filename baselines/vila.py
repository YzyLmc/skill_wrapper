"""
Reproduction of vila baseline: given image of current state and final state and the action history, return a next skill
Franka, Dorfl, and Spot need manual input of the next image after executing the proposed skill
Burger is automated.
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
# from robotouille import run_skill_sequence
# from robotouille.run_skill_sequence import exec_and_record
from src.utils import GPT4, load_from_file, save_to_file, setup_logging, get_save_fpath
from src.data_structure import Skill

@hydra.main(version_base=None, config_path="../robotouille/conf", config_name="test_config")
def main(cfg: DictConfig):
    model = GPT4(engine=cfg.model)
    prompt = load_from_file("prompts/vila_prompt.yaml")[cfg.env]

    task_config = load_from_file(f"task_config/{cfg.env}.yaml")
    log_dir = f"results/baselines/vila/{task_config['env']}/log"
    setup_logging(log_dir, task_config["env"])

    if cfg.env == "dorfl":
        prompt = prompt.replace("<robot_description>", "a robot with two arms")
    elif cfg.env == "spot":
        prompt = prompt.replace("<robot_description>", "a quadruped robot with a single arm")
    elif cfg.env == "franka":
        prompt = prompt.replace("<robot_description>", "a single-armed robot mounted on a table")
    elif cfg.env == "burger":
        breakpoint()
        kwcfg = OmegaConf.to_container(cfg.game, resolve=True)
        environment_name = kwcfg.pop('environment_name')
        prompt = prompt.replace("<robot_description>", "a kitchen robot with a single arm and a torso")

    # -- let's formulate the prompt to include the skills and objects for the robot:
    skills = [str(task_config["skills"][P]) for P in task_config["skills"]]
    skills = [f"{sk+1}. {skills[sk]}" for sk in range(len(skills))]
    prompt = prompt.replace("<skills>", "\n".join(skills))

    objects = [f"- {O}: {task_config['objects'][O]['types']}" for O in task_config["objects"]]
    prompt = prompt.replace("<objects>", "\n".join(objects))

    # -- we will keep track of all actions proposed by
    skill_sequence = []

    current_img = cfg.init_img
    goal_img = cfg.goal_img
    while True:

        new_prompt = str(prompt)

        if len(skill_sequence):
            new_prompt += f" Your last set of actions were:\n"
            for y in range(len(skill_sequence)):
                new_prompt += f"{y+1}. {skill_sequence[y]}\n"
        resp = model.generate_multimodal(new_prompt, imgs=[current_img, goal_img])
        print(resp[0])

        if "impossible" in resp[0].lower():
            logging.info("impossible")
            return ["impossible"]
        skill_string = resp[0].strip().split('\n\n')[1].split('\n')[0].strip()
        if "done" in skill_string.lower():
            logging.info("done")
            break

        skill = Skill.from_string(skill_string)
        logging.info(f"Proposed skill: {str(skill)}")
        skill_sequence.append(skill)

        if cfg.env == "burger":
            last_img_path = run_burger(environment_name, skill_sequence, cfg, **kwcfg)
            # find the last image in the tmp_dir
            next_img = last_img_path
        else:
            next_img = input("Enter path to the current image (or type 'done' to finish): ").strip()

            if not os.path.exists(next_img):
                logging("Image path does not exist. Try again.")
                continue
            logging.info(f"Next image: {next_img}")

        current_img = next_img

    save_results(skill_sequence, cfg)
    logging.info(f"{[str(s) for s in skill_sequence]}")

def run_burger(environment_name, skill_sequence, cfg, **kwcfg):
    "Take in skill sequence and execute them, save files to tmp_dir"
    img_save_path = robotouille.run_skill_sequence.exec_and_record(environment_name, skill_sequence, cfg.tmp_dir, **kwcfg)
    return img_save_path


def save_results(plan, cfg, save_dir="results/baselines/vila/"):
    save_dir = f"{save_dir}/{cfg.env}/plans/"
    os.makedirs(save_dir, exist_ok=True)
    task_dir = os.path.dirname(cfg.init_img).split('/')[-1]
    save_path = get_save_fpath(save_dir, f"plan_{task_dir}", "yaml")
    save_to_file({"plan": plan}, save_path)
    print(f"Plan saved to {save_path}")


if __name__ == "__main__":
    """
    Arguments & Default values:
    env: dorfl
    model: gpt-4o
    max_steps: 10
    init_img: null
    goal_img: null
    tmp_dir: tmp/
    """
    main()