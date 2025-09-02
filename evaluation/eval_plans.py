"For all problems, run the result skill sequence, return Done if the goal state is reached, and log them."
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append(f".") # if you run this script from the root directory
from robotouille.robotouille.robotouille_env import create_robotouille_env
from robotouille.utils.helper_functions import save_to_file, load_from_file
from robotouille.agents import NAME_TO_AGENT

@hydra.main(version_base=None, config_path="../robotouille/conf", config_name="test_config")
def main(cfg: DictConfig):
    # list all directories under problem_dir
    for root, dirs, files in os.walk(cfg.problem_dir):
        for d in dirs:
            problem_dir = os.path.join(root, d)
            root_components = root.split(os.sep)[-2:]
            root_path = os.sep.join(root_components)
            environment_name = os.path.join(root_path, d, "problem")
            
def get_successfulness(env, plan) -> bool:
    pass

if __name__ == "__main__":
    """
    Arguments & Default values:
    problem_dir: The directory containing problem files, either test, seen, or unseen.
    save_fpath: File path for saving evaluation results.
    """
    main()