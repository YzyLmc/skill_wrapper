"For all problems, run the result skill sequence, return Done if the goal state is reached, and log them."
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append(f".") # if you run this script from the root directory
import robotouille
from robotouille.robotouille.robotouille_env import create_robotouille_env
from robotouille.utils.helper_functions import save_to_file, load_from_file
from robotouille.agents import NAME_TO_AGENT

@hydra.main(version_base=None, config_path="../robotouille/conf", config_name="test_config")
def main(cfg: DictConfig):
    # list all directories under problem_dir
    results = {}
    for root, dirs, files in os.walk(cfg.problem_dir):
        for d in dirs:
            problem_dir = os.path.join(root, d)
            root_components = root.split(os.sep)[-2:]
            root_path = os.sep.join(root_components)

            environment_name = os.path.join(root_path, d, "problem")
            env = create_robotouille_env(environment_name, cfg.game.seed)

            plan_fpath = os.path.join("results", cfg.baseline, cfg.env, "plans", f"{d}.json")
            suc = get_successfulness(env, load_from_file(plan_fpath))
            results[d] = suc

    save_to_file(results, os.path.join("results", cfg.baseline, cfg.env, f"{cfg.env}_{root_components[-1]}_results.json"))

def get_successfulness(env, plan) -> bool:
    # run the plan and return true if the goal state is reached at the end of the plan
    skill_manager = robotouille.skills.SkillManager(env)
    for i, skill in enumerate(plan):
        suc = skill_manager.execute_skill(skill)
        # Ugly way of getting "done"
        wait = [a for a in skill_manager.env.current_state.get_valid_actions_and_str()[0] if a[0].name == "wait"][0]
        _, _, done, _ = env.step([])
    return done

if __name__ == "__main__":
    """
    Arguments & Default values:
    problem_dir: The directory containing problem files, either test, seen, or unseen.
    baseline: Name of the baseline {"FMinvent", "oracle", "random_explore", "vila", "skillwrapper"} 
    env: Name of the environment. For now only burger is supported.
    """
    main()